import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import json
from pathlib import Path
import random
import time
import os

from model import (
    POINT_DIM, COND_DIM, DEVICE,
    ROOT, G_WEIGHT_PATH, D_WEIGHT_PATH,
    CondGenerator, Discriminator, load_model_weights,
    normalize_cond, normalize_traj,
    set_canvas_size, set_max_delta_t, save_config, to_device
)

# ========== 训练超参 ==========
BATCH = 128
LR = 1e-4
N_CRITIC = 5
LAMBDA_GP = 10.0        # 梯度惩罚权重
MAX_BOUND_W = 2.0       # 首尾坐标约束最大权重
MAX_GRAD_NORM = 1.0
TOTAL_EPOCHS = 60

LOG_FILE = os.path.join(ROOT, "train_log.txt")
TRAIN_FOLDER = os.path.join(ROOT, "../data/click")

# 数据划分
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SPLIT_SEED = 42

SAFE_SCALE = 1.05

def mask_padding(traj, seq_lengths):
    """
    将轨迹中超出真实长度的padding位置批量清零, 与真实轨迹格式对齐向量化实现
    Args:
        traj: (batch, max_len, point_dim) 原始轨迹
        seq_lengths: (batch,) 每条样本的真实长度
    Returns:
        traj_masked: (batch, max_len, point_dim) padding位清零后的轨迹
    """
    batch_size, max_len, _ = traj.shape
    device = traj.device
    # 构造位置掩码: [batch, max_len, 1], 有效位置为1, padding为0
    idx = torch.arange(max_len, device=device).unsqueeze(0)  # [1, max_len]
    mask = (idx < seq_lengths.unsqueeze(1)).unsqueeze(-1).float()
    return traj * mask

# 数据集类
class TrajDataset(Dataset):
    def __init__(self, sample_list=None, data_dir=None):
        self.all_samples = []
        self.max_x = 0.0
        self.max_y = 0.0
        self.max_dt = 0.0

        if sample_list is not None:
            self.all_samples = sample_list
            for item in sample_list:
                self._update_stats(item)
        elif data_dir is not None:
            data_path = Path(data_dir)
            json_files = list(data_path.glob("*.json"))
            if len(json_files) == 0:
                raise FileNotFoundError(f"目录 {data_dir} 无json数据")

            print(f"[{data_dir}] 共{len(json_files)}个文件加载中...")
            for file in json_files:
                try:
                    with open(file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        for item in data:
                            self.all_samples.append(item)
                            self._update_stats(item)
                    print(f"  加载成功 {file.name}, 累计样本 {len(self.all_samples)}")
                except Exception as e:
                    print(f"  [警告] 跳过损坏文件 {file.name}: {str(e)}")

            if len(self.all_samples) == 0:
                raise RuntimeError("所有JSON文件均加载失败, 无有效训练数据")
        else:
            raise ValueError("必须传入 sample_list 或 data_dir 其中一个")

    def _update_stats(self, item):
        x0, y0, x1, y1 = item["cond"]
        self.max_x = max(self.max_x, x0, x1)
        self.max_y = max(self.max_y, y0, y1)
        for point in item["traj"]:
            x, y, dt = point
            self.max_x = max(self.max_x, x)
            self.max_y = max(self.max_y, y)
            self.max_dt = max(self.max_dt, dt)

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        item = self.all_samples[idx]
        cond = torch.tensor(normalize_cond(item["cond"]), dtype=torch.float32)
        traj = torch.tensor(normalize_traj(item["traj"]), dtype=torch.float32)
        seq_len = len(item["traj"])
        return cond, traj, seq_len

    def get_canvas_size(self):
        return self.max_x, self.max_y

    def get_max_dt(self):
        return self.max_dt

def collate_fn(batch):
    '''
    变长序列批次整理, DataLoader 默认要求所有样本 shape 一致, 轨迹长度不等必须手动 padding
    '''
    
    batch.sort(key=lambda x: x[2], reverse=True)
    cond_list, traj_list, len_list = zip(*batch)

    batch_size = len(batch)
    max_len = max(len_list)
    traj_batch = torch.zeros(batch_size, max_len, POINT_DIM, dtype=torch.float32)
    cond_batch = torch.stack(cond_list, dim=0)
    seq_lengths = torch.tensor(len_list, dtype=torch.long)

    for i in range(batch_size):
        traj_batch[i, :len_list[i], :] = traj_list[i]
    return cond_batch, traj_batch, seq_lengths

def compute_gradient_penalty(D, real_traj, fake_traj, cond, seq_lengths):
    '''
    WGAN-GP 梯度惩罚, 不裁剪权重, 只通过损失反向修正梯度, 训练稳定很多
    '''
    
    # 真假轨迹之间随机插值
    batch_size = real_traj.size(0)
    alpha = torch.rand(batch_size, 1, 1, device=real_traj.device)
    alpha = alpha.expand_as(real_traj)
    interpolates = alpha * real_traj + (1 - alpha) * fake_traj
    interpolates.requires_grad_(True)

    # 判别器预测插值样本打分
    with torch.backends.cudnn.flags(enabled=False):
        d_out = D(interpolates, cond, seq_lengths)

    # 求插值轨迹对打分的梯度
    gradients = torch.autograd.grad(
        outputs=d_out,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_out),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # 计算梯度惩罚: 约束梯度2-范数接近 1
    gradients = gradients.reshape(batch_size, -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp

def get_bound_weight(epoch, total_epochs):
    '''
    渐进式首尾约束权重
    '''
    
    warmup_epochs = int(total_epochs * 0.2)
    if epoch < warmup_epochs:
        return 0.0
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return MAX_BOUND_W * min(1.0, progress)

def calculate_eval_loss(G, D, loader):
    '''
    评估函数
    '''
    
    G.eval()
    D.eval()
    total_d, total_g, total_bound = 0.0, 0.0, 0.0
    batch_num = 0
    with torch.no_grad():
        for cond, real_traj, seq_lengths in loader:
            cond, real_traj, seq_lengths = to_device(cond, real_traj, seq_lengths)
            max_len = real_traj.shape[1]

            score_real = D(real_traj, cond, seq_lengths)
            # 生成轨迹后清零 padding, 与真实轨迹格式对齐
            fake_traj_full = G(cond, max_len)
            fake_traj = mask_padding(fake_traj_full, seq_lengths)
            score_fake = D(fake_traj, cond, seq_lengths)

            loss_d = torch.mean(score_fake) - torch.mean(score_real)
            loss_g = -torch.mean(score_fake)

            # 首尾坐标约束(仅xy, 取真实末尾点)
            start_target = cond[:, 0:2]
            end_target = cond[:, 2:4]
            fake_start = fake_traj[:, 0, 0:2]
            end_indices = (seq_lengths - 1).view(-1, 1, 1).expand(-1, 1, 2)
            fake_end = torch.gather(fake_traj[:, :, 0:2], 1, end_indices).squeeze(1)
            loss_bound = torch.mean((fake_start - start_target) ** 2) + \
                        torch.mean((fake_end - end_target) ** 2)

            total_d += loss_d.item()
            total_g += loss_g.item()
            total_bound += loss_bound.item()
            batch_num += 1

    avg_d = total_d / max(batch_num, 1)
    avg_g = total_g / max(batch_num, 1)
    avg_bound = total_bound / max(batch_num, 1)
    G.train()
    D.train()
    return avg_d, avg_g, avg_bound

def save_weights(model, path):
    torch.save(model.state_dict(), path)

def write_log(epoch, test_d, test_g, test_bound):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"Epoch {epoch:3d} | D_loss: {test_d:.4f} | G_loss: {test_g:.4f} | Bound: {test_bound:.4f}\n")

if __name__ == "__main__":
    # 加载全部数据并统计值域
    full_dataset = TrajDataset(data_dir=TRAIN_FOLDER)
    total_samples = len(full_dataset)
    print(f"原始数据集总样本量: {total_samples}")

    # 动态更新全局归一化参数并保存配置
    canvas_w, canvas_h = full_dataset.get_canvas_size()
    max_dt = full_dataset.get_max_dt()
    set_canvas_size(canvas_w * SAFE_SCALE, canvas_h * SAFE_SCALE)
    set_max_delta_t(max_dt * SAFE_SCALE)
    save_config()

    # 数据集划分
    random.seed(SPLIT_SEED)
    all_indices = list(range(total_samples))
    random.shuffle(all_indices)

    train_size = int(total_samples * TRAIN_RATIO)
    val_size = int(total_samples * VAL_RATIO)
    test_size = total_samples - train_size - val_size

    train_idx = all_indices[:train_size]
    val_idx = all_indices[train_size : train_size + val_size]
    test_idx = all_indices[train_size + val_size :]

    print(f"\n训练集: {len(train_idx)} | 验证集: {len(val_idx)} | 测试集: {len(test_idx)}")

    # 构造 DataLoader
    train_set = Subset(full_dataset, train_idx)
    val_set = Subset(full_dataset, val_idx)
    test_set = Subset(full_dataset, test_idx)

    train_loader = DataLoader(
        train_set, batch_size=BATCH, shuffle=True, drop_last=True,
        collate_fn=collate_fn, num_workers=4,
        pin_memory=True, prefetch_factor=2, persistent_workers=True
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH, shuffle=False,
        collate_fn=collate_fn, num_workers=2,
        pin_memory=True, persistent_workers=True
    )
    test_loader = DataLoader(
        test_set, batch_size=BATCH, shuffle=False,
        collate_fn=collate_fn, num_workers=2,
        pin_memory=True, persistent_workers=True
    )

    # \初始化模型与优化器
    G = CondGenerator().to(DEVICE)
    D = Discriminator().to(DEVICE)
    load_model_weights(G, G_WEIGHT_PATH)
    load_model_weights(D, D_WEIGHT_PATH)

    opt_g = optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_d = optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

    # 清空历史日志
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("")

    # 训练循环
    start_time = time.time()

    for epoch in range(TOTAL_EPOCHS):
        bound_w = get_bound_weight(epoch, TOTAL_EPOCHS)

        for idx, (cond_in, real_traj, seq_lengths) in enumerate(train_loader):
            cond_in, real_traj, seq_lengths = to_device(cond_in, real_traj, seq_lengths)
            max_len = real_traj.shape[1]

            # ========== 更新判别器 N 次 ==========
            for _ in range(N_CRITIC):
                opt_d.zero_grad()
                s_real = D(real_traj, cond_in, seq_lengths)

                # 生成轨迹后清零padding, 再 detach 送入判别器
                fake_t_full = G(cond_in, max_len)
                fake_t = mask_padding(fake_t_full, seq_lengths).detach()
                s_fake = D(fake_t, cond_in, seq_lengths)

                gp = compute_gradient_penalty(D, real_traj, fake_t, cond_in, seq_lengths)
                loss_d = torch.mean(s_fake) - torch.mean(s_real) + LAMBDA_GP * gp

                loss_d.backward()
                nn.utils.clip_grad_norm_(D.parameters(), MAX_GRAD_NORM)
                opt_d.step()

            # ========== 更新生成器 ==========
            opt_g.zero_grad()
            fake_t_full = G(cond_in, max_len)
            fake_t = mask_padding(fake_t_full, seq_lengths)
            s_fake = D(fake_t, cond_in, seq_lengths)
            loss_g = -torch.mean(s_fake)

            # 首尾约束(基于mask后的轨迹, 首尾均在有效区间内, 结果一致)
            start_target = cond_in[:, 0:2]
            end_target = cond_in[:, 2:4]
            fake_start = fake_t[:, 0, 0:2]
            end_indices = (seq_lengths - 1).view(-1, 1, 1).expand(-1, 1, 2)
            fake_end = torch.gather(fake_t[:, :, 0:2], 1, end_indices).squeeze(1)
            loss_bound = torch.mean((fake_start - start_target) ** 2) + \
                        torch.mean((fake_end - end_target) ** 2)

            loss_g_total = loss_g + bound_w * loss_bound
            loss_g_total.backward()
            nn.utils.clip_grad_norm_(G.parameters(), MAX_GRAD_NORM)
            opt_g.step()

            # 打印训练进度
            if idx % 3 == 0:
                print(
                    f"Epoch {epoch:2d} Batch {idx:3d} | D:{loss_d.item():.4f} | "
                    f"G:{loss_g.item():.4f} | Bound:{(bound_w*loss_bound).item():.4f} | "
                    f"GP:{gp.item():.4f} | W:{bound_w:.2f}"
                )

        # Epoch 结束评估
        test_d, test_g, test_bound = calculate_eval_loss(G, D, test_loader)
        write_log(epoch, test_d, test_g, test_bound)

        print(f"\n---------- Epoch {epoch} 测试集 ----------")
        print(f"D_loss: {test_d:.4f} | G_loss: {test_g:.4f} | Bound_loss: {test_bound:.4f}")

        # 保存权重
        save_weights(G, G_WEIGHT_PATH)
        save_weights(D, D_WEIGHT_PATH)
        print(f"权重已保存, 耗时: {time.time()-start_time:.1f}s\n")

    # 最终验证集评估
    print("==================== 训练完成, 验证集最终评估 ====================")
    val_d, val_g, val_bound = calculate_eval_loss(G, D, val_loader)
    print(f"Val D_loss: {val_d:.4f} | Val G_loss: {val_g:.4f} | Val Bound: {val_bound:.4f}")
    print(f"总训练耗时: {time.time()-start_time:.1f}s")
    print("================================================================")