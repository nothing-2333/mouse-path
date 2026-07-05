import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path

from wgan_model import (
    SEQ_LEN, POINT_DIM, COND_DIM, HIDDEN, DEVICE,
    G_WEIGHT_PATH, D_WEIGHT_PATH,
    CondGenerator, Discriminator, load_model_weights,
    normalize_cond, normalize_traj
)

# ========== 训练超参 ==========
BATCH = 128
LR = 2e-4
N_CRITIC = 3
LAMBDA_GP = 10.0       # 梯度惩罚权重，标准推荐值
MAX_BOUND_W = 5.0      # 首尾约束最大权重
TOTAL_EPOCHS = 60

# 数据目录
TRAIN_FOLDER = "./data/train"
TEST_FOLDER = "./data/test"
VAL_FOLDER = "./data/val"

# ---------------------- 数据集类 ----------------------
class TrajDataset(Dataset):
    def __init__(self, data_dir):
        self.all_samples = []
        data_path = Path(data_dir)
        json_files = list(data_path.glob("*.json"))
        if len(json_files) == 0:
            raise FileNotFoundError(f"目录 {data_dir} 无json数据")
        print(f"[{data_dir}] 共{len(json_files)}个文件加载中...")
        for file in json_files:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.all_samples.extend(data)
            print(f"  加载 {file.name}，累计样本 {len(self.all_samples)}")

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        item = self.all_samples[idx]
        cond_raw = item["cond"]
        cond = torch.tensor(normalize_cond(cond_raw), dtype=torch.float32)
        traj_raw = item["traj"]
        traj_norm = normalize_traj(traj_raw)
        traj = torch.tensor(traj_norm, dtype=torch.float32)
        return cond, traj

# ---------------------- WGAN-GP 梯度惩罚 ----------------------
def compute_gradient_penalty(D, real_traj, fake_traj, cond):
    batch_size = real_traj.size(0)
    # 随机插值系数
    alpha = torch.rand(batch_size, 1, 1, device=real_traj.device)
    alpha = alpha.expand_as(real_traj)
    # 构造真假插值样本
    interpolates = alpha * real_traj + (1 - alpha) * fake_traj
    interpolates.requires_grad_(True)

    # 临时禁用 CuDNN，支持 LSTM 二阶反向传播
    with torch.backends.cudnn.flags(enabled=False):
        d_out = D(interpolates, cond)

    # 计算梯度
    gradients = torch.autograd.grad(
        outputs=d_out,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_out),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.reshape(batch_size, -1)
    # 梯度范数约束到 1，满足 1-Lipschitz 条件
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp

# ---------------------- 渐进式首尾约束权重 ----------------------
def get_bound_weight(epoch, total_epochs):
    """前 30% epoch 不加约束，先学轨迹分布；之后线性增加到最大值"""
    warmup_epochs = int(total_epochs * 0.3)
    if epoch < warmup_epochs:
        return 0.0
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return MAX_BOUND_W * min(1.0, progress)

# ---------------------- 评估函数 ----------------------
def calculate_eval_loss(G, D, loader):
    G.eval()
    D.eval()
    total_d = 0.0
    total_g = 0.0
    total_bound = 0.0
    batch_num = 0
    with torch.no_grad():
        for cond, real_traj in loader:
            cond = cond.to(DEVICE)
            real_traj = real_traj.to(DEVICE)

            score_real = D(real_traj, cond)
            fake_traj = G(cond)
            score_fake = D(fake_traj, cond)

            loss_d = torch.mean(score_fake) - torch.mean(score_real)
            loss_g = -torch.mean(score_fake)

            # 计算首尾约束损失
            start_target = cond[:, 0:2]
            end_target = cond[:, 2:4]
            fake_start = fake_traj[:, 0, :]
            fake_end = fake_traj[:, -1, :]
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

# ---------------------- 主训练流程 ----------------------
if __name__ == "__main__":
    # 初始化模型
    G = CondGenerator().to(DEVICE)
    D = Discriminator().to(DEVICE)

    # 加载权重
    load_model_weights(G, G_WEIGHT_PATH)
    load_model_weights(D, D_WEIGHT_PATH)

    # WGAN-GP 标准优化器配置
    opt_g = optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_d = optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

    # 加载数据集
    print("===== 训练集 =====")
    train_set = TrajDataset(TRAIN_FOLDER)
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    print(f"训练样本总数：{len(train_set)}\n")

    print("===== 测试集 =====")
    test_set = TrajDataset(TEST_FOLDER)
    test_loader = DataLoader(
        test_set, batch_size=BATCH, shuffle=False,
        num_workers=2, pin_memory=True, persistent_workers=True
    )
    print(f"测试样本总数：{len(test_set)}\n")

    print("===== 验证集 =====")
    val_set = TrajDataset(VAL_FOLDER)
    val_loader = DataLoader(
        val_set, batch_size=BATCH, shuffle=False,
        num_workers=2, pin_memory=True, persistent_workers=True
    )
    print(f"验证样本总数：{len(val_set)}\n")

    best_w_distance = float("inf")

    for epoch in range(TOTAL_EPOCHS):
        bound_w = get_bound_weight(epoch, TOTAL_EPOCHS)

        for idx, (cond_in, real_traj) in enumerate(train_loader):
            cond_in = cond_in.to(DEVICE)
            real_traj = real_traj.to(DEVICE)

            # ========== 多次更新判别器（Critic） ==========
            for _ in range(N_CRITIC):
                opt_d.zero_grad()
                s_real = D(real_traj, cond_in)
                fake_t = G(cond_in).detach()
                s_fake = D(fake_t, cond_in)

                # 计算梯度惩罚
                gp = compute_gradient_penalty(D, real_traj, fake_t, cond_in)
                # WGAN-GP 判别器损失
                loss_d = torch.mean(s_fake) - torch.mean(s_real) + LAMBDA_GP * gp

                loss_d.backward()
                opt_d.step()

            # ========== 更新生成器 ==========
            opt_g.zero_grad()
            fake_t = G(cond_in)
            s_fake = D(fake_t, cond_in)
            loss_g = -torch.mean(s_fake)

            # 首尾点约束
            start_target = cond_in[:, 0:2]
            end_target = cond_in[:, 2:4]
            fake_start = fake_t[:, 0, :]
            fake_end = fake_t[:, -1, :]
            loss_start = torch.mean((fake_start - start_target) ** 2)
            loss_end = torch.mean((fake_end - end_target) ** 2)
            loss_bound = loss_start + loss_end

            loss_g_total = loss_g + bound_w * loss_bound
            loss_g_total.backward()
            opt_g.step()

            # 打印日志
            print(
                f"[Train] Epoch {epoch:2d} Batch {idx:3d}"
                f" | Loss_D:{loss_d.item():.4f}"
                f" | Loss_G:{loss_g.item():.4f}"
                f" | BoundLoss:{(bound_w * loss_bound).item():.4f}"
                f" | GP:{gp.item():.4f}"
                f" | BoundW:{bound_w:.2f}"
            )

        # ========== 每轮 epoch 测试集评估 ==========
        test_d, test_g, test_bound = calculate_eval_loss(G, D, test_loader)
        print(f"\n---------- Epoch {epoch} 测试集平均损失 ----------")
        print(f"Test Loss_D = {test_d:.4f} | Test Loss_G = {test_g:.4f} | Test Bound = {test_bound:.4f}")

        # 保存当前 epoch 权重
        save_weights(G, G_WEIGHT_PATH)
        save_weights(D, D_WEIGHT_PATH)
        print(f"Epoch {epoch} 权重已保存\n")

    # ========== 全部训练完成，验证集最终评估 ==========
    print("==================== 全部训练结束，验证集最终评估 ====================")
    val_d, val_g, val_bound = calculate_eval_loss(G, D, val_loader)
    print(f"Val Loss_D = {val_d:.4f} | Val Loss_G = {val_g:.4f} | Val Bound = {val_bound:.4f}")
    print("======================================================================")