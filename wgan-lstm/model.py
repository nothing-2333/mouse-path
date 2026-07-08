import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math
import os
import json

# ========== 全局超参(训练时由数据集自动统计覆盖) ==========
POINT_DIM = 3       # 轨迹点维度: x, y, Δt
COND_DIM = 4        # 条件维度: 起点(x0,y0) + 终点(x1,y1)
HIDDEN = 96
Z_DIM = 32          # 全局噪声维度
NUM_LAYERS = 2      # LSTM 层数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 画布尺寸(默认值, 数据集统计后自动覆盖)
CANVAS_W = 1920.0
CANVAS_H = 1080.0
# Δt 归一化上限(默认值, 数据集统计后自动覆盖)
MAX_DELTA_T = 50.0
# 首点dt均值(数据集统计后自动覆盖, 用于自回归初始化)
FIRST_DT_MEAN = 0.0

ROOT = os.path.dirname(os.path.abspath(__file__))
# 权重文件路径
G_WEIGHT_PATH = os.path.join(ROOT, "generator_wgan.pth")
D_WEIGHT_PATH = os.path.join(ROOT, "discriminator_wgan.pth")
# 配置文件路径
CONFIG_PATH = os.path.join(ROOT, "model_config.json")

# 全局配置动态设置
def set_canvas_size(width, height):
    """动态设置画布尺寸, 由数据集统计后调用"""
    global CANVAS_W, CANVAS_H
    CANVAS_W = float(width)
    CANVAS_H = float(height)
    print(f"[配置更新] 画布尺寸已设为: {CANVAS_W:.1f} x {CANVAS_H:.1f}")

def set_max_delta_t(value):
    """动态设置Δt归一化上限, 由数据集统计后调用"""
    global MAX_DELTA_T
    MAX_DELTA_T = float(value)
    print(f"[配置更新] Δt归一化上限已设为: {MAX_DELTA_T:.3f}")
    
def set_first_dt_mean(value):
    """动态设置首点dt均值, 由数据集统计后调用"""
    global FIRST_DT_MEAN
    FIRST_DT_MEAN = float(value)
    print(f"[配置更新] 首点dt均值已设为: {FIRST_DT_MEAN:.3f}")

def save_config():
    """保存当前所有归一化配置到文件, 供推理时加载对齐"""
    config = {
        "CANVAS_W": CANVAS_W,
        "CANVAS_H": CANVAS_H,
        "MAX_DELTA_T": MAX_DELTA_T,
        "FIRST_DT_MEAN": FIRST_DT_MEAN,
        "POINT_DIM": POINT_DIM,
        "COND_DIM": COND_DIM,
        "HIDDEN": HIDDEN,
        "Z_DIM": Z_DIM,
        "NUM_LAYERS": NUM_LAYERS
    }
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"[配置保存] 归一化参数已写入 {CONFIG_PATH}")

def load_config():
    """从文件加载归一化配置, 保证推理与训练参数完全一致"""
    if not os.path.exists(CONFIG_PATH):
        print(f"[警告] 未找到配置文件 {CONFIG_PATH}, 使用默认参数")
        return False
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    global CANVAS_W, CANVAS_H, MAX_DELTA_T, FIRST_DT_MEAN
    CANVAS_W = float(config["CANVAS_W"])
    CANVAS_H = float(config["CANVAS_H"])
    MAX_DELTA_T = float(config["MAX_DELTA_T"])
    FIRST_DT_MEAN = float(config["FIRST_DT_MEAN"]) 
    print(f"[配置加载] 画布 {CANVAS_W:.1f}x{CANVAS_H:.1f} | Δt上限 {MAX_DELTA_T:.3f} | 首点dt均值 {FIRST_DT_MEAN:.3f}")
    return True

def to_device(*tensors):
    """批量迁移张量到设备"""
    return [t.to(DEVICE) for t in tensors]

# 归一化工具函数
def normalize_xy(x, y):
    x_norm = (2 * x - CANVAS_W) / CANVAS_W
    y_norm = (2 * y - CANVAS_H) / CANVAS_H
    return x_norm, y_norm

def denormalize_xy(xn, yn):
    x = (xn + 1) * CANVAS_W / 2
    y = (yn + 1) * CANVAS_H / 2
    return round(x, 2), round(y, 2)

def normalize_dt(dt):
    return (2.0 * dt / MAX_DELTA_T) - 1.0

def denormalize_dt(dtn):
    dt = (dtn + 1.0) * MAX_DELTA_T / 2.0
    return round(dt, 3)

def normalize_traj(traj_pixel_list):
    """输入: [[x, y, dt], ...] 像素级轨迹 → 输出归一化轨迹"""
    res = []
    for x, y, dt in traj_pixel_list:
        xn, yn = normalize_xy(x, y)
        dtn = normalize_dt(dt)
        res.append([xn, yn, dtn])
    return res

def denormalize_traj(traj_norm_list):
    """输入: 归一化轨迹 → 输出像素级轨迹"""
    res = []
    for xn, yn, dtn in traj_norm_list:
        x, y = denormalize_xy(xn, yn)
        dt = denormalize_dt(dtn)
        res.append([x, y, dt])
    return res

def normalize_cond(cond_pixel):
    x0, y0, x1, y1 = cond_pixel
    x0n, y0n = normalize_xy(x0, y0)
    x1n, y1n = normalize_xy(x1, y1)
    return [x0n, y0n, x1n, y1n]

# 生成器
class CondGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(COND_DIM + Z_DIM, HIDDEN),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1)
        )
        
        self.lstm = nn.LSTM(
            input_size=POINT_DIM + HIDDEN,
            hidden_size=HIDDEN,
            num_layers=NUM_LAYERS,
            batch_first=True
        )
        # 预测下一个轨迹点
        self.out_fc = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN // 2, POINT_DIM),
            nn.Tanh()
        )

    def forward(self, cond, seq_len, real_traj=None):
        """
        Args:
            cond: [B, COND_DIM] 归一化起止条件
            seq_len: int 生成序列总长度
            real_traj: [B, seq_len, POINT_DIM] 可选, 传入则启用 Teacher Forcing 并行训练模式
        Returns:
            traj: [B, seq_len, POINT_DIM] 完整生成轨迹
        """
        b_size = cond.shape[0]
        device = cond.device

        # 1. 全局特征: 噪声 + 条件
        z = torch.randn(b_size, Z_DIM, device=device)
        z_cond = torch.cat([cond, z], dim=-1)
        global_feat = self.input_proj(z_cond)  # [B, HIDDEN]

        if real_traj is not None:
            # Teacher Forcing 模式
            # 输入: 前 seq_len-1 个真实点, 预测后 seq_len-1 个点
            input_traj = real_traj[:, :-1, :]  # [B, L-1, 3]
            # 全局特征扩展到每个时间步
            global_expand = global_feat.unsqueeze(1).expand(-1, seq_len - 1, -1)
            lstm_input = torch.cat([input_traj, global_expand], dim=-1)  # [B, L-1, 3+HIDDEN]
            
            # LSTM 一次性前向, 完全并行
            lstm_out, _ = self.lstm(lstm_input)  # [B, L-1, HIDDEN]
            pred_points = self.out_fc(lstm_out)  # [B, L-1, 3]
            
            # 拼接起点, 直接使用真实轨迹首点
            start_point = real_traj[:, 0:1, :]  # [B, 1, 3]
            traj = torch.cat([start_point, pred_points], dim=1)  # [B, L, 3]
        
        else:
            # 推理/生成假样本用
            # 首点xy强制对齐条件起点
            start_xy = cond[:, 0:2]  # [B, 2]
            # dt初始值根据真实数据首点dt分布调整
            dt_mean_norm = normalize_dt(FIRST_DT_MEAN)
            dt_noise = torch.randn(b_size, 1, device=device) * 0.05
            dt_init = dt_mean_norm + dt_noise
            # 裁剪到归一化合法范围 [-1, 1]，避免越界
            dt_init = torch.clamp(dt_init, -1.0, 1.0)
            prev_point = torch.cat([start_xy, dt_init], dim=-1)  # [B, 3]

            # 初始化 LSTM 隐藏状态
            h = torch.zeros(NUM_LAYERS, b_size, HIDDEN, device=device)
            c = torch.zeros(NUM_LAYERS, b_size, HIDDEN, device=device)

            # 逐点自回归生成
            traj_list = [prev_point.unsqueeze(1)]
            for _ in range(1, seq_len):
                lstm_in = torch.cat([prev_point, global_feat], dim=-1).unsqueeze(1)  # [B, 1, 3+96]
                output, (h, c) = self.lstm(lstm_in, (h, c))
                next_point = self.out_fc(output.squeeze(1))  # [B, 3]
                traj_list.append(next_point.unsqueeze(1))
                prev_point = next_point

            traj = torch.cat(traj_list, dim=1)  # [B, seq_len, 3]
        
        return traj

# 判别器
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # LSTM 同步移除 dropout, 保持训练/推理行为一致
        self.lstm = nn.LSTM(
            input_size=POINT_DIM + COND_DIM,
            hidden_size=HIDDEN,
            num_layers=NUM_LAYERS,
            batch_first=True
        )
        self.class_head = nn.Sequential(
            spectral_norm(nn.Linear(HIDDEN, HIDDEN // 2)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            spectral_norm(nn.Linear(HIDDEN // 2, 1))
        )

    def forward(self, traj_seq, cond, seq_lengths):
        batch_size, max_len, _ = traj_seq.shape
        device = traj_seq.device

        # 条件复制到每一步时序
        cond_expand = cond.unsqueeze(1).expand(-1, max_len, -1)
        x = torch.cat([traj_seq, cond_expand], dim=-1)

        # 打包变长序列, 仅有效位置参与LSTM计算
        packed_x = pack_padded_sequence(
            x, seq_lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        packed_feat, _ = self.lstm(packed_x)
        feat, _ = pad_packed_sequence(packed_feat, batch_first=True, total_length=max_len)

        # 取每个样本最后一个有效时间步的隐藏状态
        end_indices = (seq_lengths - 1).view(-1, 1, 1).expand(-1, 1, HIDDEN)
        last_feat = torch.gather(feat, 1, end_indices).squeeze(1)  # [B, HIDDEN]

        score = self.class_head(last_feat)
        return score

# 权重工具 
def load_model_weights(model, weight_path):
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        print(f"成功加载权重: {weight_path}")
        return True
    else:
        print(f"未找到权重文件: {weight_path}, 使用随机初始化")
        return False

# 推理函数
def generate_single_trajectory(cond_pixel_list, seq_len):
    if not isinstance(seq_len, int) or seq_len < 2:
        raise ValueError(f"序列长度必须≥2, 当前输入: {seq_len}")

    gen = CondGenerator().to(DEVICE)
    load_model_weights(gen, G_WEIGHT_PATH)
    gen.eval()

    cond_norm = normalize_cond(cond_pixel_list)
    with torch.no_grad():
        cond_tensor = torch.tensor([cond_norm], dtype=torch.float32).to(DEVICE)
        pred_norm = gen(cond_tensor, seq_len)

    traj_norm = pred_norm.squeeze(0).cpu().numpy().tolist()
    traj_pixel = denormalize_traj(traj_norm)
    return traj_pixel