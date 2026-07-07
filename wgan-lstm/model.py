import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm, pack_padded_sequence, pad_packed_sequence
import math
import os
import json

# ========== 全局超参（训练时由数据集自动统计覆盖） ==========
POINT_DIM = 3       # 轨迹点维度：x, y, Δt
COND_DIM = 4        # 条件维度：起点(x0,y0) + 终点(x1,y1)
HIDDEN = 96
Z_DIM = 32          # 全局噪声维度
NUM_LAYERS = 2      # LSTM 层数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 画布尺寸（默认值，数据集统计后自动覆盖）
CANVAS_W = 1920.0
CANVAS_H = 1080.0
# Δt 归一化上限（默认值，数据集统计后自动覆盖）
MAX_DELTA_T = 50.0

# 权重文件路径
G_WEIGHT_PATH = "generator_wgan.pth"
D_WEIGHT_PATH = "discriminator_wgan.pth"
# 配置文件路径（保存画布、Δt等归一化参数）
CONFIG_PATH = "model_config.json"

# ---------------------- 全局配置动态设置 ----------------------
def set_canvas_size(width, height):
    """动态设置画布尺寸，由数据集统计后调用"""
    global CANVAS_W, CANVAS_H
    CANVAS_W = float(width)
    CANVAS_H = float(height)
    print(f"[配置更新] 画布尺寸已设为: {CANVAS_W:.1f} x {CANVAS_H:.1f}")

def set_max_delta_t(value):
    """动态设置Δt归一化上限，由数据集统计后调用"""
    global MAX_DELTA_T
    MAX_DELTA_T = float(value)
    print(f"[配置更新] Δt归一化上限已设为: {MAX_DELTA_T:.3f}")

def save_config():
    """保存当前所有归一化配置到文件，供推理时加载对齐"""
    config = {
        "CANVAS_W": CANVAS_W,
        "CANVAS_H": CANVAS_H,
        "MAX_DELTA_T": MAX_DELTA_T,
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
    """从文件加载归一化配置，保证推理与训练参数完全一致"""
    if not os.path.exists(CONFIG_PATH):
        print(f"[警告] 未找到配置文件 {CONFIG_PATH}，使用默认参数")
        return False
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    global CANVAS_W, CANVAS_H, MAX_DELTA_T
    CANVAS_W = float(config["CANVAS_W"])
    CANVAS_H = float(config["CANVAS_H"])
    MAX_DELTA_T = float(config["MAX_DELTA_T"])
    print(f"[配置加载] 画布 {CANVAS_W:.1f}x{CANVAS_H:.1f} | Δt上限 {MAX_DELTA_T:.3f}")
    return True

def to_device(*tensors):
    """批量迁移张量到设备，减少冗余代码"""
    return [t.to(DEVICE) for t in tensors]

# ---------------------- 正弦位置编码 ----------------------
class SinusoidalPositionalEncoding(nn.Module):
    """为生成器注入绝对位置信息，解决静态初始特征时序多样性不足问题"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

# ---------------------- 归一化工具函数 ----------------------
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

# ---------------------- 生成器 ----------------------
class CondGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(COND_DIM + Z_DIM, HIDDEN),
            nn.LeakyReLU(0.2)
        )
        self.pos_enc = SinusoidalPositionalEncoding(HIDDEN + COND_DIM)
        self.lstm = nn.LSTM(
            input_size=HIDDEN + COND_DIM,
            hidden_size=HIDDEN,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=0.1
        )
        self.out_fc = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN // 2, POINT_DIM),
            nn.Tanh()
        )

    def forward(self, cond, seq_len):
        b_size = cond.shape[0]
        device = cond.device

        z = torch.randn(b_size, Z_DIM, device=device)
        z_cond = torch.cat([cond, z], dim=-1)
        hidden_feat = self.input_proj(z_cond)

        hidden_seq = hidden_feat.unsqueeze(1).expand(-1, seq_len, -1)
        cond_expand = cond.unsqueeze(1).expand(-1, seq_len, -1)
        lstm_input = torch.cat([hidden_seq, cond_expand], dim=-1)

        lstm_input = self.pos_enc(lstm_input)
        lstm_out, _ = self.lstm(lstm_input)
        traj = self.out_fc(lstm_out)
        return traj

# ---------------------- 判别器 ----------------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=POINT_DIM + COND_DIM,
            hidden_size=HIDDEN,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=0.1
        )
        self.class_head = nn.Sequential(
            spectral_norm(nn.Linear(HIDDEN, HIDDEN // 2)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(HIDDEN // 2, 1))
        )

    def forward(self, traj_seq, cond, seq_lengths):
        batch_size, max_len, _ = traj_seq.shape
        device = traj_seq.device

        cond_expand = cond.unsqueeze(1).expand(-1, max_len, -1)
        x = torch.cat([traj_seq, cond_expand], dim=-1)

        packed_x = pack_padded_sequence(
            x, seq_lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        packed_feat, _ = self.lstm(packed_x)
        feat, _ = pad_packed_sequence(packed_feat, batch_first=True, total_length=max_len)

        # 掩码平均池化，屏蔽padding无效位
        mask = torch.arange(max_len, device=device).unsqueeze(0) < seq_lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).float()
        feat_sum = (feat * mask).sum(dim=1)
        feat_avg = feat_sum / seq_lengths.unsqueeze(-1).float()

        score = self.class_head(feat_avg)
        return score

# ---------------------- 权重工具 ----------------------
def load_model_weights(model, weight_path):
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        print(f"成功加载权重：{weight_path}")
        return True
    else:
        print(f"未找到权重文件：{weight_path}，使用随机初始化")
        return False

# ---------------------- 推理函数 ----------------------
def generate_single_trajectory(cond_pixel_list, seq_len):
    if not isinstance(seq_len, int) or seq_len < 2:
        raise ValueError(f"序列长度必须≥2，当前输入: {seq_len}")

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