import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import os

# ========== 全局固定超参 ==========
SEQ_LEN = 1000
POINT_DIM = 2
COND_DIM = 4
HIDDEN = 96
Z_DIM = 32       # 全局噪声维度，控制生成多样性
NUM_LAYERS = 2   # LSTM 层数，小数据集 2 层足够，避免过拟合
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 权重文件路径
G_WEIGHT_PATH = "generator_wgan.pth"
D_WEIGHT_PATH = "discriminator_wgan.pth"

# 画布尺寸
CANVAS_W = 1920
CANVAS_H = 1080

# ---------------------- 归一化工具函数 ----------------------
def normalize_xy(x, y):
    x_norm = (2 * x - CANVAS_W) / CANVAS_W
    y_norm = (2 * y - CANVAS_H) / CANVAS_H
    return x_norm, y_norm

def denormalize_xy(xn, yn):
    x = (xn + 1) * CANVAS_W / 2
    y = (yn + 1) * CANVAS_H / 2
    return round(x, 2), round(y, 2)

def normalize_traj(traj_pixel_list):
    res = []
    for x, y in traj_pixel_list:
        xn, yn = normalize_xy(x, y)
        res.append([xn, yn])
    return res

def denormalize_traj(traj_norm_list):
    res = []
    for xn, yn in traj_norm_list:
        x, y = denormalize_xy(xn, yn)
        res.append([x, y])
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
        # 条件 + 全局噪声 → 投影为初始序列特征
        self.input_proj = nn.Sequential(
            nn.Linear(COND_DIM + Z_DIM, SEQ_LEN * HIDDEN),
            nn.LeakyReLU(0.2)
        )

        # LSTM：每个时间步拼接条件向量，保证长序列条件不遗忘
        self.lstm = nn.LSTM(
            input_size=HIDDEN + COND_DIM,
            hidden_size=HIDDEN,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=0.1
        )

        # 输出头：LeakyReLU 保留负值梯度，Tanh 严格约束到 [-1,1]
        self.out_fc = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN // 2, POINT_DIM),
            nn.Tanh()
        )

    def forward(self, cond):
        b_size = cond.shape[0]
        device = cond.device

        # 生成全局高斯噪声（整个序列共享，保证轨迹平滑性）
        z = torch.randn(b_size, Z_DIM, device=device)
        # 拼接条件与噪声
        z_cond = torch.cat([cond, z], dim=-1)

        # 投影为初始隐藏序列
        hidden_seq = self.input_proj(z_cond).view(b_size, SEQ_LEN, HIDDEN)
        # 条件向量广播到每个时间步，与隐藏特征拼接
        cond_expand = cond.unsqueeze(1).expand(-1, SEQ_LEN, -1)
        lstm_input = torch.cat([hidden_seq, cond_expand], dim=-1)

        # LSTM 前向传播
        lstm_out, _ = self.lstm(lstm_input)
        # 输出轨迹，范围严格 [-1, 1]
        traj = self.out_fc(lstm_out)
        return traj

# ---------------------- 判别器（条件联合判别） ----------------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # LSTM：每个时间步输入 = 轨迹点 + 条件向量（真正的条件判别）
        self.lstm = nn.LSTM(
            input_size=POINT_DIM + COND_DIM,
            hidden_size=HIDDEN,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=0.1
        )

        # 分类头：谱归一化辅助 Lipschitz 约束，配合 WGAN-GP 更稳定
        self.class_head = nn.Sequential(
            spectral_norm(nn.Linear(HIDDEN, HIDDEN // 2)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(HIDDEN // 2, 1))
        )

    def forward(self, traj_seq, cond):
        # 条件向量广播到每个时间步，与轨迹点拼接
        cond_expand = cond.unsqueeze(1).expand(-1, SEQ_LEN, -1)
        x = torch.cat([traj_seq, cond_expand], dim=-1)

        # LSTM 提取时序特征
        feat, _ = self.lstm(x)
        # 全局平均池化：聚合所有时间步信息，解决长序列遗忘问题
        feat = feat.mean(dim=1)
        # 输出真假分数（无激活，WGAN 要求实数输出）
        score = self.class_head(feat)
        return score

# ---------------------- 权重加载工具 ----------------------
def load_model_weights(model, weight_path):
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        print(f"成功加载权重：{weight_path}")
        return True
    else:
        print(f"未找到权重文件：{weight_path}，使用随机初始化")
        return False

# ---------------------- 推理函数 ----------------------
def generate_single_trajectory(cond_pixel_list):
    gen = CondGenerator().to(DEVICE)
    load_model_weights(gen, G_WEIGHT_PATH)
    gen.eval()

    cond_norm = normalize_cond(cond_pixel_list)
    with torch.no_grad():
        cond_tensor = torch.tensor([cond_norm], dtype=torch.float32).to(DEVICE)
        pred_norm = gen(cond_tensor)

    traj_norm = pred_norm.squeeze(0).cpu().numpy().tolist()
    traj_pixel = denormalize_traj(traj_norm)
    return traj_pixel