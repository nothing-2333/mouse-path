import json
import random
import model
from model import generate_single_trajectory, load_config, ROOT

# 生成配置
OUT_JSON = os.path.join(ROOT, "generated_traj.json")
GEN_COUNT = 10
MARGIN = 80
MIN_DIST = 150
MIN_SEQ_LEN = 40
MAX_SEQ_LEN = 200

def random_cond_pixel():
    """基于统计后的画布尺寸随机生成合法起终点"""
    canvas_w = model.CANVAS_W
    canvas_h = model.CANVAS_H
    while True:
        x0 = random.uniform(MARGIN, canvas_w - MARGIN)
        y0 = random.uniform(MARGIN, canvas_h - MARGIN)
        x1 = random.uniform(MARGIN, canvas_w - MARGIN)
        y1 = random.uniform(MARGIN, canvas_h - MARGIN)
        dist = ((x1-x0)**2 + (y1-y0)**2) ** 0.5
        if dist >= MIN_DIST:
            return [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)]

if __name__ == "__main__":
    # 加载训练配置, 保证归一化参数与训练一致
    load_config()
    output_samples = []
    print(f"\n开始生成 {GEN_COUNT} 条轨迹...\n")

    for i in range(GEN_COUNT):
        cond = random_cond_pixel()
        seq_len = random.randint(MIN_SEQ_LEN, MAX_SEQ_LEN)
        traj_pixel = generate_single_trajectory(cond, seq_len)
        sample = {
            "cond": cond,
            "traj": traj_pixel,
            "seq_len": seq_len
        }
        output_samples.append(sample)
        print(f"第{i+1:2d}条 | 长度:{seq_len:3d} | 起终点: {cond}")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output_samples, f, indent=2, ensure_ascii=False)

    print(f"\n生成完成, 结果已写入 {OUT_JSON}")