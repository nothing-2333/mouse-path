import json
import random
from wgan_model import generate_single_trajectory, CANVAS_W, CANVAS_H

# ========== 生成配置 ==========
OUT_JSON = "generated_traj.json"
GEN_COUNT = 10
MARGIN = 80
MIN_DIST = 150

def random_cond_pixel():
    """随机生成像素级起终点，确保两点距离足够远"""
    while True:
        x0 = random.uniform(MARGIN, CANVAS_W - MARGIN)
        y0 = random.uniform(MARGIN, CANVAS_H - MARGIN)
        x1 = random.uniform(MARGIN, CANVAS_W - MARGIN)
        y1 = random.uniform(MARGIN, CANVAS_H - MARGIN)
        dx = x1 - x0
        dy = y1 - y0
        dist = (dx**2 + dy**2) ** 0.5
        if dist >= MIN_DIST:
            return [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)]

if __name__ == "__main__":
    output_samples = []
    print(f"开始随机生成 {GEN_COUNT} 条像素轨迹...")

    for i in range(GEN_COUNT):
        cond = random_cond_pixel()
        traj_pixel = generate_single_trajectory(cond)
        sample = {
            "cond": cond,
            "traj": traj_pixel
        }
        output_samples.append(sample)
        print(f"第{i+1}条，起点终点：{cond}")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output_samples, f, indent=2, ensure_ascii=False)

    print(f"\n生成完成，文件输出：{OUT_JSON}")