import os
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

def process_dataset(background_color="black", batch_size=20):
    """
    对 SYSU-MM01 数据集进行实例分割，保留人及其携带物（背包、手提包、行李箱等），
    每个类别仅保留置信度最高的一个实例。
    不同类别使用不同置信度阈值：
        - person(0): conf >= 0.0
        - 其他类别(24,25,26,27,28): conf >= 0.1
    分割结果保存到指定输出路径。
    """
    input_root = Path(r"E:\python\projects\PedestrianSegmentation\SYSU-MM01-pure")
    output_root = Path(r"E:\python\projects\PedestrianSegmentation\SYSU-MM01-Pedestrian")

    # 加载模型
    model = YOLO(r"./yolo11x-seg.pt")

    # 背景颜色：黑色=0，白色=255
    bg_value = 0 if background_color == "black" else 255

    # 需要检测的类别ID（COCO对应ID）
    target_classes = [0, 24, 25, 26, 27, 28]

    # 各类别置信度阈值
    conf_thresholds = {
        0: 0.0,   # person
        24: 0.1,  # backpack
        25: 0.1,  # umbrella
        26: 0.1,  # handbag
        27: 0.1,  # tie
        28: 0.1   # suitcase
    }

    # 收集所有图像路径
    image_paths = [Path(root) / f for root, _, files in os.walk(input_root)
                   for f in files if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    progress_bar = tqdm(range(0, len(image_paths), batch_size), desc="Processing Batches")

    for idx in progress_bar:
        batch_paths = image_paths[idx:idx + batch_size]
        images = [cv2.imread(str(p)) for p in batch_paths]

        # 过滤读取失败的图像
        valid_data = [(p, img) for p, img in zip(batch_paths, images) if img is not None]
        if not valid_data:
            continue

        paths, imgs = zip(*valid_data)

        # 统一用最小置信度（0.0）推理，后续再手动过滤
        results = model.predict(
            source=list(imgs),
            classes=target_classes,
            conf=0.0,  # 全放开
            batch=batch_size,
            device="cuda:0",
            retina_masks=True,
            verbose=False
        )

        # 逐张处理结果
        for img_path, img, result in zip(paths, imgs, results):
            relative_path = img_path.relative_to(input_root)
            output_path = output_root / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if result.masks is None or len(result.boxes.conf) == 0:
                print(f"低置信度或未检测: {relative_path}")
                cv2.imwrite(str(output_path), img)
                continue

            # 取出检测结果
            cls = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            masks = result.masks.data.cpu().numpy()

            # 创建背景图
            output = np.full_like(img, bg_value)

            # 按类别挑选置信度最高的目标
            for c in target_classes:
                indices = np.where(cls == c)[0]
                if len(indices) == 0:
                    continue

                # 根据类别应用不同置信度过滤
                valid_indices = [i for i in indices if confs[i] >= conf_thresholds[c]]
                if len(valid_indices) == 0:
                    continue

                # 在符合置信度的对象中选置信度最高的一个
                best_idx = valid_indices[np.argmax(confs[valid_indices])]
                best_mask = masks[best_idx].astype(np.uint8)

                output[best_mask > 0] = img[best_mask > 0]

            cv2.imwrite(str(output_path), output)

    print(f"\n✅ 分割完成，结果已保存至：{output_root}")



if __name__ == "__main__":

    process_dataset()