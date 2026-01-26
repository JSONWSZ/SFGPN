"""
生成 Multi-branch Grad-CAM 并保存（按 SYSU-MM01 原始目录结构）

说明：
- 适配你的 embed_net（ResNet50 + MultiScaleModule），MultiScaleModule 在 batch 维度做 cat，
  因而 layer4 的输出在 batch 维度上有 3B 个样本（当 B=1 时即 3 个样本：origin/dilated/ppm）。
- 我们对每张输入图片（B=1）:
    1) 让 model 在 training 模式下做一次前向（no_grad）以获得 classifier logits 对应的 3 个样本的预测类 index
    2) 使用 pytorch-grad-cam，target_layers 指向 layer4[-1]，并用三个预测类作为 targets，
       计算出 3 个 Grad-CAM heatmaps（对应 origin/dilated/ppm）
    3) 将 3 个 heatmaps 做平均得到 fusion heatmap
    4) 将每个 heatmap overlay 到原图并保存到对应目录（SFGPN1..SFGPN4）
- 目录结构和命名尽量遵循你给出的格式（use 4-digit ID folder like '0333'）

依赖：
pip install grad-cam
（以及你已经具备 pytorch, torchvision, PIL 等常见库）
"""

import os
import sys
import shutil
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import transforms as T
from tqdm import tqdm

# grad-cam
# pip install grad-cam
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ---------- ========== 需要你修改的变量（把路径写死到这里） ========== ----------
# 根数据集路径（你保证 SYSU-MM01 标准目录结构）
DATA_ROOT = r"/media/ssd_2t/home/wsz/python/projects/Datasets/SYSU-MM01"
# 输出 grad-cam 图片保存根目录（会创建 camX/ID/... 结构）
OUT_ROOT_ORIGIN = r"/media/ssd_2t/home/wsz/python/projects/Datasets/SYSU-MM01-Grad-SFGPN1"  # origin branch
OUT_ROOT_DILATED = r"/media/ssd_2t/home/wsz/python/projects/Datasets/SYSU-MM01-Grad-SFGPN2"  # dilated branch
OUT_ROOT_PPM = r"/media/ssd_2t/home/wsz/python/projects/Datasets/SYSU-MM01-Grad-SFGPN3"      # ppm branch
OUT_ROOT_FUSION = r"/media/ssd_2t/home/wsz/python/projects/Datasets/SYSU-MM01-Grad-SFGPN4"   # fusion (avg of three)

# 你要可视化的测试集 ID 列表（手工填写）
# 示例：VIS/IR ID 333 则写 333
TARGET_IDs = [6]  # <-- 把你要做的 ID 写到这里（示例）
# 如果你希望只处理部分摄像头，请修改 cams_list，否则脚本会遍历 cam1..cam6
cams_list = ['cam1', 'cam2', 'cam3', 'cam4', 'cam5', 'cam6']

# checkpoint 路径（你的模型权重）
MODEL_CHECKPOINT = r"save_model/sysu_agw_p4_n6_lr_0.1_seed_0_best.t"  # 修改为你的权重路径
# model 构造时的 class number（训练时的 ID 数量）
NUM_CLASSES = 395  # 依据你训练时使用的 class 数
# image size（与你训练时 Resize 一致）
IMG_W, IMG_H = 144, 384
# 设备
DEVICE = 'cuda:3' if torch.cuda.is_available() else 'cpu'
# ---------- ========== 以上为可配置部分结束 ========== ----------


# ------------------- 一些辅助函数 -------------------
def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def is_img_file(fname):
    lower = fname.lower()
    return lower.endswith('.jpg') or lower.endswith('.png') or lower.endswith('.jpeg') or lower.endswith('.bmp')


def list_image_files_in_folder(folder):
    if not os.path.isdir(folder):
        return []
    return sorted([os.path.join(folder, f) for f in os.listdir(folder) if is_img_file(f)])


def id_to_foldername(pid):
    # SYSU 示例用了四位 ID 文件夹（例如 333 -> '0333'）
    return f"{int(pid):04d}"


def determine_modal_from_cam(cam_name):
    # SYSU 约定：cam1245为可见（visible），cam36 为红外（thermal）
    cam_lower = cam_name.lower()
    if 'cam3' in cam_lower or 'cam6' in cam_lower or cam_lower.endswith('3') or cam_lower.endswith('6'):
        return 2  # thermal
    else:
        return 1  # visible


# ------------------- 加载你的模型（embed_net） -------------------
# 假设你的 embed_net 定义在 model.py 并且可以 import
# 如果名称或路径不同，请修改下面的 import 语句
try:
    # 你的工程中应该有 model.py 并且其中定义了 embed_net
    from model import embed_net
except Exception as e:
    print("无法 import embed_net，请确保 model.py 在 PYTHONPATH 中并且定义了 embed_net 类。")
    print("错误信息：", e)
    raise

# 创建 model 实例并加载权重
print("==> 构建模型并加载权重...")
model = embed_net(NUM_CLASSES, gm_pool='off', arch='resnet50')
# 移到设备
model.to(DEVICE)
# 加载 checkpoint
if os.path.isfile(MODEL_CHECKPOINT):
    print(f"==> loading checkpoint {MODEL_CHECKPOINT}")
    ck = torch.load(MODEL_CHECKPOINT, map_location=DEVICE,weights_only=False)
    # 根据你保存 checkpoint 的格式调整
    # 你之前提供的训练/测试脚本里 checkpoint['net'] 是 state_dict
    if 'net' in ck:
        model.load_state_dict(ck['net'])
    else:
        # 有些 checkpoint 可能直接是 state_dict
        model.load_state_dict(ck)
    print("==> loaded checkpoint")
else:
    raise FileNotFoundError(f"找不到模型权重文件：{MODEL_CHECKPOINT}")

# set default eval mode (we will temporarily set train mode when we need classifier logits)
model.eval()

# target layer for Grad-CAM
# hook 在 base_resnet.base.layer4[-1]
try:
    target_layer = model.base_resnet.base.layer4[-1]
except Exception as e:
    print("尝试获取 target_layer 失败，请确认模型结构中存在 model.base_resnet.base.layer4[-1]")
    raise

# 预处理（与你训练时一致）
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# utility: convert tensor -> numpy image (0~1 float) for show_cam_on_image
def tensor_to_numpy_image(tensor):
    # tensor: (3,H,W) normalized
    t = tensor.detach().cpu().numpy()
    # unnormalize
    mean = np.array([0.485, 0.456, 0.406])[:, None, None]
    std = np.array([0.229, 0.224, 0.225])[:, None, None]
    img = (t * std + mean)
    img = np.transpose(img, (1, 2, 0))  # H W C
    img = np.clip(img, 0, 1)
    return img  # float32 in [0,1]

# ------------------- 创建输出目录（保留 SYSU 结构） -------------------
for root in [OUT_ROOT_ORIGIN, OUT_ROOT_DILATED, OUT_ROOT_PPM, OUT_ROOT_FUSION]:
    ensure_dir_exists(root)

# ------------------- Model wrapper 用于 GradCAM（返回 classifier logits） -------------------
# 解释：pytorch-grad-cam 期望 model(inputs) 返回 logits，我们的 embed_net 在 training 模式下会返回 (x_pool, classifier_logits, heatmaps)
# 因此 wrapper.forward 会调用 embed_net 并返回 classifier logits（以便 grad-cam 计算对每个样本的类别梯度）
class Model_for_CAM(torch.nn.Module):
    def __init__(self, backbone_model, modal):
        super(Model_for_CAM, self).__init__()
        self.model = backbone_model
        self.modal = modal

    def forward(self, x):
        """
        x: tensor (B,3,H,W) — 我们会把同一张图传入 visible 和 thermal 的两个输入槽（即 model(x,x, modal=...))
        返回： classifier logits tensor of shape (N, num_classes)
        """
        out = self.model(x, x, modal=self.modal)
        # 当 model.training == True 时，out 应该是 (x_pool, classifier_logits, heatmaps)
        if isinstance(out, (list, tuple)):
            # classifier logits 在 out[1]
            logits = out[1]
            return logits
        else:
            # 如果模型处于 eval 模式直接返回嵌入，则这是不可行的。因此要求在使用 wrapper 前把模型设为 training
            return out


# ------------------- 主流程：遍历指定 ID 并生成 Grad-CAM -------------------
def process_one_image(img_path, out_paths_tuple):
    """
    img_path: 原图路径
    out_paths_tuple: (out_origin_path, out_dilated_path, out_ppm_path, out_fusion_path)
    """
    # 1. 读取原图（PIL）并原尺寸保存的副本用于 overlay
    img_pil = Image.open(img_path).convert('RGB')
    img_np = np.array(img_pil)  # H W C (0-255)
    # transform -> tensor
    img_tensor = transform_test(img_np)  # (3,H,W), normalized
    input_tensor = img_tensor.unsqueeze(0).to(DEVICE)  # (1,3,H,W)

    # 2. 确定 modal（依据 camX 名称）
    # 假设 img_path 中包含 'cam1'..'cam6' 子串
    cam_name = None
    for cam in cams_list:
        if cam in img_path:
            cam_name = cam
            break
    if cam_name is None:
        # 若没找到 cam 信息，尝试通过路径上层文件夹名猜（例如 path/.../cam6/0333/0001.jpg）
        parts = img_path.replace('\\', '/').split('/')
        for p in parts:
            if p.lower().startswith('cam'):
                cam_name = p
                break
    if cam_name is None:
        print("无法从路径判断 cam，默认 modal=1 (visible). 路径:", img_path)
        modal = 1
    else:
        modal = determine_modal_from_cam(cam_name)

    # 3. 为了获得 classifier logits（3 个样本），需要让原 model 处于 training 模式（使 forward 返回 logits）
    prev_train_flag = model.training
    model.train()  # set True
    # 禁用梯度以获取预测类（不需要 grad）
    with torch.no_grad():
        out = model(input_tensor, input_tensor, modal=modal)
        if isinstance(out, (list, tuple)):
            # out[1] 是 classifier logits (shape [3, num_classes])
            # 注意：当 model 在 train() 并且输入是 (1,3,H,W)，因为 MultiScaleModule 会把 batch 扩为 3，
            # 所以 out[1].shape[0] == 3
            classifier_logits = out[1].detach().cpu()  # (3, num_classes)
        else:
            # 如果模型没有按期待返回 logits，抛出错误
            raise RuntimeError("模型在 train 模式下没有返回 logits，无法继续。请检查 embed_net.forward 的实现。")

    # 得到每个分支预测的 class index
    preds = torch.argmax(classifier_logits, dim=1).tolist()  # length should be 3

    # 4. 现在用 pytorch-grad-cam 计算三张 CAM（需要梯度，所以不要用 no_grad）
    #    我们通过 wrapper 让 grad-cam 看到“只返回 logits 的模型”
    cam_model = Model_for_CAM(model, modal=modal)
    cam_model.to(DEVICE)
    # grad-cam 的 target layer 仍然引用真实模型的 layer（不要用 wrapper 的 attribute）
    cam = GradCAM(model=cam_model, target_layers=[target_layer])

    # 构建 targets（针对每个 batch item 的类别）
    targets = [ClassifierOutputTarget(int(t)) for t in preds]  # length 3

    # 输入 grad-cam 的 tensor 需要是输入到 wrapper.forward 的内容（即 (B,3,H,W)）
    # 但 wrapper.forward 会把 x 复制为 (x,x,modal) 并返回 logits size (3,num_classes)
    # 此处 input_batch 仍然是 (1,3,H,W)：grad-cam 会在 forward 中触发真实模型的 forward 并计算梯度
    input_for_cam = input_tensor

    # 计算 cam，返回 ndarray shape (batch, H, W) -> 这里 batch == 1 because we passed B=1,
    # BUT 注意我们的 model 在 forward 内会扩 batch 到 3 (MultiScale)，grad-cam 会返回 cams for each internal sample
    # pytorch-grad-cam 会在内部处理 batch 和 target 的对齐 —— 它会基于 wrapper.forward 的输出（logits）来分配梯度。
    # 结果 grayscale_cam 将是一个 ndarray，shape == (batch_size_passed_to_cam, H, W).
    # 在我们的设置下，batch_size_passed_to_cam == 1, 但是 wrapper.forward 返回 logits shape (3, num_classes),
    # 而我们传入 targets 长度为 3，对应的是 wrapper.forward 输出的每个内部样本（origin/dilated/ppm）。
    # grad-cam 会把这些结果合并到返回的 grayscale_cam 中，使得返回的 ndarray 形状为 (3, H, W)。
    # 因此我们期望 grayscale_cam.shape[0] == 3 -> cam for origin/dilated/ppm respectively.
    grayscale_cam = cam(input_for_cam, targets=targets)  # ndarray: (3, H_feat, W_feat) -> resized automatically later?

    # NOTE: pytorch-grad-cam 返回的 cam 是与输入图像大小一致（默认会 upscale）
    # 测试确认 shape:
    if len(grayscale_cam.shape) == 3:
        # case: (3, H, W)
        cams_arr = grayscale_cam  # index 0: origin, 1: dilated, 2: ppm
    elif len(grayscale_cam.shape) == 4:
        # Sometimes returns (batch, num_internal, H, W). flatten to (num_internal, H, W)
        cams_arr = grayscale_cam.reshape(-1, grayscale_cam.shape[-2], grayscale_cam.shape[-1])
    else:
        raise RuntimeError("Unexpected grayscale_cam shape: {}".format(grayscale_cam.shape))

    # 如果返回的 cams 数量不是 3，我们也继续处理，但给出警告
    if cams_arr.shape[0] < 3:
        print("警告：得到的 CAM 数量 < 3，实际为", cams_arr.shape[0], "路径:", img_path)

    # 5. overlay 并保存：origin, dilated, ppm, fusion
    # prepare raw image normalized [0,1] for show_cam_on_image
    img_for_overlay = tensor_to_numpy_image(img_tensor)  # H W C float [0,1]
    # ensure same spatial size as cam; grad-cam 的 show_cam_on_image 要求 img 和 cam shape 匹配
    # pytorch-grad-cam 返回的 cam 已经是 input image 大小 (H,W) —— 若不是，请插值到 img size
    saved_paths = []
    branch_names = ['origin', 'dilated', 'ppm']
    for idx in range(min(3, cams_arr.shape[0])):
        cam_map = cams_arr[idx]
        # if cam_map size != img size, resize
        if cam_map.shape != (img_for_overlay.shape[0], img_for_overlay.shape[1]):
            import cv2
            cam_map = cv2.resize(cam_map, (img_for_overlay.shape[1], img_for_overlay.shape[0]), interpolation=cv2.INTER_LINEAR)
        # overlay
        visualization = show_cam_on_image(img_for_overlay, cam_map, use_rgb=True)
        # save
        out_path = out_paths_tuple[idx]
        out_dir = os.path.dirname(out_path)
        ensure_dir_exists(out_dir)
        # save visualization (PIL)
        Image.fromarray(visualization).save(out_path)
        saved_paths.append(out_path)

    # fusion: average first 3 cams (or all available)
    # num_used = min(3, cams_arr.shape[0])
    # fusion_map = np.mean(cams_arr[:num_used, :, :], axis=0)
    # ----------- Fusion CAM 改进版：平均 + 再做 Min-Max Normalization ----------- #

    num_used = min(3, cams_arr.shape[0])
    fusion_map_raw = np.mean(cams_arr[:num_used, :, :], axis=0)

    # Min-Max 归一化（如果最大值==最小值，避免除0）
    fusion_min = fusion_map_raw.min()
    fusion_max = fusion_map_raw.max()
    if fusion_max - fusion_min < 1e-7:
        fusion_map = np.zeros_like(fusion_map_raw)
    else:
        fusion_map = (fusion_map_raw - fusion_min) / (fusion_max - fusion_min)

    # resize if needed
    if fusion_map.shape != (img_for_overlay.shape[0], img_for_overlay.shape[1]):
        import cv2
        fusion_map = cv2.resize(fusion_map, (img_for_overlay.shape[1], img_for_overlay.shape[0]), interpolation=cv2.INTER_LINEAR)
    fusion_vis = show_cam_on_image(img_for_overlay, fusion_map, use_rgb=True)
    out_fusion_path = out_paths_tuple[3]
    ensure_dir_exists(os.path.dirname(out_fusion_path))
    Image.fromarray(fusion_vis).save(out_fusion_path)
    saved_paths.append(out_fusion_path)

    # 恢复 model 的 train/eval 状态
    if not prev_train_flag:
        model.eval()
    # 返回保存的路径
    return saved_paths


def main():
    # 遍历每个 cam 和 target id，寻找每张图片并生成 CAM
    total_images = 0
    for pid in TARGET_IDs:
        id_folder = id_to_foldername(pid)  # '0333'
        # for each cam
        for cam in cams_list:
            cam_dir = os.path.join(DATA_ROOT, cam, id_folder)
            if not os.path.isdir(cam_dir):
                # 没有该 ID 在该 cam 下
                continue
            # 创建对应输出目录（保持 SYSU 结构）
            out_origin_dir = os.path.join(OUT_ROOT_ORIGIN, cam, id_folder)
            out_dilated_dir = os.path.join(OUT_ROOT_DILATED, cam, id_folder)
            out_ppm_dir = os.path.join(OUT_ROOT_PPM, cam, id_folder)
            out_fusion_dir = os.path.join(OUT_ROOT_FUSION, cam, id_folder)
            ensure_dir_exists(out_origin_dir)
            ensure_dir_exists(out_dilated_dir)
            ensure_dir_exists(out_ppm_dir)
            ensure_dir_exists(out_fusion_dir)

            img_files = list_image_files_in_folder(cam_dir)
            print(f"Processing ID={pid} cam={cam}  {len(img_files)} images ...")
            for img_path in tqdm(img_files):
                # 构造保存路径（同名）
                fname = os.path.basename(img_path)
                out_origin_path = os.path.join(out_origin_dir, fname)
                out_dilated_path = os.path.join(out_dilated_dir, fname)
                out_ppm_path = os.path.join(out_ppm_dir, fname)
                out_fusion_path = os.path.join(out_fusion_dir, fname)

                try:
                    saved = process_one_image(img_path, (out_origin_path, out_dilated_path, out_ppm_path, out_fusion_path))
                except Exception as e:
                    print("处理图片失败：", img_path, " 错误：", e)
                total_images += 1

    print("全部完成。总共处理图片数约：", total_images)


if __name__ == "__main__":
    main()
