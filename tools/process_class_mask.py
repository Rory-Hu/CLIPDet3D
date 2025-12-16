import os
import torch
from tqdm import tqdm
from pathlib import Path
import numpy as np
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
import colorsys
from PIL import Image
from multiprocessing import Pool
def _do_paste_mask(masks, boxes, img_h, img_w, skip_empty=True):
    device = masks.device
    if skip_empty:
        x0_int = torch.clamp(boxes[:, 0].min().floor() - 1, min=0).to(dtype=torch.int32)
        y0_int = torch.clamp(boxes[:, 1].min().floor() - 1, min=0).to(dtype=torch.int32)
        x1_int = torch.clamp(boxes[:, 2].max().ceil() + 1, max=img_w).to(dtype=torch.int32)
        y1_int = torch.clamp(boxes[:, 3].max().ceil() + 1, max=img_h).to(dtype=torch.int32)
        
        if x1_int <= x0_int or y1_int <= y0_int:
            return torch.zeros((1, img_h, img_w), (slice(0,0), slice(0,0)))
    else:
        x0_int, y0_int = 0, 0
        x1_int, y1_int = img_w, img_h
    
    x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)
    N = masks.shape[0]

    img_y = torch.arange(y0_int, y1_int, device=device, dtype=torch.float32) + 0.5
    img_x = torch.arange(x0_int, x1_int, device=device, dtype=torch.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1

    if not torch.onnx.is_in_onnx_export():
        img_x = torch.nan_to_num(img_x)
        img_y = torch.nan_to_num(img_y)

    gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
    gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
    grid = torch.stack([gx, gy], dim=3)

    img_masks = F.grid_sample(masks.to(dtype=torch.float32), grid, align_corners=False)
    return (img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))) if skip_empty else (img_masks[:, 0], ())

def process_single_mask(src_path):
    # 保持与loading.py完全相同的处理流程
    data = torch.load(src_path, map_location='cpu')
    
    # 数据加载与预处理
    mask_pred = data['masks'].float() / 255
    bboxes = data['bboxes'][:, :4]
    labels = data['labels']
    img_shape = (900, 1600)  # 固定图像尺寸
    
    # 实例掩码生成（与数据加载模块严格一致）
    inst_full = torch.zeros((len(mask_pred), *img_shape), dtype=torch.float32)
    for i in range(len(mask_pred)):
        inst_mask, spatial = _do_paste_mask(
            mask_pred[i:i+1], 
            bboxes[i:i+1], 
            img_shape[0], 
            img_shape[1]
        )
        ys, xs = spatial
        inst_full[i, ys, xs] = inst_mask.squeeze(0)
    
    # 类别掩码生成（严格对齐数据加载模块）
    class_masks = torch.zeros((10, *img_shape), dtype=torch.float32)  # 移除batch维度
    for cls_id in torch.unique(labels):
        if cls_id < 10:  # 添加范围校验
            idxs = (labels == cls_id).nonzero(as_tuple=True)[0]
            if idxs.numel() > 0:
                class_masks[cls_id] = inst_full[idxs].amax(dim=0)  # 使用amax方法
    
    # 保存到同级class_mask目录
    dst_path = src_path.replace('samples_instance_mask', 'samples_class_mask').replace('.pt', '.npy')
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    np.save(dst_path, class_masks.numpy())

    # 可视化逻辑（与loading.py严格对齐）
    import matplotlib.pyplot as plt
    from datetime import datetime
    
    class_names = [
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
        'barrier', 'motorcycle', 'bicycle', 'pedestrian','traffic_cone'
    ]
    
    # 创建5x2子图画布
    fig, axes = plt.subplots(5, 2, figsize=(20, 25))
    class_masks_np = class_masks.numpy()
    
    for cls_id in range(len(class_names)):  # 动态处理类别数量
        row = cls_id // 2
        col = cls_id % 2
        ax = axes[row, col]
        mask = class_masks_np[cls_id]  # 移除batch维度索引
        ax.imshow(mask > 0.5, cmap='gray')
        title = f'{class_names[cls_id]} (cls:{cls_id})' if cls_id < len(class_names) else f'Unknown (cls:{cls_id})'
        ax.set_title(title, fontsize=8)

    # 保存路径处理（与数据目录结构一致）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.basename(src_path).replace('.pt', '')
    vis_path = os.path.join('vis/gt_class_all', f'{base_name}_{timestamp}.jpg')
    
    plt.suptitle(f'Class Masks - {base_name}')
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # 调整顶部间距
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()

def batch_process(src_root):
    # 多进程处理逻辑
    file_list = []
    for root, _, files in os.walk(src_root):
        for file in files:
            if file.endswith('.pt'):
                file_list.append(os.path.join(root, file))
    
    with Pool(8) as p:
        list(tqdm(p.imap(process_single_mask, file_list), total=len(file_list)))


if __name__ == "__main__":
    batch_process(
        src_root="/root/BEV/GeoBEV/data/nuscenes/",
    )