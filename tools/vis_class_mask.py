import numpy as np
import matplotlib.pyplot as plt

def visualize_binary_masks(mask_path, class_names=None):
    """
    直接可视化二值类别掩码
    :param mask_path: .npy文件路径
    :param class_names: 类别名称列表（可选）
    """
    class_mask = np.load(mask_path)  # shape: (C, H, W)
    num_classes = class_mask.shape[0]
    
    # 创建自适应子图布局
    cols = 5
    rows = int(np.ceil(num_classes / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    
    # 遍历所有类别
    for cls_idx in range(num_classes):
        row = cls_idx // cols
        col = cls_idx % cols
        ax = axs[row, col] if rows > 1 else axs[col]
        
        # 显示二值掩码（白色为前景）
        ax.imshow(class_mask[cls_idx], cmap='gray', vmin=0, vmax=1)
        
        # 添加标题
        title = f'C{cls_idx}'
        if class_names and cls_idx < len(class_names):
            title += f':{class_names[cls_idx]}'
        ax.set_title(title, fontsize=8)
        ax.axis('off')
    
    # 隐藏多余的子图
    for cls_idx in range(num_classes, rows*cols):
        row = cls_idx // cols
        col = cls_idx % cols
        axs[row, col].axis('off')
    
    # 创建保存路径
    from pathlib import Path
    save_dir = Path("vis/gt_class_mask") / Path(mask_path).parent.name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / (Path(mask_path).stem + ".jpg")

    # 保存图像并关闭绘图
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"可视化结果已保存至: {save_path}")

# 使用示例
if __name__ == "__main__":
    visualize_binary_masks(
        "/root/BEV/GeoBEV/data/nuscenes/samples_class_mask/CAM_FRONT_RIGHT/n008-2018-05-21-11-06-59-0400__CAM_FRONT_RIGHT__1526915644419956.npy",
        class_names = [
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]  # Added missing closing bracket
    ) 