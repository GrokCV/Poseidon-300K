# import torch
# import cv2
# import mmcv
# import numpy as np
# from mmdet.apis import init_detector, inference_detector
# from mmdet.registry import VISUALIZERS
# import os

# # ==============================================================================
# # 1. 配置路径
# # ==============================================================================
# CONFIG_FILE = 'work_dirs/tood_r50_fpn_1x_poseidon/20251017_151752/vis_data/config.py'
# CHECKPOINT_FILE = 'work_dirs/tood_r50_fpn_1x_poseidon/20251017_151752/epoch_12.pth'
# # CONFIG_FILE = 'work_dirs/tood_r50_fpn_1x_poseidon/20251219_091922/vis_data/config.py'
# # CHECKPOINT_FILE = 'work_dirs/tood_r50_fpn_1x_poseidon/20251219_091922/epoch_12.pth'
# IMG_PATH = 'datasets/Poseidon-300K/images/test/31129.jpg'  # 你想要测试的图片路径
# DEVICE = 'cuda:4'       # 或 'cpu'
# SCORE_THR = 0.3        # 置信度阈值，低于这个分数的框不会画出来

# # ==============================================================================
# # 2. 推理与可视化逻辑
# # ==============================================================================
# def draw_detection_results():
#     print(f"正在初始化模型...")
#     # 初始化模型
#     model = init_detector(CONFIG_FILE, CHECKPOINT_FILE, device=DEVICE)
    
#     print(f"正在处理图片: {IMG_PATH} ...")
#     # 执行推理
#     # inference_detector 会自动处理 Resize(keep_ratio=False, 512x512)
#     # 并将预测结果的坐标【自动映射回】原图尺寸
#     result = inference_detector(model, IMG_PATH)

#     # ---------------------------------------------------------
#     # 方法 A: 使用 MMDetection 官方 Visualizer (推荐，画风标准)
#     # ---------------------------------------------------------
#     # 初始化可视化器
#     visualizer = VISUALIZERS.build(model.cfg.visualizer)
#     # 加载原图 (用于绘制底图)
#     img = mmcv.imread(IMG_PATH)
#     img = mmcv.imconvert(img, 'bgr', 'rgb')
    
#     # 设置数据集元数据（类别名称、调色盘）
#     visualizer.dataset_meta = model.dataset_meta
    
#     # 绘制预测结果
#     visualizer.add_datasample(
#         'prediction_result',
#         img,
#         data_sample=result,
#         draw_gt=False,        # 不画真值框
#         show=False,           # 不直接弹窗显示
#         wait_time=0,
#         pred_score_thr=SCORE_THR  # 过滤低分框
#     )
    
#     # 获取绘制好的图像
#     res_img = visualizer.get_image()
#     res_img = cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR)

#     # 保存结果
#     output_name = f"pred_{os.path.basename(IMG_PATH)}"
#     cv2.imwrite(output_name, res_img)
#     print(f"检测结果已保存为: {output_name}")

#     # ---------------------------------------------------------
#     # 打印详细信息（可选）
#     # ---------------------------------------------------------
#     pred_instances = result.pred_instances
#     # 过滤掉低于阈值的索引
#     valid_idx = pred_instances.scores > SCORE_THR
#     scores = pred_instances.scores[valid_idx].cpu().numpy()
#     labels = pred_instances.labels[valid_idx].cpu().numpy()
#     classes = model.dataset_meta.get('classes')
    
#     print(f"\n检测摘要 (Threshold > {SCORE_THR}):")
#     for i in range(len(scores)):
#         class_name = classes[labels[i]] if classes else f"Class_{labels[i]}"
#         print(f"目标 {i+1}: 类别={class_name:12} | 置信度={scores[i]:.4f}")

# if __name__ == '__main__':
#     draw_detection_results()


import os
import cv2
import torch
import numpy as np
import glob
from tqdm import tqdm  # 进度条库，如果没有请 pip install tqdm
from PIL import Image, ImageDraw, ImageFont # 用于绘制高清字体

from mmdet.apis import init_detector, inference_detector

# ==============================================================================
# 1. 配置区域
# ==============================================================================
CONFIG_FILE = 'work_dirs/tood_r50_fpn_1x_poseidon/20251219_091922/vis_data/config.py'
CHECKPOINT_FILE = 'work_dirs/tood_r50_fpn_1x_poseidon/20251219_091922/epoch_12.pth'
# CONFIG_FILE = 'work_dirs/tood_r50_fpn_1x_poseidon/20251017_151752/vis_data/config.py'
# CHECKPOINT_FILE = 'work_dirs/tood_r50_fpn_1x_poseidon/20251017_151752/epoch_12.pth'
# 输入图片文件夹路径
INPUT_FOLDER = 'datasets/Poseidon-300K/images/test2' 
# 输出结果保存文件夹
OUTPUT_FOLDER = 'detection_results_vis1/'

DEVICE = 'cuda:4'
SCORE_THR = 0.30        # 置信度阈值
FONT_SIZE = 25        # 字体大小 (根据图片分辨率调整)
LINE_THICKNESS = 3     # 框的粗细

# 指定字体路径 (Times New Roman)
# 如果脚本同级目录下没有 times.ttf，请修改为系统字体路径
# Windows 示例: 'C:/Windows/Fonts/times.ttf'
FONT_PATH = 'times.ttf' 

# ==============================================================================
# 2. 颜色生成器 (为不同类别生成固定颜色)
# ==============================================================================
FIXED_COLOR_MAP = {
    7: (255, 0, 0),      # ID 7 (比如海龟) -> 红色
    6: (81, 38,102),      # ID 0 (比如乌贼) -> 绿色
    0: (197, 90,17),      # ID 0 (比如holo) -> 橙色
    4:(160,64,189),     #鱼
    3:(47,85,151),  #海星
    5: (148,19, 126),    # ID 5 (比如潜水员) -> 紫色
    11: (255, 255, 0),   # ID 11 (比如塑料垃圾) -> 黄色
}
def get_color(idx):
    #以此保证同一类别的颜色总是相同的
    if idx in FIXED_COLOR_MAP:
        return FIXED_COLOR_MAP[idx]
    np.random.seed(idx)
    color = np.random.randint(0, 255, size=3).tolist()
    return tuple(color)

# ==============================================================================
# 3. 高清绘图函数 (使用 PIL 实现 Times New Roman)
# ==============================================================================
def draw_detections_pil(img_array, bboxes, labels, scores, class_names, font_path):
    """
    img_array: cv2 读取的图片 (BGR numpy array)
    """
    # 1. cv2 (BGR) 转 PIL (RGB)
    img_pil = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # 2. 加载字体
    try:
        font = ImageFont.truetype(font_path, FONT_SIZE)
    except IOError:
        print(f"[Warning] 找不到字体文件 {font_path}，将使用默认字体。")
        font = ImageFont.load_default()

    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox.astype(int)
        score = scores[i]
        label = labels[i]
        
        # 获取类别名称
        class_name = class_names[label] if class_names else str(label)
        # 显示文本内容: Class: 0.95
        text_str = f"{class_name}: {score:.2f}"
        
        # 获取颜色 (RGB)
        color = get_color(label)
        
        # --- A. 画框 ---
        # PIL 的 rectangle 不支持直接设置线条宽度(旧版本)，这里用一种变通方法或直接画线
        draw.rectangle([x1, y1, x2, y2], outline=color, width=LINE_THICKNESS)
        
        # --- B. 画文字背景 (为了清晰) ---
        # 计算文字宽高
        try:
            # Pillow >= 10.0.0
            left, top, right, bottom = font.getbbox(text_str)
            text_w = right - left
            text_h = bottom - top
        except AttributeError:
            # 旧版 Pillow
            text_w, text_h = draw.textsize(text_str, font)
            
        # 确保文字背景不画出图片外
        text_bg_x1 = x1
        text_bg_y1 = max(0, y1 - text_h - 5)
        text_bg_x2 = x1 + text_w + 10
        text_bg_y2 = max(text_h + 5, y1)
        
        # 画实心背景 (颜色同框，但稍微暗一点或者直接用框的颜色)
        draw.rectangle([text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2], fill=color)
        
        # --- C. 画文字 (白色，Times New Roman) ---
        # 文字位置微调居中
        draw.text((x1 + 5, text_bg_y1-4), text_str, fill=(255, 255, 255), font=font)

    # 3. PIL (RGB) 转回 cv2 (BGR)
    result_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return result_img

# ==============================================================================
# 4. 主程序
# ==============================================================================
def main():
    # 0. 准备工作
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    # 查找文件夹下所有图片 (支持 jpg, png, jpeg)
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    img_list = []
    for ext in extensions:
        img_list.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))
    
    img_list = sorted(img_list)
    if len(img_list) == 0:
        print(f"错误: 在 {INPUT_FOLDER} 下未找到图片。")
        return

    print(f"1. 初始化模型...")
    model = init_detector(CONFIG_FILE, CHECKPOINT_FILE, device=DEVICE)
    # 获取类别名称列表
    class_names = model.dataset_meta.get('classes', None)

    print(f"2. 开始处理 {len(img_list)} 张图片...")
    
    # 使用 tqdm 显示进度条
    for img_path in tqdm(img_list):
        # --- 推理 ---
        # inference_detector 自动处理了 Resize/Normalize，返回的原图坐标结果
        result = inference_detector(model, img_path)
        
        # --- 解析结果 ---
        pred_instances = result.pred_instances
        # 筛选置信度
        valid_mask = pred_instances.scores > SCORE_THR
        
        valid_bboxes = pred_instances.bboxes[valid_mask].cpu().numpy()
        valid_scores = pred_instances.scores[valid_mask].cpu().numpy()
        valid_labels = pred_instances.labels[valid_mask].cpu().numpy()
        
        # --- 读取原图 ---
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # --- 绘制 (使用 PIL 高清绘图) ---
        vis_img = draw_detections_pil(
            img, 
            valid_bboxes, 
            valid_labels, 
            valid_scores, 
            class_names,
            FONT_PATH
        )
        
        # --- 保存 ---
        file_name = os.path.basename(img_path)
        save_path = os.path.join(OUTPUT_FOLDER, f"vis1_{file_name}")
        cv2.imwrite(save_path, vis_img)

    print(f"\n全部完成！结果保存在: {OUTPUT_FOLDER}")

if __name__ == '__main__':
    main()