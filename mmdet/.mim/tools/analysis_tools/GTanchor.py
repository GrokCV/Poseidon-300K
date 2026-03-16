# #!/usr/bin/env python3
# """
# 可视化 COCO 格式 GT 框
# python vis_coco_gt.py
# """

# import os, cv2, numpy as np
# from pycocotools.coco import COCO

# # ========== 1. 改成你的路径 ==========
# JSON_PATH = 'datasets/Poseidon-300K/annotations/COCO/train_fixed.json'   # 修复后的 json
# IMG_DIR   = 'datasets/Poseidon-300K/images/train'           # 图片文件夹
# IMG_NAME  = '00001.jpg'                   # 想画的单张图
# WORKDIR   = 'work_dirs/feature_vis'
# # 想一次画很多图就把 IMG_NAME 设为 None，会自动随机抽 10 张
# # =======================================

# # 2. 随机配色
# np.random.seed(42)
# CAT_COLORS = {}   # id -> (b,g,r)

# def get_color(cat_id):
#     if cat_id not in CAT_COLORS:
#         CAT_COLORS[cat_id] = tuple(map(int, np.random.randint(0,255,3)))
#     return CAT_COLORS[cat_id]

# def vis_one(coco, img_name, save_dir='.'):
#     img_id  = next(i['id'] for i in coco.imgs.values() if i['file_name']==img_name)
#     img_info = coco.imgs[img_id]
#     annIds   = coco.getAnnIds(imgIds=img_id)
#     anns     = coco.loadAnns(annIds)

#     img_path = os.path.join(IMG_DIR, img_name)
#     img = cv2.imread(img_path)
#     assert img is not None, f'读图失败：{img_path}'

#     for ann in anns:
#         x, y, w, h = ann['bbox']
#         cat_id     = ann['category_id']
#         cat_name   = coco.cats[cat_id]['name']
#         color      = get_color(cat_id)
#         cv2.rectangle(img, (int(x),int(y)), (int(x+w),int(y+h)), color, 2)
#         cv2.putText(img, cat_name, (int(x), int(y)-5),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
#     save_path = os.path.join('work_dirs/feature_vis', 'gt_'+img_name)
#     os.makedirs('work_dirs/feature_vis', exist_ok=True)   # 确保目录存在
#     cv2.imwrite(save_path, img)
#     print(f'已保存：{save_path}   框数={len(anns)}')

# def main():
#     coco = COCO(JSON_PATH)
#     if IMG_NAME:                       # 单张
#         vis_one(coco, IMG_NAME)
#     else:                              # 随机 10 张
#         sampled = list(coco.imgs.values())[:10]
#         for info in sampled:
#             vis_one(coco, info['file_name'])

# if __name__ == '__main__':
#     main()

##################################可以正常使用####################################
# from pycocotools.coco import COCO
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from PIL import Image
# import os

# # -----------------------------
# # 配置路径
# # -----------------------------
# ann_file = 'datasets/Poseidon-300K/annotations/COCO/train_fixed.json'
# img_dir = 'datasets/Poseidon-300K/images/train'
# save_dir = 'work_dirs/feature_vis'
# os.makedirs(save_dir, exist_ok=True)

# file_name = '14581.jpg'

# # -----------------------------
# # 加载 COCO 注释
# # -----------------------------
# coco = COCO(ann_file)
# img_ids = coco.getImgIds()
# img_info = coco.loadImgs([i for i in img_ids if coco.loadImgs([i])[0]['file_name'] == file_name])[0]
# image_id = img_info['id']
# ann_ids = coco.getAnnIds(imgIds=image_id)
# anns = coco.loadAnns(ann_ids)

# # -----------------------------
# # 打开图片
# # -----------------------------
# img_path = os.path.join(img_dir, file_name)
# image = Image.open(img_path)
# width, height = image.size

# # -----------------------------
# # 可视化检测框
# # -----------------------------
# fig, ax = plt.subplots(1, figsize=(width/100, height/100))  # 按图片大小设置画布
# ax.imshow(image)

# for ann in anns:
#     x, y, w, h = ann['bbox']
#     # 绘制红色矩形框
#     rect = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor='r', facecolor='none')
#     ax.add_patch(rect)

#     # 显示类别名称
#     cat_id = ann['category_id']
#     cat_name = coco.loadCats(cat_id)[0]['name']
#     ax.text(x, y-10, cat_name, color='yellow', fontsize=12, weight='bold')

# # 去掉坐标轴
# ax.axis('off')

# # 保存可视化
# save_path = os.path.join(save_dir, file_name)
# plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=100)
# plt.close(fig)

# print(f"可视化检测框已保存到: {save_path}")

#!/usr/bin/env python3
"""
修复 COCO 格式中 annotations.image_id 为字符串导致匹配失败的问题
python fix_coco_str_id.py


import json, os, sys
from pycocotools.coco import COCO

# ========== 1. 改成你的路径 ==========
SRC_JSON = 'datasets/Poseidon-300K/annotations/COCO/train.json'          # 原始 json
DST_JSON = 'datasets/Poseidon-300K/annotations/COCO/train_fixed.json'    # 修复后保存位置
TEST_IMG = '00001.jpg'                    # 任意一张图，用于验证
# =====================================

def fix_json(src, dst):
    print('Loading', src)
    with open(src) as f:
        data = json.load(f)

    print('annotations 条目数:', len(data['annotations']))
    print('images      条目数:', len(data['images']))

    # 1. 修正 annotations.image_id
    for ann in data['annotations']:
        ann['image_id'] = int(ann['image_id'])

    # 2. 修正 images.id（保险）
    for img in data['images']:
        img['id'] = int(img['id'])

    # 3. 写入新文件
    with open(dst, 'w') as f:
        json.dump(data, f, separators=(',', ':'))
    print('Fixed json 已保存至:', dst)

def verify(json_path, img_name):
    coco = COCO(json_path)
    img_id = next(i['id'] for i in coco.imgs.values() if i['file_name'] == img_name)
    n = len(coco.getAnnIds(imgIds=img_id))
    print(f'验证：img_id={img_id}  file_name={img_name}  框数={n}')
    return n

if __name__ == '__main__':
    if not os.path.isfile(SRC_JSON):
        sys.exit('SRC_JSON 文件不存在！')
    fix_json(SRC_JSON, DST_JSON)
    verify(DST_JSON, TEST_IMG)"""
# import json
# import cv2
# import os
# import numpy as np
# import random

# # ==============================================================================
# # 1. 配置路径 (请根据你的实际路径修改)
# # ==============================================================================
# # 图片所在的文件夹路径 (例如: datasets/Poseidon-300K/images/train/)
# IMG_ROOT = 'datasets/Poseidon-300K/images/test' 

# # COCO 格式的标注文件路径 (例如: datasets/Poseidon-300K/annotations/COCO/train.json)
# ANN_FILE = 'datasets/Poseidon-300K/annotations/COCO/test.json'

# # 你想要可视化的具体图片文件名
# TARGET_FILENAME = '51255.jpg' 

# # ==============================================================================
# # 2. 辅助函数：生成随机颜色
# # ==============================================================================
# def get_color(idx):
#     # 基于类别ID生成固定的颜色，保证同一类别颜色一致
#     random.seed(idx)
#     return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

# # ==============================================================================
# # 3. 主逻辑
# # ==============================================================================
# def visualize_coco_gt():
#     print(f"正在加载标注文件: {ANN_FILE} ... (文件较大时可能需要几秒)")
#     with open(ANN_FILE, 'r') as f:
#         coco_data = json.load(f)
    
#     # 1. 构建索引：找到目标图片的信息
#     target_img_info = None
#     for img_info in coco_data['images']:
#         if img_info['file_name'] == TARGET_FILENAME:
#             target_img_info = img_info
#             break
            
#     if target_img_info is None:
#         print(f"错误: 在标注文件中未找到图片 {TARGET_FILENAME}")
#         return

#     image_id = target_img_info['id']
#     img_height = target_img_info['height']
#     img_width = target_img_info['width']
    
#     print(f"找到图片: ID={image_id}, Size={img_width}x{img_height}")

#     # 2. 构建类别映射 (ID -> Name)
#     cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}

#     # 3. 找到属于该图片的所有标注
#     # 注意：COCO json 中 annotation 的 'image_id' 对应 image 的 'id'
#     img_anns = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]
    
#     print(f"该图片共有 {len(img_anns)} 个目标。")

#     # 4. 读取原图
#     img_path_full = os.path.join(IMG_ROOT, TARGET_FILENAME)
#     if not os.path.exists(img_path_full):
#         print(f"错误: 图片文件不存在 {img_path_full}")
#         return
        
#     img = cv2.imread(img_path_full)
    
#     # 5. 绘制 GT 框
#     for ann in img_anns:
#         bbox = ann['bbox'] # COCO 格式: [x_min, y_min, width, height]
#         cat_id = ann['category_id']
#         cat_name = cat_id_to_name.get(cat_id, 'unknown')
        
#         # 转换坐标: [x, y, w, h] -> [x1, y1, x2, y2]
#         x1 = int(bbox[0])
#         y1 = int(bbox[1])
#         x2 = int(bbox[0] + bbox[2])
#         y2 = int(bbox[1] + bbox[3])
        
#         # 获取颜色
#         color = get_color(cat_id)
        
#         # 画矩形框 (线条宽度 2)
#         cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
#         # 画标签背景和文字
#         label_text = f"{cat_name}"
#         (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
#         cv2.rectangle(img, (x1, y1 - text_h - 5), (x1 + text_w, y1), color, -1) # 实心背景
#         cv2.putText(img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

#     # 6. 保存结果
#     save_name = f"gt_{TARGET_FILENAME}"
#     cv2.imwrite(save_name, img)
#     print(f"可视化完成！结果已保存为: {save_name}")

# if __name__ == '__main__':
#     visualize_coco_gt()
import json
import cv2
import os
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont

# ==============================================================================
# 1. 配置路径
# ==============================================================================
IMG_ROOT = 'datasets/Poseidon-300K/images/test' 
ANN_FILE = 'datasets/Poseidon-300K/annotations/COCO/test.json'
TARGET_FILENAME = '49487.jpg' 

# 字体配置
FONT_PATH = 'times.ttf'   # 请确保该文件存在，或修改为系统路径 'C:/Windows/Fonts/times.ttf'
FONT_SIZE = 40            # 字体大小
LINE_THICKNESS = 3        # 边框粗细

# ==============================================================================
# 2. 颜色配置 (自定义配色方案)
# ==============================================================================
# 格式: { 类别ID : (R, G, B) }
FIXED_COLOR_MAP = {
    # 7: (255, 0, 0),      # Turtle (海龟) -> 红色
    # 6: (81, 38, 102),    # Cuttlefish (乌贼) -> 深紫色
    # 0: (197, 90, 17),    # Holothurian (海参) -> 橙色/棕色
    # 4: (160, 64, 189),   # Fish (鱼) -> 浅紫色
    # 3: (47, 85, 151),    # Starfish (海星) -> 蓝色
    # 5: (148, 19, 126),   # Diver (潜水员) -> 紫红色
    # 11: (255, 255, 0),   # Plastic Trash -> 黄色
    1: (197, 90, 17), 
    2: (197, 90, 17),
    3: (160, 64, 189),
4: (160, 64, 189),
    5: (160, 64, 189),
    
    7: (255, 0, 0),
    8: (255, 0, 0),
    9:(195,85,238),
    10:(195,85,238),
    11:(195,85,238),
    12:(195,85,238),

}

def get_color(idx):
    """
    根据 idx 返回颜色。
    优先查表，查不到则随机生成。
    """
    # 1. 优先使用自定义颜色表
    if idx in FIXED_COLOR_MAP:
        return FIXED_COLOR_MAP[idx]
    
    # 2. 兜底：随机生成
    random.seed(idx)
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

# ==============================================================================
# 3. 主逻辑
# ==============================================================================
def visualize_coco_gt():
    print(f"1. 正在加载标注文件: {ANN_FILE} ...")
    try:
        with open(ANN_FILE, 'r') as f:
            coco_data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到标注文件 {ANN_FILE}")
        return
    
    # --- 查找图片信息 ---
    target_img_info = None
    for img_info in coco_data['images']:
        if img_info['file_name'] == TARGET_FILENAME:
            target_img_info = img_info
            break
            
    if target_img_info is None:
        print(f"错误: 在标注文件中未找到图片 {TARGET_FILENAME}")
        return

    image_id = target_img_info['id']
    print(f"   -> 找到图片 ID: {image_id}")

    # --- 构建映射 ---
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    img_anns = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]
    print(f"   -> 该图片共有 {len(img_anns)} 个真值目标。")

    # --- 读取原图 ---
    img_path_full = os.path.join(IMG_ROOT, TARGET_FILENAME)
    if not os.path.exists(img_path_full):
        print(f"错误: 图片文件不存在 {img_path_full}")
        return
        
    img_cv2 = cv2.imread(img_path_full)
    
    # ==========================================================================
    # [核心绘制] 切换到 PIL 进行高清绘图
    # ==========================================================================
    
    # 1. 转换格式: BGR (OpenCV) -> RGB (PIL)
    img_pil = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # 2. 加载字体
    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except IOError:
        print(f"[警告] 找不到字体 {FONT_PATH}，将使用默认字体(可能不好看)。")
        font = ImageFont.load_default()

    # 3. 循环绘制
    for ann in img_anns:
        bbox = ann['bbox'] # COCO格式: [x_min, y_min, width, height]
        cat_id = ann['category_id']
        cat_name = cat_id_to_name.get(cat_id, 'unknown')
        
        # 坐标转换
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[0] + bbox[2])
        y2 = int(bbox[1] + bbox[3])
        
        # 获取颜色 (使用自定义配色)
        color = get_color(cat_id)
        
        # A. 画框 (outline=颜色, width=粗细)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=LINE_THICKNESS)
        
        # B. 准备文字
        label_text = f"{cat_name}"
        
        # C. 计算文字背景框大小
        try:
            # Pillow 新版本 (>= 10.0.0)
            left, top, right, bottom = font.getbbox(label_text)
            text_w = right - left
            text_h = bottom - top
        except AttributeError:
            # Pillow 旧版本兼容
            text_w, text_h = draw.textsize(label_text, font)
            
        # 设置文字背景的位置 (优先放在框的左上角上方)
        text_bg_x1 = x1
        text_bg_y1 = y1 - text_h - 8  # 往上提一点
        text_bg_x2 = x1 + text_w + 10
        text_bg_y2 = y1
        
        # 如果文字跑出图片上边界，就画在框内部
        if text_bg_y1 < 0:
            text_bg_y1 = y1
            text_bg_y2 = y1 + text_h + 8
            
        # D. 画实心背景 (fill=颜色)
        draw.rectangle([text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2], fill=color)
        
        # E. 画文字 (白色 fill=(255,255,255))
        # 稍微调整文字位置以居中
        draw.text((text_bg_x1 + 5, text_bg_y1 - 2), label_text, fill=(255, 255, 255), font=font)

    # 4. 转换回 OpenCV 格式: RGB -> BGR
    img_result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # --- 保存结果 ---
    save_name = f"gt_high_res_{TARGET_FILENAME}"
    cv2.imwrite(save_name, img_result)
    print(f"可视化完成！结果已保存为: {save_name}")

if __name__ == '__main__':
    visualize_coco_gt()
# import json
# import cv2
# import os
# import numpy as np
# import random
# from PIL import Image, ImageDraw, ImageFont  # 引入 PIL 进行高清绘图

# # ==============================================================================
# # 1. 配置路径
# # ==============================================================================
# IMG_ROOT = 'datasets/Poseidon-300K/images/test' 
# ANN_FILE = 'datasets/Poseidon-300K/annotations/COCO/test.json'
# TARGET_FILENAME = '49487.jpg' 

# # 字体配置 (美化核心)
# FONT_PATH = 'times.ttf'   # 请确保该文件存在，或修改为系统路径 'C:/Windows/Fonts/times.ttf'
# FONT_SIZE = 40          # 字体大小
# LINE_THICKNESS = 3        # 边框粗细

# # ==============================================================================
# # 2. 辅助函数
# # ==============================================================================
# FIXED_COLOR_MAP = {
#     7: (255, 0, 0),      # ID 7 (比如海龟) -> 红色
#     6: (81, 38,102),      # ID 0 (比如乌贼) -> 绿色
#     0: (197, 90,17),      # ID 0 (比如holo) -> 橙色
#     4:(160,64,189),     #鱼
#     3:(47,85,151),  #海星
#     5: (148,19, 126),    # ID 5 (比如潜水员) -> 紫色
#     11: (255, 255, 0),   # ID 11 (比如塑料垃圾) -> 黄色
# }
# def get_color(idx):
#     # 基于类别ID生成固定的颜色
#     if idx in FIXED_COLOR_MAP:
#         return FIXED_COLOR_MAP[idx]
#     random.seed(idx)
#     # 生成颜色 (R, G, B)
#     return (random.randint(0, 200), random.randint(0, 200), random.randint(0, 200))

# # ==============================================================================
# # 3. 主逻辑
# # ==============================================================================
# def visualize_coco_gt():
#     print(f"1. 正在加载标注文件: {ANN_FILE} ...")
#     with open(ANN_FILE, 'r') as f:
#         coco_data = json.load(f)
    
#     # --- 查找图片信息 ---
#     target_img_info = None
#     for img_info in coco_data['images']:
#         if img_info['file_name'] == TARGET_FILENAME:
#             target_img_info = img_info
#             break
            
#     if target_img_info is None:
#         print(f"错误: 在标注文件中未找到图片 {TARGET_FILENAME}")
#         return

#     image_id = target_img_info['id']
#     print(f"   -> 找到图片 ID: {image_id}")

#     # --- 构建映射 ---
#     cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
#     img_anns = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]
#     print(f"   -> 该图片共有 {len(img_anns)} 个真值目标。")

#     # --- 读取原图 ---
#     img_path_full = os.path.join(IMG_ROOT, TARGET_FILENAME)
#     if not os.path.exists(img_path_full):
#         print(f"错误: 图片文件不存在 {img_path_full}")
#         return
        
#     img_cv2 = cv2.imread(img_path_full)
    
#     # ==========================================================================
#     # [核心修改] 切换到 PIL 进行高清绘图
#     # ==========================================================================
    
#     # 1. 转换格式: BGR (OpenCV) -> RGB (PIL)
#     img_pil = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
#     draw = ImageDraw.Draw(img_pil)
    
#     # 2. 加载字体
#     try:
#         font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
#     except IOError:
#         print(f"[警告] 找不到字体 {FONT_PATH}，将使用默认字体(可能不好看)。")
#         font = ImageFont.load_default()

#     # 3. 循环绘制
#     for ann in img_anns:
#         bbox = ann['bbox'] # [x, y, w, h]
#         cat_id = ann['category_id']
#         cat_name = cat_id_to_name.get(cat_id, 'unknown')
        
#         # 坐标转换
#         x1 = int(bbox[0])
#         y1 = int(bbox[1])
#         x2 = int(bbox[0] + bbox[2])
#         y2 = int(bbox[1] + bbox[3])
        
#         # 获取颜色
#         color = get_color(cat_id)
        
#         # A. 画框 (outline=颜色, width=粗细)
#         draw.rectangle([x1, y1, x2, y2], outline=color, width=LINE_THICKNESS)
        
#         # B. 准备文字
#         label_text = f"{cat_name}"
        
#         # C. 计算文字背景框大小
#         try:
#             # Pillow 新版本
#             left, top, right, bottom = font.getbbox(label_text)
#             text_w = right - left
#             text_h = bottom - top
#         except AttributeError:
#             # Pillow 旧版本兼容
#             text_w, text_h = draw.textsize(label_text, font)
            
#         # 设置文字背景的位置 (优先放在框的左上角上方，如果出界则放在框内)
#         text_bg_x1 = x1
#         text_bg_y1 = y1 - text_h - 8  # 往上提一点
#         text_bg_x2 = x1 + text_w + 10
#         text_bg_y2 = y1
        
#         # 如果文字跑出图片上边界，就画在框内部
#         if text_bg_y1 < 0:
#             text_bg_y1 = y1
#             text_bg_y2 = y1 + text_h + 8
            
#         # D. 画实心背景 (fill=颜色)
#         draw.rectangle([text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2], fill=color)
        
#         # E. 画文字 (白色 fill=(255,255,255))
#         # 稍微调整文字位置以居中
#         draw.text((text_bg_x1 + 5, text_bg_y1 - 2), label_text, fill=(255, 255, 255), font=font)

#     # 4. 转换回 OpenCV 格式: RGB -> BGR
#     img_result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

#     # --- 保存结果 ---
#     save_name = f"gt_high_res_{TARGET_FILENAME}"
#     cv2.imwrite(save_name, img_result)
#     print(f"可视化完成！结果已保存为: {save_name}")

# if __name__ == '__main__':
#     visualize_coco_gt()