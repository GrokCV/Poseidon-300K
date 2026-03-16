from pycocotools.coco import COCO
import pandas as pd

# === 修改这里为你的标注文件路径 ===
ann_file = '/data/lzm/mmdection/mmdetection-main/datasets/Poseidon-300K/annotations/COCO/test' \
'.json'

# === 类别名称 ===
CLASSES = (
    'holothurian',
    'echinus',
    'scallop',
    'starfish',
    'fish',
    'diver',
    'cuttlefish',
    'turtle',
    'jellyfish',
    'crab',
    'shrimp',
    'plastic trash',
    'rov',
    'fabric trash',
    'fishing trash',
    'metal trash',
    'paper trash',
    'rubber trash',
    'wood trash',
)

# --- 初始化 COCO ---
coco = COCO(ann_file)

# --- 目标统计字典 ---
stats = {cls: {'small': 0, 'medium': 0, 'large': 0, 'total': 0} for cls in CLASSES}
small, medium, large = 0, 0, 0

# --- 遍历每个标注 ---
for ann in coco.dataset['annotations']:
    cat_id = ann['category_id']
    cat_name = coco.loadCats(cat_id)[0]['name']
    w, h = ann['bbox'][2], ann['bbox'][3]
    area = w * h

    # 判断尺度（COCO定义）
    if area < 32 ** 2:
        size = 'small'
    elif area < 96 ** 2:
        size = 'medium'
    else:
        size = 'large'

    # 更新统计
    if cat_name in stats:
        stats[cat_name][size] += 1
        stats[cat_name]['total'] += 1

    # 统计全局总量
    if size == 'small':
        small += 1
    elif size == 'medium':
        medium += 1
    else:
        large += 1

# --- 转换为DataFrame便于查看 ---
df = pd.DataFrame(stats).T
df['small%'] = (df['small'] / df['total'].replace(0, 1) * 100).round(1)
df['medium%'] = (df['medium'] / df['total'].replace(0, 1) * 100).round(1)
df['large%'] = (df['large'] / df['total'].replace(0, 1) * 100).round(1)
df = df.sort_values('total', ascending=False)

print(df)
print('\n=== Overall Statistics ===')
print(f'Total small:  {small}')
print(f'Total medium: {medium}')
print(f'Total large:  {large}')
print(f'Total all:    {small + medium + large}')

# 可选：保存结果
df.to_csv('object_size_distribution.csv', index=True)
print("\nSaved to object_size_distribution.csv")

import matplotlib.pyplot as plt

df[['small%', 'medium%', 'large%']].plot(kind='bar', stacked=True, figsize=(12,6))
plt.title('Proportion of Small/Medium/Large Objects per Class')
plt.ylabel('Percentage (%)')
plt.xlabel('Class')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()