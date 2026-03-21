<div align="center">
  <h1><b>Poseidon-300K: Towards a Large-Scale Open Benchmark for Underwater Object Detection</b></h1>
  <p>
    <a href="https://arxiv.org/abs/2603.xxxxx"><img src="https://img.shields.io/badge/arXiv-2603.xxxxx-b31b1b.svg" alt="arXiv"></a>
   <a href="https://github.com/your-username/HistEqDet">
    <img src="https://img.shields.io/badge/Github-page-yellow.svg?logo=github" alt="Github">
  </a>
     <a href="https://pan.baidu.com/s/1uPXBMNPL0Amgg7bHbf7vHQ?pwd=sajp">
    <img src="https://img.shields.io/badge/BaiduNetdisk-dataset-0066FF.svg?logo=baidu" alt="BaiduNetdisk">
  </a>
     <a href="https://1drv.ms/u/c/869d5f1d401ac2ec/EZekCm-CurxGpYvC6isW8tgBUvvpJ6hurMe85xgqWK2qvw">
    <img src="https://img.shields.io/badge/OneDrive-dataset-0078D4.svg?logo=microsoft-onedrive" alt="OneDrive">
  </a>
</p>
  </p>

  <p>
    If you find our work helpful, please consider giving us a ⭐!
  </p>

  <img src="docs/p2.png" width="950" alt="Teaser Image">
  <br>
</div>

---

<!-- ## 🔥 Highlights -->


<!-- --- -->

## 📝 Abstract
Underwater object detection (UOD) plays a crucial role in marine ecosystem monitoring and environmental analysis, yet its progress is significantly hindered by the scarcity of large-scale standardized benchmarks and the severe feature ambiguity caused by aquatic image degradation.
To overcome these issues, we introduce **Poseidon-300K**, a large-scale open-source UOD benchmark comprising **85,000** images with over **300,000** detection instances, together with **258,000** annotated question-answer pairs for visual-semantic learning. To our knowledge, it constitutes the largest publicly accessible dataset in this field. 
Given that contrast attenuation constitutes the primary bottleneck in underwater vision, we contend that enhancing the feature-level contrast between foreground targets and backgrounds within the detection framework can fundamentally resolve the performance degradation. Consequently, we introduce HistEqDet, an embarrassingly simple yet effective framework that pioneers the integration of a dynamic Histogram Equalization (HE) mechanism directly within the detection architecture. This mechanism effectively suppresses background interference while accentuating target-relevant features. Furthermore, we adopt a top-down vision-language pretraining paradigm to internalize underwater-specific visual priors, providing a semantically grounded initialization for underwater tasks.
Extensive experiments on Poseidon-300K demonstrate that HistEqDet achieves state-of-the-art performance with 67.8\% mAP, validating the effectiveness of the method.

<p align="center">
  <img src="docs/P1.png" width="1000" alt="Pipeline">
  <br>
</p>



---

## 🌊 Poseidon-300K Dataset
**Poseidon-300K** is a large-scale and high-quality benchmark dataset for underwater object detection.Poseidon-300K comprises two specialized subsets: Poseidon-Det and Poseidon-VQA.

### 🖼️ Visualization
**Poseidon-Det** is a long-tail distributed benchmark comprising **19 marine categories**. The following figure presents the  representative examples from **Poseidon-VQA**.

<p align="center">
  <img src="docs/Det-VQA.png" width="1000" alt="Dataset Visualization">
</p>

### 📊 Dataset Statistics
We compare Poseidon-300K with existing benchmarks to highlight its scale and diversity:
| Dataset | Image | Instance | Categories | COCO-Annotation | Pairs | Uneven Lighting | Blurriness | Low Contrast | Color Distortion |
|---------|------:|---------:|:-----:|:----------:|:-----------:|:---------------:|:----------:|:------------:|:---------------:|
| F4K-species | 27,370 | 27,370 | 23 | Y | - | ✅ | ✅ | ❌ | ❌ |
| UTDAC | 5,643 | 46,685 | 4 | Y | - | ✅ | ✅ | ✅ | ✅ |
| DUO | 7,782 | 74,515 | 4 | Y | - | ✅ | ✅ | ✅ | ✅ |
| UODD | 3,194 | 19,212 | 3 | Y | - | ✅ | ✅ | ✅ | ✅ |
| RUOD | 14,000 | 74,903 | 10 | Y | - | ✅ | ✅ | ✅ | ✅ |
| Brackish | 14,518 | 25,613 | 6 | N | - | ✅ | ✅ | ❌ | ❌ |
| Trash ICRA19 | 7,684 | 11,061 | 3 | Y | - | ❌ | ❌ | ✅ | ❌ |
| TrashCan | 7,212 | 12,480 | 16 | N | - | ❌ | ❌ | ✅ | ❌ |
| **Poseidon-300K** | **84,735** | **300,112** | **19** | **Y** | **258K (VQA)** | ✅ | ✅ | ✅ | ✅ |

### 📥 Download
- **Poseidon-Det**: [[Google Drive]](https://1drv.ms/u/c/869d5f1d401ac2ec/EZekCm-CurxGpYvC6isW8tgBUvvpJ6hurMe85xgqWK2qvw) | [[Baidu Pan]](https://pan.baidu.com/s/1uPXBMNPL0Amgg7bHbf7vHQ?pwd=sajp) 
- **Poseidon-VQA**: [[Link]](https://pan.baidu.com/s/1eMru9zBXrZJxEUtCi-swQQ?pwd=39pz)



---
## 💡 HistEqDet:Histogram Equalization Detection Model
<p align="center">
  <img src="docs/HistEqDet.png" width="850" alt="HistEqDet Framework">
  <br>
</p>

### Introduction
This repository is the official implementation of Histogram Equalization Detection Model in "Poseidon-300K: Towards a Large-Scale Open Benchmark for Underwater Object Detection"

Download the pretrained file and place it in the pretrained/ directory.[pretrained file](https://pan.baidu.com/s/1qiVlnYvlG-Rl7l5lAwkN7w?pwd=mayd)

---

## 📊 Results and Models
Comparison of different methods on Poseidon-300K.
| Method |mAP|@50   | @75 | @s | @m | @l |Config | Download |
| :-------| :------: | :------: | :------: | :------:| :------: | :------:| :------:| :------: |
|YOLOF |48.6 | 73.1 | 54.3  | 39.1 | 49.5 | 51.9|[Config](https://pan.baidu.com/s/12nkYYOttV9Xqes3hPZ25KQ?pwd=rq9p)|[Download](https://pan.baidu.com/s/1kU7Hn-T9svLKIj8vNfUkrA?pwd=kxg5)
|Foveabox|52.6 | 79.2 | 58.8 |38.1|46.9|56.6|[Config](https://pan.baidu.com/s/1yZJEhg0xCQ9Lx4hZxZCBJA?pwd=a2zn)|[Download](https://pan.baidu.com/s/12TwzX9Mbe_fhgy16o0wzog?pwd=nrvs)
|RetinaNet | 56.4 | 81.9 | 63.3 | 40.6 | 50.7 | 61.1 |[Config](https://pan.baidu.com/s/1MzPCmDlwu8LYdHY6fKxaAw?pwd=hweu)|[Download](https://pan.baidu.com/s/1K9bAxFdSKxYJQi_6mS4P4Q?pwd=72ve)
|FSAF |57.0	|83.7	|64.5	|45.9	|50.5	|62.5|[Config](https://pan.baidu.com/s/1Ys981tTvEL-iqn92hGwA2w?pwd=vdqz)|[Download](https://pan.baidu.com/s/11jTPh__3JCjIYa3p38AKsg?pwd=i1kv)
|RepPoints |59.0 |85.8 | 67.0 |43.8 |53.0 | 64.8|[Config](https://pan.baidu.com/s/1_5HsilTnkSHuOzItj2IQAA?pwd=h968)|[Download](https://pan.baidu.com/s/1GM6I7NYGLHd_Bm9DDG7S7w?pwd=bjk1)
|FCOS | 60.2 | 88.2 | 67.8 | 44.8 | 54.9 | 64.2|[Config](https://pan.baidu.com/s/1JFhPsWRtiqN260j_xIRh7g?pwd=7xgk)|[Download](https://pan.baidu.com/s/1btnLjiT-Rbp-lovRSxc77w?pwd=8cwr)
| AutoAssign|62.9 | 89.0 | 71.2  |52.3 | 56.6 | 67.3 |[Config](https://pan.baidu.com/s/1qTIVJOWb40lo4WSoqpGxIA?pwd=ryu8)|[Download](https://pan.baidu.com/s/1SmSjkb7p8LQD0ZnDGEPFsw?pwd=6x42)
|VFNet|63.7 | 87.1 | 72.2 | 50.2 | 57.1 | 68.5 |[Config](https://pan.baidu.com/s/1cmMGZgohXocyBG0tiJVWxA?pwd=njf6)|[Download](https://pan.baidu.com/s/1Ttuz2C5XbLleIyIoBHzKYw?pwd=qbdy)
|GFL| 64.0 | 87.4 | 72.5 | 49.5 | 57.0 | 69.2|[Config](https://pan.baidu.com/s/1VQ96t1uijp1_1koqetteAw?pwd=wsjn)|[Download](https://pan.baidu.com/s/1VoHz2q39HVGxpNEKPqetYA?pwd=dvg4)
|ATSS | 64.3 |  88.8 | 72.9 | 49.8 | 57.4 | 68.8|[Config](https://pan.baidu.com/s/1IKnYz0VOepbT0JCrsyYwTA?pwd=xbmh)|[Download](https://pan.baidu.com/s/1V7IpvvXnPQPfUspT4tKJ3g?pwd=t6u5)
|TOOD |64.8| 89.1  |73.6 | 50.9  |57.7  |69.8|[Config](https://pan.baidu.com/s/1PyKc-58nzxTzEaIULA2ZQA?pwd=umf6)|[Download](https://pan.baidu.com/s/14zihn2y6LT5EijpM6eCrdg?pwd=cae9)
|Faster RCNN | 61.4 | 87.7 | 69.9 | 49.8 | 55.3 | 65.0|[Config](https://pan.baidu.com/s/1O74yioBmTJ8YCj_ZACHlxg?pwd=djae)|[Download](https://pan.baidu.com/s/1QDZhwrSnmTx8ZpuARxudbA?pwd=dwgi)
|Libra RCNN |61.7 | 87.9 | 70.9 | 48.7 | 55.7 | 66.2|[Config](https://pan.baidu.com/s/19-c4mP3o5XoZVF0m1k1ITw?pwd=v24f)|[Download](https://pan.baidu.com/s/10YWQq8S33Zmth_mIcga41w?pwd=5ngw)
|Grid RCNN| 62.3 | 87.0 | 71.2 | 50.5 | 56.3 | 66.5|[Config](https://pan.baidu.com/s/1H6G1Y_U6upj8_duLFReG2g?pwd=e6mc)|[Download](https://pan.baidu.com/s/18vTvKcDW84CKmbw6Mgg8lA?pwd=epqu)
| Dynamic RCNN| 63.8 | 87.7 | 72.9 | 50.4 | 57.1 | 67.5|[Config](https://pan.baidu.com/s/1kGweCjtikvH2IAZXtWn_5w?pwd=cpvp)|[Download](https://pan.baidu.com/s/1UOpN3BDcl2RNROr1TsDl-g?pwd=v5jp)
|Cascade RCNN | 63.9 | 87.6 | 72.5 | 50.5 | 57.5 | 68.7|[Config](https://pan.baidu.com/s/14MG2C_DtlL5N85199nIDhA?pwd=9shw)|[Download](https://pan.baidu.com/s/1vsrKCv3jCdPfPKMfI6bp9A?pwd=84kk)
| Sparse RCNN|56.8 | 83.6 | 63.5 | 42.9 | 49.4 | 62.1 |[Config](https://pan.baidu.com/s/1CD8dabhmvs7TfLiWUPALwQ?pwd=ehrs)|[Download](https://pan.baidu.com/s/1Ecj19HcnTIuJgmE7Vpd3qQ?pwd=uv9n)
|GCCNet |53.8 | 82.2 | 59.7 | 46.8 | 49.2 | 56.3 |[Config](https://pan.baidu.com/s/1TVy5kIErZc1oELpwra8cKA?pwd=3mkg)|[Download](https://pan.baidu.com/s/16cyNojQCNnyfF95bfQFH_Q?pwd=db8v)
|AMSP-UOD|55.9|81.8|74.0|41.1|48.4|65.0|[Config](https://pan.baidu.com/s/1AE0b4x8t0mghS-OoDutdIg?pwd=kjq6)|[Download](https://pan.baidu.com/s/1_eWCsclNSJmKl-QlDUQx-g?pwd=dqfe)
| Boost RCNN |57.6 | 84.3 | 66.1 | 41.8 | 50.1 | 62.3 |[Config](https://pan.baidu.com/s/1ZeFzaKKRfXq6V1K8X7JteA?pwd=zdym)|[Download](https://pan.baidu.com/s/1luEUqafC069VlP8hIfIVwQ?pwd=mjhw)
| ERLNet|58.5 | 83.1 | 65.8 | 44.8 | 52.8 | 60.3 |[Config](https://pan.baidu.com/s/1KLe39FtpDsVDCCgS5KZlAQ?pwd=scd5)|[Download](https://pan.baidu.com/s/1Pde-YnDl80ogCJYOMqBwbw?pwd=4fbx)
|UnitModule |58.8 | 86.0 | 64.8 | 45.8| 51.8 | 66.7|[Config](https://pan.baidu.com/s/1lE0eV8quNeM2Dw0Q7riz-w?pwd=e3iy)|[Download](https://pan.baidu.com/s/18V7Y_HyGW8smjXgjz9DX1g?pwd=re4s)
|HistEqDet |67.8|88.2| 76.8 | 54.3 | 61.4 | 70.2|[Config](https://pan.baidu.com/s/1OXEXsttuOqnO1BBg4q4dlg?pwd=r84u)|[Download](https://pan.baidu.com/s/1acdIsYJabAehO829zlJbKA?pwd=3kjd)
---

## 🛠️ Installation
Our code depends on [PyTorch](https://pytorch.org/), [MMCV](https://github.com/open-mmlab/mmcv) and [MMDetection](https://github.com/open-mmlab/mmdetection). Below are quick steps for installation. Please refer to the [Install Guide](https://mmdetection.readthedocs.io/en/latest/get_started.html) for more detailed instructions.

---
### Step 1: Create a conda environment

```shell
conda create --name poseidon python=3.8
conda activate poseidon
```

### Step 2: Install the required packages

```shell
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -U openmim dadaptation cmake lit --no-input
mim install mmengine "mmcv==2.0.1" "mmdet==3.0.0" "mmrotate==1.0.0rc1"
```

### Step 3: Insatall flash attention
```shell
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout v0.2.8
pip install ninja
python setup.py install
cd ..
```

### Step 4: Install poseidon
```shell
pip install -r requirements.txt
pip install -e .
```

### Step 5: Compile custom operators
```shell
cd ops
sh make.sh
cd ..
```
---
### 🚀 Train
```shell
sh tools/dist_train.sh configs/HistEq/config_vit.py 4
```
### 🔍 Test
```shell
python tools/test.py configs/HistEq/config_vit.py <path/to/checkpoints>
```

## Citation
If you use this toolbox or benchmark in your research, please cite this project.
