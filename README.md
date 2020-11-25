# RSGNet: Relation based Skeleton Graph Network for Crowded Scenes Pose Estimation
# Introduction
This is an official pytorch implementation of " RSGNet: Relation based Skeleton Graph Network for Crowded Scenes Pose Estimation". In this work, we are committed to solving the problem of multi-person pose estimation under the condition of “crowded scenes”, where RGB images capture complex real-world scenes with highly-overlapped people, severe occlusions and diverse postures. In particular, we focus on two main problems: 1) how to design an effective pipeline for crowded scenes pose estimation; and 2) how to equip this pipeline with the ability of relation modeling for interference resolving. To tackle these problems, we propose a new pipeline named Relation based **Skeleton Graph Network (RSGNet)**. Unlike existing works that directly predict joints-of-target by labeling joints-of-interference as false positive, we first encourage all joints to be predicted. And then, a **Target-aware Relation Parser (TRP)** is designed to model the relation over all predicted joints, resulting in a target-aware encoding. This new pipeline will largely relieve the confusion of the joints estimation model when seeing identical joints with totally distinct labels (e.g., the identical hand exists in two bounding boxes). Furthermore, we introduce a **Skeleton Graph Machine (SGM)** to model the skeleton-based commonsense knowledge, aiming to estimate the target pose with the constraint of human body structure. Such skeleton-based constraint can help to deal with the challenges in crowded scenes from a reasoning perspective. Solid experiments on pose estimation benchmarks demonstrate that our method outperforms existing state-of-the-art methods.
![](https://github.com/vikki-dai/RSGNet/blob/main/figures/framework_RSGNet.png)
# Main Results
## Results on CrowdPose test dataset
![](https://github.com/vikki-dai/RSGNet/blob/main/visualization/main_results_CrowdPose.png)
**Note**:
1. Flip test is used.
2. Person detector has person AP of 71.0 on CrowdPose test dataset.
3. GFLOPs is for convolution and linear layers only.
## Results on  COCO val2017 dataset
![](https://github.com/vikki-dai/RSGNet/blob/main/visualization/main_results_COCOval.png)
**Note**:
1. Flip test is used.
2. Person detector has person AP of 56.4 on COCO val2017 dataset.
3. GFLOPs is for convolution and linear layers only.
## Results on COCO test-dev2017 dataset
![](https://github.com/vikki-dai/RSGNet/blob/main/visualization/main_results_COCO_testdev.png)
**Note**:
1. Flip test is used.
2. Person detector has person AP of 60.9 on COCO test-dev2017 dataset.
3. GFLOPs is for convolution and linear layers only.
# Environment
The code is developed based on the [HRNet project](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch). NVIDIA GPUs are needed. The code is developed and tested using 4 NVIDIA RTX GPU cards. Other platforms or GPU cards are not fully tested.
# Installation
1. Install pytorch >= v1.0.0 following official instruction. Note that if you use pytorch's version < v1.0.0, you should following the instruction at https://github.com/Microsoft/human-pose-estimation.pytorch to disable cudnn's implementations of BatchNorm layer. We encourage you to use higher pytorch's version(>=v1.0.0)
2. Clone this repo, and we'll call the directory that you cloned as ${POSE_ROOT}.
3. Install requirments：
```python
  pip install -r requirements.txt
```
4. Make libs:
```python
  cd ${POSE_ROOT}/lib
  make
```
5. Install [COCOAPI](https://github.com/cocodataset/cocoapi):
```python
  # COCOAPI=/path/to/clone/cocoapi
  git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
  cd $COCOAPI/PythonAPI
  # Install into global site-packages
  make install
  # Alternatively, if you do not have permissions or prefer
  # not to install the COCO API into global site-packages
  python3 setup.py install --user 
```
6. Install [CrowdPoseAPI](https://github.com/Jeff-sjtu/CrowdPose)
```python
  Install CrowdPoseAPI exactly the same as COCOAPI.
  Reverse the bug stated in https://github.com/Jeff-sjtu/CrowdPose/commit/785e70d269a554b2ba29daf137354103221f479e**
```
7. Init output and log directory:
```python
  mkdir output 
  mkdir log
```
# Data Preparation
* For **COCO data**, please download from [COCO download](https://cocodataset.org/#download), 2017 Train/Val is needed for COCO keypoints training and validation. We also provide person detection result of COCO val2017 and test-dev2017 to reproduce our multi-person pose estimation results. Please download and extract them under {POSE_ROOT}/data.  

* For **CrowdPose data**, please download from [CrowdPose download](https://github.com/Jeff-sjtu/CrowdPose#dataset), Train/Val is needed for CrowdPose keypoints training and validation. Please download and extract them under {POSE_ROOT}/data.
# Training and Testing
* Testing on CrowdPose dataset using [model zoo's models](https://github.com/vikki-dai/RSGNet/blob/main/model_zoo.txt)
```python
  CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/cp_train.py \
  --cfg experiments/crowdpose/hrnet/rsgnet_w32_256x192_adam_lr1e-3.yaml \
```
* Training on CrowdPose dataset
```python
  CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/cp_test.py \
  --cfg experiments/crowdpose/hrnet/rsgnet_w32_256x192_adam_lr1e-3.yaml \
  TEST.MODEL_FILE cp_rsgnet_w32_256.pth
```
* Testing on COCO-val dataset using [model zoo's models](https://github.com/vikki-dai/RSGNet/blob/main/model_zoo.txt)
```python
  CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/rsgnet_train.py \
  --cfg experiments/coco/hrnet/rsgnet_w32_256x192_adam_lr1e-3.yaml \
```
* Training on COCO-val dataset
```python
  CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/rsgnet_test.py \
  --cfg experiments/cocoe/hrnet/rsgnet_w32_256x192_adam_lr1e-3.yaml \
  TEST.MODEL_FILE coco_rsgnet_w32_256.pth
```
