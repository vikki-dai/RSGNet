# RSGNet: Relation based Skeleton Graph Network for Crowded Scenes Pose Estimation
# Introduction
This is an official pytorch implementation of " RSGNet: Relation based Skeleton Graph Network for Crowded Scenes Pose Estimation". In this work, we are committed to solving the problem of multi-person pose estimation under the condition of “crowded scenes”, where RGB images capture complex real-world scenes with highly-overlapped people, severe occlusions and diverse postures. In particular, we focus on two main problems: 1) how to design an effective pipeline for crowded scenes pose estimation; and 2) how to equip this pipeline with the ability of relation modeling for interference resolving. To tackle these problems, we propose a new pipeline named Relation based **Skeleton Graph Network (RSGNet)**. Unlike existing works that directly predict joints-of-target by labeling joints-of-interference as false positive, we first encourage all joints to be predicted. And then, a **Target-aware Relation Parser (TRP)** is designed to model the relation over all predicted joints, resulting in a target-aware encoding. This new pipeline will largely relieve the confusion of the joints estimation model when seeing identical joints with totally distinct labels (e.g., the identical hand exists in two bounding boxes). Furthermore, we introduce a **Skeleton Graph Machine (SGM)** to model the skeleton-based commonsense knowledge, aiming to estimate the target pose with the constraint of human body structure. Such skeleton-based constraint can help to deal with the challenges in crowded scenes from a reasoning perspective. Solid experiments on pose estimation benchmarks demonstrate that our method outperforms existing state-of-the-art methods.
![](https://github.com/vikki-dai/RSGNetfigures/framework_RSGNet.png)
# Main Results
## Results on CrowdPose test set
![](https://github.com/vikki-dai/RSGNet/blob/main/visualization/main_results_CrowdPose.png)
## Note:
* Flip test is used.
* Person detector has person AP of 71.0 on CrowdPose test dataset.
* GFLOPs is for convolution and linear layers only.
