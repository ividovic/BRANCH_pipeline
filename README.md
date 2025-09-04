# Towards Robotic Pruning:  Automated Annotation and Prediction of Branches for Pruning on Trees Reconstructed using RGB-D Images
This is the official repository for the pipeline presented in the paper "Towards Robotic Pruning: Automated Annotation and Prediction of Branches for Pruning on Trees Reconstructed using RGB-D Images" submitted to [Sensors](https://www.mdpi.com/journal/sensors) on 8 August 2025.

Authors: Jana Dukić, Petra Pejić, Ivan Vidović and Emmanuel Karlo Nyarko from [Faculty of Electrical Engineering, Computer Science and Information Technology Osijek, Croatia](https://www.ferit.unios.hr/about-ferit/general)

**NOTE**: Currently, the repository contains only the BRANCH_v2 dataset presented in the paper. Upon the paper is accepted and published we will add code for the registration and prediciton part of the proposed pipeline.

## Abstract
This paper presents a comprehensive pipeline for automated prediction of branches to be pruned, integrating 3D reconstruction of fruit trees, automatic branch labeling, and pruning prediction. The workflow begins with capturing multi-view RGB-D images in orchard settings, followed by generating and preprocessing point clouds to reconstruct partial 3D models of pear trees using the TEASER++ algorithm. Differences between pre- and post-pruning models are used to automatically label branches to be pruned, creating a valuable dataset for both reconstruction methods and training machine learning models. A neural network based on PointNet++ is trained to predict branches to be pruned directly on point clouds, with performance evaluated through quantitative metrics and visual inspections. The pipeline demonstrates promising results, enabling real-time prediction suitable for robotic implementation. While some inaccuracies remain, this work lays a solid foundation for future advancements in autonomous orchard management, aiming to improve precision, speed, and practicality of robotic pruning systems.

## Dataset BRANCH_v2 
The dataset is available [here](https://puh.srce.hr/s/EoPqgASGerLapne).

The dataset is designed for research in 3D point cloud registration and reconstruction of trees, as well as for the detection of branches to be pruned \[1\].
This dataset constitutes an enhanced version of the BRANCH dataset \[2\].

## 3D registration and reconstruction

The dataset includes RGB-D images of pear trees captured both before and after pruning within an orchard. Each tree is captured from multiple viewpoints.

Using these images and the reconstruction methodology outlined in \[2\], each tree is reconstructed into a 3D model represented by a point cloud. Corresponding models are provided for each tree both pre- and post-pruning.

Point clouds of individual trees, before and after pruning, are registered, and annotations are assigned following the process described in \[2\]. Each point in the cloud is labeled as follows: 1 if it belongs to a pruned branch, or 0 if it pertains to other parts of the tree that are not subject to pruning.

This dataset supports research in point cloud registration for creating accurate 3D models of real trees without relying on existing ground truth data. It allows for various preprocessing steps, such as removing grass, background noise, neighboring trees, etc. Additionally, the dataset can facilitate registration between pre- and post-pruning tree models, enabling data-driven annotation of branches to be pruned.

_Description, instructions and code will be added after the paper publication._

## Prediction of branches to be pruned

The dataset includes annotations identifying points belonging to branches to be pruned on tree models prior to pruning. For neural network training and branch prediction tasks, the annotated dataset is available in multiple formats: the original version, a normalized and centered version, and eight variants that are normalized, centered, and voxelized. The voxel size in these variants ranges from 2.5 mm to 2 cm. 

These annotated point clouds can be used as input for training models and as ground truth for evaluating branch prediction accuracy.

_Description, instructions and code will be added after the paper publication._

## Citation

If you use the code and/or dataset, please cite:

\[1\] Dukić, J.; Pejić, P.; Vidović, I.; Nyarko, E.K. Towards Robotic Pruning: Automated Annotation and Prediction of Branches for Pruning on Trees Reconstructed using RGB-D Images, _to be published_

and additionally:

\[2\] Dukić, J.; Pejić, P.; Bošnjak, A.; Nyarko, E.K. BRANCH - a Labeled Dataset of RGB-D Images and 3D Models for Autonomous Tree Pruning. In Proceedings of the 2024 International Conference on Smart Systems and Technologies (SST), 2024, pp. 57–64. 754 https://doi.org/10.1109/SST61991.2024.10755414
