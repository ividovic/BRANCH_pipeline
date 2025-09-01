# Towards Robotic Pruning:  Automated Annotation and Prediction of Branches for Pruning on Trees Reconstructed using RGB-D Images
This is the official repository for the pipeline presented in the paper "Towards Robotic Pruning: Automated Annotation and Prediction of Branches for Pruning on Trees Reconstructed using RGB-D Images" submitted to [Sensors](https://www.mdpi.com/journal/sensors) on 8 August 2025.

Authors: Jana Dukić, Petra Pejić, Ivan Vidović and Emmanuel Karlo Nyarko from [Faculty of Electrical Engineering, Computer Science and Information Technology Osijek, Croatia](https://www.ferit.unios.hr/about-ferit/general)

**NOTE**: Currently, the repository contains only the BRANCH_v2 dataset presented in the paper. Upon the paper is accepted and published we will add code for the registration and prediciton part of the proposed pipeline.

## Abstract
This paper presents a comprehensive pipeline for automated prediction of branches to be pruned, integrating 3D reconstruction of fruit trees, automatic branch labeling, and pruning prediction. The workflow begins with capturing multi-view RGB-D images in orchard settings, followed by generating and preprocessing point clouds to reconstruct partial 3D models of pear trees using the TEASER++ algorithm. Differences between pre- and post-pruning models are used to automatically label branches to be pruned, creating a valuable dataset for both reconstruction methods and training machine learning models. A neural network based on PointNet++ is trained to predict branches to be pruned directly on point clouds, with performance evaluated through quantitative metrics and visual inspections. The pipeline demonstrates promising results, enabling real-time prediction suitable for robotic implementation. While some inaccuracies remain, this work lays a solid foundation for future advancements in autonomous orchard management, aiming to improve precision, speed, and practicality of robotic pruning systems.

## Dataset
TODO - add description

## Point clouds registration
Description, instructions and code will be added after the paper publication.

## Prediction of branches for pruning
Description, instructions and code will be added after the paper publication.
