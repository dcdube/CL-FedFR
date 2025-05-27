# Head Pose-Based Curriculum Learning for Federated Face Recognition

This repository introduces a curriculum learning framework for Federated Face Recognition (FR) based on **head pose difficulty ranking (HR)**.

Paper: [OpenFace paper](https://ieeexplore.ieee.org/abstract/document/7477553)
Code: [OpenFace repository](https://github.com/TadasBaltrusaitis/OpenFace)

<img src="HeadPose_sort.png"> 

## Overview

In this design, we sort the dataset images from **easy to difficult** based on **head pose angles**. The difficulty is estimated using the **sum of the absolute values** of the **pitch**, **yaw**, and **roll** angles of the head pose, as illustrated in **Figure 4**.

### Dataset & Pose Estimation

- The FR model is trained using a subset of the **MS-Celeb-1M** dataset [49].
- **OpenFace 2.2.0** is used to estimate head pose angles (pitch, yaw, roll) for each identity.
- This version improves upon **OpenFace 2.0** [50].

### Data Analysis

- **Figure 5** shows histograms of the absolute head pose angles and their sum.
  - **Yaw** angles have the widest distribution.
  - **Roll** angles are mostly under 10°.
  - Only about **1% of images** have a total absolute head pose angle exceeding **50°**.

### Curriculum Strategy

1. **Sorting**: For each client, images are ranked based on the total absolute head pose angle.
2. **Subset Splitting**: Images are divided into:
   - **Easy samples**: Top 50% with the smallest pose angles.
   - **Difficult samples**: Bottom 50% with larger pose angles.
3. **Training Process**:
   - **Stage 1**: Train on easy samples.
   - **Stage 2**: Use model parameters from Stage 1 as initialization and train on all samples.

This training pipeline is illustrated in **Algorithm 1** and **Figure 3**.

## References

- [49] MS-Celeb-1M Dataset  
- [50] OpenFace 2.0

---
