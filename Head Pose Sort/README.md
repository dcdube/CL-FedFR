<!-- # Sorting Images Using Head Pose Angles

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
- [50] OpenFace 2.0 -->

# Sorting Images Using Head Pose Angles

This section explains how the **head pose angles** which we estimated using **OpenFace** were used as a curriculum to sort the [MS-Celeb-1M](https://doi.org/10.1007/978-3-319-46487-9_6) dataset images from "easy" to "difficult." To understand how to estimate the **head pose angles**, please refer to the paper and repository below:

- 📄 **Paper**: [OpenFace: An open source facial behavior analysis toolkit](https://ieeexplore.ieee.org/abstract/document/7477553)  
- 💻 **Code**: [OpenFace GitHub Repository](https://github.com/TadasBaltrusaitis/OpenFace)

<img src="HeadPose_sort.png" alt="Head Pose Sorting Visualization">

## Overview

We sort dataset images from "easy" to "difficult" based on head pose angles. The difficulty is estimated by computing the sum of the absolute values of the pitch, yaw, and roll angles using the [OpenFace 2.0](https://github.com/TadasBaltrusaitis/OpenFace) toolkit.

## Curriculum Strategy

- **Sorting**: For each client, images are ranked by the sum of absolute head pose angles.
- **Subset Splitting**:
We tried different splitting criteria but the best results were obtained using the following as shown in our [preliminary work](https://www.scitepress.org/Papers/2024/125740/125740.pdf):
   - 🟩 **Easy samples**: Top 50% with the low head pose angles.
   - 🟥 **Difficult samples**: Bottom 50% with high head pose angles.
- **Training Process**:
   - **Stage 1**: Train the model on easy samples.
   - **Stage 2**: Initialize with Stage 1 parameters and continue training on all samples.

## References

Baltrušaitis, T., Robinson, P., & Morency, L.P. (2016). OpenFace: An open source facial behavior analysis toolkit. In *2016 IEEE Winter Conference on Applications of Computer Vision (WACV)* (pp. 1–10).
