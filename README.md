# Curriculum Learning for Federated Face Recognition

This work proposes a federated FR framework that uses curriculum learning to sort the images from “easy” to “difficult” during training. Two distinct curricula are considered: Face image quality assessment (FIQA) scores and head rotation. The performance of these curriculum designs is assessed both for fully-supervised and semi-supervised federated face recognition setups.

<img src="CL_FedFR.png"> 

## Head Rotation-based Curriculum
We sort dataset images from "easy" to "difficult" based on head pose angles. The difficulty is estimated by computing the sum of the absolute values of the pitch, yaw, and roll angles using the [OpenFace 2.0](https://github.com/TadasBaltrusaitis/OpenFace) toolkit.

## Curriculum Strategy

- **Sorting**: For each client, images are ranked by the sum of absolute head pose angles.
- **Subset Splitting**:
We tried different splitting criteria but the best results were obtained using the following as shown in our [preliminary work](https://www.scitepress.org/Papers/2024/125740/125740.pdf):
   - 🟩 **Easy samples**: Top 50% (with low head pose angles).
   - 🟥 **Difficult samples**: Bottom 50% (with high head pose angles).

See the [Head Pose Sort](https://github.com/dcdube/CL-FedFR/tree/main/Head%20Pose%20Sort) folder for more details.

## Face Image Quality-based Curriculum

See the [CR-FIQA Sort](https://github.com/dcdube/CL-FedFR/tree/main/CR-FIQA%20Sort) folder for more details.

## Fully-Supervised Training
We use the [FedFR](https://github.com/jackie840129/FedFR) algorithm for all the fully-supervised training approaches.

## Semi-Supervised Training

See the [SS-FedFR](https://github.com/dcdube/CL-FedFR/tree/main/SS-FedFR) folder for more details.

## References

Baltrušaitis, T., Robinson, P., & Morency, L.P. (2016). OpenFace: An open source facial behavior analysis toolkit. In *2016 IEEE Winter Conference on Applications of Computer Vision (WACV)* (pp. 1–10).

Boutros, F., Fang, M., Klemt, M., Fu, B., & Damer, N. (2023). CR-FIQA: Face Image Quality Assessment by Learning Sample Relative Classifiability. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 5836–5845).

Liu, C. T., Wang, C. Y., Chien, S. Y., & Lai, S. H. (2022, June). FedFR: Joint optimization federated framework for generic and personalized face recognition. In *Proceedings of the AAAI Conference on Artificial Intelligence* (Vol. 36, No. 2, pp. 1656-1664).
