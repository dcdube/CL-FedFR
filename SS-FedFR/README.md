# SS-FedFR: Semi-Supervised Federated Face Recognition

This section explains how the SS-FedFR model is trained. This work is based on the **FedFR** algorithm below:

- 📄 **Paper**: [FedFR: Joint Optimization Federated Framework for Generic and Personalized Face Recognition](https://ojs.aaai.org/index.php/AAAI/article/view/20057)  
- 💻 **Code**: [FedFR GitHub Repository](https://github.com/jackie840129/FedFR)

## Obtain Training Dataset

We used a similar strategy to select a subset of the [MS-Celeb-1M](https://doi.org/10.1007/978-3-319-46487-9_6) as in FedFR. Please follow the steps in [split_dataset](https://github.com/jackie840129/FedFR/tree/main/split_dataset) to obtain the dataset.

## Prerequisite

Put the pretrained model (["backbone.pth"](https://drive.google.com/file/d/19d-Qm-RkBh9E2P1o_ZbdrHAyoZocFZbK/view?usp=sharing)) under the `pretrain/` folder.

## Semi-Supervised Learning

- Pseudo-labeling:
  - We added two new parameters to the `run.sh` command as *threshold* and *split_ratio*. The threshold is the pseudo-labeling confidence and the split_ratio defines the proportion of labeled and unlabeled data in the clients (e.g. 0.2 → 20% labeled 80% unlabeled).
  - We added **get_pseudo_labeled_data** function to `client.py` file. It gets the threshold and backbone model as parameters. By using unlabeled data in the client's **train_unlabeled_loader** attribute, it pseudo-labels and filters the samples by threshold and returns the pseudo-labeled dataset. 
  - We updated **train_with_public_data** function in `client.py` file. Before every local epoch it calls **get_pseudo_labeled_data** function and gets the pseudo-labeled data and combines it with its labeled dataset.

- Training:
  1. In `config.py`, first change the path of `config.rec` and `config.local_rec`.
  2. Run with command `sh run.sh`.
  3. After training, the models will be saved in the checkpoint directory, eg. `ckpt/FedFR/`.

## Personalized Evaluation

The personalized performance was evaluated on the [MS-Celeb-1M](https://doi.org/10.1007/978-3-319-46487-9_6). Each model is trained at the client before being aggregated at the server using FedAvg. We evaluated at a specific checkpoint for some epoch and single type of evaluation ('1:1' or '1:n'). Make sure to change the paths.

### 1:1 Evaluation
```
python3 local_all.py --backbone 'multi' --task '1:1' --ckpt_path '/home/master/SS-FedFR/ckpt/FedFR' --data_dir '/home/master/SS-FedFR/ms1m_split/local_veri_4000' --gallery '/home/master/SS-FedFR/ms1m_split/local_gallery_4000' --epoch -1 --num_client 40 --gpu 0 1 2 3
```

### 1:n Evaluation
```
python3 local_all.py --backbone 'multi' --task '1:n' --ckpt_path '/home/master/SS-FedFR/ckpt/FedFR' --data_dir '/home/master/SS-FedFR/ms1m_split/local_veri_4000' --gallery '/home/master/SS-FedFR/ms1m_split/local_gallery_4000' --epoch -1 --num_client 40 --gpu 0 1 2 3
```

## Generic Evaluation

The generic performance was evaluated on the [IJB-C](https://ieeexplore.ieee.org/abstract/document/8411217) dataset. We performed 'both' '1:1' or '1:n generic evaluations. Make sure to change the paths.
```
python3 ijbc_conti.py --root_path '/media/master/SS-FedFR/IJBC' --ckpt_dir '/home/master/SS-FedFR/ckpt/FedFR' --epoch 17 18 19 --gpu 0 1 2 3 --job 'both'
```

## References

Liu, C. T., Wang, C. Y., Chien, S. Y., & Lai, S. H. (2022, June). FedFR: Joint optimization federated framework for generic and personalized face recognition. In *Proceedings of the AAAI Conference on Artificial Intelligence* (Vol. 36, No. 2, pp. 1656-1664).