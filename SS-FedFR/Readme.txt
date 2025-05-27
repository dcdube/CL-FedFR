We have added two new parameters to the run.sh command as threshold and split_ratio.
Threshold is used for pseudo-labeling part.
Split ratio is used to define split_ratio of labeled and unlabeled data in the clients. (0.2 %20 labeled %80 unlabeled)

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 \
train.py --pretrained_root './pretrain'  --network 'sphnet' --output_dir './ckpt/FedFR' --loss 'CosFace'  \
--batch_size 64 --num_client 40 --client_sampled_ratio 1.0 --lr 0.001 --total_round 20 --local_epoch 10 --fedface \
--add_pretrained_data --combine_dataset  --contrastive_bb --return_all --BCE_local --adaptive_local_epoch --threshold 0.5 --split_ratio 0.1

Changes
-We have added get_pseudo_labeled_data function to client.py file. It gets threshold and backbone model as parameters. By using unlabeled data in client train_unlabeled_loader attribute it 
pseudo-labels and filter the samples by threshold value and returns the pseudo-labeled dataset. 
-In client folders in ckpt folder pseudo-labeling logs can be checked for each client.

-We have updated train_with_public_data function in client.py file. Before every local epoch it calls get_pseudo_labeled_data function and gets the pseudo-labeled data and combines it with its labeled dataset.

-We have added MXFaceDataset_Split_PseudoLabeled class into dataset.py file.
-We have updated MXFaceDataset_Combine class.
-We have updated creating_each_client function in All_Client_Dataset class to also initialize clients unlabeled dataset and other required attributes.

Evaluation Commands

python3 ijbc_conti.py --root_path '/media/master/BACKUP/Master_Backup/IJBC' --ckpt_dir '/home/master/FedFR-semi/ckpt/FedFR' --epoch 19 --gpu 0 1 2 3 --job 'both'   

Personalized Evaluation: 1:1 Evaluation
python3 local_all.py --backbone 'multi' --task '1:1' --ckpt_path '/home/master/FedFR-semi/ckpt/FedFR' --data_dir '/home/master/FedFR-semi/ms1m_split/local_veri_4000' --gallery '/home/master/FedFR-semi/ms1m_split/local_gallery_4000' --epoch -1 --num_client 40 --gpu 0 1 2 3


Personalized Evaluation: 1:n Evaluation
python3 local_all.py --backbone 'multi' --task '1:n' --ckpt_path '/home/master/FedFR-semi/ckpt/FedFR' --data_dir '/home/master/FedFR-semi/ms1m_split/local_veri_4000' --gallery '/home/master/FedFR-semi/ms1m_split/local_gallery_4000' --epoch -1 --num_client 40 --gpu 0 1 2 3

