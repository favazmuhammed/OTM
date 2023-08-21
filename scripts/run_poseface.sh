CUDA_VISIBLE_DEVICES=0 python main.py \
    --data_root ./ \
    --train_data_path ../adaface_org/AdaFace/data/faces_emore/imgs_subset \
    --val_data_path data/DL23_Dataset \
    --test_data_path data/faces_webface_112x112 \
    --prefix irse50_4m_adaface \
    --use_wandb \
    --gpus 1 \
    --use_16bit \
    --arch ir_se_50 \
    --batch_size 128 \
    --num_workers 16 \
    --epochs 26\
    --lr_milestones 12,20,24 \
    --lr 0.1 \
    --head poseface \
    --m 0.4\
    --h 0.333 \
    --s 64.0 \
    --z_alpha 1.0 \
    --t_alpha 0.01 \
    --distributed_backend ddp \
    --low_res_augmentation_prob 0.2 \
    --crop_augmentation_prob 0.2 \
    --photometric_augmentation_prob 0.2\
    --custom_num_class 52158\

~  