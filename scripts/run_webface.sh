CUDA_VISIBLE_DEVICES=0 python main.py \
    --data_root ./ \
    --train_data_path ../webface_subset \
    --val_data_path data/faces_webface_112x112 \
    --prefix ir18_webface_subset \
    --use_wandb \
    --gpus 1 \
    --use_16bit \
    --arch ir_18 \
    --batch_size 256 \
    --num_workers 16 \
    --epochs 26\
    --lr_milestones 12,20,24 \
    --lr 0.1 \
    --head adaface \
    --m 0.5 \
    --h 0.333 \
    --s 64.0 \
    --distributed_backend ddp \
    --low_res_augmentation_prob 0.2 \
    --crop_augmentation_prob 0.2 \
    --photometric_augmentation_prob 0.2\
    --custom_num_class 3980\

~                                                                                                                                  ~                                                                                                                                  ~                                                                                                                                  ~                                                                                                                                  ~                                                                                                                                  ~                                                                                                                                  ~                                                                                                                                              
