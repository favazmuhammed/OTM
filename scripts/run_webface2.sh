CUDA_VISIBLE_DEVICES=0 python main.py \
    --data_root ./data \
    --train_data_path faces_webface_112x112 \
    --val_data_path faces_webface_112x112 \
    --prefix ir18_webface_adaface \
    --use_wandb \
    --use_mxrecord \
    --gpus 1 \
    --use_16bit \
    --arch ir_18 \
    --batch_size 512 \
    --num_workers 10 \
    --epochs 10\
    --lr_milestones 5,8 \
    --lr 0.1 \
    --head adaface \
    --m 0.4 \
    --h 0.333 \
    --s 64.0 \
    --t_alpha 0.1\
    --distributed_backend ddp \
    --low_res_augmentation_prob 0.2 \
    --crop_augmentation_prob 0.2 \
    --photometric_augmentation_prob 0.2\

~                                                                                                                                  ~                                                                                                                                  ~                                                                                                                                  ~                                                                                                                                  ~                                                                                                                                  ~                                                                                                                                  ~                                                                                                                                              
