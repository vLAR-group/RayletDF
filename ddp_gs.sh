
CUDA_VISIBLE_DEVICES=7,8,9 torchrun --nproc_per_node=3 --master_port=29519 train.py --model_save_path /media/HDD2/models/finals/arkit \
    --scene_model_dir /media/HDD2/ARKitScenes/ARKitScenes_3dgs/train \
    --scene_source_dir /media/HDD2/ARKitScenes/ARKitScenes_nerf/train \
    --iterations 300000

