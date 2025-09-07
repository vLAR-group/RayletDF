
CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --master_port=29518 train_pointcloud.py --model_save_path /media/HDD2/models/finals/scans \
    --scene_model_dir   /media/SSD/sparse_pts/Scannetpp/train/ /media/SSD/sparse_pts/scannet/train/ \
    --scene_source_dir  /media/SSD/datasets/Scannetpp/Scannetpp_nerf/train /media/SSD/datasets/scannet_nerf/scans\
    --iterations 300000

CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --master_port=29518 train_pointcloud.py --model_save_path /media/HDD2/models/finals/arkit \
    --scene_model_dir   /media/SSD/sparse_pts/ARKitScenes/train/ \
    --scene_source_dir /media/HDD2/ARKitScenes/ARKitScenes_nerf/train \
    --iterations 300000
