
CUDA_VISIBLE_DEVICES=$1 python test_pointcloud.py --dpt_model_path /media/HDD2/models/finals/scans/dpt_ckpt_0150000.pth \
    --scene_model_dir  /media/SSD/sparse_pts/Scannetpp/test/ /media/SSD/sparse_pts/scannet/test/  /media/SSD/sparse_pts/ARKitScenes/test/\
                        /media/SSD/sparse_pts/Multiscan/test/ /media/SSD/sparse_pts/Multiscan/train/ \
    --scene_source_dir /media/HDD2/Scannetpp/Scannetpp_nerf/test /media/HDD2/scannet_nerf/scans_test /media/HDD2/ARKitScenes/ARKitScenes_nerf/test/ \
                        /media/HDD2/Multiscan/Multiscan_nerf/test /media/HDD2/Multiscan/Multiscan_nerf/train \


CUDA_VISIBLE_DEVICES=$1 python test.py --dpt_model_path /media/HDD2/models/finals/scan_scanpp/dpt_ckpt_0100000.pth \
    --scene_model_dir  /media/HDD2/Scannetpp/Scannetpp_3dgs/test/ /media/HDD2/scannet_3dgs/test/ /media/HDD2/ARKitScenes/ARKitScenes_3dgs/test/\
                        /media/HDD2/Multiscan/Multiscan_3dgs/test /media/HDD2/Multiscan/Multiscan_3dgs/train\
    --scene_source_dir /media/HDD2/Scannetpp/Scannetpp_nerf/test /media/HDD2/scannet_nerf/scans_test /media/HDD2/ARKitScenes/ARKitScenes_nerf/test/ \
                        /media/HDD2/Multiscan/Multiscan_nerf/test /media/HDD2/Multiscan/Multiscan_nerf/train\



