# RayletDF: Raylet Distance Fields for Generalizable 3D Surface Reconstruction from Point Clouds or Gaussians (ICCV2025)

<h4 align="center">

[![License: CC-BY-NC-SA](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-blue)](./LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2506.07865-b31b1b)](https://arxiv.org/pdf/2508.09830)

This repository will contain the official implementation of the paper: *RayletDF: Raylet Distance Fields for Generalizable 3D Surface Reconstruction from Point Clouds or Gaussians*.

## ⚙️ Installation
```shell script
git clone https://github.com/vLAR-group/RayletDF.git
cd RayletDF

### CUDA 11.3
conda create -n RayletDF python=3.7
conda activate RayletDF

# install pytorch 
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
# install other dependencies
pip install -r requirements.txt
# install gaussian requirements
pip install submodules/diff-gaussian-rasterization-raylet
pip install submodules/simple-knn
```
**(Optional)** *install Voxel grid k-nearest-neighbor (Only for accelerate inference speed)*

```shell script
pip install ./torch_knnquery
```

## 💾 Datasets
Raw datasets can be download from their project website:
- [Scannet++ Dataset](https://kaldir.vc.in.tum.de/scannetpp/)
- [Scannet Dataset](https://github.com/ScanNet/ScanNet)
- [ARKitScenes Dataset](https://github.com/apple/ARKitScenes)
- [Multiscan Benchmark Dataset](https://github.com/smartscenes/multiscan)

We provide several scripts in `data_preprocess` folder to convert raw data in these datasets to NeRF format for training Gaussians and RayletDF, organised as:

```
<NeRF dataset location>
|---train
|   |---<scene_0>
|   |   |---input
|   |   |   |---<image 0>  
|   |   |   |---...
|   |   |---depth
|   |   |   |---<depth 0>  
|   |   |   |---...
|   |   |---points3d.ply
|   |   |---scene_0_mesh.ply     
|   |   |---transforms_train.json  
|   |   |---transforms_test.json  
|   |   |---transforms_all.json
|   |--...  
|---test
|   |---<scene_0>
|   |---<scene_1>
|   |---...
```


The Gaussian data is organised following the official structure:
```
<Gaussians dataset location>
|---train
|   |---<scene_0>
|   |   |---point_cloud
|   |   |   |---iteration_xxxxx
|   |   |   |   |---point_cloud.ply
|   |   |---input.ply
|   |---...
|---test
|   |---<scene_0>
|   |---<scene_1>
|   |---...
```

The sparse point clouds data is organised as the following structure:
```
<point clouds dataset location>
|---train
|   |---<scene_0>
|   |   |---point_cloud.ply
|   |---...
|---test
|   |---<scene_0>
|   |---<scene_1>
|   |---...
```

## 🔑 Usage
To train RayletDF for Gaussians, run `train.py`: 

```shell script
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 --master_port=29519 train.py 
    --model_save_path /media/HDD2/models/arkit \
    --scene_model_dir /media/HDD2/ARKitScenes/ARKitScenes_3dgs/train \
    --scene_source_dir /media/HDD2/ARKitScenes/ARKitScenes_nerf/train \
    --iterations 300000
```
and `train_pointcloud.py` for sparse point clouds:
```shell script
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29518 train_pointcloud.py 
    --model_save_path /media/HDD2/models/arkit_pts \
    --scene_model_dir   /media/SSD/sparse_pts/ARKitScenes/train/ \
    --scene_source_dir /media/HDD2/ARKitScenes/ARKitScenes_nerf/train \
    --iterations 300000
```

For testing a trained model, you can use `test.py` or `test_pointcloud.py`: 
```shell script
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
```

If you want to obtain the normals from RayletDF, set --normal to True.

For testing inference speed on Gaussians, you can run `infer_speed.py`.

Refer to ddp_gs.sh, ddp_pt.sh and test.sh for more examples.

The checkpoints are free to download from [Google Drive](https://drive.google.com/drive/folders/1bh8fUCfLHwq6eoQRvlTFacaXqRnz9R_7?usp=sharing)

## Citation
If you find our work useful in your research, please consider citing:
```bibtex   
@inproceedings{wei2025rayletdf,
title={RayletDF: Raylet Distance Fields for Generalizable 3D Surface Reconstruction from Point Clouds or Gaussians},
author={Wei, Shenxing and Li, Jinxi and Yang, Yafei and Zhou, Siyuan and Yang, Bo},
journal={Proceedings of the IEEE/CVF International Conference on Computer Vision},
year={2025}
}        
```

