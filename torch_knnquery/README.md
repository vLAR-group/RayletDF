# Voxel grid k-nearest-neighbor ray sampling in PyTorch

This is a Pytorch extension providing a CUDA voxel grid datastructure for k-nearest-neighbor point queries used in our work 
Christopher Wewer, Eddy Ilg, Bernt Schiele, Jan Eric Lenssen: [SimNP: Learning Self-Similarity between Neural Points](https://geometric-rl.mpi-inf.mpg.de/simnp/), ICCV 2023.


Based on the kernels from [Xu et al. Point-NeRF: Point-based Neural Radiance Fields](https://github.com/Xharlie/pointnerf).

Given a set of points, a voxel grid is constructed. This voxel grid can be used to efficiently obtain k-NN points for samples along rays, shot through the scene. Implementation is CUDA only.

Differences to the PointNeRF implementation:
- standalone
- implemented as a PyTorch extension instead of using pycuda, directly working on `torch.Tensor` objects
- thus, has a clean Python interface
- has batch support
- sparse ray structure supporting different numbers of valid rays for examples in a batch

## Installation

```shell
git clone git@github.com:janericlenssen/torch_knnquery.git
cd torch_knnquery
pip install .
```

## Usage

Create an empty `VoxelGrid`:

```python
import torch_knnquery

voxel_grid = VoxelGrid(
    voxel_size=(0.8, 0.8, 0.8),                 # Voxel size for each dimension
    voxel_scale=(2, 2, 2),                      # Voxel scale for each dimension
    kernel_size=(3, 3, 3),                      # Range of voxels searched for neighbors 
                                                # (default: [3, 3, 3])
    max_points_per_voxel=26,                    # Maximum number of points stored in a voxel 
                                                # (default: 26)
    max_occ_voxels_per_example=610000,          # Maximum number of occupied voxels per point cloud 
                                                # (default: 600000)
    ranges=(-10.0,-10.0,-10.0,10.0,10.0,10.0)   # Maximum ranges the VoxelGrid spans 
                                                # (default: inferred from data)
)

```

Insert a set of point clouds into the `VoxelGrid`:
```python
voxel_grid.set_pointset(
        points=points_tensor,                       # Tensor of size [B, max_num_points, 3] containing 
                                                    # B point clouds.
        actual_num_points_per_example=num_tensor    # Tensor of size [B] containing the actual
                                                    # num_points<=max_num_points for each point cloud.
        )

```

Query k-nearest neighbors for ray samples through the `VoxelGrid`:
```python
sample_point_indices, sample_locations, ray_mask = voxel_grid.query(
        raypos=raypos_tensor,           # Tensor of size [B, num_rays, num_samples_per_ray, 3] 
                                        # containing query positions.
        k=8,                            # Number of neighbors to sample for each ray sample 
        radius_limit_scale=4,           # Maximum radius to search for neighbors in
        max_shading_points_per_ray=24   # The maximum number of points per ray for which neighbors 
                                        # are sampled. The first max_shading_points_per_ray samples 
                                        # of each ray that hit occupied voxels return neighbors.
        )
```

Returns are:
```python
sample_point_indices    # Tensor of size [total_num_valid_rays, max_shading_points_per_ray, k]
                        # containing the indices of the k nearest neighbors in points_tensor
                        # for each of the B point clouds (flattened batch and ray dimensions)
sample_locations        # Tensor of size [total_num_valid_rays, max_shading_points_per_ray, 3]
                        # containing the positions of the used shading points for each
                        # of the B point clouds (flattened batch and ray dimensions)
ray_mask                # Tensor of size [B, num_original_rays], containing 1 for rays
                        # that produced shading points (valid rays) and 0 for others.
                        # Contains exactly num_valid_rays 1s.

```

The sparse indices and locations can be recovered into a dense format by
```python
dense_indices = torch.zeros((B, num_original_rays, max_shading_points_per_ray, k), dtype=torch.int32, device=device)
dense_indices[:] = -1
dense_indices[ray_mask] = sample_point_indices

dense_locations = torch.zeros((B, num_original_rays, max_shading_points_per_ray, 3), dtype=torch.float32, device=device)
dense_locations[ray_mask] = sample_locations
```
if needed.
