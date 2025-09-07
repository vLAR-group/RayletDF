#include <torch/extension.h>
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <curand_kernel.h> 

#define IS_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be CUDA tensor");
#define IS_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " is not contiguous");
#define CHECK_INPUT(x) IS_CUDA(x) IS_CONTIGUOUS(x)

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS

// --- Kernels --- //

template <typename scalar_t>
__global__ void claim_occ_kernel(
    const scalar_t* in_data,   // B * N * 3
    const int* in_actual_numpoints, // B 
    const int B,
    const int N,
    const float *d_coord_shift,     // 3
    const float *d_voxel_size,      // 3
    const int *d_grid_size,       // 3
    const int grid_size_vol,
    const int max_o,
    int* occ_idx, // B, all 0
    int *coor_2_occ,  // B * 400 * 400 * 400, all -1
    int *occ_2_coor,  // B * max_o * 3, all -1
    unsigned long seconds
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; // index of gpu thread
    int i_batch = index / N;  // index of batch
    if (i_batch >= B) { return; }
    int i_pt = index - N * i_batch;
    if (i_pt < in_actual_numpoints[i_batch]) {
        int coor[3];
        const scalar_t *p_pt = in_data + index * 3;
        coor[0] = (int) floor((p_pt[0] - d_coord_shift[0]) / d_voxel_size[0]);
        coor[1] = (int) floor((p_pt[1] - d_coord_shift[1]) / d_voxel_size[1]);
        coor[2] = (int) floor((p_pt[2] - d_coord_shift[2]) / d_voxel_size[2]);
        // printf("p_pt %f %f %f %f; ", p_pt[2], d_coord_shift[2], d_coord_shift[0], d_coord_shift[1]);
        if (coor[0] < 0 || coor[0] >= d_grid_size[0] || coor[1] < 0 || coor[1] >= d_grid_size[1] || coor[2] < 0 || coor[2] >= d_grid_size[2]) { return; }
        int coor_indx_b = i_batch * grid_size_vol + coor[0] * (d_grid_size[1] * d_grid_size[2]) + coor[1] * d_grid_size[2] + coor[2];
        
        int voxel_idx = coor_2_occ[coor_indx_b];
        if (voxel_idx == -1) {  // found an empty voxel
            int old_voxel_num = atomicCAS(
                    &coor_2_occ[coor_indx_b],
                    -1, 0
            );
            if (old_voxel_num == -1) {
                // CAS -> old val, if old val is -1
                // if we get -1, this thread is the one who obtain a new voxel
                // so only this thread should do the increase operator below
                int tmp = atomicAdd(occ_idx+i_batch, 1); // increase the counter, return old counter
                    // increase the counter, return old counter
                if (tmp < max_o) {
                    int coord_inds = (i_batch * max_o + tmp) * 3;
                    occ_2_coor[coord_inds] = coor[0];
                    occ_2_coor[coord_inds + 1] = coor[1];
                    occ_2_coor[coord_inds + 2] = coor[2];
                } else {
                    curandState state;
                    curand_init(index+2*seconds, 0, 0, &state);
                    int insrtidx = ceilf(curand_uniform(&state) * (tmp+1)) - 1;
                    if(insrtidx < max_o){
                        int coord_inds = (i_batch * max_o + insrtidx) * 3;
                        occ_2_coor[coord_inds] = coor[0];
                        occ_2_coor[coord_inds + 1] = coor[1];
                        occ_2_coor[coord_inds + 2] = coor[2];
                    }
                }
            }
        }
    }
}

__global__ void map_coor2occ_kernel(
    const int B,
    const int *d_grid_size,       // 3
    const int *kernel_size,       // 3
    const int grid_size_vol,
    const int max_o,
    int* occ_idx, // B, all -1
    int *coor_occ,  // B * 400 * 400 * 400
    int *coor_2_occ,  // B * 400 * 400 * 400
    int *occ_2_coor  // B * max_o * 3
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; // index of gpu thread
    int i_batch = index / max_o;  // index of batch
    if (i_batch >= B) { return; }
    int i_pt = index - max_o * i_batch;
    if (i_pt < occ_idx[i_batch] && i_pt < max_o) {
        int coor[3];
        coor[0] = occ_2_coor[index*3];
        if (coor[0] < 0) { return; }
        coor[1] = occ_2_coor[index*3+1];
        coor[2] = occ_2_coor[index*3+2];
        
        int coor_indx_b = i_batch * grid_size_vol + coor[0] * (d_grid_size[1] * d_grid_size[2]) + coor[1] * d_grid_size[2] + coor[2];
        coor_2_occ[coor_indx_b] = i_pt;
        // printf("kernel_size[0] %d", kernel_size[0]);
        for (int coor_x = max(0, coor[0] - kernel_size[0] / 2) ; coor_x < min(d_grid_size[0], coor[0] + (kernel_size[0] + 1) / 2); coor_x++)    {
            for (int coor_y = max(0, coor[1] - kernel_size[1] / 2) ; coor_y < min(d_grid_size[1], coor[1] + (kernel_size[1] + 1) / 2); coor_y++)   {
                for (int coor_z = max(0, coor[2] - kernel_size[2] / 2) ; coor_z < min(d_grid_size[2], coor[2] + (kernel_size[2] + 1) / 2); coor_z++) {
                    coor_indx_b = i_batch * grid_size_vol + coor_x * (d_grid_size[1] * d_grid_size[2]) + coor_y * d_grid_size[2] + coor_z;
                    if (coor_occ[coor_indx_b] > 0) { continue; }
                    atomicCAS(coor_occ + coor_indx_b, 0, 1);
                }
            }
        }   
    }
}

template <typename scalar_t>
__global__ void fill_occ2pnts_kernel(
    const scalar_t* in_data,   // B * N * 3
    const int* in_actual_numpoints, // B 
    const int B,
    const int N,
    const int P,
    const float *d_coord_shift,     // 3
    const float *d_voxel_size,      // 3
    const int *d_grid_size,       // 3
    const int grid_size_vol,
    const int max_o,
    int *coor_2_occ,  // B * 400 * 400 * 400, all -1
    int *occ_2_pnts,  // B * max_o * P, all -1
    int *occ_numpnts,  // B * max_o, all 0
    unsigned long seconds
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; // index of gpu thread
    int i_batch = index / N;  // index of batch
    if (i_batch >= B) { return; }
    int i_pt = index;
    if (i_pt - N * i_batch < in_actual_numpoints[i_batch]) {
        int coor[3];
        //const scalar_t *p_pt = in_data + index * 3;
        coor[0] = (int) floor((in_data[index*3 + 0] - d_coord_shift[0]) / d_voxel_size[0]);
        coor[1] = (int) floor((in_data[index*3 + 1] - d_coord_shift[1]) / d_voxel_size[1]);
        coor[2] = (int) floor((in_data[index*3 + 2] - d_coord_shift[2]) / d_voxel_size[2]);
        if (coor[0] < 0 || coor[0] >= d_grid_size[0] || coor[1] < 0 || coor[1] >= d_grid_size[1] || coor[2] < 0 || coor[2] >= d_grid_size[2]) { return; }
        int coor_indx_b = i_batch * grid_size_vol + coor[0] * (d_grid_size[1] * d_grid_size[2]) + coor[1] * d_grid_size[2] + coor[2];
        
        int voxel_idx = coor_2_occ[coor_indx_b];
        if (voxel_idx >= 0) {  // found an claimed coor2occ
            int occ_indx_b = i_batch * max_o + voxel_idx;
            int tmp = atomicAdd(occ_numpnts + occ_indx_b, 1); // increase the counter, return old counter
            if (tmp < P) {
                occ_2_pnts[occ_indx_b * P + tmp] = i_pt;
            } else {
                curandState state;
                curand_init(index+2*seconds, 0, 0, &state);
                int insrtidx = ceilf(curand_uniform(&state) * (tmp+1)) - 1;
                if(insrtidx < P){
                    occ_2_pnts[occ_indx_b * P + insrtidx] = i_pt;
                }
            }
        }
    }
}

    
template <typename scalar_t>        
__global__ void mask_raypos_kernel(
    scalar_t *raypos,    // [B, 2048, 400, 3]
    int *coor_occ,    // B * 400 * 400 * 400
    const int B,       // 3
    const int R,       // 3
    const int D,       // 3
    const int grid_size_vol,
    const float *d_coord_shift,     // 3
    const int *d_grid_size,       // 3
    const float *d_voxel_size,      // 3
    int *raypos_mask    // B, R, D
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; // index of gpu thread
    int i_batch = index / (R * D);  // index of batch
    if (i_batch >= B) { return; }
    int coor[3];
    coor[0] = (int) floor((raypos[index*3] - d_coord_shift[0]) / d_voxel_size[0]);
    coor[1] = (int) floor((raypos[index*3+1] - d_coord_shift[1]) / d_voxel_size[1]);
    coor[2] = (int) floor((raypos[index*3+2] - d_coord_shift[2]) / d_voxel_size[2]);
    // printf(" %f %f %f;", raypos[index*3], raypos[index*3+1], raypos[index*3+2]);
    if ((coor[0] >= 0) && (coor[0] < d_grid_size[0]) && (coor[1] >= 0) && (coor[1] < d_grid_size[1]) && (coor[2] >= 0) && (coor[2] < d_grid_size[2])) { 
        int coor_indx_b = i_batch * grid_size_vol + coor[0] * (d_grid_size[1] * d_grid_size[2]) + coor[1] * d_grid_size[2] + coor[2];
        raypos_mask[index] = coor_occ[coor_indx_b];
    }
}


template <typename scalar_t>
__global__ void get_shadingloc_kernel(
    const scalar_t *raypos,    // [B, 2048, 400, 3]
    const int *raypos_mask,    // B, R, D
    const int R_valid,       // 3
    const int total_samples_per_ray,       // 3
    const int max_samples_per_ray,       // 3
    scalar_t *sample_loc,       // B * R * SR * 3
    int *sample_loc_mask       // B * R * SR
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; // index of gpu thread
    if(index>=R_valid*total_samples_per_ray) { return; }

    int temp = raypos_mask[index];
    if (temp >= 0) {
        int r = index / total_samples_per_ray;
        int loc_inds = r * max_samples_per_ray + temp;
        sample_loc[loc_inds * 3] = raypos[index * 3];
        sample_loc[loc_inds * 3 + 1] = raypos[index * 3 + 1];
        sample_loc[loc_inds * 3 + 2] = raypos[index * 3 + 2];
        sample_loc_mask[loc_inds] = 1;
    }
}


template <typename scalar_t>
__global__ void query_along_ray_kernel(
    const scalar_t* in_data,   // B * N * 3
    const int *ray_to_batch_indices,
    const int R_valid,
    const int samples_per_ray,               // num. samples along each ray e.g., 128
    const int max_o,
    const int P,
    const int K,                // num.  neighbors
    const int grid_size_vol,
    const float radius_limit2,
    const float *d_coord_shift,     // 3
    const int *d_grid_size,
    const float *d_voxel_size,      // 3
    const int *kernel_size,
    const int *occ_numpnts,    // B * max_o
    const int *occ_2_pnts,            // B * max_o * P
    const int *coor_2_occ,      // B * 400 * 400 * 400 
    const scalar_t *sample_loc,       // B * R * SR * 3
    const int *sample_loc_mask,       // B * R * SR
    int *sample_pidx       // B * R * SR * K
) {
    int index =  blockIdx.x * blockDim.x + threadIdx.x; // index of gpu thread
    if(index>=R_valid*samples_per_ray) { return; }

    int ray = index / samples_per_ray;
    int i_batch = ray_to_batch_indices[ray];  // index of batch
    if (sample_loc_mask[index] <= 0) { return; }
    scalar_t centerx = sample_loc[index * 3];
    scalar_t centery = sample_loc[index * 3 + 1];
    scalar_t centerz = sample_loc[index * 3 + 2];
    int frustx = (int) floor((centerx - d_coord_shift[0]) / d_voxel_size[0]);
    int frusty = (int) floor((centery - d_coord_shift[1]) / d_voxel_size[1]);
    int frustz = (int) floor((centerz - d_coord_shift[2]) / d_voxel_size[2]);
                        
    int kid = 0, far_ind = 0, coor_z, coor_y, coor_x;
    scalar_t far2 = 0.0;
    scalar_t xyz2Buffer[20]; // assuming K<=20
    for (int layer = 0; layer < (kernel_size[0]+1)/2; layer++){                        
        for (int x = max(-frustx, -layer); x < min(d_grid_size[0] - frustx, layer + 1); x++) {
            coor_x = frustx + x;
            for (int y = max(-frusty, -layer); y < min(d_grid_size[1] - frusty, layer + 1); y++) {
                coor_y = frusty + y;
                for (int z =  max(-frustz, -layer); z < min(d_grid_size[2] - frustz, layer + 1); z++) {
                    coor_z = z + frustz;
                    if (max(abs(z), max(abs(x), abs(y))) != layer) continue;
                    int coor_indx_b = i_batch * grid_size_vol + coor_x * (d_grid_size[1] * d_grid_size[2]) + coor_y * d_grid_size[2] + coor_z;
                    int occ_indx = coor_2_occ[coor_indx_b] + i_batch * max_o;
                    if (occ_indx >= 0) {
                        for (int g = 0; g < min(P, occ_numpnts[occ_indx]); g++) {
                            int pidx = occ_2_pnts[occ_indx * P + g];
                            scalar_t x_v = (in_data[pidx*3]-centerx);
                            scalar_t y_v = (in_data[pidx*3 + 1]-centery);
                            scalar_t z_v = (in_data[pidx*3 + 2]-centerz);
                            scalar_t xyz2 = x_v * x_v + y_v * y_v + z_v * z_v;
                            if ((radius_limit2 == 0 || xyz2 <= radius_limit2)){
                                if (kid++ < K) {
                                    sample_pidx[index * K + kid - 1] = pidx;
                                    xyz2Buffer[kid-1] = xyz2;
                                    if (xyz2 > far2){
                                        far2 = xyz2;
                                        far_ind = kid - 1;
                                    }
                                } else {
                                    if (xyz2 < far2) {
                                        sample_pidx[index * K + far_ind] = pidx;
                                        xyz2Buffer[far_ind] = xyz2;
                                        far2 = xyz2;
                                        for (int i = 0; i < K; i++) {
                                            if (xyz2Buffer[i] > far2) {
                                                far2 = xyz2Buffer[i];
                                                far_ind = i;
                                            }
                                        }
                                    } 
                                }
                            }
                        }
                    }
                }
            }
        }
        //if (kid >= K) break;
    }
}

// --- CPP --- //

void find_occupied_voxels(
  at::Tensor points,
  at::Tensor actual_num_points_per_batch,
  int B,
  int N,
  at::Tensor d_coord_shift,
  at::Tensor scaled_vsize,
  at::Tensor scaled_vdim,
  int grid_size_vol,
  int max_o,
  at::Tensor coor_idx_tensor,
  at::Tensor coor_2_occ_tensor,
  at::Tensor occ_2_coor_tensor,
  unsigned long seconds
) {
  CHECK_INPUT(points);
  CHECK_INPUT(actual_num_points_per_batch);
  CHECK_INPUT(d_coord_shift);
  CHECK_INPUT(scaled_vsize);
  CHECK_INPUT(scaled_vdim);
  CHECK_INPUT(coor_idx_tensor);
  CHECK_INPUT(coor_2_occ_tensor);
  CHECK_INPUT(occ_2_coor_tensor);

  int units = B*N;

  AT_DISPATCH_FLOATING_TYPES(points.type(), "claim_occ_kernel", [&] {
    claim_occ_kernel<scalar_t><<<BLOCKS(units), THREADS>>>(
      points.data<scalar_t>(),
      actual_num_points_per_batch.data<int>(),
      B,
      N,
      d_coord_shift.data<float>(),
      scaled_vsize.data<float>(),
      scaled_vdim.data<int>(),
      grid_size_vol,
      max_o,
      coor_idx_tensor.data<int>(),
      coor_2_occ_tensor.data<int>(),
      occ_2_coor_tensor.data<int>(),
      seconds
    );
  });
}


void create_coor_occ_maps(
  size_t B,
  at::Tensor scaled_vdim,
  at::Tensor kernel_size,
  int grid_size_vol,
  int max_o,
  at::Tensor occ_idx_tensor,
  at::Tensor coor_occ_tensor,
  at::Tensor coor_2_occ_tensor,
  at::Tensor occ_2_coor_tensor
) {
  CHECK_INPUT(scaled_vdim);
  CHECK_INPUT(kernel_size);
  CHECK_INPUT(occ_idx_tensor);
  CHECK_INPUT(coor_occ_tensor);
  CHECK_INPUT(coor_2_occ_tensor);
  CHECK_INPUT(occ_2_coor_tensor);

  int units = B * max_o;

  //AT_DISPATCH_FLOATING_TYPES(double, "map_coor2occ_kernel", [&] {
    map_coor2occ_kernel<<<BLOCKS(units), THREADS>>>(
      B,
      scaled_vdim.data<int>(),
      kernel_size.data<int>(),
      grid_size_vol,
      max_o,
      occ_idx_tensor.data<int>(),
      coor_occ_tensor.data<int>(),
      coor_2_occ_tensor.data<int>(),
      occ_2_coor_tensor.data<int>()
    );
  //});
}

void assign_points_to_occ_voxels(
  at::Tensor points,
  at::Tensor actual_num_points_per_batch,
  int B,
  int N,
  int P,
  at::Tensor d_coord_shift,
  at::Tensor scaled_vsize,
  at::Tensor scaled_vdim,
  int grid_size_vol,
  int max_o,
  at::Tensor coor_2_occ_tensor,
  at::Tensor occ_2_pnts_tensor,
  at::Tensor occ_numpnts_tensor,
  unsigned long seconds
) {
  CHECK_INPUT(points);
  CHECK_INPUT(actual_num_points_per_batch);
  CHECK_INPUT(d_coord_shift);
  CHECK_INPUT(scaled_vsize);
  CHECK_INPUT(scaled_vdim);
  CHECK_INPUT(coor_2_occ_tensor);
  CHECK_INPUT(occ_2_pnts_tensor);
  CHECK_INPUT(occ_numpnts_tensor);

  int units = B*N;

  AT_DISPATCH_FLOATING_TYPES(points.type(), "fill_occ2pnts_kernel", [&] {
    fill_occ2pnts_kernel<scalar_t><<<BLOCKS(units), THREADS>>>(
      points.data<scalar_t>(),
      actual_num_points_per_batch.data<int>(),
      B,
      N,
      P,
      d_coord_shift.data<float>(),
      scaled_vsize.data<float>(),
      scaled_vdim.data<int>(),
      grid_size_vol,
      max_o,
      coor_2_occ_tensor.data<int>(),
      occ_2_pnts_tensor.data<int>(),
      occ_numpnts_tensor.data<int>(),
      seconds
    );
  });
}

void create_raypos_mask(
  at::Tensor raypos_tensor,
  at::Tensor coor_occ_tensor,
  int B,
  int R,
  int D,
  int grid_size_vol,
  at::Tensor d_coord_shift,
  at::Tensor scaled_vdim,
  at::Tensor scaled_vsize,
  at::Tensor raypos_mask_tensor
) {
  CHECK_INPUT(raypos_tensor);
  CHECK_INPUT(coor_occ_tensor);
  CHECK_INPUT(d_coord_shift);
  CHECK_INPUT(scaled_vsize);
  CHECK_INPUT(scaled_vdim);
  CHECK_INPUT(raypos_mask_tensor);

  int units = B * R * D;

  AT_DISPATCH_FLOATING_TYPES(raypos_tensor.type(), "mask_raypos_kernel", [&] {
    mask_raypos_kernel<scalar_t><<<BLOCKS(units), THREADS>>>(
      raypos_tensor.data<scalar_t>(),
      coor_occ_tensor.data<int>(),
      B,
      R,
      D,
      grid_size_vol,
      d_coord_shift.data<float>(),
      scaled_vdim.data<int>(),
      scaled_vsize.data<float>(),
      raypos_mask_tensor.data<int>()
    );
  });
}


void get_shadingloc(
  at::Tensor raypos_tensor,
  at::Tensor raypos_mask_tensor,
  int R_valid,
  int total_samples_per_ray,
  int max_samples_per_ray,
  at::Tensor sample_loc_tensor,
  at::Tensor sample_loc_mask_tensor
) {
  CHECK_INPUT(raypos_tensor);
  CHECK_INPUT(raypos_mask_tensor);
  CHECK_INPUT(sample_loc_tensor);
  CHECK_INPUT(sample_loc_mask_tensor);

  int units = R_valid * total_samples_per_ray;

  AT_DISPATCH_FLOATING_TYPES(raypos_tensor.type(), "get_shadingloc_kernel", [&] {
    get_shadingloc_kernel<scalar_t><<<BLOCKS(units), THREADS>>>(
      raypos_tensor.data<scalar_t>(),
      raypos_mask_tensor.data<int>(),
      R_valid,
      total_samples_per_ray,
      max_samples_per_ray,
      sample_loc_tensor.data<scalar_t>(),
      sample_loc_mask_tensor.data<int>()
    );
  });
}


void query_along_ray(
  at::Tensor points,
  at::Tensor ray_to_batch_indices,
  int R_valid,
  int samples_per_ray,
  int max_o,
  int P,
  int num_neighbors,
  int grid_size_vol,
  float radius_limit,
  at::Tensor d_coord_shift,
  at::Tensor scaled_vdim,
  at::Tensor scaled_vsize,
  at::Tensor kernel_size,
  at::Tensor occ_numpnts_tensor,
  at::Tensor occ_2_pnts_tensor,
  at::Tensor coor_2_occ_tensor,
  at::Tensor sample_loc_tensor,
  at::Tensor sample_loc_mask_tensor,
  at::Tensor sample_pidx_tensor
) {
  CHECK_INPUT(points);
  CHECK_INPUT(ray_to_batch_indices);
  CHECK_INPUT(d_coord_shift);
  CHECK_INPUT(scaled_vsize);
  CHECK_INPUT(scaled_vdim);
  CHECK_INPUT(kernel_size);
  CHECK_INPUT(occ_numpnts_tensor);
  CHECK_INPUT(occ_2_pnts_tensor);
  CHECK_INPUT(coor_2_occ_tensor);
  CHECK_INPUT(sample_loc_tensor);
  CHECK_INPUT(sample_loc_mask_tensor);
  CHECK_INPUT(sample_pidx_tensor);

  int units = R_valid * samples_per_ray;

  AT_DISPATCH_FLOATING_TYPES(points.type(), "query_along_ray_kernel", [&] {
    query_along_ray_kernel<scalar_t><<<BLOCKS(units), THREADS>>>(
      points.data<scalar_t>(),
      ray_to_batch_indices.data<int>(),
      R_valid,
      samples_per_ray,
      max_o,
      P,
      num_neighbors,
      grid_size_vol,
      radius_limit,
      d_coord_shift.data<float>(),
      scaled_vdim.data<int>(),
      scaled_vsize.data<float>(),
      kernel_size.data<int>(),
      occ_numpnts_tensor.data<int>(),
      occ_2_pnts_tensor.data<int>(),
      coor_2_occ_tensor.data<int>(),
      sample_loc_tensor.data<scalar_t>(),
      sample_loc_mask_tensor.data<int>(),
      sample_pidx_tensor.data<int>()
    );
  });
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("find_occupied_voxels", &find_occupied_voxels, "Find occupied voxels");
  m.def("create_coor_occ_maps", &create_coor_occ_maps, "Map voxel coordinates to occupied voxels and back");
  m.def("assign_points_to_occ_voxels", &assign_points_to_occ_voxels, "Assign points to occupied voxels");
  m.def("create_raypos_mask", &create_raypos_mask, "Find mask for ray positions that hit occupied voxels");
  m.def("get_shadingloc", &get_shadingloc, "Get shading locations");
  m.def("query_along_ray", &query_along_ray, "Query KNN point indices");
}