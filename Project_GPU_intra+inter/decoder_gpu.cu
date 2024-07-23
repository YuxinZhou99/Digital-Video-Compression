__global__ void reconstruct_decoder_gpu(const int *d_intra_mode, const int *d_intra_ele, const int *d_residual_blk, float *recon_blk_decoder, int num_block_col, int pad_value) {

    int single_blk_col = threadIdx.x;
    int single_blk_row = threadIdx.y;
    int blk_col = blockIdx.x;
    int blk_row = blockIdx.y;
    int ele_idx = (blk_row * gridDim.x + blk_col) * (blockDim.x * blockDim.y) + single_blk_row * blockDim.x + single_blk_col;

    int single_intra_ele;
    if (d_intra_mode[blockIdx.y * gridDim.x + blockIdx.x] == 0) {
        single_intra_ele = d_intra_ele[(blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.y];
    }
    else {
        single_intra_ele = d_intra_ele[(blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x];
    }

    recon_blk_decoder[ele_idx] = single_intra_ele + d_residual_blk[ele_idx];
}

__global__ void reconstruct_decoder_gpu_P(const float *d_prev_recon_blk, const int *d_inter_mv, const int *d_residual_blk_P, float *d_recon_blk_decoder_P, int r, int i) {

    int single_blk_col = threadIdx.x;
    int single_blk_row = threadIdx.y;
    int blk_col = blockIdx.x;
    int blk_row = blockIdx.y;
    int blk_idx = blockIdx.y * gridDim.x + blockIdx.x;

    int current_mv_x = d_inter_mv[blk_idx * 2];
    int current_mv_y = d_inter_mv[blk_idx * 2 + 1];

    float current_block = 0.0f;
    if (i == 1) current_block = (float) d_prev_recon_blk[(blk_row * blockDim.y + single_blk_row - current_mv_y) * gridDim.x * blockDim.x + (blk_col * blockDim.x + single_blk_col + current_mv_x)];
    else current_block = (float) d_prev_recon_blk[(r + blk_row * blockDim.y + single_blk_row - current_mv_y) * gridDim.x * blockDim.x + (blk_col * blockDim.x + single_blk_col + current_mv_x)];
    d_recon_blk_decoder_P[blk_idx * blockDim.x * blockDim.y + single_blk_row * blockDim.x + single_blk_col] = current_block + d_residual_blk_P[blk_idx * blockDim.x * blockDim.y + single_blk_row * blockDim.x + single_blk_col];
}