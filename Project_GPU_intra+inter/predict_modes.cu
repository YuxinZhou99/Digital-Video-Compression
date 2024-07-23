
__device__ float mae(float* single_block, float* reference_block, int size){
    float sum = 0;
    for (int i=0; i<size; i++){
        sum += fabs(single_block[i]-reference_block[i]);
    }
    return sum/size;
}


__global__ void predict_motion(float* rearrange_split_img, int *I_frame_modes, float *residual_blocks, float *intra_ele_lines, float *reconstructed_blocks, int current_frame, int current_I_frame){

    int idx = threadIdx.y * blockDim.x + threadIdx.x;
    size_t block_size = blockDim.y * blockDim.x;
    size_t blocks_single_frame = gridDim.y * gridDim.x;
    size_t frame_size = block_size * blocks_single_frame;

    size_t block_offset_I = current_I_frame * blocks_single_frame + blockIdx.y * gridDim.x + blockIdx.x;

    size_t offset = current_frame * frame_size + blockIdx.y * gridDim.x * block_size + blockIdx.x * block_size;

    // Calculate shared memory size dynamically based on the block size
    extern __shared__ float shared_mem[];
    float* shared_reference_block = &shared_mem[0];
    float* shared_horizontal_block = &shared_mem[block_size];
    float* shared_vertical_block = &shared_mem[2 * block_size];
    float* shared_predict_block = &shared_mem[3 * block_size]; // Allocating space for shared_predict_block
    float* shared_residual_block = &shared_mem[4 * block_size];
    float* shared_reconstructed_block = &shared_mem[5 * block_size];

    // Load data into shared memory
    shared_reference_block[idx] = rearrange_split_img[offset + idx];

    // Omitted: Horizontal and vertical prediction calculations
    // Assuming these are calculated and populated into shared_horizontal_block and shared_vertical_block
    // horizontal (mode 0)
    // boundry case for horizontal
    size_t horizontal_idx = offset-block_size+blockDim.x*(threadIdx.y+1)-1;
    shared_horizontal_block[idx] = (blockIdx.x == 0)?128.0:rearrange_split_img[horizontal_idx];

    // vertical (mode 1)
    // boundry case for vertical
    size_t vertical_idx = offset-gridDim.x*block_size+block_size-blockDim.x+threadIdx.x;
    shared_vertical_block[idx] = (blockIdx.y == 0)?128.0:rearrange_split_img[vertical_idx];
    __syncthreads();

    if (idx == 0) {
        float horizontal_mae = mae(shared_horizontal_block, shared_reference_block, block_size);
        float vertical_mae = mae(shared_vertical_block, shared_reference_block, block_size);

        int mode = vertical_mae < horizontal_mae ? 1 : 0;
        I_frame_modes[block_offset_I] = mode;
    }
    __syncthreads();
    shared_predict_block[idx] = I_frame_modes[block_offset_I] == 0 ? shared_horizontal_block[idx] : shared_vertical_block[idx];

    // Continue with residual and reconstructed block calculations
    shared_residual_block[idx] = shared_reference_block[idx] - shared_predict_block[idx];
    residual_blocks[offset + idx] = shared_residual_block[idx];

    shared_reconstructed_block[idx] = shared_residual_block[idx] + shared_predict_block[idx];
    reconstructed_blocks[offset + idx] = shared_reconstructed_block[idx];

    if (threadIdx.y == 0){
        size_t line_idx = block_offset_I * blockDim.x + threadIdx.x;
        size_t horizontal_line_idx = (threadIdx.x + 1) * blockDim.x - 1;
        size_t vertical_line_idx = (blockDim.x - 1) * blockDim.x + threadIdx.x;
        intra_ele_lines[line_idx] = (I_frame_modes[block_offset_I] == 0)?shared_horizontal_block[horizontal_line_idx]:shared_vertical_block[vertical_line_idx];
    }
    __syncthreads();
}

__global__ void predict_mv(float* rearrange_split_img, int *motion_vector_row, int *motion_vector_col, float *residual_blocks, float *reconstructed_blocks, int current_frame, int current_P_frame){

    int idx = threadIdx.y * blockDim.x + threadIdx.x;
    size_t block_size = blockDim.y * blockDim.x;
    size_t blocks_single_frame = gridDim.y * gridDim.x;
    size_t frame_size = block_size * blocks_single_frame;

    size_t block_offset_P = current_P_frame * blocks_single_frame + blockIdx.y * gridDim.x + blockIdx.x;

    size_t offset = current_frame * frame_size + blockIdx.y * gridDim.x * block_size + blockIdx.x * block_size;
    size_t ref_blk_dim = blockDim.x+2;

    // Calculate shared memory size dynamically based on the block size
    extern __shared__ float shared_mem[];
    float* shared_reference_block = &shared_mem[0];
    float* shared_predict_block = &shared_mem[1 * block_size]; // Allocating space for shared_predict_block
    float* shared_residual_block = &shared_mem[2 * block_size];
    float* shared_reconstructed_block = &shared_mem[3 * block_size];


    float* shared_ref_block = &shared_mem[4 * block_size];
    float* shared_ref_area = &shared_mem[5 * block_size];

    // boundry case soecial handling
    // rows
    if (blockIdx.y == 0 || blockIdx.y == gridDim.y-1 || blockIdx.x == 0 || blockIdx.x == gridDim.x-1){
        shared_predict_block[idx] = rearrange_split_img[offset+idx-frame_size];
        __syncthreads();

        shared_reference_block[idx] = rearrange_split_img[offset+idx];

        shared_residual_block[idx] = shared_reference_block[idx]-shared_predict_block[idx];
        residual_blocks[offset+idx] = shared_residual_block[idx];

        shared_reconstructed_block[idx] = shared_residual_block[idx]+shared_predict_block[idx];
        reconstructed_blocks[offset+idx] = shared_reconstructed_block[idx];
    }
    else{
        if (threadIdx.y == 0){
            shared_ref_area[idx+1] = rearrange_split_img[offset+idx-frame_size-block_size*gridDim.x+block_size-blockDim.x];
        }
        else if (threadIdx.y == blockDim.y-1){
            shared_ref_area[(threadIdx.y+2)*(blockDim.y+2)+threadIdx.x+1] = rearrange_split_img[offset+threadIdx.x-frame_size+block_size*gridDim.x];
        }
        __syncthreads();

        if (threadIdx.x == 0){
            shared_ref_area[idx+2*threadIdx.y+blockDim.x+2] = rearrange_split_img[offset+idx-frame_size-block_size+blockDim.x-1];
        }
        else if (threadIdx.x == blockDim.x-1){
            shared_ref_area[idx+2*(threadIdx.y+1)] = rearrange_split_img[offset+idx-frame_size+block_size-blockDim.x+1];
        }
        __syncthreads();
        shared_ref_area[(threadIdx.y+1)*(blockDim.x+2)+threadIdx.x+1] = rearrange_split_img[offset+idx-frame_size];
        shared_reference_block[idx] = rearrange_split_img[offset+idx];

        if (threadIdx.y == 0 && threadIdx.x == 0){
            shared_ref_area[0] = rearrange_split_img[offset-frame_size-block_size*gridDim.x-1];
            shared_ref_area[blockDim.x+2-1] = rearrange_split_img[offset-frame_size-block_size*gridDim.x+2*block_size-blockDim.x];
            shared_ref_area[(blockDim.y+1)*(blockDim.x+2)] = rearrange_split_img[offset-frame_size+block_size*gridDim.x-block_size+blockDim.x-1];
            shared_ref_area[(blockDim.y+2)*(blockDim.x+2)-1] = rearrange_split_img[offset-frame_size+block_size*gridDim.x+block_size];
        }
        __syncthreads();

        float mae_block = 0.0;
        float smallest_mae;

        //block 1
        shared_ref_block[idx] = shared_ref_area[idx+2*threadIdx.y];
        __syncthreads();
        if (threadIdx.y == 0 && threadIdx.x == 0){
            mae_block = mae(shared_ref_block, shared_reference_block, block_size);
            smallest_mae = mae_block;
            motion_vector_row[block_offset_P] = 1;
            motion_vector_col[block_offset_P] = -1;
        }
        __syncthreads();

        // block 2
        shared_ref_block[idx] = shared_ref_area[idx+2*threadIdx.y+1];
        __syncthreads();
        if (threadIdx.y == 0 && threadIdx.x == 0){
            mae_block = mae(shared_ref_block, shared_reference_block, block_size);
            if (mae_block < smallest_mae){
                smallest_mae = mae_block;
                motion_vector_row[block_offset_P] = 1;
                motion_vector_col[block_offset_P] = 0;
            }
        }
        __syncthreads();

        // block 3
        shared_ref_block[idx] = shared_ref_area[idx+2*threadIdx.y+2];
        __syncthreads();
        if (threadIdx.y == 0 && threadIdx.x == 0){
            mae_block = mae(shared_ref_block, shared_reference_block, block_size);
            if (mae_block < smallest_mae){
                smallest_mae = mae_block;
                motion_vector_row[block_offset_P] = 1;
                motion_vector_col[block_offset_P] = 1;
            }
        }
        __syncthreads();

        // block 4
        shared_ref_block[idx] = shared_ref_area[idx+2*threadIdx.y+ref_blk_dim];
        __syncthreads();
        if (threadIdx.y == 0 && threadIdx.x == 0){
            mae_block = mae(shared_ref_block, shared_reference_block, block_size);
            if (mae_block < smallest_mae){
                smallest_mae = mae_block;
                motion_vector_row[block_offset_P] = 0;
                motion_vector_col[block_offset_P] = -1;
            }
        }
        __syncthreads();

        // block 5
        shared_ref_block[idx] = shared_ref_area[idx+2*threadIdx.y+ref_blk_dim+1];
        __syncthreads();
        if (threadIdx.y == 0 && threadIdx.x == 0){
            mae_block = mae(shared_ref_block, shared_reference_block, block_size);
            if (mae_block < smallest_mae){
                smallest_mae = mae_block;
                motion_vector_row[block_offset_P] = 0;
                motion_vector_col[block_offset_P] = 0;
            }
        }
        __syncthreads();

        // block 6
        shared_ref_block[idx] = shared_ref_area[idx+2*threadIdx.y+ref_blk_dim+2];
        __syncthreads();
        if (threadIdx.y == 0 && threadIdx.x == 0){
            mae_block = mae(shared_ref_block, shared_reference_block, block_size);
            if (mae_block < smallest_mae){
                smallest_mae = mae_block;
                motion_vector_row[block_offset_P] = 0;
                motion_vector_col[block_offset_P] = 1;
            }
        }
        __syncthreads();

        // block 7
        shared_ref_block[idx] = shared_ref_area[idx+2*threadIdx.y+2*ref_blk_dim];
        __syncthreads();
        if (threadIdx.y == 0 && threadIdx.x == 0){
            mae_block = mae(shared_ref_block, shared_reference_block, block_size);
            if (mae_block < smallest_mae){
                smallest_mae = mae_block;
                motion_vector_row[block_offset_P] = -1;
                motion_vector_col[block_offset_P] = -1;
            }
        }
        __syncthreads();

        // block 8
        shared_ref_block[idx] = shared_ref_area[idx+2*threadIdx.y+2*ref_blk_dim+1];
        __syncthreads();
        if (threadIdx.y == 0 && threadIdx.x == 0){
            mae_block = mae(shared_ref_block, shared_reference_block, block_size);
            if (mae_block < smallest_mae){
                smallest_mae = mae_block;
                motion_vector_row[block_offset_P] = -1;
                motion_vector_col[block_offset_P] = 0;
            }
        }
        __syncthreads();

        // block 9
        shared_ref_block[idx] = shared_ref_area[idx+2*threadIdx.y+2*ref_blk_dim+2];
        __syncthreads();
        if (threadIdx.y == 0 && threadIdx.x == 0){
            mae_block = mae(shared_ref_block, shared_reference_block, block_size);
            if (mae_block < smallest_mae){
                smallest_mae = mae_block;
                motion_vector_row[block_offset_P] = -1;
                motion_vector_col[block_offset_P] = 1;
            }
        }
        __syncthreads();

        int row = motion_vector_row[block_offset_P];
        int col = motion_vector_col[block_offset_P];
        shared_predict_block[idx] = shared_ref_area[idx-(row-1)*ref_blk_dim+col+1+2*threadIdx.y];
        __syncthreads();

        shared_residual_block[idx] = shared_reference_block[idx]-shared_predict_block[idx];
        residual_blocks[offset+idx] = shared_residual_block[idx];

        shared_reconstructed_block[idx] = shared_residual_block[idx]+shared_predict_block[idx];
        reconstructed_blocks[offset+idx] = shared_reconstructed_block[idx];


    }

}