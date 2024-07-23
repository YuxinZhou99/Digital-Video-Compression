#include "preprocessing.h"
#include "decoder.h"
#include "write_to_txt.h"
#include "predict_modes.cu"
#include "decoder_gpu.cu"
#include <time.h>
#include <sys/time.h>

double getTimeStamp() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_usec / 1000000 + tv.tv_sec;
}

int main() {
    char *parameters[MAX_PARAMETERS];
    int num_parameters;

    readConfig("config.txt", parameters, &num_parameters);

    // Extract variables
    char path[] = "/homes/z/zhouyu76/Desktop/ECE1782/Project_GPU_intra+inter/";
    char *video = parameters[1] + 1; // Assuming parameters[1] holds the video filename
    char video_path[256]; // Assuming the maximum length of the video path is 256 characters
    strcpy(video_path, path); // Copy the base path to video_path
    strcat(video_path, video); // Concatenate the video filename to form the complete path
    int width = atoi(parameters[3]);
    int height = atoi(parameters[5]);
    int num_frame = atoi(parameters[7]);
    int pad_value = atoi(parameters[9]);
    int r = atoi(parameters[11]);
    int n = atoi(parameters[13]);
    int num_P_frame = atoi(parameters[15]);
    printf("Video: %s\nWidth: %d\nHeight: %d\nNum Frames: %d\nPad Value: %d\nr: %d\nn: %d\nnum_P_frame: %d\n", video, width, height, num_frame, pad_value, r, n, num_P_frame);

    // Preprocessing data
    float *y_only = (float *)malloc(width * height * num_frame * sizeof(float));
    yuv_read_420(width, height, num_frame, video_path, y_only);

    // Call the padding function
    float* img_pad;
    int new_width, new_height;
    padding(y_only, width, height, num_frame, pad_value, &img_pad, &new_width, &new_height);
//    saveMatrixAsPPM(img_pad, new_width, new_height, num_frame);
    // Output new width and height
    printf("New Width: %d\nNew Height: %d\n", new_width, new_height);

    // Split image into blocks
    int num_block_row = new_height / pad_value;
    int num_block_col = new_width / pad_value;
    float**** split_img = allocateBlocksArray(img_pad, num_frame, num_block_row, num_block_col, pad_value, new_width, new_height);
    // Test only
//    saveMatrixAsPPM_2(split_img, new_width, new_height, num_frame, num_block_row, num_block_col, pad_value);

    // Rearrange blocks to fit in the kernel coalesced form
    float *h_rearrange_split_img = (float *) malloc(sizeof(float) * num_frame * num_block_row * num_block_col * pad_value * pad_value);
    rearrangeAllocatedBlocks(h_rearrange_split_img, split_img, num_frame, num_block_row, num_block_col, pad_value, new_width, new_height);

    // Predict Motion
    // Motion estimation
    // Host side
    int num_I_frames = num_frame/(num_P_frame+1)+1;
    int num_P_frames = num_frame-(num_frame/(num_P_frame+1)+1);
    int *h_motion_vector_row = (int *) malloc(sizeof(int) * num_block_row * num_block_col * num_P_frames);
    int *h_motion_vector_col = (int *) malloc(sizeof(int) * num_block_row * num_block_col * num_P_frames);

    int *h_I_frame_modes = (int *) malloc(sizeof(int) * num_block_row * num_block_col * num_I_frames);
    float *h_residual_blocks = (float *) malloc(sizeof(float) * pad_value * pad_value * num_block_row * num_block_col * num_frame);
    float *h_intra_ele_lines = (float *) malloc(sizeof(float) * pad_value * num_block_row * num_block_col * num_I_frames);
    float *h_reconstructed_blocks = (float *) malloc(sizeof(float) * pad_value * pad_value * num_block_row * num_block_col * num_frame);

    // Device side
    size_t size_mode = sizeof(int) * num_block_row * num_block_col * num_I_frames;
    size_t size_mv = sizeof(int) * num_block_row * num_block_col * num_P_frames;
    size_t size_full = sizeof(float) * pad_value * pad_value * num_block_row * num_block_col * num_frame;
    size_t size_single_block = sizeof(float) * pad_value * pad_value;
    size_t size_single_ref_block = sizeof(float) * (pad_value+2) * (pad_value+2);
    size_t size_line = sizeof(float) * pad_value * num_block_row * num_block_col * num_I_frames;

    float *d_rearrange_split_img;
    cudaMalloc((void **)&d_rearrange_split_img, size_full);

    float *d_residual_blocks, *d_intra_ele_lines, *d_reconstructed_blocks;
    int *d_I_frame_modes, *d_motion_vector_row, *d_motion_vector_col;
    cudaMalloc((void **)&d_I_frame_modes, size_mode);
    cudaMalloc((void **)&d_motion_vector_row, size_mv);
    cudaMalloc((void **)&d_motion_vector_col, size_mv);
    cudaMalloc((void **)&d_residual_blocks, size_full);
    cudaMalloc((void **)&d_intra_ele_lines, size_line);
    cudaMalloc((void **)&d_reconstructed_blocks, size_full);

    // Predict motion kernel
    dim3 block(pad_value, pad_value);
    dim3 grid(num_block_col, num_block_row);

    double start = getTimeStamp();

    // I frame
    cudaStream_t streams[num_I_frames+1];
    for (int i=0; i<num_I_frames+1; i++){
        cudaStreamCreate(&streams[i]);
    }

    for (int i=1; i<num_I_frames+1; i++){
        int current_I_frame = (i-1);
        int current_frame = current_I_frame*(num_P_frame+1);
        size_t offset_block_I = current_I_frame*num_block_row * num_block_col;
        size_t offset_full = current_frame*num_block_row * num_block_col*pad_value*pad_value;
        size_t offset_line_I = current_I_frame*pad_value * num_block_row * num_block_col;

        cudaMemcpyAsync(d_rearrange_split_img+offset_full, h_rearrange_split_img+offset_full, size_full/num_frame, cudaMemcpyHostToDevice, streams[i]);
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Error: memory copy h2d error, %s, stream: %d\n", cudaGetErrorString(cudaStatus), i);
            return 1;
        }

        predict_motion<<<grid, block, 6*size_single_block, streams[i]>>>(d_rearrange_split_img, d_I_frame_modes, d_residual_blocks, d_intra_ele_lines,
                                                                         d_reconstructed_blocks, current_frame, current_I_frame);

        cudaMemcpyAsync(h_residual_blocks+offset_full, d_residual_blocks+offset_full, size_full/num_frame, cudaMemcpyDeviceToHost, streams[i]);
        cudaMemcpyAsync(h_I_frame_modes+offset_block_I, d_I_frame_modes+offset_block_I, size_mode/num_I_frames, cudaMemcpyDeviceToHost, streams[i]);
        cudaMemcpyAsync(h_intra_ele_lines+offset_line_I, d_intra_ele_lines+offset_line_I, size_line/num_I_frames, cudaMemcpyDeviceToHost, streams[i]);
        cudaMemcpyAsync(h_reconstructed_blocks+offset_full, d_reconstructed_blocks+offset_full, size_full/num_frame, cudaMemcpyDeviceToHost, streams[i]);

    }
    for (int i=0; i<num_I_frames+1; i++){
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    // P frame
    cudaStream_t streams_P[num_P_frames+1];
    cudaStreamCreate(&streams_P[0]);
    for (int i=0; i<num_P_frames+1; i++){
        cudaStreamCreate(&streams_P[i]);
    }

    for (int i=1; i<num_P_frames+1; i++){
        int current_P_frame = (i-1);
        int current_frame = current_P_frame+current_P_frame/num_P_frame+1;
        size_t offset_block_P = current_P_frame*num_block_row * num_block_col;
        size_t offset_full = current_frame*num_block_row * num_block_col*pad_value*pad_value;

        cudaMemcpyAsync(d_rearrange_split_img+offset_full, h_rearrange_split_img+offset_full, size_full/num_frame, cudaMemcpyHostToDevice, streams_P[i]);
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Error: memory copy h2d error, %s, stream: %d\n", cudaGetErrorString(cudaStatus), i);
            return 1;
        }

        predict_mv<<<grid, block, 5*size_single_block+size_single_ref_block, streams_P[i]>>>(d_rearrange_split_img, d_motion_vector_row, d_motion_vector_col, d_residual_blocks,
                                                                                             d_reconstructed_blocks, current_frame, current_P_frame);

        cudaMemcpyAsync(h_residual_blocks+offset_full, d_residual_blocks+offset_full, size_full/num_frame, cudaMemcpyDeviceToHost, streams_P[i]);
        cudaMemcpyAsync(h_motion_vector_row+offset_block_P, d_motion_vector_row+offset_block_P, size_mv/num_P_frames, cudaMemcpyDeviceToHost, streams_P[i]);
        cudaMemcpyAsync(h_motion_vector_col+offset_block_P, d_motion_vector_col+offset_block_P, size_mv/num_P_frames, cudaMemcpyDeviceToHost, streams_P[i]);
        cudaMemcpyAsync(h_reconstructed_blocks+offset_full, d_reconstructed_blocks+offset_full, size_full/num_frame, cudaMemcpyDeviceToHost, streams_P[i]);
    }
    for (int i=0; i<num_P_frames+1; i++){
        cudaStreamSynchronize(streams_P[i]);
        cudaStreamDestroy(streams_P[i]);
    }

    double stop = getTimeStamp();
    printf("Predict motion kernel process time: %f ms\n", 1000*(stop-start));

//    // Test only
    float *rearranged_reconstructed_blocks = (float *) malloc(sizeof(float) * pad_value * pad_value * num_block_row * num_block_col * num_frame);
    rearrangeReconstruct(h_reconstructed_blocks, rearranged_reconstructed_blocks, new_width, new_height, num_block_row, num_block_col, pad_value, num_frame);
//    saveMatrixAsPPM(rearranged_reconstructed_blocks, new_width, new_height, num_frame);

    // Write into txt
    write_modes_into_txt(h_I_frame_modes, num_block_row, num_block_col, num_I_frames, pad_value);
    write_mv_into_txt(h_motion_vector_row, h_motion_vector_col, num_block_row, num_block_col, num_P_frames, pad_value);
    write_rb_into_txt(h_residual_blocks, num_block_row, num_block_col, num_frame, pad_value, num_P_frame);
    write_intra_ele_into_txt(h_intra_ele_lines, num_block_row, num_block_col, num_I_frames, pad_value);

    // Encoder CPU
    free(y_only);
    free(img_pad);
    free(split_img);
    free(h_rearrange_split_img);
    free(h_I_frame_modes);
    free(h_motion_vector_row);
    free(h_motion_vector_col);
    free(h_residual_blocks);
    free(h_intra_ele_lines);
    free(h_reconstructed_blocks);
    // Encoder GPU
    cudaFree(d_I_frame_modes);
    cudaFree(d_motion_vector_row);
    cudaFree(d_motion_vector_col);
    cudaFree(d_residual_blocks);
    cudaFree(d_intra_ele_lines);

    // ---------------------------------------------------------------------------------------------------------------//
    // Decoder
    int num_sets_IP;
    if (num_frame % (num_P_frame + 1) != 0 ) num_sets_IP = num_frame / (num_P_frame + 1) +1;
    else num_sets_IP = num_frame / (num_P_frame + 1);
    int total_I_frames = num_sets_IP;
    int total_P_frames = num_frame - total_I_frames;
    int *intra_modes, *residual_blk_I, *intra_blocks_cpu;
    cudaMallocHost(&intra_modes, sizeof(int) * total_I_frames * num_block_row * num_block_col);
    cudaMallocHost(&residual_blk_I, sizeof(int) * num_frame * num_block_row * pad_value * num_block_col * pad_value);
    cudaMallocHost(&intra_blocks_cpu, sizeof(int) * total_I_frames * num_block_row * num_block_col * pad_value);
    readIntraModes("intra_modes.txt", intra_modes, num_block_row, num_block_col);
    readResidualBlks("residual_blocks_I.txt", residual_blk_I, num_block_row, num_block_col, pad_value);
    readIntraEle("intra_ele.txt", intra_blocks_cpu, num_block_row, num_block_col, pad_value);
    int *residual_blk_P, *inter_mv;
    if (num_P_frame != 0) {
        cudaMallocHost(&inter_mv, sizeof(int) * 2 * total_P_frames * num_block_row * num_block_col);
        readInterMVs("motion_vectors.txt", inter_mv, num_block_row, num_block_col);
        cudaMallocHost(&residual_blk_P, sizeof(int) * num_frame * num_block_row * pad_value * num_block_col * pad_value);
        readResidualBlks("residual_blocks_P.txt", residual_blk_P, num_block_row, num_block_col, pad_value);
    }
    else {
        cudaMallocHost(&residual_blk_P, sizeof(int) * 1 * num_block_row * pad_value * num_block_col * pad_value);
    }

    // GPU
    // I frames
//    int NSTREAMS = 1;
    int NSTREAMS = total_I_frames;
    dim3 block_decoder(pad_value, pad_value);
//    dim3 grid_decoder(num_block_col, num_block_row * total_I_frames);
    dim3 grid_decoder(num_block_col, num_block_row);
    int *d_intra_ele[NSTREAMS+1], *d_residual_blk_I[NSTREAMS+1], *d_intra_mode[NSTREAMS+1];
    float *h_recon_blk_decoder_I, *d_recon_blk_decoder_I[NSTREAMS+1];
    size_t size_intra_modes = sizeof(int) * num_block_row * num_block_col * total_I_frames;
    size_t size_intra_ele = sizeof(int) * pad_value * num_block_row * num_block_col * total_I_frames;
    size_t size_residuals_I = sizeof(int) * total_I_frames * num_block_row * pad_value * num_block_col * pad_value;
    size_t size_recon_I = sizeof(float) * total_I_frames * num_block_row * num_block_col * pad_value * pad_value;

    cudaStream_t stream[NSTREAMS + 1];
    size_t offset[NSTREAMS + 1], offset_modes[NSTREAMS + 1], offset_intra_ele[NSTREAMS + 1];
    for (int i = 1; i < NSTREAMS + 1; i++) {
        cudaStreamCreate(&stream[i]);
        offset[i] = (i - 1) * pad_value * num_block_col * pad_value * num_block_row;
        offset_modes[i] = (i - 1) * num_block_col * num_block_row;
        offset_intra_ele[i] = (i - 1) * pad_value * num_block_col * num_block_row;
    }

    cudaMallocHost((void**)&h_recon_blk_decoder_I, size_recon_I);
    clock_t total_GPU_time_start_I, total_GPU_time_end_I;
    total_GPU_time_start_I = clock();

    for (int i = 1; i < NSTREAMS + 1; i++) {
//        cudaMalloc(&d_intra_mode[i], size_intra_modes);
//        cudaMalloc(&d_intra_ele[i], size_intra_ele);
//        cudaMalloc(&d_residual_blk_I[i], size_residuals_I);
//        cudaMalloc(&d_recon_blk_decoder_I[i], size_recon_I);

        cudaMalloc(&d_intra_mode[i], size_intra_modes/total_I_frames);
        cudaMalloc(&d_intra_ele[i], size_intra_ele/total_I_frames);
        cudaMalloc(&d_residual_blk_I[i], size_residuals_I/total_I_frames);
        cudaMalloc(&d_recon_blk_decoder_I[i], size_recon_I/total_I_frames);
    }

    for (int i = 1; i < NSTREAMS + 1; i++) {
//        cudaMemcpyAsync(d_intra_mode[i], intra_modes+offset_modes[i], size_intra_modes, cudaMemcpyHostToDevice, stream[i]);
//        cudaMemcpyAsync(d_intra_ele[i], intra_blocks_cpu+offset_intra_ele[i], size_intra_ele, cudaMemcpyHostToDevice, stream[i]);
//        cudaMemcpyAsync(d_residual_blk_I[i], residual_blk_I+offset[i], size_residuals_I, cudaMemcpyHostToDevice, stream[i]);
//        reconstruct_decoder_gpu<<<grid_decoder, block_decoder, 0, stream[i]>>>(d_intra_mode[i], d_intra_ele[i], d_residual_blk_I[i], d_recon_blk_decoder_I[i], num_block_col, pad_value);
//        cudaMemcpyAsync(h_recon_blk_decoder_I+offset[i], d_recon_blk_decoder_I[i], size_recon_I, cudaMemcpyDeviceToHost, stream[i]);

        cudaMemcpyAsync(d_intra_mode[i], intra_modes+offset_modes[i], size_intra_modes/total_I_frames, cudaMemcpyHostToDevice, stream[i]);
        cudaMemcpyAsync(d_intra_ele[i], intra_blocks_cpu+offset_intra_ele[i], size_intra_ele/total_I_frames, cudaMemcpyHostToDevice, stream[i]);
        cudaMemcpyAsync(d_residual_blk_I[i], residual_blk_I+offset[i], size_residuals_I/total_I_frames, cudaMemcpyHostToDevice, stream[i]);
        reconstruct_decoder_gpu<<<grid_decoder, block_decoder, 0, stream[i]>>>(d_intra_mode[i], d_intra_ele[i], d_residual_blk_I[i], d_recon_blk_decoder_I[i], num_block_col, pad_value);
        cudaMemcpyAsync(h_recon_blk_decoder_I+offset[i], d_recon_blk_decoder_I[i], size_recon_I/total_I_frames, cudaMemcpyDeviceToHost, stream[i]);
    }
    cudaDeviceSynchronize();
    total_GPU_time_end_I = clock();
    double total_GPU_time_I = double(total_GPU_time_end_I - total_GPU_time_start_I) / 1000;

    for (int i = 1; i< NSTREAMS + 1; i++) {
        cudaStreamSynchronize(stream[i]);
        cudaStreamDestroy(stream[i]);
    }

    float *rearranged_recons_decoder_I = (float *) malloc(sizeof(float) * pad_value * pad_value * num_block_row * num_block_col * total_I_frames);
    rearrangeReconstruct(h_recon_blk_decoder_I, rearranged_recons_decoder_I, new_width, new_height, num_block_row, num_block_col, pad_value, total_I_frames);

    // P frames
//    NSTREAMS = 1;
    NSTREAMS = num_block_row;
    dim3 block_decoder_P(pad_value, pad_value);
    dim3 grid_decoder_P(num_block_col, num_block_row / NSTREAMS);
    size_t size_inter_mv = sizeof(int) * 2 * grid_decoder_P.x * grid_decoder_P.y;
    size_t size_residuals_P = sizeof(int) * grid_decoder_P.x * pad_value * grid_decoder_P.y * pad_value;
    size_t size_recon_P = sizeof(float) * grid_decoder_P.x * grid_decoder_P.y * pad_value * pad_value;
    size_t size_prev_recon_blk = sizeof(float) * grid_decoder_P.x * pad_value * (grid_decoder_P.y * pad_value + 2 * r);
    size_t size_prev_recon_blk_boundary = sizeof(float) * grid_decoder_P.x * pad_value * (grid_decoder_P.y * pad_value + r);

    float *h_recon_blk_decoder_P;
    cudaMallocHost((void**)&h_recon_blk_decoder_P, sizeof(float) * total_P_frames * num_block_row * num_block_col * pad_value * pad_value);
    int set_IP_idx = 0;
    float *rearranged_recons_decoder_P;
    cudaMallocHost((void**)&rearranged_recons_decoder_P, sizeof(float) * total_P_frames * num_block_row * num_block_col * pad_value * pad_value);

    double total_GPU_time_P;
    for (int frame = 0; frame < total_P_frames; frame++) {
        int *d_inter_mv[NSTREAMS+1], *d_residual_blk_P[NSTREAMS+1];
        float *d_recon_blk_decoder_P[NSTREAMS+1], *d_prev_recon_blk[NSTREAMS+1];

        cudaStream_t stream_P[NSTREAMS + 1];
        size_t offset_P[NSTREAMS + 1], offset_mv[NSTREAMS + 1], offset_previous[NSTREAMS + 1];

        for (int i = 1; i < NSTREAMS + 1; i++) {
            cudaStreamCreate(&stream_P[i]);

            offset_P[i] = frame * num_block_row * pad_value * num_block_col * pad_value + (i - 1) * pad_value * grid_decoder_P.x * pad_value * grid_decoder_P.y;
            offset_mv[i] = frame * 2 * num_block_row * num_block_col + (i - 1) * 2 * grid_decoder_P.x * grid_decoder_P.y;
            if (frame % num_P_frame == 0) {
                if (i == 1) offset_previous[i] = set_IP_idx * num_block_row * pad_value * num_block_col * pad_value;
                else offset_previous[i] = set_IP_idx * num_block_row * pad_value * num_block_col * pad_value + (i - 1) * pad_value * grid_decoder_P.x * pad_value * grid_decoder_P.y - r * pad_value * grid_decoder_P.x;
            }
            else {
                if (i == 1) offset_previous[i] = (frame - 1) * num_block_row * pad_value * num_block_col * pad_value;
                else offset_previous[i] = (frame - 1) * num_block_row * pad_value * num_block_col * pad_value + (i - 1) * pad_value * grid_decoder_P.x * pad_value * grid_decoder_P.y - r * pad_value * grid_decoder_P.x;
            }
        }

        clock_t total_GPU_time_start_P, total_GPU_time_end_P;
        total_GPU_time_start_P = clock();
        for (int i = 1; i < NSTREAMS + 1; i++) {
            cudaMalloc(&d_inter_mv[i], size_inter_mv);
            cudaMalloc(&d_residual_blk_P[i], size_residuals_P);
            cudaMalloc(&d_recon_blk_decoder_P[i], size_recon_P);
            if (i == 1 or i == NSTREAMS) cudaMalloc(&d_prev_recon_blk[i], size_prev_recon_blk_boundary);
            else cudaMalloc(&d_prev_recon_blk[i], size_prev_recon_blk);
        }

        for (int i = 1; i < NSTREAMS + 1; i++) {
            cudaMemcpyAsync(d_inter_mv[i], inter_mv+offset_mv[i], size_inter_mv, cudaMemcpyHostToDevice, stream_P[i]);
            cudaMemcpyAsync(d_residual_blk_P[i], residual_blk_P+offset_P[i], size_residuals_P, cudaMemcpyHostToDevice, stream_P[i]);

            if (frame % num_P_frame == 0) {
                if (i == 1 or i == NSTREAMS) {
                    cudaMemcpyAsync(d_prev_recon_blk[i], rearranged_recons_decoder_I+offset_previous[i], size_prev_recon_blk_boundary, cudaMemcpyHostToDevice, stream_P[i]);
                }
                else {
                    cudaMemcpyAsync(d_prev_recon_blk[i], rearranged_recons_decoder_I+offset_previous[i], size_prev_recon_blk, cudaMemcpyHostToDevice, stream_P[i]);
                }
            }
            else {
                if (i == 1 or i == NSTREAMS) cudaMemcpyAsync(d_prev_recon_blk[i], rearranged_recons_decoder_P+offset_previous[i], size_prev_recon_blk_boundary, cudaMemcpyHostToDevice, stream_P[i]);
                else cudaMemcpyAsync(d_prev_recon_blk[i], rearranged_recons_decoder_P+offset_previous[i], size_prev_recon_blk, cudaMemcpyHostToDevice, stream_P[i]);
            }

            reconstruct_decoder_gpu_P<<<grid_decoder_P, block_decoder_P, 0, stream_P[i]>>>(d_prev_recon_blk[i], d_inter_mv[i], d_residual_blk_P[i], d_recon_blk_decoder_P[i], r, i);
            cudaMemcpyAsync(h_recon_blk_decoder_P+offset_P[i], d_recon_blk_decoder_P[i], size_recon_P, cudaMemcpyDeviceToHost, stream_P[i]);
        }
        cudaDeviceSynchronize();
        total_GPU_time_end_P = clock();
        total_GPU_time_P += double(total_GPU_time_end_P - total_GPU_time_start_P) / 1000;

        for (int i = 1; i< NSTREAMS + 1; i++) {
            cudaStreamSynchronize(stream_P[i]);
            cudaStreamDestroy(stream_P[i]);
        }

        if (frame % num_P_frame == 0) {
            set_IP_idx++;
        }
        rearrangeReconstruct(h_recon_blk_decoder_P, rearranged_recons_decoder_P, new_width, new_height, num_block_row, num_block_col, pad_value, total_P_frames);
    }
    printf("Decoder recons kernel process time: %f ms\n", total_GPU_time_I + total_GPU_time_P);

    saveMatrixAsPPM(rearranged_reconstructed_blocks, new_width, new_height, num_frame);

    //Docoder CPU
    cudaFreeHost(intra_modes);
    cudaFreeHost(residual_blk_I);
    cudaFreeHost(intra_blocks_cpu);

    // Decoder GPU
    cudaFreeHost(h_recon_blk_decoder_I);
    cudaFree(d_intra_ele);
    cudaFree(d_residual_blk_I);
    cudaFree(d_intra_mode);
    cudaFree(d_recon_blk_decoder_I);
    free(rearranged_recons_decoder_I);

    cudaDeviceReset();

    return 0;
}