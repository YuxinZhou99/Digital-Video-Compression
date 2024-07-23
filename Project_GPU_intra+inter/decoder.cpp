#include "decoder.h"
#include "preprocessing.h"

void readIntraModes(const char *filename, int *matrix, int  num_block_row, int num_block_col) {

    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error: Unable to open file.\n");
        return;
    }

    int row = 0;
    char line[MAX_LINE_LENGTH];
    while (fgets(line, sizeof(line), file)) {
        char *token = strtok(line, " ");
        int col = 0;
        while (token) {
            if (strcmp(token, "end") != 0) {
                matrix[row * num_block_col + col++] = atoi(token);
            }
            token = strtok(NULL, " ");
        }
        row++;
    }

//    // Output the matrix
//    for (int i = 0; i < row; i++) {
//        for (int j = 0; j < num_block_col; j++) {
//            printf("%d ", matrix[i * num_block_col + j]);
//        }
//        printf("\n");
//    }
    fclose(file);
}

void readInterMVs(const char *filename, int *matrix, int  num_block_row, int num_block_col) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error: Unable to open file.\n");
        return;
    }

    int row = 0;
    char line[MAX_LINE_LENGTH];
    while (fgets(line, sizeof(line), file)) {
        char *token = strtok(line, " ");
        int col = 0;
        while (token) {
            if (strcmp(token, " ") != 0) {
                matrix[row * num_block_col * 2 + col++] = atoi(token);
            }
            token = strtok(NULL, " ");
        }
        row++;
    }

//    // Output the matrix
//    for (int i = 0; i < num_block_row; i++) {
//        for (int j = 0; j < num_block_col * 2; j++) {
//            printf("%d ", matrix[i * num_block_col * 2 + j]);
//        }
//        printf("\n");
//    }

    fclose(file);
}

void readResidualBlks(const char *filename, int *residual_blk, int num_block_row, int num_block_col, int pad_value) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error: Unable to open file.\n");
        return;
    }

    int rows = 0;
    char line[sizeof(int) * pad_value * pad_value];
    while (fgets(line, sizeof(line), file)) {
        char *token = strtok(line, " ");
        int col = 0;
        while (token) {
            if (strcmp(token, " ") != 0) {
                residual_blk[rows * pad_value * pad_value + col++] = atoi(token);
            }
            token = strtok(NULL, " ");
        }
        rows++;
    }

//    // Output the matrix
//    for (int i = 0; i < 1; i++) { //frame
//        for (int row = 1; row < 2; row++) {
//            for (int ele = 0; ele < pad_value * pad_value; ele++) { //row
//                printf("%d ", residual_blk[i * num_block_col * pad_value * num_block_row * pad_value + row * pad_value * pad_value + ele]);
//            }
//            printf("\n");
//        }
//    }

    fclose(file);
}

void readIntraEle(const char *filename, int *intra_blocks_cpu, int num_block_row, int num_block_col, int pad_value) {

    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error: Unable to open file.\n");
        return;
    }

    int rows = 0;
    char line[sizeof(int) * pad_value + 1000];
    while (fgets(line, sizeof(line), file)) {
        char *token = strtok(line, " ");
        int col = 0;
        while (token) {
            if (strcmp(token, " ") != 0) {
                intra_blocks_cpu[rows * pad_value + col++] = atoi(token);
            }
            token = strtok(NULL, " ");
        }
        rows++;
    }

//    // Output the matrix
//    for (int i = 0; i < 1; i++) { //frame
//        for (int row = 47; row < 48; row++) {
//            for (int ele = 0; ele < pad_value; ele++) { //row
//                printf("%d ", intra_blocks_cpu[i * num_block_col * num_block_row * pad_value + row * pad_value + ele]);
//            }
//            printf("\n");
//        }
//    }

    fclose(file);
}

void reconstruct_decoder(const int *intra_blocks_cpu, const int *intra_modes, const int * residual_blk_I, const int * residual_blk_P, float *recon_blk_decoder, int num_frame, int num_block_row, int num_block_col, int pad_value, int num_P_frame, const int *inter_mv) {

    int current_I = 0;
    int current_P = 0;
    for (int frame = 0; frame < num_frame; frame++) {
        if (frame % (num_P_frame + 1) == 0) {
            for (int blk_idx = 0; blk_idx < num_block_row * num_block_col; blk_idx++) {
                int single_mode = intra_modes[current_I * num_block_col * num_block_row + blk_idx];
                for (int row = 0; row < pad_value; row++) {
                    for (int col = 0; col < pad_value; col++) {
                        float current_block;
                        if (blk_idx == 0) current_block = 128.0f; // First block
                        else {
                            if (single_mode == 0) {
                                if ((blk_idx % num_block_col) == 0) current_block = 128.0f;
                                else current_block = recon_blk_decoder[frame * num_block_row * pad_value * num_block_col * pad_value + (blk_idx - 1) * pad_value * pad_value + (row + 1) * pad_value - 1];
                            }
                            else {
                                if (blk_idx < num_block_col) current_block = 128.0f;
                                else current_block = recon_blk_decoder[frame * num_block_row * pad_value * num_block_col * pad_value + (blk_idx - num_block_col) * pad_value * pad_value + pad_value * (pad_value - 1) + col];
                            }
                        }
                        recon_blk_decoder[frame * num_block_row * pad_value * num_block_col * pad_value + blk_idx * pad_value * pad_value + row * pad_value + col] = current_block + (float) residual_blk_I[current_I * num_block_row * pad_value * num_block_col * pad_value + blk_idx * pad_value * pad_value + row * pad_value + col];
                    }
                }
            }
            current_I++;
        }
        else {
            float *recon_blk_decoder_prev = (float *) malloc(sizeof(float) * 1 * num_block_row * num_block_col * pad_value * pad_value);
            float *rearranged_recons_decoder = (float *) malloc(sizeof(float) * pad_value * pad_value * num_block_row * num_block_col * 1);
            for (int blk_idx = 0; blk_idx < num_block_row * num_block_col; blk_idx++) {
                for (int row = 0; row < pad_value; row++) {
                    for (int col = 0; col < pad_value; col++) {
                        recon_blk_decoder_prev[blk_idx * pad_value * pad_value + row * pad_value + col] = (float) recon_blk_decoder[(frame - 1) * num_block_row * pad_value * num_block_col * pad_value + blk_idx * pad_value * pad_value + row * pad_value + col];
                    }
                }
            }
            rearrangeReconstruct(recon_blk_decoder_prev, rearranged_recons_decoder, pad_value * num_block_col, pad_value * num_block_row, num_block_row, num_block_col, pad_value, 1);
            for (int blk_row = 0; blk_row < num_block_row; blk_row++) {
                for (int blk_col = 0; blk_col < num_block_col; blk_col++) {
                    int blk_idx = blk_row * num_block_col + blk_col;
                    int current_mv_x = inter_mv[current_P * num_block_row * num_block_col * 2 + blk_idx * 2];
                    int current_mv_y = inter_mv[current_P * num_block_row * num_block_col * 2 + blk_idx * 2 + 1];
                    for (int row = 0; row < pad_value; row++) {
                        for (int col = 0; col < pad_value; col++) {
                            float current_block = (float) rearranged_recons_decoder[(blk_row * pad_value + row - current_mv_y) * num_block_col * pad_value + (blk_col * pad_value + col + current_mv_x)];
                            recon_blk_decoder[frame * num_block_row * pad_value * num_block_col * pad_value + blk_idx * pad_value * pad_value + row * pad_value + col] = current_block + (float) residual_blk_P[current_P * num_block_row * pad_value * num_block_col * pad_value + blk_idx * pad_value * pad_value + row * pad_value + col];
                        }
                    }
                }
            }
            current_P++;
        }
    }
}

void reconsIP(float *rearranged_IP, const float *rearranged_recons_decoder_I, const float *rearranged_recons_decoder_P, int new_width, int new_height, int num_block_row, int num_block_col, int pad_value, int total_I_frames, int total_P_frames, int num_P_frame) {

    int I_frame_count = 0, P_frame_count = 0;
    for (int frame = 0; frame < (total_I_frames + total_P_frames); frame++) {
        if (frame % (num_P_frame + 1) == 0) {
            for (int i = 0; i < new_width * new_height; i++) {
                rearranged_IP[frame * new_width * new_height + i] = rearranged_recons_decoder_I[I_frame_count * new_width * new_height + i];
            }
            I_frame_count++;
        }
        else {
            for (int i = 0; i < new_width * new_height; i++) {
                rearranged_IP[frame * new_width * new_height + i] = rearranged_recons_decoder_P[P_frame_count * new_width * new_height + i];
            }
            P_frame_count++;
        }
    }

}