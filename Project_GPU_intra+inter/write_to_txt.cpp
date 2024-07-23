#include "write_to_txt.h"

void rearrangeReconstruct(float* reconstructed_blocks, float* rearranged_reconstructed_blocks, int new_width, int new_height, int num_block_row, int num_block_col, int pad_value, int num_frame){
    int blockCount = 0;
    for (int frame = 0; frame < num_frame; frame++) {
        for (int i = 0; i < new_height; i += pad_value) {
            for (int j = 0; j < new_width; j += pad_value) {
                for (int x = 0; x < pad_value; x++) {
                    for (int y = 0; y < pad_value; y++) {
                        rearranged_reconstructed_blocks[frame * new_width * new_height + (i + x) * new_width + j + y] = reconstructed_blocks[blockCount * pad_value * pad_value + x * pad_value + y];
                    }
                }
                blockCount++;
            }
        }
    }
}

void write_modes_into_txt(int* I_frame_modes, int num_block_row, int num_block_col, int num_frame, int pad_value){
    FILE *modes_file = fopen("intra_modes.txt", "w");
    if (modes_file == NULL) {
        printf("Error opening file for writing.\n");
        return;
    }

    for (int current_frame = 0; current_frame < num_frame; current_frame++) {
        for (int current_row = 0; current_row < num_block_row; current_row++) {
            for (int current_col = 0; current_col < num_block_col; current_col++) {
                size_t block_offset = current_frame*num_block_row*num_block_col + current_row*num_block_col+current_col;
                fprintf(modes_file, "%d ", I_frame_modes[block_offset]);
                fprintf(modes_file, "end ");
            }
            fprintf(modes_file, "\n");
        }
    }

    fclose(modes_file);
}

void write_mv_into_txt(int *motion_vector_row, int *motion_vector_col, int num_block_row, int num_block_col, int num_frame, int pad_value){
    FILE *mv_file = fopen("motion_vectors.txt", "w");
    if (mv_file == NULL) {
        printf("Error opening file for writing.\n");
        return;
    }

    for (int current_frame = 0; current_frame < num_frame; current_frame++) {
        for (int current_row = 0; current_row < num_block_row; current_row++) {
            for (int current_col = 0; current_col < num_block_col; current_col++) {
                size_t block_offset = current_frame*num_block_row*num_block_col + current_row*num_block_col+current_col;
                fprintf(mv_file, "%d %d ", motion_vector_col[block_offset], motion_vector_row[block_offset]);
            }
            fprintf(mv_file, "\n");
        }
    }

    fclose(mv_file);
}

void write_rb_into_txt(float* residual_blocks, int num_block_row, int num_block_col, int num_frame, int pad_value, int num_P_frame){
    FILE *rb_file_I = fopen("residual_blocks_I.txt", "w");
    if (rb_file_I == NULL) {
        printf("Error opening file for writing.\n");
        return;
    }
    FILE *rb_file_P = fopen("residual_blocks_P.txt", "w");
    if (rb_file_P == NULL) {
        printf("Error opening file for writing.\n");
        return;
    }


    size_t block_size = pad_value*pad_value;
    size_t blocks_single_frame = num_block_row*num_block_col;
    size_t frame_size = block_size*blocks_single_frame;
    size_t total_size = frame_size*num_frame;

    for (int current_frame = 0; current_frame < num_frame; current_frame++) {
        if (current_frame%(num_P_frame+1) == 0 | num_P_frame == 0) {
            for (int current_row = 0; current_row < num_block_row; current_row++) {
                for (int current_col = 0; current_col < num_block_col; current_col++) {
                    size_t offset = current_frame * frame_size + current_row * num_block_col * block_size + current_col * block_size;
                    for(int i=0; i<block_size; i++){
                        fprintf(rb_file_I, "%d ", int(residual_blocks[offset + i]));
                    }
                    fprintf(rb_file_I, "\n");
                }
            }
        }
        else{
            for (int current_row = 0; current_row < num_block_row; current_row++) {
                for (int current_col = 0; current_col < num_block_col; current_col++) {
                    size_t offset = current_frame * frame_size + current_row * num_block_col * block_size + current_col * block_size;
                    for(int i=0; i<block_size; i++){
                        fprintf(rb_file_P, "%d ", int(residual_blocks[offset + i]));
                    }
                    fprintf(rb_file_P, "\n");
                }
            }
        }
    }

    fclose(rb_file_I);
    fclose(rb_file_P);
}

void write_intra_ele_into_txt(float* intra_ele_lines, int num_block_row, int num_block_col, int num_frame, int pad_value){
    FILE *rb_file = fopen("intra_ele.txt", "w");
    if (rb_file == NULL) {
        printf("Error opening file for writing.\n");
        return;
    }

    size_t block_size = pad_value;
    size_t blocks_single_frame = num_block_row*num_block_col;
    size_t frame_size = block_size*blocks_single_frame;
    size_t total_size = frame_size*num_frame;

    for (int current_frame = 0; current_frame < num_frame; current_frame++) {
        for (int current_row = 0; current_row < num_block_row; current_row++) {
            for (int current_col = 0; current_col < num_block_col; current_col++) {
                size_t offset = current_frame * frame_size + current_row * num_block_col * block_size + current_col * block_size;
                for(int i=0; i<block_size; i++){
                    fprintf(rb_file, "%d ", int(intra_ele_lines[offset+i]));
                }
                fprintf(rb_file, "\n");
            }
        }
    }

    fclose(rb_file);
}
