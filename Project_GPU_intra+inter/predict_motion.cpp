#include "predict_motion.h"

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

float mae(float* single_block, float* reference_block, int size){
    float sum = 0;
    for (int i=0; i<size; i++){
        sum += fabs(single_block[i]-reference_block[i]);
    }
    return sum/size;
}

//void rearrange_img()

void predict_motion(float* rearrange_split_img, int num_block_row, int num_block_col, int num_frame, int pad_value, int r, int n, int *I_frame_modes, float *mae_blocks, float *residual_blocks, float *predicted_blocks, float *reconstructed_blocks, float *intra_ele_lines){
    // these can be put into shared memory
    int modes_count;
    float *single_intra_block = (float *) malloc(sizeof(float) * pad_value * pad_value);
    float *single_vertical_intra_block = (float *) malloc(sizeof(float) * pad_value * pad_value);
    float *single_horizontal_intra_block = (float *) malloc(sizeof(float) * pad_value * pad_value);
    float *single_reference_block = (float *) malloc(sizeof(float) * pad_value * pad_value);
    float *single_predict_block = (float *) malloc(sizeof(float) * pad_value * pad_value);
    float *single_residual_block = (float *) malloc(sizeof(float) * pad_value * pad_value);
    float *single_reconstructed_block = (float *) malloc(sizeof(float) * pad_value * pad_value);

    size_t block_size = pad_value*pad_value;
    size_t blocks_single_frame = num_block_row*num_block_col;
    size_t frame_size = block_size*blocks_single_frame;
    size_t total_size = frame_size*num_frame;
    for (size_t i=0; i<total_size; i+=block_size){
        int single_block_mode = 0;
        size_t frame = i/(block_size*blocks_single_frame);
        size_t current_row = ((i%frame_size)/block_size)/num_block_col;
        size_t current_column = ((i%frame_size)/block_size)%num_block_col;
        // if gpu, put offset into device side
        size_t block_offset = frame*blocks_single_frame + current_row*num_block_col+current_column;
        size_t offset = frame * frame_size + current_row * num_block_col * block_size + current_column * block_size;
        for (size_t j=0; j<block_size; j++){
            int current_row_single_block = j/pad_value;
            int current_col_single_block = j%pad_value;

            // get the reference block
            single_reference_block[j] = rearrange_split_img[offset+j];

//            if (current_row == 0 && current_column == 0){
//                printf("%f\n", single_reference_block[j]);
//            }

            // horizontal (mode 0)
            // boundry case for hirizontal
            if (current_column == 0){
                single_horizontal_intra_block[j] = 128;
            }
                // non-boundary case
            else{
                // in gpu, remove the for loop
                for (int k = 0; k<block_size; k++){
                    single_intra_block[k] = reconstructed_blocks[offset+k-block_size];
                }
//                if (current_row == 0 && current_column == 1){
//                    printf("%f\n", single_intra_block[j]);
//                }
//                single_intra_block[j] = reconstructed_blocks[offset+j-block_size];
                // get last element of each row from previous block
                single_horizontal_intra_block[j] = single_intra_block[current_row_single_block*pad_value+pad_value-1];
            }

            // vertical (mode 1)
            // boundry case for vertical
            if (current_row == 0){
                single_vertical_intra_block[j] = 128;
            }
            else{
                // in gpu, remove the for loop
                for (int k = 0; k<block_size; k++){
                    single_intra_block[k] = reconstructed_blocks[offset+k-block_size*num_block_col];
                }
//                single_intra_block[j] = reconstructed_blocks[offset+j-block_size*num_block_col];
                single_vertical_intra_block[j] = single_intra_block[(pad_value-1)*pad_value+current_col_single_block];
            }

        }
        // find the mode that gives lowest MAE cost for one block
        // need to syn threads before, then use only one thread to process the following
        float horizontal_mae = mae(single_horizontal_intra_block, single_reference_block, block_size);
        float vertical_mae = mae(single_vertical_intra_block, single_reference_block, block_size);
        float min_mae = horizontal_mae;
        if (vertical_mae < horizontal_mae){
            single_block_mode = 1;
            min_mae = vertical_mae;
        }
        I_frame_modes[block_offset] = single_block_mode;
        mae_blocks[block_offset] = min_mae;
        for (int line_idx = 0; line_idx < pad_value; line_idx++) {
            if (single_block_mode == 0) {
                intra_ele_lines[block_offset * pad_value + line_idx] = single_horizontal_intra_block[(line_idx + 1) * pad_value - 1];
            }
            else {
                intra_ele_lines[block_offset * pad_value + line_idx] = single_vertical_intra_block[(pad_value - 1) * pad_value + line_idx];
            }
        }
//        printf("%f\n", horizontal_mae);

        for (size_t j=0; j<block_size; j++) {
            if (single_block_mode == 0) {
                single_predict_block[j] = single_horizontal_intra_block[j];
            }
            else {
                single_predict_block[j] = single_vertical_intra_block[j];
            }
            predicted_blocks[offset+j] = single_predict_block[j];

//            single_residual_block[j] = round((single_reference_block[j]-single_predict_block[j])/pow(2,n))*pow(2,n);
            single_residual_block[j] = single_reference_block[j]-single_predict_block[j];
            residual_blocks[offset+j] = single_residual_block[j];

            single_reconstructed_block[j] = single_residual_block[j]+single_predict_block[j];
            reconstructed_blocks[offset+j] = single_reconstructed_block[j];
//            if (current_row == 0 && current_column == 0){
//                printf("%f\n", single_reference_block[j]);
//            }
        }
    }

//    free(single_intra_block);
//    free(single_vertical_intra_block);
//    free(single_horizontal_intra_block);
//    free(single_reference_block);
//    free(single_predict_block);
//    free(single_residual_block);
//    free(single_reconstructed_block);

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
                fprintf(modes_file, "%d ", int(I_frame_modes[block_offset]));
                fprintf(modes_file, "end ");
            }
            fprintf(modes_file, "\n");
        }
    }

    fclose(modes_file);
}

void write_rb_into_txt(float* residual_blocks, int num_block_row, int num_block_col, int num_frame, int pad_value){
    FILE *rb_file = fopen("residual_blocks.txt", "w");
    if (rb_file == NULL) {
        printf("Error opening file for writing.\n");
        return;
    }

    size_t block_size = pad_value*pad_value;
    size_t blocks_single_frame = num_block_row*num_block_col;
    size_t frame_size = block_size*blocks_single_frame;
    size_t total_size = frame_size*num_frame;

    for (int current_frame = 0; current_frame < num_frame; current_frame++) {
        for (int current_row = 0; current_row < num_block_row; current_row++) {
            for (int current_col = 0; current_col < num_block_col; current_col++) {
                size_t offset = current_frame * frame_size + current_row * num_block_col * block_size + current_col * block_size;
                for(int i=0; i<block_size; i++){
                    fprintf(rb_file, "%d ", int(residual_blocks[offset+i]));
                }
                fprintf(rb_file, "\n");
            }
        }
    }

    fclose(rb_file);
}

//void write_intra_ele_into_txt(float* predicted_blocks, int num_block_row, int num_block_col, int num_frame, int pad_value){
//    FILE *rb_file = fopen("intra_ele.txt", "w");
//    if (rb_file == NULL) {
//        printf("Error opening file for writing.\n");
//        return;
//    }
//
//    size_t block_size = pad_value*pad_value;
//    size_t blocks_single_frame = num_block_row*num_block_col;
//    size_t frame_size = block_size*blocks_single_frame;
//    size_t total_size = frame_size*num_frame;
//
//    for (int current_frame = 0; current_frame < num_frame; current_frame++) {
//        for (int current_row = 0; current_row < num_block_row; current_row++) {
//            for (int current_col = 0; current_col < num_block_col; current_col++) {
//                size_t offset = current_frame * frame_size + current_row * num_block_col * block_size + current_col * block_size;
//                for(int i=0; i<block_size; i++){
//                    fprintf(rb_file, "%d ", int(predicted_blocks[offset+i]));
//                }
//                fprintf(rb_file, "\n");
//            }
//        }
//    }
//
//    fclose(rb_file);
//}

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