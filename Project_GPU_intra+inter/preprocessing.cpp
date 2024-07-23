#include "preprocessing.h"

void readConfig(const char *filename, char *parameters[], int *num_parameters) {
    FILE *fileID_config;
    char formatSpec[10];
    char config[MAX_LINE_LENGTH];

    fileID_config = fopen(filename, "r");
    if (fileID_config == NULL) {
        printf("Error opening config.txt file.\n");
        return;
    }

    // Read format specification
    fscanf(fileID_config, "%s", formatSpec);

    // Read configuration parameters
    int i = 0;
    while (fgets(config, MAX_LINE_LENGTH, fileID_config) != NULL && i < MAX_PARAMETERS) {
        // Remove newline character
        config[strcspn(config, "\n")] = 0;
        char *line = strtok(config, ";");
        while (line != NULL) {
            char *line_split = strtok(line, "=");
            while (line_split != NULL) {
                // Allocate memory for the parameter
                parameters[i] = (char *)malloc((strlen(line_split) + 1) * sizeof(char));
                if (parameters[i] == NULL) {
                    printf("Error: Memory allocation failed.\n");
                    return;
                }
                strcpy(parameters[i], line_split);
                i++;
                line_split = strtok(NULL, "=");
            }
            line = strtok(NULL, ";");
        }
    }
    fclose(fileID_config);
    *num_parameters = i;
}

void yuv_read_420(int width, int height, int num_frame, const char *video, float *y_only) {
    FILE *file = fopen(video, "rb"); // Open the file in binary mode for reading
    if (file == NULL) {
        printf("Error opening file: %s\n", video);
        return;
    }

    // Read the entire file as a stream
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    rewind(file);
    unsigned char *stream_video = (unsigned char *)malloc(file_size * sizeof(unsigned char));
    fread(stream_video, sizeof(unsigned char), file_size, file);
    fclose(file);

    // Reshape and extract Y component
    long frame_length = width * height * 1.5;
    for (int frame = 0; frame < num_frame; frame++) {
        unsigned char *frame_data = stream_video + frame * frame_length;
        unsigned char *y_data = frame_data;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                y_only[frame * width * height + i * width + j] = y_data[i * width + j];
            }
        }
    }

    free(stream_video);
}

void padding(float* y_only, int width, int height, int num_frame, int pad_value, float** img_pad, int* new_width, int* new_height) {
    *new_width = 0;
    *new_height = 0;

    if ((width % pad_value != 0) || (height % pad_value != 0)) {
        *new_width = pad_value * ((width + pad_value - 1) / pad_value);
        *new_height = pad_value * ((height + pad_value - 1) / pad_value);
    } else {
        *new_width = width;
        *new_height = height;
    }

    *img_pad = (float*)malloc(sizeof(float) * (*new_height) * (*new_width) * num_frame);

    for (int frame = 0; frame < num_frame; ++frame) {
        for (int i = 0; i < *new_height; ++i) {
            for (int j = 0; j < *new_width; ++j) {
                if (i < height && j < width) {
                    (*img_pad)[frame * (*new_height) * (*new_width) + i * (*new_width) + j] = y_only[frame * height * width + i * width + j];
                } else {
                    (*img_pad)[frame * (*new_height) * (*new_width) + i * (*new_width) + j] = 128;
                }
            }
        }
    }
}

void saveMatrixAsPPM(const float *matrix, int width, int height, int num_frame) {
    // Write the pixel data
    for (int frame = 0; frame < num_frame; frame++) {
        char img_name[256]; // Assuming the maximum length of the video path is 256 characters
        sprintf(img_name, "./img/img%d.pp", frame);

        FILE *fp = fopen(img_name, "wb");
        if (!fp) {
            printf("Error opening file for writing: %s\n", img_name);
            return;
        }

        // Write the PPM header
        fprintf(fp, "P6\n%d %d\n255\n", width, height);

        for (int i = 0; i < width * height; i++) {
            fputc(matrix[frame * width * height + i], fp);  // Assuming each value in the matrix is a pixel value (0-255)
            fputc(matrix[frame * width * height + i], fp);
            fputc(matrix[frame * width * height + i], fp);
        }

        fclose(fp);
    }
}

void saveMatrixAsPPM_2(float ****split_img, int width, int height, int num_frame, int num_block_row, int num_block_col, int pad_value) {
    // Write the pixel data
    for (int frame = 0; frame < num_frame; frame++) {

        for (int i = 0; i < num_block_row; ++i) {
            for (int j = 0; j < num_block_col; ++j) {
                float* block = split_img[frame][i][j];

                char img_name[256]; // Assuming the maximum length of the video path is 256 characters
                sprintf(img_name, "./img/img%d_%d_%d.pp", frame, i, j);

                FILE *fp = fopen(img_name, "wb");
                if (!fp) {
                    printf("Error opening file for writing: %s\n", img_name);
                    return;
                }

                // Write the PPM header
                fprintf(fp, "P6\n%d %d\n255\n", pad_value, pad_value);

                for (int k = 0; k < pad_value * pad_value; k++) {
                    fputc(block[k], fp);  // Assuming each value in the matrix is a pixel value (0-255)
                    fputc(block[k], fp);
                    fputc(block[k], fp);
                }

                fclose(fp);
            }
        }

    }
}

// Function to allocate memory for the blocks array
float**** allocateBlocksArray(float* img_pad, int num_frame, int num_block_row, int num_block_col, int pad_value, int new_width, int new_height) {
    float ****blocks = (float ****) malloc(sizeof(float ***) * num_frame);
    for (int frame = 0; frame < num_frame; frame++) {
        blocks[frame] = (float ***) malloc(sizeof(float **) * num_block_row);
        for (int i = 0; i < num_block_row; ++i) {
            blocks[frame][i] = (float **) malloc(sizeof(float *) * num_block_col);
            for (int j = 0; j < num_block_col; ++j) {
                blocks[frame][i][j] = (float *) malloc(sizeof(float) * pad_value * pad_value);
                for (int m = 0; m < pad_value; ++m) {
                    for (int n = 0; n < pad_value; ++n) {
                        blocks[frame][i][j][m * pad_value + n] = img_pad[frame * new_height * new_width + (i * pad_value + m) * new_width + j * pad_value + n];
                    }
                }
            }
        }
    }

    return blocks;
}

// Function to rearrange blocks to fit in the kernel coalesced form
float* rearrangeAllocatedBlocks(float* rearrange_split_img, float**** split_img, int num_frame, int num_block_row, int num_block_col, int pad_value, int new_width, int new_height) {
    int rearrange_index = 0;
    for (int frame = 0; frame < num_frame; frame++) {
        for (int i = 0; i < num_block_row; ++i) {
            for (int j = 0; j < num_block_col; ++j) {
                for (int m = 0; m < pad_value; ++m) {
                    for (int n = 0; n < pad_value; ++n) {
                        rearrange_split_img[rearrange_index] = split_img[frame][i][j][m * pad_value + n];
                        rearrange_index++;
                    }
                }
            }
        }
    }

    return rearrange_split_img;
}