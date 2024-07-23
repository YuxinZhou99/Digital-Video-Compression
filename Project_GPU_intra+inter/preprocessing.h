#ifndef PROJECT_PREPROCESSING_H
#define PROJECT_PREPROCESSING_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_PARAMETERS 16
#define MAX_LINE_LENGTH 100

void readConfig(const char *, char **, int *);
void yuv_read_420(int, int, int, const char *, float *);
void padding(float*, int, int, int, int, float**, int*, int*);
void saveMatrixAsPPM(const float *, int, int, int);
void saveMatrixAsPPM_2(float ****, int, int, int, int, int, int);
float**** allocateBlocksArray(float*, int, int, int, int, int, int);
float* rearrangeAllocatedBlocks(float*, float****, int, int, int, int, int, int);

#endif //PROJECT_READ_CONFIG_H
