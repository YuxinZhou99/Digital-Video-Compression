#ifndef PROJECT_PREDICT_MOTION_H
#define PROJECT_PREDICT_MOTION_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void rearrangeReconstruct(float* ,float*, int, int, int, int, int, int);
void write_modes_into_txt(int*, int, int, int, int);
void write_mv_into_txt(int*, int*, int, int, int, int);
void write_rb_into_txt(float*, int, int, int, int, int);
void write_intra_ele_into_txt(float*, int, int, int, int);

#endif //PROJECT_PREDICT_MOTION_H
