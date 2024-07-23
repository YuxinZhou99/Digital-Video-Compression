#ifndef PROJECT_DECODER_H
#define PROJECT_DECODER_H
#include <math.h>
#include "write_to_txt.h"

void readIntraModes(const char *, int *, int, int);
void readInterMVs(const char *, int *, int, int);
void readResidualBlks(const char *, int *, int, int, int);
void readIntraEle(const char *, int *, int, int, int);
void reconstruct_decoder(const int *, const int *, const int *, const int *, float *, int, int, int, int, int, const int *);
void reconsIP(float *, const float *, const float *, int, int, int, int, int, int, int, int);
//void prepare_current_ele(const int *, float *, int, int, int, int);

#endif //PROJECT_DECODER_H
