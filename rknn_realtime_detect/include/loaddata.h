#include <cstdio>
#ifndef _RKNN_PREPROCESS_LOAD_MODEL_H_
#define _RKNN_PREPROCESS_LOAD_MODEL_H_

static int saveFloat(const char* file_name, float* output, int element_size);

static unsigned char* load_data(FILE* fp, size_t ofst, size_t sz);

static unsigned char* load_model(const char* filename, int* model_size);

#endif //_RKNN_PREPROCESS_LOAD_MODEL_H_
