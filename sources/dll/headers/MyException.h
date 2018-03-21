#pragma once

#include <string>

#include "cuda_runtime.h"

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

class MyException
{
public:
	CUDA_CALLABLE_MEMBER MyException(std::string msg);
	CUDA_CALLABLE_MEMBER ~MyException();
	std::string message;
};

