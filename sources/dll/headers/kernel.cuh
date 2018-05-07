#ifndef KERNEL_CUH
#define KERNEL_CUH

#include "Scene.h"
#include "BelnderScriptCreator.h"

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

void Init();

void ProcessInput();

void StartCPU();

void StartGPU();

void WriteOutput();

void Close();

#endif