#include "kernel.cuh"

#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Point.h"

__device__ Point* d_P;
Point P(1, 1, 1);

__global__ void ASD()
{
	d_P->X++;
}

void Init()
{

	cudaMemcpyToSymbol(d_P, &P, sizeof(Point));

	ASD << <1, 1 >> > ();

	cudaMemcpyFromSymbol(&P, d_P, sizeof(Point));

	std::cout << P.X << std::endl;

	std::getchar();
}
