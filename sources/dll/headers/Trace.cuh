#pragma once

#include "Scene.h"

cudaError CopyToDevice(Scene *s);

void startSequential();

__global__ void SequentialTrace(Triangle **dev_triangles);

__global__ void ParallelTrace();

cudaError CopyFromDevice(Scene *s);