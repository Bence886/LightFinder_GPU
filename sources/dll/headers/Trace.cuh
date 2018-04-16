#pragma once

#include "Scene.h"

cudaError CopyToDevice(Scene *s);

__global__ void SequentialTrace();

__global__ void ParallelTrace();

cudaError CopyFromDevice(Scene *s);