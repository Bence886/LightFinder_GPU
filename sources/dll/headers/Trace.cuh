#pragma once

#include "Scene.h"

cudaError CopyToDevice(Scene *s);

void StartSequential();

void startParallel(int block, int thread);

__global__ void SequentialTrace(Triangle *dev_triangles, LightSource *dev_lights, Camera *dev_cameras);

__global__ void ParallelTrace(Triangle *dev_triangles, LightSource *dev_lights, Camera *dev_cameras);

cudaError CopyFromDevice(Scene *s);