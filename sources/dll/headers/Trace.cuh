#pragma once

#include "Scene.h"

cudaError CopyToDevice(Scene *s);

void StartSequential();

void startParallel();

__global__ void SequentialTrace(Triangle *dev_triangles, LightSource *dev_lights, Camera *dev_cameras);

__global__ void ParallelTrace();

cudaError CopyFromDevice(Scene *s);