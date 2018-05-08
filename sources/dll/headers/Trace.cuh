#pragma once

#include "Scene.h"

cudaError CopyToDevice(Scene *s);

void StartSequential();

void startParallel(int block, int thread);

__global__ void SequentialTrace(Triangle *dev_triangles, LightSource *dev_lights, Camera *dev_cameras, int dev_triangles_len, int dev_lights_len, int dev_cameras_len);

__device__ float Trace(LightSource* lights, Triangle *triangles, Vector *ray, int dept, int dev_triangles_len, int dev_lights_len, int dev_cameras_len);

__global__ void ParallelTrace(Triangle *dev_triangles, LightSource *dev_lights, Camera *dev_cameras, int dev_triangles_len, int dev_lights_len, int dev_cameras_len);

cudaError CopyFromDevice(Scene *s);