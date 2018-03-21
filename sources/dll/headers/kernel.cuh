#ifndef KERNEL_CUH
#define KERNEL_CUH

#include "Scene.h"
#include "BelnderScriptCreator.h"

void Init();

void ProcessInput();

void StartCPU();

void StartGPU();

void WriteOutput();

void Close();

#endif