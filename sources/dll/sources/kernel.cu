#include "kernel.cuh"

#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Trace.cuh"

#include "Log.h"

#define OUTPUT_NAME "Blender.txt"
#define INPUT_NAME "In.xml"

#define cudaCheckError() {cudaError_t e = cudaGetLastError(); WriteLog(std::string("CUDA result:") + cudaGetErrorString(e), true, Log::Exception);}

Scene *myScene;

void ProcessInput()
{
	WriteLog(std::string("Started reading input from: ") + INPUT_NAME, true, Log::Debug);
	myScene = new Scene(INPUT_NAME);
	WriteLog(std::string("Finished reading input from: ") + INPUT_NAME, true, Log::Debug);
}

void StartCPU()
{
	myScene->StartTrace_CPU();
	WriteLog("Finished CPU trace.", true, Log::Debug);
}

void StartGPU()
{

	myScene->cameras[0]->lookDirections[0] = Point(1, 2, 3);

	WriteLog("Started copy to GPU", true, Log::Trace);
	CopyToDevice(myScene);
	cudaCheckError();

	StartSequential();
	cudaCheckError();

	WriteLog("Started copy from GPU", true, Log::Trace);
	CopyFromDevice(myScene);
	cudaCheckError();
}

void WriteOutput()
{
	WriteLog(std::string("Started writing belnder scripts to: ") + OUTPUT_NAME, true, Log::Debug);
	BlenderScriptCreator bs(OUTPUT_NAME);

	for (Camera *item : myScene->cameras)
	{
		bs.CreateObject(item->lookDirections, "Camera", item->lookNum);
	}
	WriteLog(std::string("Finished writing belnder scripts to: ") + OUTPUT_NAME, true, Log::Debug);
}

void Close()
{
	delete myScene;
}