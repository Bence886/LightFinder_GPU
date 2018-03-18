#include "kernel.cuh"

#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Scene.h"
#include "BelnderScriptCreator.h"

void Start()
{
	Scene s("In.xml");

	s.StartTrace_CPU();

	BlenderScriptCreator bs("Blender.txt");

	for (Camera *item : s.cameras)
	{
		bs.CreateObject(item->lookDirections, "Camera");
	}	
}
