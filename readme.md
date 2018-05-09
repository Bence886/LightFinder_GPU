# LightFinder_GPU 

CMake device link is not working!?
Fix:
Visual Studio
	LightFinder_GPU_DLL/Propertyes/CUDA Linker/General/Perform Device Link = Yes (-dlink)
	[Camera.cpp, Point.cpp, Vector.cpp, Triangle.cpp]/Propertyes/Item Type = CUDA C/C++