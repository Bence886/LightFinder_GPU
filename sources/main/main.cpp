#include "main.h"

#include <iostream>

#include "kernel.cuh"
#include "Log.h"

int main()
{
	Log::InitLog();
	WriteLog("Program started!", true, Log::Message);

	Start();
	
	WriteLog("Program finished!", true, Log::Message);
	std::getchar();
	Log::CloseLog();
	return 0;
}
