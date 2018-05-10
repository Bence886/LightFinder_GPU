#include "main.h"

#include <iostream>

#include "kernel.cuh"
#include "Log.h"

void Init()
{
	Log::InitLog();
	WriteLog(std::string("Log initialized, log level: ") , true, Log::Message);
}


int main()
{
	Init();

	ProcessInput();
	
	//StartCPU();
	StartGPU();

	WriteOutput();

	Close();
	
	WriteLog("Program finished!", true, Log::Message);
	//std::getchar();
	Log::CloseLog();
	return 0;
}
