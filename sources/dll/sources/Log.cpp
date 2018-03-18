#include "Log.h"

#include <iostream>

bool Log::DoLog = true;
Log::LogLevel Log::currentLogLevel = Log::Trace;
std::ofstream Log::ofs;

void Log::WriteLogFunction(std::string msg, bool console, LogLevel level)
{
	if (DoLog)
	{
		if (console && level >= currentLogLevel)
		{
			std::cout << level << " : " << msg << std::endl;
		}
		if (ofs.is_open())
		{
			ofs << level << " : " << msg << std::endl;
		}
	}
}

void Log::InitLog()
{
	ofs.open("Log.txt");
}

void Log::CloseLog()
{
	if (ofs.is_open())
	{
		ofs.close();
	}
}