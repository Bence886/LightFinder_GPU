#include "Log.h"


bool Log::DoLog = true;
Log::LogLevel Log::currentLogLevel = Log::Exception;
//FILE *f;

void Log::WriteLogFunction(std::string msg, bool console, LogLevel level)
{
	if (DoLog)
	{
		if (console && level >= currentLogLevel)
		{
			printf("%d : %s\n", level, msg.c_str());
		}
		//if (f)
		{
			//fprintf(f, "%d : %s\n", (int)level, msg);
		}
	}
}

void Log::InitLog()
{
	//f = fopen("Log.txt", "w");
}

void Log::CloseLog()
{
	//if (f)
	{
		//fclose(f);
	}
}