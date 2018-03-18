#pragma once

#include <string>
#include <fstream>

#define LOG_ON

#ifdef LOG_ON
#define WriteLog(msg, console, level) Log::WriteLogFunction(msg, console, level)
#else	WriteLog
#define WriteLog(msg, console, level)
#endif

static class Log
{
public:
	enum LogLevel
	{
		Exception, Trace, Debug, Message, Warning, Error
	};

	static LogLevel currentLogLevel;
	static bool DoLog;

	static void WriteLogFunction(std::string msg, bool console, LogLevel level);

	static void InitLog();
	static void CloseLog();

private:
	static std::ofstream ofs;
};