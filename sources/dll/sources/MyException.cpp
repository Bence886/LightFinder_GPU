#include "MyException.h"

#include "Log.h"

MyException::MyException(std::string msg):message(msg)
{
	WriteLog(msg, true, Log::Exception);
}

MyException::~MyException()
{
}
