#pragma once

#include <string>

class Exception
{
public:
	Exception(std::string msg);
	~Exception();
	std::string message;
};

