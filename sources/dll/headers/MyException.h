#pragma once

#include <string>

class MyException
{
public:
	MyException(std::string msg);
	~MyException();
	std::string message;
};

