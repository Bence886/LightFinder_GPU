#include "BelnderScriptCreator.h"

BelnderScriptCreator::BelnderScriptCreator(std::string filename)
{
	ofs.open(filename);
	ofs << def;
}


BelnderScriptCreator::~BelnderScriptCreator()
{
	if (ofs.is_open())
	{
		ofs.close();
	}
}