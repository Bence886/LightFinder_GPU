#include "BelnderScriptCreator.h"

BlenderScriptCreator::BlenderScriptCreator(std::string filename)
{
	ofs.open(filename);
	ofs << def;
}


BlenderScriptCreator::~BlenderScriptCreator()
{
	if (ofs.is_open())
	{
		ofs.close();
	}
}

void BlenderScriptCreator::CreateObject(Point *points, std::string objName, int pointsNum)
{
	if (ofs.is_open())
	{
		ofs << "\nverts = [\n";
		for (int i = 0; i < pointsNum; i++)
		{
			ofs << points[i].ToFile();
			ofs << ",\n";
		}
		ofs << "]\n";

		ofs << "create_Vertices(\"";
		ofs << objName;
		ofs << "\", verts)\n";
	}
}
