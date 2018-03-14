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

void BlenderScriptCreator::CreateObject(std::vector<Point> points, std::string objName)
{
	if (ofs.is_open())
	{
		ofs << "\nverts = [\n";
		for (Point item : points)
		{
			ofs << item.ToFile();
			ofs << ",\n";
		}
		ofs << "]\n";

		ofs << "create_Vertices(";
		ofs << objName;
		ofs << ", verts)\n";
	}
}
