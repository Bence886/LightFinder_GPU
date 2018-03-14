#pragma once

#include <string>
#include <fstream>
#include <vector>

#include "Point.h"

class BelnderScriptCreator
{
public:
	BelnderScriptCreator(std::string filename);
	~BelnderScriptCreator();

	void CreateObject(std::vector<Point> points, std::string objName);

private:
	std::string def = 
		"import bpy\n"
		"import bmesh\n"
		"def create_Vertices(name, verts) :\n"
			"\tme = bpy.data.meshes.new(name + 'Mesh')\n"
			"\tob = bpy.data.objects.new(name, me)\n"
			"\tob.show_name = True\n"
			"\tbpy.context.scene.objects.link(ob)\n"
			"\tme.from_pydata(verts, [], [])\n"
			"\tme.update()\n"
			"\treturn ob\n";

	std::ofstream ofs;

};
