#include "clang/Omp2Ocl/OpenCLKernelName.h"
#include <stdio.h>

vector<KernelIndex> OpenCLKernelName::oclKernelIdxs;

string OpenCLKernelName::getOpenCLKernelName(FunctionDecl* f)
{
	string s;
	char buf[128];

	s = f->getNameInfo().getAsString();
	snprintf(buf, 1024, "_%u", getOCLKernelIdx(s));

	s = s + buf;

	return s;
}

unsigned int OpenCLKernelName::getOCLKernelIdx(string funcName)
{
	for (vector<KernelIndex>::iterator iter = oclKernelIdxs.begin(); iter != oclKernelIdxs.end(); iter++)
	{
		if (iter->function == funcName)
		{
			unsigned int index = iter->index;
			iter->index++;
			return index;
		}
	}

	KernelIndex g(funcName);
	oclKernelIdxs.push_back(g);

	return 0;
}
