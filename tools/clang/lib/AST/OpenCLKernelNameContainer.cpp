#include "clang/Omp2Ocl/OpenCLKernelNameContainer.h"
using namespace clang;

vector<string> OpenCLKernelNameContainer::kernelNames;

void OpenCLKernelNameContainer::addKernelName(string name)
{
	kernelNames.push_back(name);
}
