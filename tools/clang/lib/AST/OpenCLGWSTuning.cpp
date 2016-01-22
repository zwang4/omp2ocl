#include "clang/Omp2Ocl/OpenCLGWSTuning.h"

void OpenCLGWSTuning::addKernelMacros(string kernelName, unsigned num)
{
	OS << "[ \"" << kernelName << "\",[";
	for (unsigned i=0; i<num; i++)
	{
		if (i > 0)
			OS << ",";
		OS << "\"WGS_" << kernelName << "_" << i << "\"";
	}

	OS << "] ]";
}

void OpenCLGWSTuning::doIt()
{
	for (unsigned int i=0; i<oclLoops.size(); i++)
	{
		ForStmt* l = oclLoops[i]->getForStmt();
		addKernelMacros(l->getKernelName(), l->getLoopIndex().size());
		if (i < oclLoops.size() - 1)
			OS << ",\n";
	}
}

