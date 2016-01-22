#include "clang/Omp2Ocl/OpenCLCompilerOptions.h"
#include "clang/Omp2Ocl/OpenCLLocalMemOpt.h"
#include "clang/Omp2Ocl/OpenCLCommon.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
using namespace clang;
using namespace std;

bool OCLCompilerOptions::UseLocalMemory = true;
bool OCLCompilerOptions::EnableSoftwareCache = true;
unsigned OCLCompilerOptions::DefaultParallelDepth = PARALLEL_LOOP_DEPTH;
bool OCLCompilerOptions::UserDefParallelDepth = false;
bool OCLCompilerOptions::EnableLoopInterchange = true;
bool OCLCompilerOptions::EnableMLFeatureCollection = false;
bool OCLCompilerOptions::EnableDebugCG = false;
bool OCLCompilerOptions::UseArrayLinearize = false;
bool OCLCompilerOptions::EnableGPUTLs = false;
bool OCLCompilerOptions::StrictTLSChecking = true;
bool OCLCompilerOptions::TLSCheckAtProgramEnd = true;
bool OCLCompilerOptions::printLinearMacros = false;
bool OCLCompilerOptions::GenProfilingFunc = true;
bool OCLCompilerOptions::OclTLSMechanism = false;

bool OCLCompilerOptionAction::isLocalVar(ValueDecl* D)
{
	if (OCLCompilerOptions::UseLocalMemory)
	{
		return OpenCLLocalMemOpt::isLocalVar(D);	
	}

	return false;
}

bool OCLCompilerOptionAction::isLocalVar(OpenCLKernelLoop* loop, ValueDecl* D)
{
	if (OCLCompilerOptions::UseLocalMemory)
	{
		return OpenCLLocalMemOpt::isLocalVar(loop, D);	
	}

	return false;
}

void OCLCompilerOptions::printCompilerOptions()
{
	std::cout << endl << "----------------------------------------------------------" << endl;
	std::cout << "Compiler options:" << endl;
	std::cout << "	Software Cache\t" << ((OCLCompilerOptions::EnableSoftwareCache == true) ? "true" : "false") << endl;
	std::cout << "	Local Memory\t" << ((OCLCompilerOptions::UseLocalMemory == true) ? "true" : "false") << endl;
	std::cout << "	DefaultParallelDepth\t" << OCLCompilerOptions::DefaultParallelDepth << endl;
	std::cout << "	UserDefParallelDepth\t" << ((OCLCompilerOptions::UserDefParallelDepth == true) ? "true" : "false") << endl;
	std::cout << "	EnableLoopInterchange\t"<< ((OCLCompilerOptions::EnableLoopInterchange == true) ? "true" : "false") << endl;
	std::cout << "	Generating debug/profiling code\t"<< ((OCLCompilerOptions::EnableDebugCG == true) ? "true" : "false") << endl;
	std::cout << "	EnableMLFeatureCollection\t"<< ((OCLCompilerOptions::EnableMLFeatureCollection == true) ? "true" : "false") << endl;
	std::cout << "	Array Linearization\t"<< ((OCLCompilerOptions::UseArrayLinearize == true) ? "true" : "false") << endl;
	std::cout << "	GPU TLS\t"<< ((OCLCompilerOptions::EnableGPUTLs == true) ? "true" : "false") << endl;
	std::cout << "	Strict TLS Checking\t"<< ((OCLCompilerOptions::StrictTLSChecking == true) ? "true" : "false") << endl;
	std::cout << "	Check TLS Conflict at the end of program execution\t"<< ((OCLCompilerOptions::TLSCheckAtProgramEnd == true) ? "true" : "false") << endl;
	std::cout << "	Use OCL TLS \t"<< ((OCLCompilerOptions::OclTLSMechanism == true) ? "true" : "false") << endl;
	std::cout << "----------------------------------------------------------" << endl;
}

void OCLCompilerOptions::commentCompilerOptions(llvm::raw_fd_ostream& out)
{
	out << "//Compiler options: \n";
	out << "//	Software Cache\t" << ((OCLCompilerOptions::EnableSoftwareCache == true) ? "true" : "false") << "\n";
	out << "//	Local Memory\t" << ((OCLCompilerOptions::UseLocalMemory == true) ? "true" : "false") << "\n";
	out << "//	DefaultParallelDepth\t" << OCLCompilerOptions::DefaultParallelDepth << "\n";
	out << "//	UserDefParallelDepth\t" << ((OCLCompilerOptions::UserDefParallelDepth == true) ? "true" : "false") << "\n";
	out << "//	EnableLoopInterchange\t"<< ((OCLCompilerOptions::EnableLoopInterchange == true) ? "true" : "false") << "\n";
	out << "//	Generating debug/profiling code\t"<< ((OCLCompilerOptions::EnableDebugCG == true) ? "true" : "false") << "\n";
	out << "//	EnableMLFeatureCollection\t"<< ((OCLCompilerOptions::EnableMLFeatureCollection == true) ? "true" : "false")  << "\n";
	out << "//	Array Linearization\t"<< ((OCLCompilerOptions::UseArrayLinearize == true) ? "true" : "false")  << "\n";
	out << "//	GPU TLs\t"<< ((OCLCompilerOptions::EnableGPUTLs == true) ? "true" : "false") << "\n";
	out << "//	Strict TLS Checking\t"<< ((OCLCompilerOptions::StrictTLSChecking == true) ? "true" : "false") << "\n";
	out << "//	Check TLS Conflict at the end of function\t"<< ((OCLCompilerOptions::TLSCheckAtProgramEnd == true) ? "true" : "false") << "\n";
	out << "//	Use OCL TLS \t"<< ((OCLCompilerOptions::OclTLSMechanism == true) ? "true" : "false") << "\n";
}
