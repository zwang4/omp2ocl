#include "clang/Omp2Ocl/OpenCLLocalMemOpt.h"
#include "clang/Omp2Ocl/OpenCLGlobalInfoContainer.h"
#include "clang/Omp2Ocl/OpenCLCopyInRoutine.h"
#include "clang/Omp2Ocl/OpenCLRoutine.h"
#include <iostream>

using namespace std;
using namespace clang;

bool OpenCLLocalMemOpt::isLocalVar(ValueDecl* d)
{
	vector<OCLLocalVar>& oclVars = OpenCLGlobalInfoContainer::getLocalMemVars(); 
	string name = d->getNameAsString();
	bool found = false;

	for (unsigned i=0; i<oclVars.size(); i++)
	{
		if (name == oclVars[i].getName())
		{
			found = true;
			break;
		}
	}

	//local declared variables can only be a __local
	//memory var when it is in the omp parallel for pragma
	//list of the loop
	//
	if (!d->isDefinedOutsideFunctionOrMethod() && found)
	{
		cerr << endl << "Warning: " << name 
			<< " is declared inside a function (as told by clang)" << endl;
	}

	return found;
}

bool OpenCLLocalMemOpt::isLocalVar(OpenCLKernelLoop* loop, ValueDecl* d)
{
	return false;
}

void OpenCLLocalMemOpt::printLocalVars()
{
	vector<OCLLocalVar>& oclVars = OpenCLGlobalInfoContainer::getLocalMemVars(); 

	cerr << endl << "Local memory variables: ";	

	for (unsigned i=0; i<oclVars.size(); i++)
	{
		cerr << oclVars[i].getName();
	}

	cerr << endl;
}

void OpenCLLocalMemOpt::genPreloadCode(llvm::raw_ostream& Out, ValueDecl* d, string passInName)
{
	string type = getCononicalType(d);
	vector<unsigned> arrayDef = getArrayDef(type);	
	assert(arrayDef.size() >= 1 && "This is not a buffer!");

	Out << "      /*Copyin code for " << d->getNameAsString() << " (BEGIN)*/\n";
	OpenCLCopyInRoutine::genLocalCopyInCode(Out, d, passInName, "__local");
	Out << "      /*Copyin code for " << d->getNameAsString() << " (END)*/\n\n";
}

//Declare local vars
void OpenCLLocalMemOpt::declareLocalVar(llvm::raw_ostream& Out, ValueDecl* d)		
{
	Out << "__local ";
	string type = getCononicalType(d);
	string gtype = getGlobalType(type);
	vector<unsigned> arrayDef = getArrayDef(type);	

	Out << gtype << " " << d->getNameAsString();

	for (unsigned i=0; i<arrayDef.size(); i++)
	{
		Out << "[" << arrayDef[i] << "]";
	}

	Out << ";\n";
}
