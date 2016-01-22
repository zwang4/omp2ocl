#include "clang/Omp2Ocl/OpenCLKernelLoop.h"
#include "clang/Omp2Ocl/OpenCLCommon.h"
#include "clang/Omp2Ocl/OpenCLGloballInfoContainer.h"

vector<DeclRefExpr*> OpenCLKernelLoop::functions;

OpenCLKernelLoop::OpenCLKernelLoop(ForStmt* f) { 
	for_stmt = f; 
	arrayV = NULL; 
	bReductionLoop = f->isParallelReduction();
	hasGlobalCopyInBuf = false;
	Kernel = NULL;
	optimised = false;
	useDefLocalWorkGroup = false;
}

void OpenCLKernelLoop::setUseDefLocalWorkGroup()
{
	useDefLocalWorkGroup = true;
}

bool OpenCLKernelLoop::isUseDefLocalWorkGroup()
{
	return useDefLocalWorkGroup;
}

bool OpenCLKernelLoop::hasOptimised()
{
	return optimised;
}

void OpenCLKernelLoop::setOptimised()
{
	optimised = true;
}

void OpenCLKernelLoop::addCopyInBuffer(ValueDecl* d, bool isGlobalBuf)
{
	string name = d->getNameAsString();
	copyInBuffers.push_back(CopyInBuffer(d, isGlobalBuf));
}

bool OpenCLKernelLoop::isACopyIn(string& name)
{
	assert(for_stmt && "current for statement is not set yet!");
	return (for_stmt->getOMPFor().isACopyInVar(name));
}

void OpenCLKernelLoop::addGlobalMemoryVar(OCLGlobalMemVar o)
{
	ValueDecl* v = o.v;
	string name = v->getNameAsString();

	bool isLocalLevel = false;
	if (!v->isDefinedOutsideFunctionOrMethod())
	{
		func->addFunctionLevelOCLBuffer(o);	
		isLocalLevel = true;
	}

	for (vector<OCLGlobalMemVar>::iterator iter = globalMemoryVariables.begin(); iter != globalMemoryVariables.end(); iter++)
	{
		if (name == iter->getNameAsString())
		{
			return;
		}
	}	

	o.isFLevel = isLocalLevel;

	globalMemoryVariables.push_back(o);
	for_stmt->addAGlobalMemoryVariable(o);
}

void OpenCLKernelLoop::addDeclParam(Decl* S)
{
	innerDecls.push_back(S);
}

bool OpenCLKernelLoop::isInnerDecl(DeclRefExpr* df)
{
	//FIXED ME, This maybe a problem
	//Maybe I should use isDefinedOutsideFunctionOrMethod()
	for (vector<Decl*>::iterator iter = innerDecls.begin(); iter != innerDecls.end(); iter++)
	{
		if ((*iter) == df->getDecl())
		{
			return true;
		}
	}
	return false;
}

void OpenCLKernelLoop::removeFromGMVarList(string name)
{
	for (vector<OCLGlobalMemVar>::iterator iter = globalMemoryVariables.begin(); iter != globalMemoryVariables.end(); iter++)
	{
		ValueDecl* d = iter->getDecl();
		if (name == d->getNameAsString())
		{
			globalMemoryVariables.erase(iter);
		}
	}
}

bool OpenCLKernelLoop::isAGlobalMemoryVariable(DeclRefExpr* v)
{
	std::string name = v->getNameInfo().getAsString();
	for (vector<OCLGlobalMemVar>::iterator iter = globalMemoryVariables.begin(); iter != globalMemoryVariables.end(); iter++)
	{
		ValueDecl* d = iter->getDecl();
		if (name == d->getNameAsString())
			return true;
	}
	return false;
}

void OpenCLKernelLoop::addOpenCLNDRangeVar(OpenCLNDRangeVar g)
{
	for (vector<OpenCLNDRangeVar>::iterator iter = gV.begin(); iter != gV.end(); iter++)
	{
		if (iter->variable == g.variable)
			return;
	}

	for_stmt->addOpenCLNDRangeVar(g);

	gV.push_back(g);
}

bool OpenCLKernelLoop::isAOpenCLNDRangeVar(string name)
{
	for (unsigned i=0; i<gV.size(); i++)
	{
		if (gV[i].variable == name)
			return true;
	}

	return false;
}

//Whether a variable is in a user provided list of TLS Variables
bool OpenCLKernelLoop::isInTLSVarList(ValueDecl* d)
{
	string name = d->getName();
	return isInTLSVarList(name);
}

bool OpenCLKernelLoop::isInTLSVarList(string name)
{
	vector<OMPTLSVariable> vars = for_stmt->getOMPFor().getTLSVars();
	for (unsigned i=0; i<vars.size(); i++)
	{
		if (name == vars[i].getVariable())
			return true;
	}

	return false;
}

bool OpenCLKernelLoop::isEnableTLSAutoCheck()
{
	return for_stmt->getOMPFor().isEnableAutoTLSTrack();
}

//Is this a reduction variable
bool OpenCLKernelLoop::isAReductionVariable(string name)
{
	vector<OMPReductionObj>& reducObjs = for_stmt->getOMPFor().getReductionObjs();
	for (vector<OMPReductionObj>::iterator iter = reducObjs.begin(); iter != reducObjs.end(); iter++)
	{
		if (name == iter->getVariable())
			return true;
	}

	return false;
}

void OpenCLKernelLoop::addFuncParam(DeclRefExpr* ref)
{
	const std::string ref_name = ref->getNameInfo().getAsString();
	ValueDecl* decl = ref->getDecl();
	//Functions will add to another category
	if (decl->getKind() == Decl::Function)
	{
		for (vector<DeclRefExpr*>::iterator iter = functions.begin(); iter != functions.end(); iter++)
		{
			if ((*iter)->getNameInfo().getAsString() == ref_name)
			{
				return;
			}	
		}

		functions.push_back(ref);
	}
}

unsigned int OpenCLKernelLoop::getLineNumber(ASTContext& Context)
{
	SourceLocation Soc = getForStmt()->getForLoc();
	return OCLCommon::getLineNumber(Context, Soc);	
}

//has this variable been written in the OpenCLKernel
bool OpenCLKernelLoop::isVariableBeenWrited(DeclRefExpr* expr)
{
	string name = expr->getNameInfo().getAsString();
	for (unsigned i=0; i<RWSet.size(); i++)
	{
		if (RWSet[i].getNameAsString() == name)
		{
			return RWSet[i].isWrite;
		}
	}

	//cerr << "Warning: could not find the read/write status of " << name << endl;
        return false;	
}

void OpenCLKernelLoop::addParam(DeclRefExpr *ref)
{
	const std::string ref_name = ref->getNameInfo().getAsString();
	ValueDecl* decl = ref->getDecl();
	//Functions will add to another category
	if (decl->getKind() == Decl::Function)
	{
		addFuncParam(ref);
	}
	else
	{		
		for (vector<PLoopParam>::iterator iter = params.begin(); iter != params.end(); iter++)
		{
			//FIxed me: this is compared by names only
			if (iter->declRef->getNameInfo().getAsString() == ref_name)
			{
				return;
			}
		}

		params.push_back(PLoopParam(ref, isVariableBeenWrited(ref)));
	}
}

void OpenCLKernelLoop::addSubLoop(ForStmt* node)
{
	subLoops.push_back(node);
}
