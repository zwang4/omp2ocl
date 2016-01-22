#include "clang/Omp2Ocl/OpenCLCommon.h"
#include "clang/Omp2Ocl/OpenCLRoutine.h"
#include "clang/Omp2Ocl/OpenCLKernel.h"
#include "clang/Omp2Ocl/OpenCLCompilerOptions.h"
#include "clang/AST/DeclPrinter.h"
#include "clang/Basic/SourceManager.h"
#include <vector>
#include <string>
#include <iostream>
#include "clang/Omp2Ocl/OpenCLGlobalInfoContainer.h"
using namespace std;

vector<OCLGlobalMemVar> OCLCommon::globalMemoryObjs;
FunctionDecl* OCLCommon::CurrentVisitFunction = NULL;
vector<VLoadVarInfo> OCLCommon::vInfo;
int OCLCommon::arch;

string VLoadVarInfo::getDeclareName()
{
	string n = name + "_" + uint2String(c);
  	this->c++;
	return n;
}

bool OCLCommon::isOMP2OpenCLBuiltInRoutine(string name)
{
	if (name == "init_ocl_runtime" ||
			name == "create_ocl_buffers" ||
			name == "release_ocl_buffers" ||
			name == "sync_ocl_buffers" ||
			name == "flush_ocl_buffers" ||
			name == "dump_profiling" ||
			name == "oclHostReads" ||
			name == "oclHostWrites" ||
			name == "oclDevWrites" ||
			name == "oclSync" ||
			name == "oclDevReads" ||
			name == "dump_ml_features"
	   )
	{
		return true;
	}

	return false;
}

bool OCLCommon::isAGlobalMemObj(string name)
{
	for (unsigned i=0; i<globalMemoryObjs.size(); i++)
	{
		if (globalMemoryObjs[i].getNameAsString() == name)
		{
			return true;
		}
	}

	return false;
}


bool OCLCommon::isAGlobalMemObj(VarDecl* d)
{
	return isAGlobalMemObj(d->getNameAsString());
}

LoopIndex* OCLCommon::getLoopIndex(ForStmt* Node)
{
	if (dyn_cast<DeclStmt>(Node->getInit()))
	{
		cerr << "The loop index is a DeclStmt which is not supported yet" << endl;
		exit(-1);
	}
	else
	{
		Expr* expr = cast<Expr>(Node->getInit());
		if (expr->getStmtClass() == Stmt::BinaryOperatorClass)
		{
			BinaryOperator *op = (BinaryOperator*) expr;
			LoopIndex* l = new LoopIndex(op->getLHS(), op->getRHS(), Node);
			return l;
		}
		else
		{
			cerr << "I can only handle a BinaryOperatorClass" << endl;
		}
	}

	return NULL;
}

RecordDecl* OCLCommon::getRecordDecl(const QualType& type)
{
	string ty = getGlobalType(type.getAsString());
	vector<RecordDecl*>& recordDecls = DeclPrinter::recordDecls;

	for (unsigned i=0; i<recordDecls.size(); i++)
	{
		RecordDecl* RD = recordDecls[i];
		IdentifierInfo* II = RD->getIdentifier();

		if (!II)
		{
			continue;
		}

		string str = RD->getKindName ();
		str = str + ' ';
		str = str + II->getName().data();

		if (str == ty)
		{
			return RD;
		}
	}

	cerr << "Could not find the definition of " << ty << endl;
	exit(-1);

	return NULL;

}


OpenCLInputArgu::OpenCLInputArgu(ValueDecl* decl, bool isb, bool isCopyIn, bool isG, FunctionDecl* D)
{
	this->decl = decl;
	isBuffer = isb;
	this->isCopyIn = isCopyIn;
	this->isGlobalThMem = isG;
	this->D = D;
	this->isFLevel = !decl->isDefinedOutsideFunctionOrMethod();
}

string OpenCLInputArgu::getType()
{
	return getCononicalType(decl);
}

string OpenCLInputArgu::getNameAsString()
{
	return decl->getNameAsString();
}

bool OpenCLInputArgu::isAGlobalThMemVar()
{
	return isGlobalThMem;
}

bool OpenCLInputArgu::isNeedWriteSync()
{
	if (!isBuffer)
		return false;

	if (isCopyIn)
		return false;

	if (isGlobalThMem)
		return false;

	return true;
}

bool OpenCLInputArgu::isNeedReadSync()
{
	if (isCopyIn)
		return true;
	return isNeedWriteSync();
}

bool OpenCLInputArgu::canbeDeclareAsGlobal()
{
	if (!isBuffer)
		return false;
	if (isGlobalThMem)
		return false;
	if (isCopyIn)
		return false;
	return (decl->isDefinedOutsideFunctionOrMethod());
}

string OpenCLInputArgu::getGType()
{
	return getGlobalType(getCononicalType(decl));
}


OCLGlobalMemVar::OCLGlobalMemVar(ValueDecl* d, bool isG, bool isFLevel, bool isThreadPrivate)
{
	this->v = d;
	isGlobalThreadPrivate = isG;
	this->isFLevel = isFLevel;
	if (!isFLevel)
	{
		this->isFLevel = !(d->isDefinedOutsideFunctionOrMethod());
	}

	this->isThreadPrivate = isThreadPrivate;

	string type = getCononicalType(d);
	unsigned dim = getArrayDimension(type);

	if (dim == 0)
	{
		isArray = false;
	}
	else
	{
		isArray = true;
	}

}

OCLGlobalMemVar::OCLGlobalMemVar(FuncProtoExt& P)
{
	this->v = P.expr->getDecl();
	this->isFLevel = true;
	this->isGlobalThreadPrivate = P.isGTP();
	this->isThreadPrivate = false;
}

string OCLGlobalMemVar::getNameAsString() { 
	if (declaredName.length()) 
		return declaredName; 

	return v->getNameAsString(); 
}

bool OCLCommon::isAOMPPrivateVariable(string name, ForStmt* for_stmt)
{
	return for_stmt->getOMPFor().isVariablePrivate(name);
}

bool OCLCommon::isAThreadPrivateVariable(string& name, vector<OMPThreadPrivateObject>& threadPrivates, bool& isGlobal)
{
	isGlobal = false;
	for (vector<OMPThreadPrivateObject>::iterator iter = threadPrivates.begin(); iter != threadPrivates.end(); iter++)
	{
		if (iter->getVariable() == name)
		{
			isGlobal = iter->isUseGlobalMem();
			return true;
		}
	}

	return false;
}

bool OCLCommon::isAGlobalMemThreadPrivateVar(string& name)
{
	bool isGlobal = false;
	vector<OMPThreadPrivateObject>& threadPrivates = OpenCLGlobalInfoContainer::getThreadPrivateVars();
	isAThreadPrivateVariable(name, threadPrivates, isGlobal);
	if (isGlobal) 
		return true;
	else 
		return false;
}

bool OCLCommon::isAGlobalMemThreadPrivateVar(ValueDecl* D)
{
	bool isGlobal = false;

	//Threadprivate buffers should be global static variables
	if (!D->isDefinedOutsideFunctionOrMethod())
	{
		return false;
	}
	
	string name = D->getNameAsString();
	string type = getCononicalType(D);
	unsigned dim = getArrayDimension(type);

	if (dim <= 0) 
		return false;


	vector<OMPThreadPrivateObject>& threadPrivates = OpenCLGlobalInfoContainer::getThreadPrivateVars();
	isAThreadPrivateVariable(name, threadPrivates, isGlobal);
	if (isGlobal) 
		return true;
	else 
		return false;
}

bool OCLCommon::isAGTPVariable(string name, vector<OCLGlobalMemVar>& globalMemoryVariables, vector<OMPThreadPrivateObject>& threadPrivates)
{
	//	if  (isAGMThreadPrivateVariable(name, threadPrivates))
	//	{
	//		return true;
	//	}

	for (vector<OCLGlobalMemVar>::iterator iter = globalMemoryVariables.begin(); iter != globalMemoryVariables.end(); iter++)
	{
		if (name == iter->getNameAsString())
		{
			return iter->isGlobalThreadPrivate;
		}
	}

	return false;
}

string OpenCLNDRangeVar::getCondString(ASTContext& Context)
{
	string cond;
	if (isCondInt)
	{
		string buf = getStringStmt(Context, Cond);
		if (cond_string.length() == 0)
		{
			return buf;
		}
		else //This is mainly for ">= and > " opcode
		{
			return cond_string;
		}
	}
	else
	{
		BinaryOperator* oc = dyn_cast<BinaryOperator>(Cond);
		string opcode = BinaryOperator::getOpcodeStr(oc->getOpcode());
		string buf = getStringStmt(Context, oc->getLHS());

		if (opcode == "<" || opcode == "<=")
		{
			buf = buf + opcode;
		}
		else
		{
			buf = cond_string;
			//There must be something wrong if I reach here
			if (cond_string.length() == 0)
			{
				cerr << "Warning: Something must be wrong when printing the conditions" << endl;
			}
		}

		return buf + "__ocl_" + variable + "_bound";
	}
}

/*!
 * Retrive the line number from Loc
 * return 0 if fail
 */
unsigned int OCLCommon::getLineNumber(ASTContext& Context, SourceLocation& Loc)
{
	SourceManager &SM = Context.getSourceManager();
	PresumedLoc PLoc = SM.getPresumedLoc(Loc);
	return PLoc.isValid()? PLoc.getLine() : 0;
}

/*!
 * Retrive the file name from Loc
 * return NULL if fail
 */
const char* OCLCommon::getFileName(ASTContext& Context, SourceLocation& Loc)
{
	SourceManager &SM = Context.getSourceManager();
	PresumedLoc PLoc = SM.getPresumedLoc(Loc);
	return PLoc.getFilename();
}

/*!
 * Check whether the for_stmt is a perfect nested loop
 *
 */
bool OCLCommon::isAPerfectNestedLoop(ForStmt* for_stmt)
{
	Stmt* stmt = for_stmt->getBody();

	if (CompoundStmt *CS = dyn_cast<CompoundStmt>(stmt))
	{
		unsigned int c = 0;
		for (CompoundStmt::body_iterator I = CS->body_begin(), E = CS->body_end();
				I != E; ++I)
		{
			stmt = (*I);
			//The first statement should be a for statement, otherwise, it won't be
			//a perfect nest loop.
			if (!dyn_cast<ForStmt>(stmt))
			{
				return false;
			}

			c++;
		}

		if (c > 1)
		{
			return false;
		}
	}
	else
	{	
		if (!dyn_cast<ForStmt>(stmt)) {
			return false;
		}
	}

	return true;
}

void OCLCommon::setArch(string carch)
{
	if (carch == "AMD")
	{
		arch = ARCH_AMD;
	}
	else
		if (carch == "NVIDIA")
		{
			arch = ARCH_NVIDIA;
		}
		else
		if (carch == "CPU")
		{
			arch = ARCH_CPU;
			defaultCPUSetting();
		}
		else
		{
			cerr << "Unknown architecture: " << carch << ", set to NVIDIA" << endl;
			arch = ARCH_NVIDIA;
		}
}

/*
 * Reset the compiler setting when arch=CPU
 *
 */
void OCLCommon::defaultCPUSetting()
{
	OCLCompilerOptions::EnableSoftwareCache = false;
	OCLCompilerOptions::UseLocalMemory = false;
	OCLCompilerOptions::EnableLoopInterchange = false;
}

string OCLCommon::getArchString()
{
	if (arch == ARCH_AMD)
	{
		return "AMD";
	}
	else
	if (arch == ARCH_CPU)
	{
		return "Intel";
	}
	else
	{
		return "NVIDIA";
	}
}

int OCLCommon::getArch()
{
	return arch;
}

bool OCLCommon::isAGTPVariable(string name, vector<OMPThreadPrivateObject>& objs)
{

	for (unsigned i=0; i<objs.size(); i++)
	{
		if (objs[i].getVariable() == name)
		{
			return objs[i].isUseGlobalMem();
		}
	}

	return false;
}

SourceLocation getPrivateVLoc(string& name, vector<OMPThreadPrivateObject>& objs)
{

	SourceLocation loc;	
	for (vector<OMPThreadPrivateObject>::iterator iter = objs.begin(); iter != objs.end(); iter++)
	{
		if (iter->getVariable() == name)
			return iter->loc;
	}

	return loc;
}

bool OCLCommon::isAThreadPrivateVariable(string& name)
{
	bool isG;
	return isAThreadPrivateVariable(name, OpenCLGlobalInfoContainer::getThreadPrivateVars(), isG);
}

bool OCLCommon::isAThreadPrivateVariable(ValueDecl* d)
{
	bool isG;
	string name = d->getNameAsString();

	bool is = isAThreadPrivateVariable(name, OpenCLGlobalInfoContainer::getThreadPrivateVars(), isG);

	if (!d->isDefinedOutsideFunctionOrMethod() && is)
	{
		cerr << "Warning: " << name << " is a local variable, but is treated as threadprivate" << endl;
	}
	
	return is;
}

//Get a declare name for a vload variable
string OCLCommon::getVLoadVariableName(string name)
{
	for (unsigned i=0; i<vInfo.size(); i++)
	{
		if (vInfo[i].getName() == name)
		{
			return vInfo[i].getDeclareName();		
		}
	}

	VLoadVarInfo v(name);
	vInfo.push_back(v);
	unsigned id = vInfo.size() - 1;

	return vInfo[id].getDeclareName();
}

unsigned OCLCommon::getMemAlignSize(string type)
{
	if (type == "char")
		return 8;
	if (type == "double" || type == "long" || type == "unsigned long")
		return 64;

	return 32;
}
