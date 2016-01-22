#include "clang/Omp2Ocl/Pragma.h"
#include "clang/Omp2Ocl/OpenCLCompilerOptions.h"
#include "clang/Lex/LexDiagnostic.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/CrashRecoveryContext.h"
#include "llvm/Support/ErrorHandling.h"
#include "clang/Omp2Ocl/OpenCLGlobalInfoContainer.h"

using namespace clang;

bool OMPThreadPrivate::isAThreadPrivateVariable(string& name)
{
	for (vector<OMPThreadPrivateObject>::iterator iter = objs.begin(); iter != objs.end(); iter++)
	{
		if (iter->getVariable() == name)
			return true;
	}

	return false;
}

bool OMPThreadPrivate::isAGlobalMemThreadPrivateVar(string& name)
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


SourceLocation OMPThreadPrivate::getLoc(string& name)
{
	SourceLocation loc;	
	for (vector<OMPThreadPrivateObject>::iterator iter = objs.begin(); iter != objs.end(); iter++)
	{
		if (iter->getVariable() == name)
			return iter->loc;
	}

	return loc;
}

void OMPFor::print(llvm::raw_ostream& Out)
{
	Out << "#pragma omp parallel for ";
	if (reducObjs.size())
	{
		Out << " reduction(" << reducObjs[0].getOperatorCode() << ":";
		PRINT(reducObjs);
		Out << ") ";
	}

	if (privates.size())
	{
		Out << " private(";
		PRINT(privates);
		Out << ") ";
	}

	if (first_privates.size())
	{
		Out << " firstprivate(";
		PRINT(first_privates);
		Out << ") ";
	}

	if (copyins.size())
	{
		Out << " copyin(";
		PRINT(copyins);
		Out << ") ";
	}

	if (schedule.length())
		Out << " schedule( " << schedule << ")";

}

void OMPFor::set2Reduction()
{
	isReduction = true;
}

vector<OMPMultIterIndex>& OMPFor::getMultIterIndex()
{
	return multIterIndex;
}

void OMPFor::addMultIterIndex(OMPMultIterIndex& O)
{
	multIterIndex.push_back(O);
}

void OMPFor::addReductionVariable(OMPReductionObj& obj)
{
	reducObjs.push_back(obj);
}

vector<OMPReductionObj>& OMPFor::getReductionObjs()
{
	return reducObjs;
}

bool OMPFor::isACopyInVar(string& name)
{
	for (vector<OMPCopyIn>::iterator iter = copyins.begin(); iter != copyins.end(); iter++)
	{
		if (name == iter->getVariable())
			return true;
	}

	return false;
}

bool OMPFor::isReductionFor() { return isReduction; }

bool OMPFor::getSwap() { return swap; }
void OMPFor::setSwap(bool l) { this->swap = l; }

OMPParallelDepth& OMPFor::getParallelDepth() { 
	return depth; 
}

void OMPFor::addPrivateVariable(OMPPrivate &val) {
	privates.push_back(val);
}

void OMPFor::addThreadPrivateVariable(OMPThreadPrivateObject& v)
{
	//FIXME, ZHENG: I should overload the == operator
	for (vector<OMPThreadPrivateObject>::iterator iter = threadPrivates.begin(); iter != threadPrivates.end(); iter++)
	{
		if (v.getVariable() == iter->getVariable())
			return;
	}

	threadPrivates.push_back(v);
}

bool OMPFor::isAThreadPrivateVariable(string &v)
{
	for (vector<OMPThreadPrivateObject>::iterator iter = threadPrivates.begin(); iter != threadPrivates.end(); iter++)
	{
		if (v == iter->getVariable())
			return true;
	}

	return false;
}

void OMPFor::setParallelDepth(OMPParallelDepth depth) { 
	this->depth = depth;
}
void OMPFor::addFirstPrivateVariable(OMPFirstPrivate &val) {first_privates.push_back(val);}
void OMPFor::addCopyInVariable(OMPCopyIn &val) {copyins.push_back(val);}
void OMPFor::addTLSVariable(OMPTLSVariable &val) {tlsVars.push_back(val); enable_auto_tls_track=false;}
void OMPFor::addSwapIndex(OMPSwapIndex &val) { swapIndexs.push_back(val);}
void OMPFor::addSchedule(std::string sch) {schedule = sch; }
void OMPFor::print() {}

std::vector<OMPSwapIndex> OMPFor::getSwapIndexs() { return swapIndexs; }
std::vector<OMPPrivate> OMPFor::getPrivate() {return privates;}
std::vector<OMPFirstPrivate> OMPFor::getFirstPrivate() {return first_privates;}
std::vector<OMPCopyIn> OMPFor::getCopyIn() {return copyins;}
std::vector<OMPThreadPrivateObject> OMPFor::getThreadPrivate() {return threadPrivates;}
std::string OMPFor::getSchedule() {return schedule;}

bool OMPFor::isVariablePrivate(std::string& v)
{
	for (std::vector<OMPPrivate>::iterator iter = privates.begin(); iter != privates.end(); iter++)
	{
		if (iter->getVariable() == v)
		{
			return true;
		}
	}

	return false;			
}

bool OMPFor::isVariableFirstPrivate(std::string& v)
{
	for (std::vector<OMPFirstPrivate>::iterator iter = first_privates.begin(); iter != first_privates.end(); iter++)
	{
		if (iter->getVariable() == v)
		{
			return true;
		}
	}

	return false;			
}

bool OMPFor::isVariableCopyIns(std::string& v)
{
	for (std::vector<OMPCopyIn>::iterator iter = copyins.begin(); iter != copyins.end(); iter++)
	{
		if (iter->getVariable() == v)
		{
			return true;
		}
	}

	return false;
}

OMPFor OMPFor::operator=(OMPFor& rhs)
{
	if (this != &rhs)
	{
		this->swap = rhs.getSwap();
		this->isReduction = rhs.isReductionFor();
		this->reducObjs = rhs.getReductionObjs();	
		this->privates = rhs.getPrivate();
		this->first_privates = rhs.getFirstPrivate();
		this->copyins = rhs.getCopyIn();
		this->swapIndexs = rhs.getSwapIndexs();
		this->multIterIndex = rhs.getMultIterIndex();
		this->threadPrivates = rhs.getThreadPrivate();
		this->schedule = rhs.getSchedule();
		this->depth = rhs.getParallelDepth();
		this->tlsVars = rhs.getTLSVars();
		this->tls_check = rhs.isTLSCheck();
	}

	return *this;
}

void OMPParallelDepth::init(Diagnostic &Diag, SourceLocation& loc, unsigned depth)
{
	if (depth == 0)
	{
		Diag.Report(loc, diag::warn_parallel_depth_set_to_pvalue) << 0;
	}

	if (OCLCompilerOptions::UserDefParallelDepth)
	{
		if ((int)depth > OCLCompilerOptions::DefaultParallelDepth)
		{
			if (depth != PARALLEL_LOOP_DEPTH)
			{
				Diag.Report(loc, diag::warn_overwrite_parallel_depth)
					<< depth << OCLCompilerOptions::DefaultParallelDepth;
			}
	
			depth = OCLCompilerOptions::DefaultParallelDepth;
		}
	}
	
	this->userSetDepth = false;
	this->depth = depth;
}

//OMPParallelDepth
OMPParallelDepth::OMPParallelDepth(unsigned d, Diagnostic &Diag, SourceLocation& loc)
{
	init(Diag, loc, d);
	customDepth();
}

OMPParallelDepth::OMPParallelDepth(string depth, vector<string>& seq, Diagnostic &Diag, SourceLocation& loc)
{
	type = ocl_parallel_depth;
	init (Diag, loc, atoi(depth.c_str()));
	seqs = seq;
	customDepth();
}

OMPParallelDepth::OMPParallelDepth()
{
	type = ocl_parallel_depth;
	depth = OCLCompilerOptions::DefaultParallelDepth;
	userSetDepth = false;
}
