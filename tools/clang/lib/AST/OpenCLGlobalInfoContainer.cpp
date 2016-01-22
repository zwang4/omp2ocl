#include "clang/Omp2Ocl/OpenCLGlobalInfoContainer.h"
#include "clang/Lex/Pragma.h"
#include "clang/AST/GlobalCallArgPicker.h"
#include "clang/AST/StmtPicker.h"

vector<RecordDecl*> OpenCLGlobalInfoContainer::RDs;
vector<TypedefDecl*> OpenCLGlobalInfoContainer::TDs;
vector<FunctionDecl*> OpenCLGlobalInfoContainer::candidateFuncs;
vector<FunctionDecl*> OpenCLGlobalInfoContainer::revisedFuncs;
vector<FunctionDecl*> OpenCLGlobalInfoContainer::calleeFuncs;
vector<FunctionDecl*> OpenCLGlobalInfoContainer::allRecordFuncs;
vector<RenamedFuncInfo> OpenCLGlobalInfoContainer::rnFuncs;
vector<OMPThreadPrivateObject> OpenCLGlobalInfoContainer::threadPrivateVars;
vector<OpenCLKernelLoop*> OpenCLGlobalInfoContainer::oclLoops;
vector<Decl*> OpenCLGlobalInfoContainer::globalDecls;
vector<OCLLocalVar> OpenCLGlobalInfoContainer::oclLocalVars;
vector<ValueDecl*> OpenCLGlobalInfoContainer::_writeGlobalMemObjs;
vector<RenameFuncGPUTLS> OpenCLGlobalInfoContainer::rfGPUTLS;

void OpenCLGlobalInfoContainer::addRenameFuncGPUTLS(string func_name, ValueDecl* d, unsigned i)
{
	rfGPUTLS.push_back(RenameFuncGPUTLS(func_name, d, i));
}

vector<RenameFuncGPUTLS> OpenCLGlobalInfoContainer::getRenameFuncGPUTLs()
{
	return rfGPUTLS;
}

bool OpenCLGlobalInfoContainer::isFuncHasGPUTLSLog(string func_name)
{
	for (unsigned i=0; i<rfGPUTLS.size(); i++)
	{
		if (rfGPUTLS[i].getFuncName() == func_name)
		{
			return true;
		}
	}

	return false;
}

void OpenCLGlobalInfoContainer::addRecordDecl(RecordDecl* rd)
{
	RDs.push_back(rd);
}

vector<RecordDecl*>& OpenCLGlobalInfoContainer::getRecordDecls()
{
	return RDs;
}

void OpenCLGlobalInfoContainer::addTypedefDecl(TypedefDecl* TD)
{
	TDs.push_back(TD);
}

vector<TypedefDecl*>& OpenCLGlobalInfoContainer::getTypedefDecls()
{
	return TDs;
}

vector<FunctionDecl*>& OpenCLGlobalInfoContainer::getCandidateFuncs()
{
	return candidateFuncs;
}

vector<RenamedFuncInfo>& OpenCLGlobalInfoContainer::getRenameFuncs()
{
	return rnFuncs;
}

void OpenCLGlobalInfoContainer::addFuncDecl(FunctionDecl* D)
{
	candidateFuncs.push_back(D);
}

void OpenCLGlobalInfoContainer::addRenameFuncInfo(RenamedFuncInfo& info)
{
	rnFuncs.push_back(info);
}

void OpenCLGlobalInfoContainer::addThreadPrivate(OMPThreadPrivateObject& obj)
{
	threadPrivateVars.push_back(obj);
}

vector<OMPThreadPrivateObject>& OpenCLGlobalInfoContainer::getThreadPrivateVars()
{
	return threadPrivateVars;
}

vector<FunctionDecl*> OpenCLGlobalInfoContainer::getRevisedFuncs()
{
	return revisedFuncs;
}

void OpenCLGlobalInfoContainer::addRevisedFunc(FunctionDecl* D)
{
	revisedFuncs.push_back(D);
}

void OpenCLGlobalInfoContainer::addRecordFunc(FunctionDecl* D)
{
	if (D->getStorageClass() != SC_Extern)
	{
		string name = D->getNameInfo().getAsString();
		for (unsigned i=0; i<allRecordFuncs.size(); i++)
		{
			if (name == allRecordFuncs[i]->getNameInfo().getAsString())
			{
				return;
			}
		}

		allRecordFuncs.push_back(D);
	}
}

vector<FunctionDecl*>& OpenCLGlobalInfoContainer::getRecordFuncs()
{
	return allRecordFuncs;
}

vector<FunctionDecl*>& OpenCLGlobalInfoContainer::getCalleeFuncs()
{
	return calleeFuncs;
}

void OpenCLGlobalInfoContainer::addCalleeFunc(FunctionDecl* D)
{
	if (D->getKind() == Decl::Function)
	{
		string name = D->getNameInfo().getAsString();
		for (unsigned i=0; i<calleeFuncs.size(); i++)
		{
			if (name == calleeFuncs[i]->getNameInfo().getAsString())
			{
				return;
			}
		}

		calleeFuncs.push_back(D);
	}
}

vector<OpenCLKernelLoop*>& OpenCLGlobalInfoContainer::getOclLoops()
{
	return oclLoops;
}

void OpenCLGlobalInfoContainer::addOclLoop(OpenCLKernelLoop* loop)
{
	oclLoops.push_back(loop);
}

vector<Decl*> OpenCLGlobalInfoContainer::getGlobalDecls()
{
	return globalDecls;
}

void OpenCLGlobalInfoContainer::addGlobalDecl(Decl* D)
{
	globalDecls.push_back(D);
}

static bool isInUnIntVector(vector<globalVarIndex>& arrays, globalVarIndex gv)
{
	for (vector<globalVarIndex>::iterator iter=arrays.begin(); iter != arrays.end(); iter++)
	{
		if (iter->i == gv.i && iter->isPointerAccess == gv.isPointerAccess)
			return true;
	}

	return false;
}

//put an global buffer that is written into the vector. This is used to support GPU TLS
void OpenCLGlobalInfoContainer::addwriteGlobalMemObj(ValueDecl* d)
{
	string name = d->getName();
	for (unsigned i=0; i<_writeGlobalMemObjs.size(); i++)
	{
		if (name == _writeGlobalMemObjs[i]->getName())
		{
			return;
		}
	}

	//Only a variable declared outside functions will be recorded here
	if (d->isDefinedOutsideFunctionOrMethod())
	{
		_writeGlobalMemObjs.push_back(d);
	}
}

string OpenCLGlobalInfoContainer::getAName4RenamedFunc(ASTContext& Context, CallArgInfoContainer* cArg, vector<FunctionDecl*>& funcsNeed2Revised)
{
	CallExpr* Node = cArg->ce;

	//Get callee name
	string funcName;
	llvm::raw_string_ostream os_v(funcName);
	StmtPicker p(os_v, Context, NULL, Context.PrintingPolicy);
	p.PrintExpr(Node->getCallee());
	os_v.flush();

	FunctionDecl* D = Node->getDirectCallee();

	if (!D)
	{
		cerr << "WARNING: Rename for function: " << funcName << " may introduce bugs!" << endl;
	}

	//collect the extended id
	unsigned numParam = D->getNumParams();
	vector<extendVarIndex> EXTID;

	for (unsigned i = 0; i < numParam; ++i) {
		ParmVarDecl* parm = D->getParamDecl(i);
		if (parm->isOCLExtended())
		{
			EXTID.push_back(extendVarIndex(i));	
		}
	}

	//The index of the pointer access arguments
	vector<globalVarIndex> PIs;
	for (unsigned ii=0; ii<cArg->gCallArgs.size(); ii++)
	{
		PIs.push_back(globalVarIndex(cArg->gCallArgs[ii])); 
	}

	//Check whether we already have one funcs
	for (vector<RenamedFuncInfo>::iterator iter=rnFuncs.begin(); iter != rnFuncs.end(); iter++)
	{
		if (iter->origFuncName != funcName)
			continue;	

		if (iter->hasGlobalMemThreadPrivate != cArg->hasGlobalMemThreadPrivate)
			continue;

		bool matched = true;

		for (unsigned ii=0; ii<PIs.size(); ii++)
		{
			if (!isInUnIntVector(iter->globalArugIds, PIs[ii]))
			{
				if (iter->isInExtendId(PIs[ii].i))
					continue;

				matched = false;
				break;
			}
		}
		
		if (matched)
		{
			//Check extid
			for (unsigned ii=0;  ii<EXTID.size(); ii++)
			{
				if (!iter->isInExtendId(EXTID[ii]))
				{
					matched = false;
					break;
				}
			}
		}

		if (matched)
		{
			return iter->newName;
		}
	}

	//OK, I need to allocate a name
	string newName = funcName;
	for (unsigned ii=0; ii<PIs.size(); ii++)	
	{
		char buf[64];
		if (PIs[ii].isPointerAccess)
		{
			snprintf(buf, 64, "p%u", PIs[ii].i);
		}
		else
		{
			snprintf(buf, 64, "g%u", PIs[ii].i);
		}

		newName = newName + "_" + buf;
	}

	for (unsigned ii=0; ii<EXTID.size(); ii++)
	{
		bool found = false;
		for (unsigned ip=0; ip<PIs.size(); ip++)
		{
			if (PIs[ip].i == EXTID[ii].id)
			{
				found = true;
				break;
			}
		}	

		if (!found)
		{
			char buf[64];
			snprintf(buf, 64, "e%u", EXTID[ii].id);
			newName = newName + "_" + buf;
		}
	}

	if (cArg->hasGlobalMemThreadPrivate)
	{
		newName = newName + "_gtp";
	}

	RenamedFuncInfo ri(funcName, PIs, EXTID, newName, cArg->hasGlobalMemThreadPrivate);

	//FIXE ME THIS MAYBE WRONG
	bool found = false;
	for (unsigned i=0; i<rnFuncs.size(); i++)
	{
		if (rnFuncs[i].newName == newName)
		{
			found = true;
			break;
		}	
	}

	if (!found)
	{
		rnFuncs.push_back(ri);
	}

	return newName;
}



void OpenCLGlobalInfoContainer::addLocalMemVar(OCLLocalVar& v)
{
	oclLocalVars.push_back(v);
}

vector<OCLLocalVar>& OpenCLGlobalInfoContainer::getLocalMemVars()
{
	return oclLocalVars;
}


