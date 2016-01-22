#include "clang/Omp2Ocl/OpenCLCollectCalledFuncs.h"
#include "clang/AST/DeclPrinter.h"
#include "clang/AST/StmtPrinter.h"
#include "clang/Omp2Ocl/OpenCLGlobalInfoContainer.h"
#include "clang/Omp2Ocl/OpenCLRoutine.h"
#include "clang/Omp2Ocl/OpenCLCommon.h"
#include "clang/Omp2Ocl/OpenCLCompilerOptions.h"

/*
 * This function visits the functions that need to be generated for the 
 * OpenCL kernel. It then collects the callee functions used by the funcation.
 *
 * The "reqFuncs" vector stores those functions that are required by the ocl kernels
 *
 */
void OpenCLCollectCalledFuncs::collectCandidateFuncs()
{
	unsigned int ss;
	ASTContext& Context = getContext();
	vector<FunctionDecl*>& calleeFuncs = OpenCLGlobalInfoContainer::getCalleeFuncs();
	vector<FunctionDecl*>& functionDefs = OpenCLGlobalInfoContainer::getRecordFuncs();

	DeclPrinter dp(llvm::nulls(), Context, Context.PrintingPolicy, 4);
	StmtPrinter::isCollectedCallee = true;

	do
	{
		ss = calleeFuncs.size();
	
		for (vector<FunctionDecl*>::iterator iter = calleeFuncs.begin(); iter != calleeFuncs.end(); iter++)
		{
			string func_name = (*iter)->getNameInfo().getAsString();
			//Find the definiton of the function and then travere it
			for (vector<FunctionDecl*>::iterator itt=functionDefs.begin(); itt!=functionDefs.end(); itt++)
			{
				if ((*itt)->getNameInfo().getAsString() == func_name)
				{
					dp.VisitFunctionDecl(*itt);
				}
			}

			if (ss != calleeFuncs.size())
			{
				break;
			}

		}	

	}while(ss != calleeFuncs.size()); /**Do it until no new function has been added*/

	StmtPrinter::isCollectedCallee = false;
}


//FIXME: I should replace FuncProtoExt with globalIndexVar!! They are identical
void OpenCLCollectCalledFuncs::reviseFunctionWithOpenCLNDRangeVar(FunctionDecl* D, vector<DeclRefExpr*> globalVariables)
{
	unsigned int numParam = D->getNumParams();
	vector<QualType> ParamsTypes;

	vector<ParmVarDecl*> Params;
	for (unsigned i = 0; i < numParam; ++i) {
		ParmVarDecl* parm = D->getParamDecl(i);
		Params.push_back(parm);
		ParamsTypes.push_back(parm->getType());
	}

	/*This is the global variables, I need to replace array with buffer*/
	for (unsigned i=0; i<globalVariables.size(); i++)
	{
		ValueDecl* decl = globalVariables[i]->getDecl();
		string st = getCononicalType(decl);
		unsigned int dim = getArrayDimension(st);
		bool isLocalVar = false;

		//This buffer is treated as a local variable	
		if (OCLCompilerOptionAction::isLocalVar(decl))
		{
			isLocalVar = true;
		}
		
		string name = globalVariables[i]->getNameInfo().getAsString();

		ParmVarDecl* parm = 
			ParmVarDecl::Create(D->getASTContext(),
					D, 
					decl->getLocStart(), decl->getLocEnd(), decl->getIdentifier(),
					decl->getType(), 0, SC_None, SC_None, 0);

		//This argument should be printed with __global keyword in
		//the OpenCL kernel
		if (dim &&  !OCLCommon::isAThreadPrivateVariable(name))
		{
			parm->set2GlobalBuf();
		}

		parm->setOCLExtended();

		Params.push_back(parm);
		ParamsTypes.push_back(decl->getType());

		D->addAOpenCLNDRangeVar(FuncProtoExt(globalVariables[i], i+numParam, true, false, isLocalVar));
	}

	//Revise the function prototype
	FunctionProtoType::ExtProtoInfo fpi;
	fpi.Variadic = D->isVariadic();

	QualType newFT = D->getASTContext().getFunctionType(D->getResultType(), ParamsTypes.data(), ParamsTypes.size(), fpi);
	D->setType(newFT);
	D->setParams(Params.data(), Params.size()); 
}

/*
 * This function will holist the global variables to the function arguments
 * Return true if a holis action was happened
 */
bool OpenCLCollectCalledFuncs::holistOpenCLNDRangeVarInFunction(FunctionDecl* D)
{
	string buf;
	llvm::raw_string_ostream O(buf);

	vector<ParmVarDecl*> FuncParams;

	//Make sure that function arugments are not treated as global variables
	for (unsigned i = 0; i< D->getNumParams();  ++i) {
		FuncParams.push_back(D->getParamDecl(i));
	}

	if (D->getBody())
	{
		OpenCLNDRangeVarPicker gp(O, getContext(), NULL, getContext().PrintingPolicy, 4, FuncParams);
		gp.Visit(D->getBody());

		//Damn, I need to revise the function prototype and definition!!!
		if (gp.getOpenCLNDRangeVars().size() > 0)
		{
			reviseFunctionWithOpenCLNDRangeVar(D, gp.getOpenCLNDRangeVars());
			return true;
		}
	}
	return false;
}

vector<FunctionDecl*> OpenCLCollectCalledFuncs::findExpendedFunc(vector<FunctionDecl*>& candidateFuncs)
{
	vector<FunctionDecl*> funcNeed2Revised;

	//holistic the global variables. This may revise the function prototype and definition
	for (vector<FunctionDecl*>::iterator iter = candidateFuncs.begin(); iter != candidateFuncs.end(); iter++)
	{
		if (holistOpenCLNDRangeVarInFunction(*iter))
		{
			//Recored the revised functions,
			//because any call arguments to these functions 
			//needed to be revised later
			funcNeed2Revised.push_back(*iter);

			(*iter)->setBodyHasGlobalVar();
		}
	}

	return funcNeed2Revised;
}


/*
 * Visit the callarg to find out function arguments that refer as
 * a pointer to the global memory
 *
 */
bool OpenCLCollectCalledFuncs::findCall2GlobalBuffer(Stmt* Body, vector<OCLGlobalMemVar>& globalMemoryVariables, 
		vector<OMPThreadPrivateObject>& threadPrivates, vector<FunctionDecl*>& funcsNeed2Revised)
{
	bool need2Revised = false;

	GlobalCallArgPicker gp(llvm::nulls(), getContext(), NULL, getContext().PrintingPolicy, globalMemoryVariables, threadPrivates);
	gp.PrintStmt(Body);

	vector<CallArgInfoContainer*>& cf = gp.getCalledFuncs();

	for (vector<CallArgInfoContainer*>::iterator iter=cf.begin(); iter != cf.end(); iter++)
	{
		if ((*iter)->gCallArgs.size())
		{
			CallExpr *ca = (*iter)->ce;
			string newFuncName = OpenCLGlobalInfoContainer::getAName4RenamedFunc(getContext(), (*iter), funcsNeed2Revised);
			ca->setRenameInfo((*iter), newFuncName, (*iter)->hasGlobalMemThreadPrivate);

			need2Revised = true;
		}
	}

	return need2Revised;
}

FunctionDecl* OpenCLCollectCalledFuncs::PickFuncDeclByName(vector<FunctionDecl*>& candidateFuncs, string name)
{
	for (vector<FunctionDecl*>::iterator iter = candidateFuncs.begin(); iter != candidateFuncs.end(); iter++)
	{
		if ((*iter)->getNameInfo().getAsString() == name)
			return (*iter);	
	}

	cerr << "Could find the definition of function: " << name << endl;
	exit(-1);

	return NULL;
}

void OpenCLCollectCalledFuncs::scanRenameFuncs()
{
	vector<FunctionDecl*>& calleeFuncs = OpenCLGlobalInfoContainer::getCalleeFuncs();
	vector<FunctionDecl*> revisedFuncs = findExpendedFunc(calleeFuncs);
	for (unsigned i=0; i<revisedFuncs.size(); i++)
	{
		addRevisedFunc(revisedFuncs[i]);
	}

	//rescan the loops
	_do();

	//First time scan to make sure I have collected the functions 
	vector<OpenCLKernelLoop*>& ocls = getOclLoops();
	for (unsigned i=0; i<ocls.size(); i++)
	{
		OpenCLKernelLoop* l = ocls[i];	
		ForStmt* for_stmt = l->getForStmt();

		findCall2GlobalBuffer(for_stmt->getBody(), l->getGlobalMemVars(),
					OpenCLGlobalInfoContainer::getThreadPrivateVars(), revisedFuncs);
	}

	//DO the rename function
	vector<RenamedFuncInfo>& rnFuncs = OpenCLGlobalInfoContainer::getRenameFuncs();
	unsigned rsize;
	do
	{
		rsize = rnFuncs.size();

		vector<RenamedFuncInfo>& RFs = rnFuncs;
		for (unsigned i=0; i<RFs.size(); i++)
		{
			RenamedFuncInfo& RI = RFs[i];
			vector<globalVarIndex>& globalArugIds = RI.globalArugIds;
			FunctionDecl* D = PickFuncDeclByName(calleeFuncs, RI.origFuncName);
			
			vector<OCLGlobalMemVar> gvs;
			
			for (unsigned j=0; j<globalArugIds.size(); j++)
			{
				unsigned index = globalArugIds[j].i;
				ParmVarDecl* parm = D->getParamDecl(index);
				bool isFL = parm->isDefinedOutsideFunctionOrMethod();
				bool isTP = (globalArugIds[j].isGTP);
			
				if (OCLCompilerOptionAction::isLocalVar(parm))
				{
					globalArugIds[j].set2LocalBuf();
					globalArugIds[j].isPointerAccess = false;
				}
				else
				{
					OCLGlobalMemVar oc(parm, globalArugIds[j].isGTP, isFL, isTP);
					gvs.push_back(oc);
				}
			}

			findCall2GlobalBuffer(D->getBody(), gvs, 
					OpenCLGlobalInfoContainer::getThreadPrivateVars(), revisedFuncs);
		}
	
	}while(rsize != rnFuncs.size());
}

void OpenCLCollectCalledFuncs::scan(OpenCLKernelLoop* loop)
{
	ASTContext& Context = getContext();
	StmtPrinter::isCollectedCallee = true;
	StmtPrinter tp(llvm::nulls(), Context, NULL, Context.PrintingPolicy, 4);

	//Visit the kernel to record the function calls
	tp.Visit(loop->getKernel());

	StmtPrinter::isCollectedCallee = false;
}

void OpenCLCollectCalledFuncs::collectCallees()
{
	vector<OpenCLKernelLoop*>& ocls = getOclLoops();

	for (unsigned i=0; i<ocls.size(); i++)
	{
		OpenCLKernelLoop* l = ocls[i];
		scan(l);
	}
	
	collectCandidateFuncs();
}

void OpenCLCollectCalledFuncs::doIt()
{
	collectCallees();	
	scanRenameFuncs();
}
