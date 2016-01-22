#include "clang/Omp2Ocl/OpenCLKernelSchedule.h"
#include "clang/Basic/SourceManager.h"
#include "clang/AST/StmtPrinter.h"
#include "clang/Omp2Ocl/OpenCLRoutine.h"
#include "clang/Omp2Ocl/OpenCLCommon.h"
#include "clang/Omp2Ocl/OpenCLCopyInRoutine.h"
#include "clang/Omp2Ocl/OpenCLLoopInterChange.h"
#include "clang/Omp2Ocl/OpenCLKernelName.h"
#include "clang/Omp2Ocl/OpenCLReductionKernelGenerator.h"
#include "clang/Omp2Ocl/OpenCLKernelCodeGenerator.h"

vector<OpenCLKernelLoop*> OpenCLKernelSchedule::openclLoops;
vector<DeclRefExpr*> OpenCLKernelSchedule::collectedFunctions;
vector<OMPThreadPrivateObject> OpenCLKernelLoop::threadPrivates;
vector<string> OpenCLKernelSchedule::oclKernelNames;

//  Stmt printing methods.
//===----------------------------------------------------------------------===//
//OpenCL related methods
void OpenCLKernelSchedule::recordArraySub(DeclRefExpr* e)
{
	if (getCurLoop()->arrayV)
	{
		getCurLoop()->arrayV->addElement(e);
	}
}

void OpenCLKernelSchedule::setArrayBase(DeclRefExpr* base)
{
	if (getCurLoop()->arrayV)
	{
		getCurLoop()->arrayV->setBase(base);
	}
}

DeclRefExpr* OpenCLKernelSchedule::getArrayBase()
{
	if (getCurLoop()->arrayV)
	{
		return getCurLoop()->arrayV->getBase();
	}

	return NULL;
}

void OpenCLKernelSchedule::enableTrackingArraySubExpr() { 
	isTArraySubExpr = true; 
	isMetTheArrayDecl = false;
}


void OpenCLKernelSchedule::metArrayDecl(DeclRefExpr* expr) { 
	isMetTheArrayDecl = true; 
	setArrayBase(expr);
}


void OpenCLKernelSchedule::newOpenCLCurrentLoop(ForStmt* forNode, const FunctionDecl* func) { 
	this->curLoop = new OpenCLKernelLoop(forNode); 
	this->curLoop->func = const_cast<FunctionDecl*>(func);
}

void OpenCLKernelSchedule::setOpenCLOut(llvm::raw_fd_ostream* opencl)
{
	this->fOpenCL = opencl;
}

string OpenCLKernelSchedule::OpenCLIndent(unsigned int level)
{
	string str = "	";
	for (unsigned int i = 0; i<level; i++)
		str = str + "	";

	return str;
}

void OpenCLKernelSchedule::disableTrackingArraySubExpr() { 
	isTArraySubExpr = false; 
	isMetTheArrayDecl=false;
}

bool OpenCLKernelSchedule::isExpVariableAlreadyInParameterList(string name, vector<string>& ParamList)
{
	for (vector<string>::iterator iter = ParamList.begin(); iter != ParamList.end(); iter++)
	{
		if (name == (*iter))
			return true;
	}

	return false;
}


//This function collect arguments that should be passed into the OpenCL Kernels as global memory buffe
void OpenCLKernelSchedule::CollectGlobalInputParameters(OpenCLKernelLoop* curLoop)
{
	std::string name;
	vector<string> ParamList;

	for (vector<PLoopParam>::iterator iter = curLoop->params.begin(); iter != curLoop->params.end(); iter++)
	{
		name = iter->declRef->getNameInfo().getAsString();

		std::string type = getCononicalType(iter->declRef->getDecl());

		//make sure this is not a private variable
		if ((curLoop->for_stmt->getOMPFor().isVariablePrivate(name)) == true)
		{
			continue;
		}

		//This variable is declared inside the loop
		if (curLoop->isInnerDecl(iter->declRef))
		{
			continue;
		}

		if (curLoop->isAOpenCLNDRangeVar(name))
			continue;

		bool tp = OCLCommon::isAGlobalMemThreadPrivateVar(name);
		//Check if is an array? FIXME: THIS IS STUPID
		if (type.find('[') != string::npos)
		{
			ValueDecl* d = iter->declRef->getDecl();
			bool isFL = d->isDefinedOutsideFunctionOrMethod();
			//Only threadprivate variables decleared in the __global memory will be treated as global memory
			if (OCLCommon::isAThreadPrivateVariable(name))
			{
				curLoop->addGlobalMemoryVar(OCLGlobalMemVar(d, tp, isFL, true));
			}
			else
			{
				curLoop->addGlobalMemoryVar(OCLGlobalMemVar(d, tp, isFL, false));
			}
		}
	}
}

//Scan thread privated variables of the loop body
void OpenCLKernelSchedule::ScanThreadPrivate(OpenCLKernelLoop* loop)
{	
	string string_buf;
	llvm::raw_string_ostream Out(string_buf);
	//dr visits the kernel to record the DeclRef information of variables
	StmtPicker dr (Out, Context, NULL, Context.PrintingPolicy, 0);
	dr.Visit(loop->Kernel);
	vector<DeclRefExpr*>& decls = dr.getDecl();

	for (vector<DeclRefExpr*>::iterator iter = decls.begin(); iter != decls.end(); iter++)
	{
		string name = (*iter)->getNameInfo().getAsString();
		ValueDecl* decl = (*iter)->getDecl();
		VarDecl* varDecl = dyn_cast<VarDecl>(decl);

		//I have met a threadprivate name that has the same name and it is 
		//not declared locally
		if (otp.isAThreadPrivateVariable(name))
		{
			bool pri=true;
			if (varDecl)
			{
				if (varDecl->isLocalVarDecl())
				{
					pri=false;
				}
			}
			if (pri)
			{
				//Do worry, the addPrivateVariable will check whether a variable
				//with the same name has already existed or not!
				OMPThreadPrivateObject obj(name, otp.getLoc(name), otp.isAGlobalMemThreadPrivateVar(name));
				//loop->addThreadPrivateVariable(obj);
				cerr << "This execution path is obselated\n" << endl;
			}
		}
	}
}

void OpenCLKernelSchedule::initArraySubRecord(ArraySubscriptExpr* Node)
{
	if (getCurLoop()->arrayV)
	{
		if (getCurLoop()->arrayV->getBase() != NULL)
		{
			getCurLoop()->pushCurrentArraySubV();
		}
		getCurLoop()->newArraySubVariable();
	}
	else
	{
		getCurLoop()->newArraySubVariable();
	}

}

void OpenCLKernelSchedule::forLoopEndRoutine()
{
	if (getCurLoop()->arrayV)
	{
		if (getCurLoop()->arrayV->getBase() != NULL)
		{
			getCurLoop()->pushCurrentArraySubV();
		}
	}
}



/// ------------------------------------------------------------------------------
//
//	OpenCL Related Routines Starts
//
//  ------------------------------------------------------------------------------

//Zheng: I should implement the OpenCL dump here.
/*
 * Check whether a loop is a perfect nest loop
 * A perfect nest looks like:
 * 	for (...)
 * 		for (...)
 * 			for (...)
 *
 */
bool OpenCLKernelSchedule::isPerfectNestLoop(ForStmt* forLoop)
{
	Stmt* stmt = forLoop->getBody(); 

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

/*
 * Check whether a variable is defined as a global variable in the kernel
 *
 */
bool OpenCLKernelSchedule::isAnOpenCLOpenCLNDRangeVar(OpenCLKernelLoop* curLoop, string& name)
{
	vector<OpenCLNDRangeVar> GV = curLoop->getOclLoopIndexs();
	for (vector<OpenCLNDRangeVar>::iterator iter = GV.begin(); iter != GV.end(); iter++)
	{
		if (iter->variable == name)
			return true;
	}

	return false;
}

/*
 * This function performs:
 * a) loop interchange, if possible
 * b) recording the loop indexs
 * c) This function returns the opencl kernel
 */
Stmt* OpenCLKernelSchedule::ScheduleLoops(OpenCLKernelLoop* curLoop, vector<SwapLoopInfo>& innerLoops, bool& swaped)
{
	return NULL;
}

static string declareAPrivateVariable(DeclRefExpr* declRef)
{
	std::string type = getCononicalType(declRef->getDecl());
	string name = declRef->getNameInfo().getAsString();

	//array
	if (type.find('[') != string::npos)
	{
		string t;
		unsigned int i;
		for (i = 0; i<type.length(); i++)
		{
			if (type[i] == ' ')
			{
				i++;
				break;
			}
			t = t + type[i];
		}

		for (; i<type.length(); i++)
		{
			name = name + type[i];
		}

		type = t;
	}

	type = type + " " + name + ";";

	return type;
}

/*
 * This function is used to generate private 
 *
 */
void OpenCLKernelSchedule::GeneratePrivateVariables(OpenCLKernelLoop* loop)
{
	cerr << "This function is obselated.\n" << endl;
}

/*!
 * This function retuns a name of a function whose argument are
 * passed as pointer to global memory and offsets
 */
string OpenCLKernelSchedule::getReNameFuncName(CallArgInfoContainer* cArg, vector<FunctionDecl*>& funcsNeed2Revised)
{
	string newName;
	return newName;
}

/*
 * Visit the callarg to find out function arguments that refer as
 * a pointer to the global memory
 *
 */
bool OpenCLKernelSchedule::findCall2GlobalBuffer(Stmt* Body, vector<OCLGlobalMemVar>& globalMemoryVariables, 
		vector<OMPThreadPrivateObject>& threadPrivates, vector<FunctionDecl*>& funcsNeed2Revised)
{
	string v_buf;
	llvm::raw_string_ostream os_v(v_buf);
	bool need2Revised = false;

	GlobalCallArgPicker gp(os_v, Context, NULL, Context.PrintingPolicy, globalMemoryVariables, threadPrivates);
	gp.PrintStmt(Body);

	vector<CallArgInfoContainer*>& cf = gp.getCalledFuncs();

	for (vector<CallArgInfoContainer*>::iterator iter=cf.begin(); iter != cf.end(); iter++)
	{
		if ((*iter)->gCallArgs.size())
		{
			CallExpr *ca = (*iter)->ce;
			string newFuncName = getReNameFuncName((*iter), funcsNeed2Revised);
			ca->setRenameInfo((*iter), newFuncName, (*iter)->hasGlobalMemThreadPrivate);

			need2Revised = true;
		}
	}

	return need2Revised;

}


static void addLoopParams(vector<DeclRefExpr*>& decls, OpenCLKernelLoop* loop)
{
	for (unsigned i=0; i<decls.size(); i++)
	{
		loop->addParam(decls[i]);
	}
}

TypedefDecl* OpenCLKernelSchedule::getTypeDefRef(string ty)
{

	for (unsigned i=0; i<typeDefs.size(); i++)
	{
		TypedefDecl* D = typeDefs[i];
		if (D->getUnderlyingType().getAsString() == ty)
			return D;
	}

	return NULL;
}

//Scanning to find any DeclRefExprs whose type are non-primitive
void OpenCLKernelSchedule::ScanNonPrimitiveType(OpenCLKernelLoop* loop, vector<DeclRefExpr*>& decls)
{
	for (unsigned i=0; i<decls.size(); i++)
	{
		DeclRefExpr* d = decls[i];
		const QualType type = d->getType().getCanonicalType();
		string ty = getGlobalType(type.getAsString());

		//skip functions
		if (d->getDecl()->getKind() == Decl::Function)
			continue;

		if (!isOCLPremitiveType(ty))
		{
			bool found=false;
			for (unsigned j=0; j<qtypes.size(); j++)
			{
				if (getGlobalType(qtypes[j].getAsString()) == ty)
				{
					found = true;
					break;
				}
			}

			if (!found)
			{
				this->qtypes.push_back(type);
				RecordDecl* rd = OCLCommon::getRecordDecl(type);
				TypedefDecl* TD = getTypeDefRef(ty);

				if (rd)
				{
					this->usedRDs.push_back(rd);
				}

				if (TD)
				{
					this->usedDefs.push_back(TD);
				}
			}	
		}	
	}	

}


/*!
 * This function pre-scan the loops and performs loop swaption and call argument revision
 *
 */
void OpenCLKernelSchedule::scanLoop(OpenCLKernelLoop* loop, vector<FunctionDecl*>& funcsNeed2Revised)
{
	bool swaped = false;
	vector<SwapLoopInfo> innerLoops;

	Stmt* Kernel = ScheduleLoops(loop, innerLoops, swaped);
	loop->Kernel = Kernel;
	loop->for_stmt->setOpenCLKernel(Kernel);
	loop->innerLoops = innerLoops;


	//Scan thread private variables
	//FIXME: THIS MAYBE A BUG!!!!
	ScanThreadPrivate(loop);

	reviseCalledArgs(funcsNeed2Revised, Kernel, loop);
	CollectGlobalInputParameters(loop);

	findCall2GlobalBuffer(loop->for_stmt->getBody(), loop->globalMemoryVariables, loop->threadPrivates, funcsNeed2Revised);

	//FIXME: I should remove the scan process in StmtPrinter	
	//Scan the declaration of the loop kernel
	string string_buf;
	llvm::raw_string_ostream Out(string_buf);
	//dr visits the kernel to record the DeclRef information of variables
	StmtPicker dr (Out, Context, NULL, Context.PrintingPolicy, 0);
	dr.Visit(Kernel);
	Out.flush();
	//Record read/write set. FIXME: ALL THE DECLREF in getLHS() will be treated as write
	loop->for_stmt->setRWS(dr.getRWS());
	loop->setRWSet(dr.getRWS());

	//Scan non-primitive types of structures
	ScanNonPrimitiveType(loop, dr.getDecl());

	vector<OpenCLNDRangeVar>& gvs = loop->getOclLoopIndexs();
	for (unsigned i=0; i<gvs.size(); i++)
	{
		dr.Visit(gvs[i].Init);
	}

	for (unsigned i=0; i<innerLoops.size(); i++)
	{
		dr.VisitForStmtHeader(innerLoops[i].for_stmt);
	}

	vector<DeclRefExpr*>& decls = dr.getDecl();
	addLoopParams(decls, loop);

	loop->swaped = swaped;
}

// Actual kernel generation routine
//
//
//
void OpenCLKernelSchedule::_generateKernel(OpenCLKernelLoop* loop, vector<FunctionDecl*>& funcsNeed2Revised)
{
	llvm::raw_ostream& Out = (*fOpenCL);
	bool bReduction = loop->isReductionLoop();

	if (bReduction)
	{	
		OpenCLReductionKernelGenerator g(Out, Context, loop, funcsNeed2Revised, pDMF);
		g.doIt();
	}
	else
	{
		OpenCLKernelCodeGenerator g(Out, Context, loop, pDMF);
		g.doIt();
	}
}

static string genCalcMarco(unsigned dim)
{
	string str = "#define CALC_" + uint2String(dim) + "D_INX(";

	for (unsigned i=1; i<=dim; i++)
	{
		if (i > 1)
			str = str + ",";
		str = str + "M" + uint2String(i);
	}

	for (unsigned i=1; i<=dim; i++)
	{
		str = str + ",m" + uint2String(i);
	}

	str = str + ") (";

	for (unsigned i=1; i<=dim; i++)
	{
		if (i > 1)
			str = str + "+";

		str = str + "((m" + uint2String(i) + ")";
		for (unsigned j=i+1; j<=dim; j++)
		{
			str = str + "*(M" + uint2String(j) + ")";
		}

		str = str + ")";
	}

	str = str + ")\n";

	return str;
}

string OpenCLKernelSchedule::_generateCommandRoutine()
{
	string routine;

	routine = routine + "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
	//routine = routine + "#pragma OPENCL EXTENSION cl_amd_printf : enable\n\n";
	routine = routine + "//These are some common routines\n";
	for (unsigned i=2; i<=8; i++)
	{
		routine = routine + genCalcMarco(i);
	}

	char buf[64];
	snprintf(buf, 64, "%u", DEFAULT_GROUP_SIZE);

	routine = routine + "#define GROUP_SIZE	";
	routine = routine + buf;
	routine = routine + "\n";
	//routine = routine + "typedef unsigned size_t;\n";
	routine = routine + "\n";

	//Print out non-primitive data structure
	for (unsigned i=0; i<usedRDs.size(); i++)
	{
		RecordDecl* rd = usedRDs[i];

		string string_buf;
		llvm::raw_string_ostream Out(string_buf);
		DeclPrinter dp(Out, Context, Context.PrintingPolicy, 0);
		dp.VisitRecordDecl(rd);
		Out.flush();

		string_buf = string_buf + ";\n";

		routine = routine + string_buf;
	}

	routine = routine + "\n";

	for (unsigned i=0; i<usedDefs.size(); i++)
	{
		TypedefDecl* D = usedDefs[i];

		string S = D->getNameAsString();
		S = "typedef " + D->getUnderlyingType().getAsString() + " " + S;

		routine = routine + S + ";\n";
	}

	routine = routine + "\n";

	return routine;
}

/**
 * This will be call by StmtPrinter when collecting callee functions
 * See VisitDeclRefExpr in StmtPrinter
 */
void OpenCLKernelSchedule::addCollectedFunction(DeclRefExpr* expr)
{
	string ref_name = expr->getNameInfo().getAsString();
	ValueDecl* decl = expr->getDecl();
	//Functions will add to another category
	if (decl->getKind() == Decl::Function)
	{
		for (vector<DeclRefExpr*>::iterator iter = collectedFunctions.begin(); iter != collectedFunctions.end(); iter++)
		{
			if ((*iter)->getNameInfo().getAsString() == ref_name)
			{
				return;
			}	
		}

		collectedFunctions.push_back(expr);
	}
}

//
// This prints out the function declaration
//
//
void OpenCLKernelSchedule::generateFuncPrototype(llvm::raw_ostream& Out, FunctionDecl* D)
{
	QualType BackType = D->getType();
	vector<ParmVarDecl*> BackParams;


	//Func proto has changed. I will backup and restore it back again
	if (D->hasFuncProtoChanged())
	{
		unsigned numParam = D->getNumParams();

		for (unsigned i = 0; i < numParam; ++i) {
			ParmVarDecl* parm = D->getParamDecl(i);
			BackParams.push_back(parm);
		}

		D->RestoreOCLRevisedParams();
	}

	switch (D->getStorageClass()) {
		case SC_None: break;
		case SC_Extern: Out << "extern "; break;
		case SC_Static: Out << "static "; break;
		case SC_PrivateExtern:
		case SC_Auto:
		case SC_Register:
				break;
	}

	if (D->isInlineSpecified())  Out << "inline ";

	std::string Proto = D->getNameInfo().getAsString();

	QualType Ty = D->getResultType();

	Out << Ty.getAsString() << " ";
	Out << Proto << " (";

	DeclPrinter ParamPrinter(Out, Context, Context.PrintingPolicy, 0, true);
	for (unsigned i = 0, e = D->getNumParams(); i != e; ++i) {
		if (i > 0)
		{
			Out << ", ";
		}

		ParamPrinter.VisitParmVarDecl(D->getParamDecl(i));
	}

	Out << ")";

	//Roll back to the modified version
	if (D->hasFuncProtoChanged())
	{
		D->setType(BackType);
		D->setParams(BackParams.data(), BackParams.size());
	}
}

/*
 * This function visits the functions that need to be generated for the 
 * OpenCL kernel. It then collects the callee functions used by the funcation.
 *
 */
void OpenCLKernelSchedule::collectCallees(vector<DeclRefExpr*>& functions, vector<FunctionDecl*>& functionDefs)
{
	unsigned int ss = functions.size();
	string v_buf;
	llvm::raw_string_ostream os_v(v_buf);
	DeclPrinter dp(os_v, Context, Context.PrintingPolicy, 4);
	StmtPrinter::isCollectedCallee = true;

	do
	{
		for (vector<DeclRefExpr*>::iterator iter = functions.begin(); iter != functions.end(); iter++)
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

		}	


		for (vector<DeclRefExpr*>::iterator iter = collectedFunctions.begin(); iter != collectedFunctions.end(); iter++)
		{
			string cfn = (*iter)->getNameInfo().getAsString();

			bool found = false;
			for (vector<DeclRefExpr*>::iterator itt = functions.begin(); itt != functions.end(); itt++)
			{
				string name = (*itt)->getNameInfo().getAsString();
				if (name == cfn)
				{
					found = true;
					break;
				}
			}
			if (!found)
			{
				functions.push_back(*iter);
				ss++;
			}
		}

	}while(ss != functions.size());

	StmtPrinter::isCollectedCallee = false;
}


/*
 * Revise the function arguments once the function accesses to global variables
 *
 */
void OpenCLKernelSchedule::reviseFunctionWithOpenCLNDRangeVar(FunctionDecl* D, vector<DeclRefExpr*> globalVariables)
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

		string name = globalVariables[i]->getNameInfo().getAsString();

		ParmVarDecl* parm = 
			ParmVarDecl::Create(D->getASTContext(),
					D, 
					decl->getLocStart(), decl->getLocEnd(), decl->getIdentifier(),
					decl->getType(), 0, SC_None, SC_None, 0);

		//This argument should be printed with __global keyword in
		//the OpenCL kernel
		if (dim &&  !otp.isAThreadPrivateVariable(name))
		{
			parm->set2GlobalBuf();
		}

		parm->setOCLExtended();

		Params.push_back(parm);
		ParamsTypes.push_back(decl->getType());

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
bool OpenCLKernelSchedule::holistOpenCLNDRangeVarInFunction(FunctionDecl* D)
{
	string buf;
	llvm::raw_string_ostream O(buf);

	vector<ParmVarDecl*> FuncParams;

	//Make sure that function arugments are not treated as global variables
	for (unsigned i = 0; i< D->getNumParams();  ++i) {
		FuncParams.push_back(D->getParamDecl(i));
	}

	OpenCLNDRangeVarPicker gp(O, Context, NULL, Context.PrintingPolicy, 4, FuncParams);
	gp.Visit(D->getBody());

	//Damn, I need to revise the function prototype and definition!!!
	if (gp.getOpenCLNDRangeVars().size() > 0)
	{
		reviseFunctionWithOpenCLNDRangeVar(D, gp.getOpenCLNDRangeVars());
		return true;
	}

	return false;
}

/**
 * This function recursively visits all the code to replace
 * the call args to a function whose prototype has changed.
 *
 */
void OpenCLKernelSchedule::reviseCalledArgs(vector<FunctionDecl*>& functionDefs, Stmt* E, OpenCLKernelLoop* loop)
{
	string v;
	llvm::raw_string_ostream buf(v);

	ForStmt *topF = loop->for_stmt;

	CallArgReviseAction ca(buf, Context, NULL, Context.PrintingPolicy, 0, functionDefs);
	ca.Visit(E);
	vector<DeclRefExpr*>& expV = ca.getExpVariables();

	/*
	 * Recorded the expended variables (if any) so that they will be added into 
	 * the OpenCL Kernel arguments.
	 *
	 */
	for (vector<DeclRefExpr*>::iterator iter = expV.begin(); iter != expV.end(); iter++)
	{
		string type = getCononicalType((*iter)->getDecl());

		string passInName = (*iter)->getNameInfo().getAsString();
		string localName = passInName;

		unsigned dim = getArrayDimension(type);
		//This maybe a global memory variable
		if (dim)
		{
			if (!topF->getOMPFor().isVariablePrivate(passInName))
			{
				loop->addParam((*iter));
			}
		}	

		ExpendedCallArg arg(passInName, localName, (*iter));
		topF->addedExpendedCallArg(arg);
	}	
}

/*!
 * Collect functions that will be used by the OpenCL kernels
 */
vector<FunctionDecl*> OpenCLKernelSchedule::collectCandidateFunc(vector<FunctionDecl*>& functionDefs)
{
	vector<FunctionDecl*> candidateFuncs;
	//Collect callees
	collectCallees(OpenCLKernelLoop::getFunctions(), functionDefs);

	for (vector<DeclRefExpr*>::iterator iter = OpenCLKernelLoop::getFunctions().begin(); iter != OpenCLKernelLoop::getFunctions().end(); iter++)
	{
		string func_name = (*iter)->getNameInfo().getAsString();
		/*Do not added functions when printting the code*/
		for (vector<FunctionDecl*>::iterator itt=functionDefs.begin(); itt!=functionDefs.end(); itt++)
		{
			if ((*itt)->getNameInfo().getAsString() == func_name)
			{
				candidateFuncs.push_back(*itt);
			}
		}
	}

	return candidateFuncs;
}

/*!
 *
 *
 */
vector<FunctionDecl*> OpenCLKernelSchedule::findExpendedFunc(vector<FunctionDecl*>& candidateFuncs)
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
		}
	}

	return funcNeed2Revised;
}



/*!
 *
 * Generating the declartion of a function that has been renamed.
 * The function is renamed is becasue a global memory object is passed as a pointer
 *
 */
string OpenCLKernelSchedule::genProtoType4RenamedFunc(FunctionDecl* D, RenamedFuncInfo& r)
{
	string output_buffer;
	llvm::raw_string_ostream Out (output_buffer);

	switch (D->getStorageClass()) {
		case SC_None: break;
		case SC_Extern: Out << "extern "; break;
		case SC_Static: Out << "static "; break;
		case SC_PrivateExtern:
		case SC_Auto:
		case SC_Register:
				break;
	}

	if (D->isInlineSpecified())  Out << "inline ";

	std::string Proto = r.newName;

	QualType Ty = D->getResultType();

	Out << Ty.getAsString() << " ";
	Out << Proto << " (";

	vector<globalVarIndex> gIds =  r.globalArugIds;

	DeclPrinter ParamPrinter(Out, Context, Context.PrintingPolicy, 0);
	for (unsigned i = 0, e = D->getNumParams(); i != e; ++i) {
		if (i > 0)
		{
			Out << ", ";
		}

		//Checking whether "
		bool found = false;
		vector<globalVarIndex>::iterator iter;
		for (iter = gIds.begin(); iter != gIds.end(); iter++)
		{
			if (iter->i == i)
			{
				found = true;
				break;
			}


		}

		ParmVarDecl* decl = D->getParamDecl(i);

		if (!found)
		{
			ParamPrinter.VisitParmVarDecl(decl);
		}
		else //This arguments should be renamed, simply replace the argument to "__global" will do 
			//the trick
		{
			decl->setOCLGlobal();
			if (iter->isPointerAccess)
			{
				decl->setOCLPointerAccess();
			}

			Out << "__global ";
			string type = decl->getType().getAsString();
			Out << getGlobalType(type) << "* ";
			Out << decl->getNameAsString();	
		}
	}

	//FIXME: Where are they come from?
	//From: reviseFunctionWithOpenCLNDRangeVar()
	vector<FuncProtoExt>& aGVs = D->getAddedOpenCLNDRangeVars();

	for (unsigned i=0; i<aGVs.size(); i++)
	{
		if (aGVs[i].hasRevised())
			continue;

		DeclRefExpr* expr = aGVs[i].expr;
		string type = getCononicalType(expr->getDecl());
		string name = expr->getNameInfo().getAsString();
		unsigned int dim = getArrayDimension(type);

		if (dim > 0)
		{
			string t = getGlobalType(type);
			if (D->getNumParams() > 0 || i > 0)
			{
				Out << ", __global " << t << " *" << name;
			}
			else
			{
				Out << " __global " << t << " *" << name;
			}
		}	
	}

	for (unsigned i=0; i<gIds.size(); i++)
	{
		if (gIds[i].isPointerAccess)
		{
			Out << ", unsigned arg_" << gIds[i].i << "_offset";
		}
	}

	if (r.hasGlobalMemThreadPrivate)
	{
		Out << ", unsigned int " << COPYIN_MULTI_FACTOR_NAME;
		Out << ", unsigned int " << COPYIN_ADD_OFFSET_NAME;
	}

	Out << ")";

	Out.flush();

	return output_buffer;
}

FunctionDecl* OpenCLKernelSchedule::PickFuncDeclByName(vector<FunctionDecl*>& candidateFuncs, string name)
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

//Print the prototype for Renamed func
void OpenCLKernelSchedule::generatePrototypeForRenamedFunc(llvm::raw_ostream& Out, vector<FunctionDecl*>& candidateFuncs, vector<RenamedFuncInfo>& rnFuncs)
{
	for (vector<RenamedFuncInfo>::iterator iter=rnFuncs.begin(); iter!=rnFuncs.end(); iter++)
	{
		FunctionDecl* decl = PickFuncDeclByName(candidateFuncs, iter->origFuncName);
		Out << 	genProtoType4RenamedFunc(decl, *iter) << ";\n";
	}
}

/*
 * This generates the definition for a renamed function
 *
 */
void OpenCLKernelSchedule::generateDef4RenamedFunc(llvm::raw_ostream& Out, vector<FunctionDecl*>& candidateFuncs, vector<RenamedFuncInfo>& rnFuncs)
{
	for (vector<RenamedFuncInfo>::iterator iter=rnFuncs.begin(); iter!=rnFuncs.end(); iter++)
	{
		FunctionDecl* D = PickFuncDeclByName(candidateFuncs, iter->origFuncName);
		Out << 	genProtoType4RenamedFunc(D, *iter);

		OpenCLPrinter tp(Out, Context, NULL, Context.PrintingPolicy, 4);
		tp.setGlobalMemoryVariables(D->getAddedOpenCLNDRangeVars());

		//This parameter should be declared as global memory objects as well
		vector<globalVarIndex> gIds = iter->globalArugIds;
		for (unsigned i=0; i<gIds.size(); i++)
		{
			if (gIds[i].i >= D->getNumParams())
			{
				cerr << "Warning: somthing is wrong when renaming function: " << D->getNameAsString() << " : " << gIds[i].i << " : " << D->getNumParams() << endl;
				continue;
			}

			ParmVarDecl* param = D->getParamDecl(gIds[i].i);

			if (gIds[i].isPointerAccess)
			{
				string offset_string = "arg_";
				char buf[64];
				snprintf(buf, 64, "%u", gIds[i].i);
				offset_string = offset_string + buf;
				offset_string = offset_string + "_offset";
				param->setOffsetString(offset_string);
			}

			bool isFLevel = false;
			if (param->isLocalVarDecl())
				isFLevel = true;
			if (param->isFunctionOrMethodVarDecl())
			       isFLevel = true;

			//FIXME:!!
			bool isTP = (gIds[i].isGTP) ? true: false;
			tp.addAGlobalMemoryVariables(OCLGlobalMemVar(param, gIds[i].isGTP, isFLevel, isTP));
		}

		tp.PrintStmt(D->getBody());

		Out << "\n";


		//FIXME: This is urgely!
		//Reset the offset string
		for (unsigned i=0; i<gIds.size(); i++)
		{
			if (gIds[i].i >= D->getNumParams())
				continue;

			ParmVarDecl* param = D->getParamDecl(gIds[i].i);
			string offset_string;
			param->setOffsetString(offset_string);
		}

		D->RestoreOCLRevisedParams();
	}
}

/*
 * Generating the functions that will be called by the OpenCL Kernels
 *
 *
 */
void OpenCLKernelSchedule::genrerateCalledFunctions(llvm::raw_ostream& Out, vector<FunctionDecl*>& candidateFuncs, vector<RenamedFuncInfo>& rnFuncs)
{
	DeclPrinter dp(Out, Context, Context.PrintingPolicy, 4);

	generatePrototypeForRenamedFunc(Out, candidateFuncs, rnFuncs);

	//traverses the functions that are needed to be generated
	for (vector<FunctionDecl*>::iterator iter = candidateFuncs.begin(); iter != candidateFuncs.end(); iter++)
	{
		//o not generate functions whose origial body
		//has access to global variables
		if ((*iter)->isBodyHasGlobalVar())
			continue;

		generateFuncPrototype(Out, *iter);
		Out << ";\n";
	}

	Out << "\n";

	for (vector<FunctionDecl*>::iterator iter = candidateFuncs.begin(); iter != candidateFuncs.end(); iter++)
	{
		if ((*iter)->isBodyHasGlobalVar())
			continue;

		generateFuncPrototype(Out, *iter);

		OpenCLPrinter tp(Out, Context, NULL, Context.PrintingPolicy, 0);
		tp.setGlobalMemoryVariables((*iter)->getAddedOpenCLNDRangeVars());
		tp.PrintStmt((*iter)->getBody());

		Out << "\n";
	}

	//Generate definitions of renamed functions
	generateDef4RenamedFunc(Out, candidateFuncs, rnFuncs);

	Out << "\n";
}

/*!
 * This function generates funcs prototype and definitions that will be used by
 * the OpenCL kernels
 * 
 * */
vector<FunctionDecl*> OpenCLKernelSchedule::generateFuncRoutines(string& string_buf)
{
	llvm::raw_string_ostream Out(string_buf);
	vector<FunctionDecl*> fRevised;

	//This run will collect functions used by the loop kernel
	for (vector<OpenCLKernelLoop*>::iterator iter = openclLoops.begin(); iter != openclLoops.end(); iter++)
	{
		this->curLoop = (*iter);
		scanLoop(*iter, fRevised);
	}

	vector<FunctionDecl*> candidateFuncs = collectCandidateFunc(DeclPrinter::functions);

	//First scan to see whether the boody accesses to global memory variables
	for (unsigned i=0; i<candidateFuncs.size(); i++)
	{
		FunctionDecl* D = candidateFuncs[i];

		string string_buf;
		llvm::raw_string_ostream Out(string_buf);
		//dr visits the kernel to record the DeclRef information of variables
		StmtPicker dr (Out, Context, NULL, Context.PrintingPolicy, 0);
		dr.Visit(D->getBody());
		vector<DeclRefExpr*>& decls = dr.getDecl();

		for (unsigned j=0; j<decls.size(); j++)
		{
			ValueDecl* d = decls[j]->getDecl();	
			if (d->isDefinedOutsideFunctionOrMethod())
			{
				D->setBodyHasGlobalVar();
				break;
			}
		}
	}

	//Find functions whose arguments have to be expended
	//Warning, this will also revise the Function Proto
	fRevised = findExpendedFunc(candidateFuncs);

	//This run re-collect global memory variables in cases some variables have been added as extended variables (because
	// a function may access a global variable)
	for (vector<OpenCLKernelLoop*>::iterator iter = openclLoops.begin(); iter != openclLoops.end(); iter++)
	{
		this->curLoop = (*iter);
		scanLoop(*iter, fRevised);
	}

	unsigned rsize;
	do
	{
		rsize = RenamedFuncs.size();
		//Once again scan the kernel of Revised Function, to see whether there are any call argument to global variable
		vector<RenamedFuncInfo> RFs = RenamedFuncs;
		for (unsigned i=0; i<RFs.size(); i++)
		{
			RenamedFuncInfo RI = RFs[i];
			vector<globalVarIndex> globalArugIds = RI.globalArugIds;
			FunctionDecl* D = PickFuncDeclByName(candidateFuncs, RI.origFuncName);
			
			vector<OCLGlobalMemVar> gvs;
			
			for (unsigned j=0; j<globalArugIds.size(); j++)
			{
				unsigned index = globalArugIds[j].i;
				ParmVarDecl* parm = D->getParamDecl(index);
				bool isFL = parm->isDefinedOutsideFunctionOrMethod();
				bool isTP = (globalArugIds[j].isGTP) ? true : false;

				OCLGlobalMemVar oc(parm, globalArugIds[j].isGTP, isFL, isTP);
				gvs.push_back(oc);
			}

			findCall2GlobalBuffer(D->getBody(), gvs, OpenCLKernelLoop::threadPrivates, fRevised);
		}


	}while(rsize != RenamedFuncs.size());

	Out << "//Functions that will be used by the kernels (START)\n\n";
	genrerateCalledFunctions(Out, candidateFuncs, RenamedFuncs);

	Out << "//Functions that will be used by the kernels (END)\n\n";

	Out.flush();

	return fRevised;
}

/**
 * Generate OpenCL Kernels
 *
 */
void OpenCLKernelSchedule::GenerateOpenCLLoopKernel()
{
	llvm::raw_ostream& Out = (*fOpenCL);
	string func_buf;

	vector<FunctionDecl*> fRevised = generateFuncRoutines(func_buf);
	string command = _generateCommandRoutine();

	Out << command << "\n" << func_buf;

	for (vector<OpenCLKernelLoop*>::iterator iter = openclLoops.begin(); iter != openclLoops.end(); iter++)
	{
		this->curLoop = (*iter);
		_generateKernel(*iter, fRevised);
	}

	//rollback revised functions
	for (unsigned i=0; i<fRevised.size(); i++)
	{
		FunctionDecl* D = fRevised[i];
		D->RestoreOCLRevisedParams();
	}

	pDMF->flush();
}


unsigned int OpenCLKernelSchedule::getLineNumber(SourceLocation Loc)
{
	SourceManager &SM = Context.getSourceManager();
	PresumedLoc PLoc = SM.getPresumedLoc(Loc);
	return PLoc.isValid()? PLoc.getLine() : 0;
}


const char* OpenCLKernelSchedule::getFileName(SourceLocation Loc)
{
	SourceManager &SM = Context.getSourceManager();
	PresumedLoc PLoc = SM.getPresumedLoc(Loc);
	return PLoc.getFilename();
}

/// ------------------------------------------------------------------------------
//
//	OpenCL Related Routines
//
//  ------------------------------------------------------------------------------


