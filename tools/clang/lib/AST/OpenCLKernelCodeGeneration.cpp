#include "clang/Omp2Ocl/OpenCLKernelCodeGeneration.h"
#include "clang/Omp2Ocl/OpenCLCommon.h"
#include "clang/Omp2Ocl/OpenCLRoutine.h"
#include "clang/Omp2Ocl/OpenCLGlobalInfoContainer.h"
#include "clang/AST/DeclPrinter.h"
#include "clang/Omp2Ocl/OpenCLPrinter.h"
#include "clang/Omp2Ocl/OpenCLReductionKernelGenerator.h"
#include "clang/Omp2Ocl/OpenCLKernelCodeGenerator.h"
#include "clang/Omp2Ocl/OpenCLLoopScan.h"
#include "clang/Omp2Ocl/OpenCLCollectCalledFuncs.h"
#include "clang/Omp2Ocl/OpenCLKernelCodeOpt.h"
#include "clang/Omp2Ocl/OpenCLCompilerOptions.h"

void OpenCLKernelCodeGeneration::genNVIDIAMacros()
{
	Out << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
}

void OpenCLKernelCodeGeneration::genAMDMacros()
{
	Out << "#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n";
}

void OpenCLKernelCodeGeneration::calculateThreadId()
{
	Out << "int calc_thread_id_1() {\n";
	Out << " 	return get_global_id(0);\n";
	Out << "}\n\n";
	
	Out << "int calc_thread_id_2() {\n";
	Out << " 	return (get_global_id(1) * get_global_size(0) + get_global_id(0));\n";
	Out << "}\n\n";
	
	Out << "int calc_thread_id_3() {\n";
	Out << " 	return (get_global_id(2) * (get_global_size(1) * get_global_size(0)) + \n (get_global_id(1) * get_global_size(0) + get_global_id(0)));\n";
	Out << "}\n\n";
}

//TLS Reads
void OpenCLKernelCodeGeneration::genSpecRead(string type)
{
	Out << "double spec_read_" << type << "(__global " << type << "* a, " << 
		"__global int* wr_log, __global int *read_log, int thread_id, __global int *invalid)\n";
	Out << "{\n";
	Out << type << " value;\n";
	Out << "atom_max((__global int*)read_log, thread_id);\n";
	Out << "value = a[0];\n\n";

	Out << "if (wr_log[0] > thread_id)\n";
	Out << "{\n";
	Out << "	*invalid = 1;\n";
	Out << "}\n";

	Out<< "return value;\n";
	Out << "}\n\n";
}

//TLS Writes
void OpenCLKernelCodeGeneration::genSpecWrite(string type)
{
#if 1
	Out << "double spec_write_" << type << "(__global " << type << "* a, " 
		"__global int* wr_log, __global int* read_log, int thread_id, __global int *invalid," << type << " value)\n";
	Out << "{\n";
	Out << "if (atom_max((__global int*)wr_log, thread_id) > thread_id)\n";
	Out << "{\n";
	Out <<"	*invalid = 1;\n";
	Out << "}\n\n";

	Out << "a[0] = value;\n";

	Out << "if (read_log[0] > thread_id)\n";
	Out << "{\n";
	Out << "	*invalid = 1;\n";
	Out << "}\n";
	Out << "return value;\n";
	Out << "}\n\n";
#endif
}


/*!
 * Output architecture-dependent headers
 *
 */
void OpenCLKernelCodeGeneration::genArchDepHeaders()
{
	genNVIDIAMacros();
	if(OCLCompilerOptions::EnableGPUTLs)
	{
		Out << "#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable\n";
		Out << "#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable\n";
	}
}

//array linearised routines
static string genLineariseMarco(unsigned dim)
{
	string str = "#define CALC_" + uint2String(dim) + "D_IDX(";

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

/*!
 * Output user defined data structures
 *
 */
void OpenCLKernelCodeGeneration::genDataStructures()
{
	vector<RecordDecl*> usedRDs = OpenCLGlobalInfoContainer::getRecordDecls();
	vector<TypedefDecl*> TDs = OpenCLGlobalInfoContainer::getTypedefDecls();

	DeclPrinter dp(Out, Context, Context.PrintingPolicy, 0);
	for (unsigned i=0; i<usedRDs.size(); i++)
	{
		RecordDecl* rd = usedRDs[i];

		dp.VisitRecordDecl(rd);
		Out << ";\n";
	}

	Out << "\n";

	for (unsigned i=0; i<TDs.size(); i++)
	{
		TypedefDecl* D = TDs[i];

		string S = D->getNameAsString();
		S = "typedef " + D->getUnderlyingType().getAsString() + " " + S;

		Out << S << ";\n";
	}
}

//TLS Check Routines
void OpenCLKernelCodeGeneration::TLSCheckRoutines(unsigned int dim)
{
	Out << "__kernel void TLS_Checking_" << dim << "D(";
	for (unsigned i=0; i<dim; i++)
	{
		Out << "unsigned dim" << i << ", ";
	}

	Out << "__global int* rd_log, __global int* wr_log, __global int* conflict_flag) {\n";

	Out << "	int wr, rd, index;\n";
	Out << "	int conflict = 0; \n";

	int d = (dim <= OCL_MAX_LOOP_LEVEL) ? dim : OCL_MAX_LOOP_LEVEL;

	for (int i=0; i<d; i++)
	{
		Out << "if (get_global_id(" << i << ") >= dim" << i << ") {\n return; \n }\n";
	}

	if (dim > OCL_MAX_LOOP_LEVEL)
	{
		Out << "unsigned ws[" << (dim - OCL_MAX_LOOP_LEVEL) << "];";
		for (unsigned i=OCL_MAX_LOOP_LEVEL; i<dim; i++)
		{
			unsigned id = (i - OCL_MAX_LOOP_LEVEL);
			Out << "for(ws[" << id << "]=0; ws[" << id << "] < dim" << i << "; ws[" << id << "]++)\n";
		}
		Out << "{\n";

	}

	if (d > 1)
	{
		Out << " index = CALC_" << dim << "D_IDX(";
		for (int i=dim-1; i>=0; i--)
		{
			if (i < (int)(dim-1))
				Out << ", ";
			Out << "dim" << i;
		}

		for (int i=d-1; i>=0; i--)
		{
			Out << ", get_global_id(" << i << ")";
		}

		for (unsigned i=OCL_MAX_LOOP_LEVEL; i<dim; i++)
		{
			unsigned int id = (i - OCL_MAX_LOOP_LEVEL);
			Out << ", ws[" << id << "]";
		}

		Out << ");\n";
	}
	else
	{
		Out << "	index = get_global_id(0);\n";
	}

	Out << "	wr = wr_log[index];\n"; 
	Out << "	rd = rd_log[index];\n"; 
	if (dim > OCL_MAX_LOOP_LEVEL)
	{
		Out << "	conflict = conflict | ((wr > 1) | (rd & wr)); \n";
	}
	else
	{
		Out << "	conflict = (wr > 1) | (rd & wr); \n";
	}
	Out << " 	wr_log[index]=0;\n";
	Out << " 	rd_log[index]=0;\n";

	if (dim > OCL_MAX_LOOP_LEVEL)
	{
		Out << "}\n";
	}

	Out << "\n 	if (conflict) {\n	*conflict_flag = 1;\n}";
	Out << " }\n";
}

/*!
 * Generate common macro routines
 *
 */
void OpenCLKernelCodeGeneration::genMacros()
{
	genArchDepHeaders();
	Out << "#define GROUP_SIZE " << DEFAULT_GROUP_SIZE << "\n\n";

	if (OCLCompilerOptions::UseArrayLinearize || OCLCompilerOptions::EnableGPUTLs || OCLCompilerOptions::printLinearMacros)
	{
		Out << "//-------------------------------------------\n";
		Out << "//Array linearize macros (BEGIN)\n";
		Out << "//-------------------------------------------\n";
		for (unsigned i=2; i<=8; i++)
		{
			Out << genLineariseMarco(i);		
		}
		Out << "//-------------------------------------------\n";
		Out << "//Array linearize macros (END)\n";
		Out << "//-------------------------------------------\n";
		Out << "\n";
	}
}

//Process variables in a fuction, which needed to be tracked as GPU TLS Variables
vector<ValueDecl*> OpenCLKernelCodeGeneration::funcLevelGPUTLSVars(FunctionDecl* D, RenamedFuncInfo& r, vector<ValueDecl*>& globalMemoryObjs, 
		vector<unsigned>& globalMemObjIndexs, bool recordTLSInput)
{
	vector<ValueDecl*> gR;
	vector<unsigned> gRI;
	std::string Proto = r.newName;

	StmtPicker sp(llvm::nulls(), Context, NULL, Context.PrintingPolicy, 4);
	sp.Visit(D->getBody());
	vector<OCLRWSet> rws = sp.getRWS();

	//Function level TLS should be conservative, as long as a global buffer is wirtten
	//we should track it
	//Get the read write set
	for (unsigned ii=0; ii<globalMemoryObjs.size(); ii++)
	{
		string name = globalMemoryObjs[ii]->getName();

		for (unsigned k=0; k<rws.size(); k++)
		{
			if (rws[k].getName() == name && rws[k].isWriteVar())
			{
				gR.push_back(globalMemoryObjs[ii]);
				gRI.push_back(globalMemObjIndexs[ii]);
				break;
			}
		}
	}

	for (unsigned k=0; k<gR.size(); k++)
	{
		string name = gR[k]->getName();

		string type = getCononicalType(gR[k]);
		unsigned dim = getArrayDimension(type);

		if (dim <= 1)
		{
			Out << ", __global int* rd_log_" << name;
			Out << ", __global int* wr_log_" << name;
		}
		else
		{
			vector<unsigned> dims = getArrayDef(type);
			string declare_string;

			for (unsigned j=1; j<dim; j++)
			{
				declare_string += "[" + uint2String(dims[j]) + "]";
			}

			Out << ", __global int (*rd_log_" << name << ")";
			Out << declare_string << "";

			Out << ", __global int (*wr_log_" << name << ")";
			Out << declare_string << "";
		}

		if (recordTLSInput)
		{
			unsigned index = gRI[k];
			OpenCLGlobalInfoContainer::addRenameFuncGPUTLS(Proto, gR[k], index);
		}
	}

	return gR;
}

//Expand the function arguments
vector<ValueDecl*> OpenCLKernelCodeGeneration::genProto4RenamedFunc(FunctionDecl* D, RenamedFuncInfo& r, bool showComment, bool recordTLSInput)
{
	vector<ValueDecl*> globalMemoryObjs;
	vector<unsigned> globalMemObjIndexs;

	switch (D->getStorageClass()) {
		case SC_None: break;
		case SC_Extern: break;
		case SC_Static:  break;
		case SC_PrivateExtern:
		case SC_Auto:
		case SC_Register:
						 break;
	}
	vector<globalVarIndex> gIds = r.globalArugIds;
	std::string Proto = r.newName;
	QualType Ty = D->getResultType();

	if (showComment)
	{
		vector<string> globalVs;
		vector<string> localVs;

		//This scan is used to annotate the function comments
		for (unsigned i = 0, e = D->getNumParams(); i != e; ++i) {
			//Checking whether "
			bool found = false;
			bool isALocalBuf = false;
			vector<globalVarIndex>::iterator iter;
			for (iter = gIds.begin(); iter != gIds.end(); iter++)
			{
				if (iter->i == i)
				{
					found = true;
					isALocalBuf = iter->isALocalBuf();
					break;
				}
			}

			ParmVarDecl* decl = D->getParamDecl(i);
			string name = decl->getNameAsString();

			if (!found || isALocalBuf)
			{
				if (isALocalBuf)
				{
					localVs.push_back(name);
				}
			}
			else //This argument should be renamed, simply replace the argument to "__global" will do 
				//the trick
			{
				globalVs.push_back(name);
			}
		}


		Out << "//-------------------------------------------------------------------------------\n";
		Out << "//This is an alias of function: " << r.getOrigFuncName() << "\n";
		Out << "//The input arguments of this function are expanded. \n";
		if (localVs.size() > 0)
		{
			Out << "//Local memory variables:\n";
			for (unsigned k=0; k<localVs.size(); k++)
			{
				Out << "//	" << (k) << ": " << localVs[k] << "\n";
			}
		}
		if (globalVs.size() > 0)
		{
			Out << "//Global memory variables:\n";
			for (unsigned k=0; k<globalVs.size(); k++)
			{
				Out << "//	" << (k) << ": " << globalVs[k] << "\n";
			}
		}
		Out << "//-------------------------------------------------------------------------------\n";
	}
	Out << Ty.getAsString() << " ";
	Out << Proto << " (";

	DeclPrinter ParamPrinter(Out, Context, Context.PrintingPolicy, 0);
	for (unsigned i = 0, e = D->getNumParams(); i != e; ++i) {
		if (i > 0)
		{
			Out << ", ";
		}

		//Checking whether "
		bool found = false;
		bool isALocalBuf = false;
		vector<globalVarIndex>::iterator iter;
		for (iter = gIds.begin(); iter != gIds.end(); iter++)
		{
			if (iter->i == i)
			{
				found = true;
				isALocalBuf = iter->isALocalBuf();
				break;
			}
		}

		ParmVarDecl* decl = D->getParamDecl(i);
		string type = getCononicalType(decl);
		string gtype = getGlobalType(type);
		string name = decl->getNameAsString();
		unsigned dim = getArrayDimension(type);

		if (!found || isALocalBuf)
		{
			if (isALocalBuf)
			{
				GlobalMemoryObj _g(name, type, gtype, dim, false);
				//Make sure the intel compiler can compile the code
				printGLInputParam(_g, "__local ");
			}
			else
			{
				ParamPrinter.VisitParmVarDecl(decl);
			}
		}
		else //This argument should be renamed, simply replace the argument to "__global" will do 
			//the trick
		{
			decl->setOCLGlobal();
			if (iter->isPointerAccess)
			{
				decl->setOCLPointerAccess();
			}

			if (OCLCompilerOptions::UseArrayLinearize) {
				Out << "__global ";
				Out << gtype << "* ";
				Out << name;	
			}
			else
			{
				GlobalMemoryObj _g(name, type, gtype, dim, OCLCommon::isAGlobalMemThreadPrivateVar(decl));
				printGLInputParam(_g, "__global ");
				globalMemoryObjs.push_back(decl);
				globalMemObjIndexs.push_back(i);
			}
		}
	}

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

			globalMemoryObjs.push_back(expr->getDecl());
		}	
	}

	if (OCLCompilerOptions::UseArrayLinearize) {
		for (unsigned i=0; i<gIds.size(); i++)
		{
			if (gIds[i].isPointerAccess)
			{
				Out << ", unsigned arg_" << gIds[i].i << "_offset";
			}
		}
	}

	if (r.hasGlobalMemThreadPrivate)
	{
		if (OCLCompilerOptions::UseArrayLinearize) {
			Out << ", unsigned int " << COPYIN_MULTI_FACTOR_NAME;
			Out << ", unsigned int " << COPYIN_ADD_OFFSET_NAME;
		}
	}

	//GPU TLS
	if (OCLCompilerOptions::EnableGPUTLs && r.enable_spec)
	{
		globalMemoryObjs = funcLevelGPUTLSVars(D, r, globalMemoryObjs, globalMemObjIndexs, recordTLSInput);
	}

	if (OCLCompilerOptions::OclTLSMechanism)
	{
		Out << ",__global int* tls_validflag, int tls_thread_id";
	}

	Out << ")";

	return globalMemoryObjs;
}


FunctionDecl* OpenCLKernelCodeGeneration::pickFuncDeclByName(vector<FunctionDecl*>& candidateFuncs, string name)
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


void OpenCLKernelCodeGeneration::printGLInputParam(GlobalMemoryObj& g, string qualifier)
{
	string gtype = g.getGType();
	Out << qualifier << " " << gtype;
	if ((g.getDim() > 1 ) || g.isTGBF())
	{
		vector<unsigned> dims = getArrayDef(g.getType());
		Out << " (*" << g.getName() << ")";
		for (unsigned j=1; j<dims.size(); j++)
		{
			Out << "[" << uint2String(dims[j]) << "]";
		}
	}
	else
	{
		Out << "* ";	
		Out << g.getName();
	}

}

//Generate the prototype for functions that have been renamed
void OpenCLKernelCodeGeneration::genRenamedFuncProto(vector<FunctionDecl*>& candFuncs, vector<RenamedFuncInfo>& rnFuncs)
{
	for (unsigned i=0; i<rnFuncs.size(); i++)
	{
		FunctionDecl* decl = pickFuncDeclByName(candFuncs, rnFuncs[i].getOrigFuncName());
		genProto4RenamedFunc(decl, rnFuncs[i]);
		Out << ";\n";
	}
}

/*!
 * Generate the definition for renamed functions
 *
 */
void OpenCLKernelCodeGeneration::genDef4RenamedFunc(vector<FunctionDecl*>& candidateFuncs, vector<RenamedFuncInfo>& rnFuncs)
{
	OpenCLPrinter tp(Out, Context, NULL, Context.PrintingPolicy, 4);

	for (unsigned j=0; j<rnFuncs.size(); j++)
	{
		FunctionDecl* D = pickFuncDeclByName(candidateFuncs, rnFuncs[j].getOrigFuncName());
		vector<ValueDecl*> gWriteObjs = genProto4RenamedFunc(D, rnFuncs[j], true, true);

		Out << "{\n";
		vector<FuncProtoExt>& fs = D->getAddedOpenCLNDRangeVars();
		vector<FuncProtoExt> as;
		//FIXME: This is very urgly!!!
		for (vector<FuncProtoExt>::iterator iter=fs.begin(); iter!=fs.end(); iter++)
		{
			if (!iter->isALocalVar())
			{
				as.push_back(*iter);
			}
		}

		tp.setGlobalMemoryVariables(as);

		if (OCLCompilerOptions::UseArrayLinearize) {
			//This parameter should be declared as global memory objects as well
			vector<globalVarIndex> gIds = rnFuncs[j].globalArugIds;

			for (unsigned i=0; i<gIds.size(); i++)
			{
				if (gIds[i].i >= D->getNumParams())
				{
					cerr << "Warning: somthing is wrong when renaming function: " << D->getNameAsString() << 
						" : " << gIds[i].i << " : " << D->getNumParams() << endl;
					continue;
				}
				else
					if (!gIds[i].shouldTreadAsGlobalVar())
					{
						continue;
					}

				ParmVarDecl* param = D->getParamDecl(gIds[i].i);

				if (gIds[i].isPointerAccess)
				{
					string offset_string = "arg_";
					offset_string = offset_string + uint2String(gIds[i].i);
					offset_string = offset_string + "_offset";
					param->setOffsetString(offset_string);

					string pname = param->getNameAsString();
					Out << pname << "+=" << offset_string << ";\n";
				}

				bool isFLevel = false;
				if (param->isLocalVarDecl())
					isFLevel = true;
				else
					if (param->isFunctionOrMethodVarDecl())
						isFLevel = true;

				//FIXME:!! This is very urgely!
				bool isTP = (gIds[i].isGTP) ? true: false;
				OCLGlobalMemVar var(param, gIds[i].isGTP, isFLevel, isTP);
				tp.addAGlobalMemoryVariables(var);
			}

			Out << "\n";
		}
		else
		{
			//genGlocal2LocalConvertCode(_globalInputParams);
		}

		bool track_all_g_write = false;

		//record gpu tls bufs
		if (OCLCompilerOptions::EnableGPUTLs || OCLCompilerOptions::OclTLSMechanism)
		{
			tp.setGlobalWriteBufs(gWriteObjs);
			//track_all_g_write = true;
		}

		if (!rnFuncs[j].enable_spec)
		{
			tp.DisableSpecReadWrite();
		}

		tp.SetIsPrintFunc(true);
		tp.printFuncBody(D->getBody(), false, track_all_g_write);
		tp.SetIsPrintFunc(false);

		Out << "}\n";
		Out << "\n";

	}

	//Restore function parameters
	for (unsigned i=0; i<rnFuncs.size(); i++)
	{
		FunctionDecl* D = pickFuncDeclByName(candidateFuncs, rnFuncs[i].getOrigFuncName());
		vector<globalVarIndex> gIds = rnFuncs[i].globalArugIds;

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

/*!
 * Generate prototype for a function
 *
 */
void OpenCLKernelCodeGeneration::genFuncProto(FunctionDecl* D)
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
		case SC_Extern: break;
		case SC_Static: break;
		case SC_PrivateExtern:
		case SC_Auto:
		case SC_Register:
						break;
	}

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

/*!
 * Generate callee functions
 *
 */
void OpenCLKernelCodeGeneration::genFuncs()
{
	vector<FunctionDecl*>& candFuncs = OpenCLGlobalInfoContainer::getCalleeFuncs();
	vector<RenamedFuncInfo>& rnFuncs = OpenCLGlobalInfoContainer::getRenameFuncs();

	if (OCLCompilerOptions::EnableGPUTLs)
	{
		vector<RenamedFuncInfo> bkFuncs  = rnFuncs;
		for (unsigned i=0; i<bkFuncs.size(); i++)
		{
			RenamedFuncInfo n = bkFuncs[i];
			n.newName = n.newName + "_no_spec";
			n.enable_spec = false;
			rnFuncs.push_back(n);
		}
	}

	DeclPrinter dp(Out, Context, Context.PrintingPolicy, 4);

	genRenamedFuncProto(candFuncs, rnFuncs);

	for (unsigned i=0; i<candFuncs.size(); i++)
	{
		FunctionDecl* D = candFuncs[i];
		//If the function body uses variables that are declared
		//as "__global", the function will be printed as a renamed function.
		//Therefore, it will not be printed here
		if (D->isBodyHasGlobalVar())
			continue;
		//Generate the function prototype fist
		if (D->getBody())
		{
			genFuncProto(D);
			Out << ";\n";
		}
	}

	Out << "\n";

	OpenCLPrinter tp(Out, Context, NULL, Context.PrintingPolicy, 0);

	for (unsigned i=0; i<candFuncs.size(); i++)
	{
		FunctionDecl* D = candFuncs[i];
		if (D->isBodyHasGlobalVar())
			continue;

		if (D->getBody())
		{
			genFuncProto(D);
			tp.setGlobalMemoryVariables(D->getAddedOpenCLNDRangeVars());
			tp.PrintStmt(D->getBody());
			Out << "\n";
		}
	}

	//Generate the definition of renamed function
	genDef4RenamedFunc(candFuncs, rnFuncs);
}

/*!
 * Generate ocl Kernels
 *
 */
void OpenCLKernelCodeGeneration::genKernels()
{
	vector<FunctionDecl*> revisedFuncs = OpenCLGlobalInfoContainer::getRevisedFuncs();

	for (unsigned i=0; i<oclLoops.size(); i++)
	{
		OpenCLKernelLoop* loop = oclLoops[i];

		bool bReduction = loop->isReductionLoop();

		//Generate reduction kernel
		if (bReduction)
		{	
			OpenCLReductionKernelGenerator g(Out, Context, loop, revisedFuncs, pDMF);
			g.doIt();
		}
		else
		{
			OpenCLKernelCodeGenerator g(Out, Context, loop, pDMF);
			g.doIt();
		}
	}

}

void OpenCLKernelCodeGeneration::optimisation()
{
	OpenCLKernelCodeOpt opt(Context, oclLoops);
	opt.doIt();
}

void OpenCLKernelCodeGeneration::postOptimisation()
{
	OpenCLLoopScan ocL(Context, oclLoops);
	ocL.doIt();

	OpenCLCollectCalledFuncs ocF(Context, oclLoops);
	ocF.doIt();
}

/*!
 * The main routine
 *
 */
void OpenCLKernelCodeGeneration::doIt()
{
	pDMF = new OpenCLDumpMLFeature(kernelFile);

	optimisation();
	postOptimisation();

	//Generate common routine
	genMacros();
	genDataStructures();

	//Generate TLS Checking Kernels
	if (OCLCompilerOptions::EnableGPUTLs)
	{
		Out << "//-------------------------------------------------------------------------------\n";
		Out << "//TLS Checking Routines (BEGIN)\n";
		Out << "//-------------------------------------------------------------------------------\n";

		if (OCLCompilerOptions::OclTLSMechanism)
		{
			calculateThreadId();
			genSpecRead("double");
			genSpecWrite("double");
		}
		else
		{
			for (unsigned i=1; i<=MAX_TLS_DIMENSIONS; i++)
			{
				TLSCheckRoutines(i);
			}
		}
		Out << "//-------------------------------------------------------------------------------\n";
		Out << "//TLS Checking Routines (END)\n";
		Out << "//-------------------------------------------------------------------------------\n";
	}

	Out << "//-------------------------------------------------------------------------------\n";
	Out << "//Functions (BEGIN)\n";
	Out << "//-------------------------------------------------------------------------------\n";
	genFuncs();
	Out << "//-------------------------------------------------------------------------------\n";
	Out << "//Functions (END)\n";
	Out << "//-------------------------------------------------------------------------------\n";
	Out << "\n";


	Out << "//-------------------------------------------------------------------------------\n";
	Out << "//OpenCL Kernels (BEGIN)\n";
	Out << "//-------------------------------------------------------------------------------\n\n";
	genKernels();
	Out << "//-------------------------------------------------------------------------------\n";
	Out << "//OpenCL Kernels (END)\n";
	Out << "//-------------------------------------------------------------------------------\n\n";



	if (OCLCompilerOptions::EnableMLFeatureCollection)
	{
		string err;
		llvm::raw_fd_ostream Out("ml_helper.py", err);
		pDMF->doIt(Out);
		Out.close();
	}

	delete pDMF;
}
