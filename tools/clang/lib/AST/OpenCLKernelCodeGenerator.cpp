#include "clang/Omp2Ocl/OpenCLKernelCodeGenerator.h"
#include "clang/Omp2Ocl/OpenCLKernelNameContainer.h"
#include "clang/Omp2Ocl/OpenCLRoutine.h"
#include "clang/Omp2Ocl/OpenCLCommon.h"
#include "clang/Omp2Ocl/OpenCLKernelName.h"
#include "clang/Omp2Ocl/OpenCLCopyInRoutine.h"
#include "clang/Omp2Ocl/OpenCLPrinter.h"
#include "clang/Omp2Ocl/OpenCLGloballInfoContainer.h"
#include "clang/Omp2Ocl/OpenCLLocalMemOpt.h"
#include "clang/Omp2Ocl/OpenCLCompilerOptions.h"

void OpenCLKernelCodeGenerator::genMLFeatureScript(string kernel_name)
{
	if (pDMF)
	{
		pDMF->generateScriptForKernel(kernel_name);
	}
}

string OpenCLKernelCodeGenerator::oclKernelPrefix()
{
	return OCL_KERNEL_PREFIX;
}


//Scan TLS Variables
vector<OpenCLTLSBufferAccess> OpenCLKernelCodeGenerator::scanTLSVariables()
{
	scanGlobalWriteBufs();

	OpenCLPrinter op(llvm::nulls(), Context, NULL, Context.PrintingPolicy, 4, true);
	op.setReductionObj(loop->for_stmt->getOMPFor().getReductionObjs());
	op.setGlobalMemoryVariables(loop->globalMemoryVariables);

	op.setGlobalLCWriteBufs(loop->getGlobalLCWriteBufs());
	op.setGlobalWriteBufs(loop->getGlobalWriteBufs());
	op.setOpenCLKernelLoop(loop);

	op.PrintStmt(loop->Kernel);

	return op.getActTLSVec();  
}

bool OpenCLKernelCodeGenerator::isInActTls(string name)
{
	for (unsigned i=0; i<act_tls.size(); i++)
	{
		if (name == act_tls[i].getName())
			return true;
	}

	return false;
}

//FIXED ME, this should be combined with genKernelArguments
void OpenCLKernelCodeGenerator::scanGlobalWriteBufs()
{
	vector<string> ParamList;
	string name;
	string string_buf;
	llvm::raw_string_ostream OS(string_buf);
	vector<string> gputls_params;

	for (vector<PLoopParam>::iterator iter =  loop->params.begin(); iter != loop->params.end(); iter++)
	{
		if (shouldDeclaredAsKernelArgu((*iter)))
		{
			bool __localv = false;

			name = iter->getName();

			ValueDecl* d = iter->getDecl();		
			string type = getCononicalType(d);
			unsigned dim = getArrayDimension(type);

			if (loop->isACopyIn(name))
			{
				continue;
			} //copy a buffer to a __local array
			else if (OCLCompilerOptions::UseLocalMemory && 
					(OpenCLLocalMemOpt::isLocalVar(d) || OpenCLLocalMemOpt::isLocalVar(loop, d) ) && 
					(dim >= 1)
				)
			{
				continue;
			}
			//Other, non threadprivate var (this can be a global array)	
			else if (!OCLCommon::isAThreadPrivateVariable(name))
			{
			}	

			if (dim)
			{
				ValueDecl* d = iter->declRef->getDecl();

				if (OCLCommon::isAThreadPrivateVariable(name))
				{
					continue;
				}
				//Make sure this is not a variable that will be copied to a local buffer
				else if (!__localv)
				{
					//This will generate an input buffer that will be used to track write/read for GPU TLS
					if (OCLCompilerOptions::EnableGPUTLs && loop->isInWriteSet(name))
					{
						//This is used for conflict detections if GPU speculation is enabled
						if (!d->isDefinedOutsideFunctionOrMethod())
						{
							loop->addGlobalLCWriteBuf(d);
						}
						else
						{
							loop->addGlobalWriteBuf(d);
							OpenCLGlobalInfoContainer::addwriteGlobalMemObj(d);
						}
					}
				}
			}	
		}
	}
} 

//Decide whether a vairable should appear on the argument list or not
bool OpenCLKernelCodeGenerator::shouldDeclaredAsKernelArgu(PLoopParam& P)
{
	string name = P.getName();

	if (OCLCommon::isAOMPPrivateVariable(name, loop->getForStmt()))
	{
		return false;
	}
	else
		if (loop->isInnerDecl(P.declRef))
		{
			return false;
		}
		else //e.g. int i = get_global_id(0)
			if (loop->isAOpenCLNDRangeVar(name))
			{
				return false;
			}
			else //I don't handle a reduction variable here
				if (loop->isAReductionVariable(name))
				{
					return false;
				}
				else
					if (OCLCommon::isAThreadPrivateVariable(name) &&
							//This threadprivate variable is not decalred as __global
							!OCLCommon::isAGlobalMemThreadPrivateVar(name) &&
							//Copy in variables are processed in a different way
							!loop->isACopyIn(name)
					   )

					{
						return false;
					}

	return true;
}

// gen an OpenCL Kernel argument
static string genInputParm(unsigned dim, string type, string name, OpenCLKernelLoop* loop, bool is_local_v=false)
{
	string argu;

	if (dim)
	{
		argu = argu + "__global " + getGlobalType(type) + "*  ";
	}
	else
	{
		argu = argu + type + " ";
	}

	if (!OCLCompilerOptions::UseArrayLinearize) {
		if (dim > 1 && !is_local_v)
		{
			name = "g_" + name;
		}
	}
	argu = argu + name + ", ";

	return argu;
}

// gen an OpenCL copyin argument for the kernel. 
// the copyin variable (x) is renamed as __ocl_copyin_x
static string genCopyInShadow(unsigned dim, string type, string name, OpenCLKernelLoop* loop)
{
	string argu;
	if (dim)
	{
		argu = argu + DEFAULT_COPYIN_TYPE;
		argu = argu + " " + getGlobalType(type) + "*  ";
		argu = argu + "__ocl_copyin_" + name;
	}	
	else
	{
		argu = argu + type + " " + name;
	}

	argu = argu + ", ";

	return argu;
}

void OpenCLKernelCodeGenerator::addLocalD(ValueDecl* d)
{
	string name = d->getNameAsString();

	for (unsigned i=0; i<_localD.size(); i++)
	{
		if (_localD[i]->getNameAsString() == name)
			return;	
	}

	d->setUseAsLocalVar();
	_localD.push_back(d);
}

//Find and print out arguments for an OpenCL kernel
unsigned int OpenCLKernelCodeGenerator::genKernelArguments(vector<string>& gputls_params)
{
	vector<string> ParamList;
	string name;
	string string_buf;
	llvm::raw_string_ostream OS(string_buf);
	unsigned num = 0;

	for (vector<PLoopParam>::iterator iter =  loop->params.begin(); iter != loop->params.end(); iter++)
	{
		if (shouldDeclaredAsKernelArgu((*iter)))
		{
			bool __localv = false;

			name = iter->getName();

			ValueDecl* d = iter->getDecl();		
			string type = getCononicalType(d);
			unsigned dim = getArrayDimension(type);

			OpenCLInputArgu inarg(iter->getDecl(), dim, loop->isACopyIn(name), OCLCommon::isAGlobalMemThreadPrivateVar(name), loop->getFunc());
			loop->getForStmt()->addInputArgu(inarg);

			//Print input argument, this is a little complicate
			//Threadprivate variables are passed as a __global memory 
			if (OCLCommon::isAGlobalMemThreadPrivateVar(name))
			{
				loop->hasGlobalCopyInBuf = true;
				OS << genInputParm(dim, type, name, loop);
				_globalInputParams.push_back(InputParam(d, name, dim, getGlobalType(type)));
			}

			if (loop->isACopyIn(name))
			{
				OS << genCopyInShadow(dim, type, name, loop);

			} //copy a buffer to a __local array
			else if (OCLCompilerOptions::UseLocalMemory && 
					(OpenCLLocalMemOpt::isLocalVar(d) || OpenCLLocalMemOpt::isLocalVar(loop, d) ) && 
					(dim >= 1)
				)
			{
				__localv = true;
				OS << genInputParm(dim, type, LOCAL_VAR_PASSIN_PREFIX + name, loop, true); 	
				addLocalD(d);
			}
			//Other, non threadprivate var (this can be a global array)	
			else if (!OCLCommon::isAThreadPrivateVariable(name))
			{
				OS << genInputParm(dim, type, name, loop);	
				_globalInputParams.push_back(InputParam(d, name, dim, getGlobalType(type)));
			}	

			if (dim)
			{
				ValueDecl* d = iter->declRef->getDecl();
				bool isGlobalM = OCLCommon::isAGlobalMemThreadPrivateVar(name);
				bool isFL = d->isDefinedOutsideFunctionOrMethod();

				if (loop->isACopyIn(name))
				{
					loop->addCopyInBuffer(d, isGlobalM);
				}

				if (OCLCommon::isAThreadPrivateVariable(name))
				{
					loop->addGlobalMemoryVar(OCLGlobalMemVar(d, isGlobalM, isFL, true));
				}
				//Make sure this is not a variable that will be copied to a local buffer
				else if (!__localv)
				{
					loop->addGlobalMemoryVar(OCLGlobalMemVar(iter->declRef->getDecl(), isGlobalM, isFL, false));
					//This will generate an input buffer that will be used to track write/read for GPU TLS
					if (OCLCompilerOptions::EnableGPUTLs && loop->getOMPFor().isTLSCheck() && isInActTls(name))
					{
						if (dim > 1)
						{
							string s = "__global int* g_rd_log_" + name; 
							gputls_params.push_back(s);
							s = "__global int* g_wr_log_" + name;
							gputls_params.push_back(s);
						}
						else
						{
							string s = "__global int* rd_log_" + name; 
							gputls_params.push_back(s);
							s = "__global int* wr_log_" + name;
							gputls_params.push_back(s);
						}
					}
				}
			}	

			num++;
		}
	}

	vector<OpenCLNDRangeVar> GV = loop->getOclLoopIndexs();
	for (unsigned i=0; i<GV.size(); i++)
	{
		if(GV[i].hasIncremental && !GV[i].isIncInt)
		{
			OS << GV[i].type <<  " __ocl_" << GV[i].variable << "_inc_" << GV[i].increment << ",";
			num++;
		}

		if (GV[i].Cond && !GV[i].isCondInt)
		{				
			OS << GV[i].type << " __ocl_" << GV[i].variable << "_bound,";
			num++;
		}
	}

	for (unsigned i=0; i<gputls_params.size(); i++)
	{
		if (i > 0)
			OS << ",";
		OS << gputls_params[i];
	}

	OS.flush();

	if (string_buf.length())
	{
		char c = string_buf[string_buf.length() - 1];
		while(c == ',' || c == ' ')
		{	
			string_buf.erase(string_buf.end() - 1);
			c = string_buf[string_buf.length() - 1];
		}
		Out << string_buf;
	}

	loop->for_stmt->setLoopIndex(GV);

	return num;
}

// is the variable an OpenCLNDRange var (e.g. int i = get_global_id(0);)
bool OpenCLKernelCodeGenerator::isAnOpenCLOpenCLNDRangeVar(string& name)
{
	vector<OpenCLNDRangeVar> GV = loop->getOclLoopIndexs();
	for (vector<OpenCLNDRangeVar>::iterator iter = GV.begin(); iter != GV.end(); iter++)
	{
		if (iter->variable == name)
			return true;
	}

	return false;
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

/*====================================================================
 *
 * Functions that will be called by doIt directely (BEGIN)
 *
 *
 ======================================================================*/

/*!
 * Print the OpenCL kernel prototype
 *
 */
unsigned int OpenCLKernelCodeGenerator::genKernelProto(string kernel_name, vector<string>& gputls_params)
{
	Out << oclKernelPrefix() << kernel_name;
	Out << " (";
	unsigned int num = genKernelArguments(gputls_params);

	if (OCLCompilerOptions::OclTLSMechanism)
	{
		Out << ",__global int* tls_validflag";
	}

	Out << ")";

	return num;
}


/*!
 * Output NDRange variables
 *
 */
void OpenCLKernelCodeGenerator::printNDRangeVars()
{
	vector<OpenCLNDRangeVar> GV = loop->getOclLoopIndexs();
	vector<OMPMultIterIndex> multIterIndex = loop->getForStmt()->getOMPFor().getMultIterIndex();
	bool hasMult = (multIterIndex.size() > 0) ? true : false;
	unsigned ii = 0;

	Out << "	//-------------------------------------------\n";
	Out << " 	//OpenCL global indexes (BEGIN)\n";	
	Out << "	//-------------------------------------------\n";

	for (vector<OpenCLNDRangeVar>::iterator iter = GV.begin(); iter != GV.end(); iter++)
	{
		Out << iter->type << " " << iter->variable << " = get_global_id(" << ii << ")";

		if (iter->Init)
		{
			string buf = getStringStmt(Context, iter->Init);

			if (!isAZeroInteger(buf))
			{
				Out << " + " << buf;
			}
		}

		Out << ";\n";

		if(iter->hasIncremental)
		{
			if (iter->increment != "1")
			{
				Out << iter->variable << " = " << iter->variable << " * __ocl_" << iter->variable << "_inc_" << iter->increment << ";\n";
			}		
		}

		if (hasMult)
		{
			Out << iter->type << " _ocl_" << iter->variable << "_init = " << iter->variable << ";\n";
		}

		ii++;
	}

	for (vector<OpenCLNDRangeVar>::iterator iter = GV.begin(); iter != GV.end(); iter++)
	{
		if (iter->Cond)
		{
			Out << "if (!(" << iter->getCondString(Context) << ")) {\n    return;\n}\n";
		}
	}

	Out << "	//-------------------------------------------\n";
	Out << "	//OpenCL global indexes (END)\n";	
	Out << "	//-------------------------------------------\n";
	Out << "\n";
}


/*!
 * Declare private variables
 *
 */
void OpenCLKernelCodeGenerator::declarePrivateVariables()
{
	Out << "	//-------------------------------------------\n";
	Out << "	//Pivate variables (BEGIN)\n";
	Out << "	//-------------------------------------------\n";
	for (vector<PLoopParam>::iterator iter = loop->params.begin(); iter != loop->params.end(); iter++)
	{
		string name = iter->getName();

		//This variable has been declared as and index variable
		if (isAnOpenCLOpenCLNDRangeVar(name))
		{
			continue;
		}

		if (loop->getForStmt()->getOMPFor().isVariablePrivate(name))
		{
			SourceLocation loc = iter->getDecl()->getLocation();
			string decl = declareAPrivateVariable(iter->declRef);

			Out << decl << "/* (User-defined privated variables) : Defined at " << OCLCommon::getFileName(Context, loc) << " : " << OCLCommon::getLineNumber(Context, loc) << " */" << "\n";

		}
		else
		{
			if (OCLCommon::isAThreadPrivateVariable(iter->getDecl()))
			{
				if (!OCLCommon::isAGlobalMemThreadPrivateVar(name))
				{
					SourceLocation loc = iter->declRef->getDecl()->getLocation();
					string decl = declareAPrivateVariable(iter->declRef);
					Out << decl << "/* threadprivate: defined at " << OCLCommon::getFileName(Context, loc) << " : " << OCLCommon::getLineNumber(Context, loc) << " */" << "\n";
				}
				continue;
			}
		}

	}

	Out << "	//-------------------------------------------\n";
	Out << "	//Pivate variables (END)\n";
	Out << "	//-------------------------------------------\n";
	Out << "\n";
	//	Out << " // Declare private variables (END)\n\n";	
}

void OpenCLKernelCodeGenerator::declareCopyInBuffers()
{
	OpenCLCopyInRoutine oc ( loop, Out );
	oc.declareCopyInBuffers();
}

void OpenCLKernelCodeGenerator::genCopyInCode()
{
	OpenCLCopyInRoutine co (loop, Out);
	co.doIt();
}

// gen loop headers to perform multiple iterations
void OpenCLKernelCodeGenerator::genMultIterLoopHeader()
{
	vector<OMPMultIterIndex> multIterIndex = loop->getForStmt()->getOMPFor().getMultIterIndex();
	vector<OpenCLNDRangeVar> gV = loop->getOclLoopIndexs();

	if (multIterIndex.size() <= 0)
	{
		return;
	}

	//Map the loop index to the original one (before loop interchange)
	for (unsigned i=0; i<gV.size(); i++)
	{
		OpenCLNDRangeVar& g = gV[i];

		if (g.orig_loop_index >= multIterIndex.size())
		{
			continue;
		}

		string s = multIterIndex[g.orig_loop_index].getVariable();
		int work_item = atoi( s.c_str());

		if (work_item > 1)
		{
			Out << "for (";
			Out << g.variable << "= _ocl_" << g.variable << "_init;";
			Out << g.getCondString(Context);
			Out << "; " << g.variable << " += get_global_size(" << i << ") ";

			if (g.hasIncremental)
			{
				if (g.isIncInt)
				{
					if (g.increment != "1")
					{
						Out << " * " << g.increment;
					}
				}
				else
				{
					Out << " * __ocl_" <<  g.variable << "_inc_" << g.increment;
				}
			}

			Out << ")\n";
		}
	}
}

void OpenCLKernelCodeGenerator::genKernelBodyInfo()
{
	SourceLocation lc = loop->getForStmt()->getForLoc();

	Out << "//-------------------------------------------------------------------------------\n";
	Out << "//Loop defined at line " << OCLCommon::getLineNumber(Context, lc) << " of " << OCLCommon::getFileName(Context, lc) << "\n";
	if (loop->swaped)
	{
		Out << "//The nested loops were swaped. " << "\n";
	}

	if (OCLCompilerOptions::EnableGPUTLs)
	{
		if (!loop->getOMPFor().isTLSCheck())
		{
			Out << "//GPU TLS Checking is disabled by the user. \n";
		}
	}
	Out << "//-------------------------------------------------------------------------------\n";
}

void OpenCLKernelCodeGenerator::declLocalVars()
{
	if (OCLCompilerOptions::UseLocalMemory && _localD.size())
	{
		Out << "	//-------------------------------------------\n";
		Out << "	//Declare local memory variables (END)\n";
		Out << "	//-------------------------------------------\n";

		for (unsigned i=0; i<_localD.size(); i++)
		{
			OpenCLLocalMemOpt::declareLocalVar(Out, _localD[i]);
		}

		Out << "	//-------------------------------------------\n";
		Out << "	//Declare local memory variables (END)\n";
		Out << "	//-------------------------------------------\n";
		Out << "\n";
	}
}

void OpenCLKernelCodeGenerator::genLocalVarCopyCode()
{
	if (OCLCompilerOptions::UseLocalMemory && _localD.size())
	{
		vector<OpenCLNDRangeVar> GV = loop->getOclLoopIndexs();

		Out << "	//-------------------------------------------\n";
		Out << "	//Load to local memory (BEGIN)\n";
		Out << "	//-------------------------------------------\n";

		/*
		Out << "if (";
		for (unsigned i=0; i<GV.size(); i++)
		{
			if (i > 0) 
				Out << "&&";
			Out << " (get_local_id(" << i << ") == 0) ";
		}

		Out << ") {\n";
		*/

		Out << "int __ocl_local_id = ";
		
		for (unsigned i=0; i<GV.size(); i++)
		{
			if (i > 0) 
				Out << " + ";
			Out << " get_local_id(" << i << ")";

			for (int j=i-1; j>=0; j--)
			{
				Out << " * ";
				Out << "get_local_size(" << j << ")";
			}
		}
		
		Out << ";\n";		

		Out << "int __ocl_dim_size = get_local_size (0)";
		
		for (unsigned i=1; i<GV.size(); i++)
		{
			Out << "* get_local_size(" << (i) << ")";
		}
		
		Out << ";\n";		

		for (unsigned i=0; i<_localD.size(); i++)
		{
			string passInName = LOCAL_VAR_PASSIN_PREFIX + _localD[i]->getNameAsString();
			OpenCLLocalMemOpt::genPreloadCode(Out, _localD[i], passInName);
		}

		//Out << "}\n";
		Out << "barrier(CLK_LOCAL_MEM_FENCE);\n";	

		Out << "	//-------------------------------------------\n";
		Out << "	//Load to local memory (END)\n";
		Out << "	//-------------------------------------------\n";
		Out << "\n";
	}
}

void OpenCLKernelCodeGenerator::declareVars()
{
	printNDRangeVars();
	declLocalVars();
	declarePrivateVariables();	
	declareCopyInBuffers();

	if (OCLCompilerOptions::OclTLSMechanism)
	{
		Out << "int tls_thread_id = calc_thread_id_"  << loop->getOclLoopIndexs().size() << "();\n";
	}

}

void OpenCLKernelCodeGenerator::genCommonRoutine()
{
	genCopyInCode();
	genLocalVarCopyCode();
}

void OpenCLKernelCodeGenerator::genKernelBody(bool enableTLS)
{
	//Acutal computation
	OpenCLPrinter op(Out, Context, NULL, Context.PrintingPolicy, 4, true);
	op.setReductionObj(loop->for_stmt->getOMPFor().getReductionObjs());
	op.setGlobalMemoryVariables(loop->globalMemoryVariables);
	op.setOpenCLKernelLoop(loop);
	op.DisableSpecReadWrite();

	if (OCLCompilerOptions::EnableGPUTLs && loop->getOMPFor().isTLSCheck() && enableTLS)
	{
		//These are set in genKernelArguments()
		op.setGlobalLCWriteBufs(loop->getGlobalLCWriteBufs());
		op.setGlobalWriteBufs(loop->getGlobalWriteBufs());
		op.EnableSpecReadWrite();
	}

	Out << "	//-------------------------------------------\n";
	Out << "	//OpenCL kernel (BEGIN)\n";
	Out << "	//-------------------------------------------\n";

	//print code if the user specifies that each work-item performs multiple loop iterations	
	genMultIterLoopHeader();

	//Printing out the inner loop headers if there are any
	for (vector<SwapLoopInfo>::iterator iter= loop->innerLoops.begin(); iter != loop->innerLoops.end(); iter++)
	{
		op.VisitForHeader(iter->for_stmt);	
	}

	//THe kernel
	op.PrintStmt(loop->Kernel);

	loop->for_stmt->setTLSCheckingVec(act_tls);

	Out << "	//-------------------------------------------\n";
	Out << "	//OpenCL kernel (END)\n";
	Out << "	//-------------------------------------------\n";
	Out << "\n";
}

/*====================================================================
 *
 * Functions that will be called by doIt directely (END)
 *
 ======================================================================*/

//
// Converting global memory to local variables
//
// e.g.  __global double (*u)[JMAXP+1][IMAXP+1][5] =
//    (__global double (*)[JMAXP+1][IMAXP+1][5])g_u;
//
void OpenCLKernelCodeGenerator::DeclPrivateVars4GlobalObjs()
{
	if(_globalInputParams.size() > 0)
	{
		Out << "	//-------------------------------------------\n";
		Out << "	//Convert global memory objects (BEGIN)\n";
		Out << "	//-------------------------------------------\n";
		for (unsigned i=0; i<_globalInputParams.size(); i++)
		{
			if (_globalInputParams[i].getDim() > 1)
			{
				string gtype =  getCononicalType(_globalInputParams[i].getDecl());
				vector<unsigned> dims = getArrayDef(gtype);
				Out << "__global " << _globalInputParams[i].getType();
				Out << " (*" << _globalInputParams[i].getName() << ")";

				string declare_string;

				for (unsigned j=1; j<dims.size(); j++)
				{
					declare_string += "[" + uint2String(dims[j]) + "]";
				}

				Out << declare_string << " = ";
				Out << "(__global " << _globalInputParams[i].getType() << "(*)" << declare_string << ")";
				Out << "g_" <<  _globalInputParams[i].getName() << ";\n";
			}
		}

		if (OCLCompilerOptions::EnableGPUTLs)
		{
			vector<ValueDecl*> vds = loop->getGlobalWriteBufs();
			if (vds.size() && act_tls.size())
			{
				Out << "\n";
				Out << "	//TLS Checking Buffers (BEGIN)\n";
				for (unsigned k=0; k<vds.size(); k++)
				{	
					string name = vds[k]->getName();
					string type = getCononicalType(vds[k]);
					unsigned dim = getArrayDimension(type);

					if (dim > 1 && isInActTls(name))
					{
						vector<unsigned> dims = getArrayDef(type);

						string declare_string;
						for (unsigned j=1; j<dim; j++)
						{
							declare_string += "[" + uint2String(dims[j]) + "]";
						}

						Out << "__global int (*rd_log_" << name << ")";
						Out << declare_string << " = ";
						Out << "(__global int (*)" << declare_string << ")";
						Out << "g_rd_log_" << name << ";\n";

						Out << "__global int (*wr_log_" << name << ")";
						Out << declare_string << " = ";
						Out << "(__global int (*)" << declare_string << ")";
						Out << "g_wr_log_" << name << ";\n";
					}
				}
				Out << "	//TLS Checking Buffers (END)\n\n";
			}

		}

		Out << "	//-------------------------------------------\n";
		Out << "	//Convert global memory objects (END)\n";
		Out << "	//-------------------------------------------\n\n";
	}
}


/*!
 * Generate the OpenCL kernel code
 *
 */
void OpenCLKernelCodeGenerator::doIt()
{
	string kernel_name = OpenCLKernelName::getOpenCLKernelName(loop->getFunc());
	loop->getForStmt()->setKernelName(kernel_name);
	OpenCLKernelNameContainer::addKernelName(kernel_name);
	vector<string> gputls_params;

	if (OCLCompilerOptions::EnableMLFeatureCollection)
	{
		Out << "#ifdef ENABLE_" << kernel_name << "\n";
	}

	if (OCLCompilerOptions::EnableGPUTLs)
	{
		if (loop->getOMPFor().isTLSCheck())
		{
			act_tls = scanTLSVariables();
		}
		else
		{
			act_tls.clear();
		}
	}

	genKernelBodyInfo();
	genKernelProto(kernel_name, gputls_params);

	//Local memory is used
	if (_localD.size())
	{
		loop->setUseDefLocalWorkGroup();
	}
	
	Out << " {\n";
	declareVars();

	if (!OCLCompilerOptions::UseArrayLinearize)
	{
		DeclPrivateVars4GlobalObjs();
	}

	genCommonRoutine();
	genKernelBody((gputls_params.size() > 0) ? true : false);

	Out << "}\n";

	if (OCLCompilerOptions::EnableMLFeatureCollection)
	{
		Out << "#endif\n";
	}

	if (pDMF)
	{
		pDMF->generateScriptForKernel(kernel_name, Context, loop->Kernel);
	}
}
