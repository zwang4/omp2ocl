#include "clang/Omp2Ocl/OpenCLReductionKernelGenerator.h"
#include "clang/Omp2Ocl/OpenCLKernelNameContainer.h"
#include "clang/Omp2Ocl/OpenCLRoutine.h"
#include "clang/Omp2Ocl/OpenCLCommon.h"
#include "clang/Omp2Ocl/OpenCLKernelName.h"
#include "clang/Omp2Ocl/OpenCLCopyInRoutine.h"
#include "clang/Omp2Ocl/OpenCLPrinter.h"
#include "clang/Omp2Ocl/OpenCLCompilerOptions.h"

//Generate code to write back results for the first stage
void OpenCLReductionKernelGenerator::ReductionFirstPhaseWriteBack( )
{
	string addition;
	vector<OpenCLNDRangeVar> GV = getOCLKernelL()->getOclLoopIndexs();
	OpenCLPrinter p(Out, getContext(), NULL, getContext().PrintingPolicy, 4, true);

	Out << "	//-------------------------------------------\n";
	Out << "	//Write back to the global buffer (BEGIN)\n";
	Out << "	//-------------------------------------------\n";
	Out << "{\n";
	if (GV.size() > 1)
	{
		Out << "	unsigned int __ocl_wb_idx = CALC_" << GV.size() << "D_IDX(";
		int gsize = GV.size() - 1;
		for (int j=gsize; j>=0; j--)
		{
			if (j < gsize)
				Out << ",";
			Out << "get_global_size(" << j << ")";
		}

		for (int j=GV.size()-1; j>=0; j--)
		{
			Out << ", get_global_id(" << j << ")";
		}

		Out << ");\n";
	}
	else
	{
		Out << "	unsigned int __ocl_wb_idx = get_global_id(0);\n";
	}

	vector<OMPReductionObj>& rObjs = getOCLKernelL()->getForStmt()->getOMPFor().getReductionObjs();	
	for (unsigned j=0; j<rObjs.size(); j++)
	{
		Out << "__ocl_part_" << rObjs[j].getVariable() << "[__ocl_wb_idx] = "
			<< rObjs[j].getVariable() << ";\n";
	}

	Out << "}\n";
	Out << "	//-------------------------------------------\n";
	Out << "	//Write back to the global buffer (END)\n";
	Out << "	//-------------------------------------------\n";
}

//Print kernel arguments for the firstReduction Phase
void OpenCLReductionKernelGenerator::genFirstPhaseArguments()
{
	for (unsigned i = 0; i < rVariables.size(); i++)
	{
		ValueDecl* d = rVariables[i];
		string type = getCononicalType(d);

		if (i > 0)
			Out << ", ";
		Out << "__global " << type << "* __ocl_part_" << d->getNameAsString();
	}

}

void OpenCLReductionKernelGenerator::declareReductionVars()
{
	Out << "	//-------------------------------------------\n";
	Out << "	//Declare reduction variables (BEGIN)\n";
	Out << "	//-------------------------------------------\n";
	
	for (unsigned i=0; i<rVariables.size(); i++)
	{
		ValueDecl* d = rVariables[i];
		SourceLocation loc = d->getLocation();
		
		string name = d->getNameAsString();
		string type = getGlobalType(getCononicalType(d));

		Out << type << " " << name << " = " << initValue(type) << ";";
	        Out <<	"/* reduction variable, defined at: " << OCLCommon::getFileName(getContext(), loc) << " : " << OCLCommon::getLineNumber(getContext(), loc) << " */" << "\n";

	}

	Out << "	//-------------------------------------------\n";
	Out << "	//Declare reduction variables (END)\n";
	Out << "	//-------------------------------------------\n";
	
}

//Collecting reduction variables
void OpenCLReductionKernelGenerator::collectReducVarDecls()
{
	vector<OMPReductionObj>& reducObjs = getOCLKernelL()->getForStmt()->getOMPFor().getReductionObjs();
	vector<OpenCLNDRangeVar> GV = getOCLKernelL()->getOclLoopIndexs();

	for (vector<PLoopParam>::iterator iter = getOCLKernelL()->params.begin(); iter != getOCLKernelL()->params.end(); iter++)
	{
		string name = iter->getName();
		ValueDecl* decl = iter->getDecl();

		if (getOCLKernelL()->isAReductionVariable(name))
		{
			rVariables.push_back(decl);
		}
	}

	
	assert(rVariables.size() == reducObjs.size() && "Failed to find the declarations of some reduction variables");
	getOCLKernelL()->getForStmt()->setReductionVariables(rVariables);
}

/*========================================================================
 *
 * Functions that are directly called by doIt() (BEGIN)
 *
 ========================================================================*/
void OpenCLReductionKernelGenerator::ReductionPreparePhase()
{
	string kernel_name = getOCLKernelL()->getForStmt()->getKernelName();
	kernel_name = kernel_name + "_reduction_step0";
	OpenCLKernelNameContainer::addKernelName(kernel_name);

	if (OCLCompilerOptions::EnableMLFeatureCollection)
	{
		Out << "#ifdef ENABLE_" << kernel_name << "\n";
	}

	Out << oclKernelPrefix() << kernel_name  << "(";
	genFirstPhaseArguments();
	Out << ", unsigned int offset, unsigned int bound";
	
	Out << ") {";

	Out << "	unsigned int i = get_global_id(0);\n";
	Out << "	if (i >= bound) return;\n";
	Out << "	i = i + offset;\n";

	for (unsigned i = 0; i < rVariables.size(); i++)
	{
		ValueDecl* d = rVariables[i];
		string type = getGlobalType(getCononicalType(d));
		
		Out << "__ocl_part_" + d->getNameAsString() << "[i] = " << initValue(type) << ";\n"; 
	}

	Out << "}\n";

	if (OCLCompilerOptions::EnableMLFeatureCollection)
	{
		Out << "#endif\n";
	}

	genMLFeatureScript(kernel_name);
}

void OpenCLReductionKernelGenerator::ReductionFirstPhase()
{
	string kernel_name = getOCLKernelL()->for_stmt->getKernelName();
	kernel_name = kernel_name + "_reduction_step1";
	OpenCLKernelNameContainer::addKernelName(kernel_name);
	vector<string> gpu_params;

	if (OCLCompilerOptions::EnableMLFeatureCollection)
	{
		Out << "#ifdef ENABLE_" << kernel_name << "\n";
	}

	genKernelBodyInfo();

	Out << oclKernelPrefix() << kernel_name << "(";

	genKernelArguments(gpu_params);
	Out << ",";
	genFirstPhaseArguments();

	Out << ") {\n";

	declareVars();
	DeclPrivateVars4GlobalObjs();
	declareReductionVars();
	Out << "\n";
	
	genCommonRoutine();
	genKernelBody((gpu_params.size() > 0) ? true : false);

	//Generate code for writting the results back
	ReductionFirstPhaseWriteBack();

	Out << "}\n";	

	if (OCLCompilerOptions::EnableMLFeatureCollection)
	{
		Out << "#endif\n";
	}
	
	genMLFeatureScript(kernel_name);
}

void OpenCLReductionKernelGenerator::ReductionSecondPhase()
{
	string kernelName = getOCLKernelL()->getForStmt()->getKernelName() + "_reduction_step2";
	OpenCLKernelNameContainer::addKernelName(kernelName);

	if (OCLCompilerOptions::EnableMLFeatureCollection)
	{
		Out << "#ifdef ENABLE_" << kernelName << "\n";
	}

	Out << "__kernel void " << kernelName << "(";
	vector<OMPReductionObj>& reductionObjs = getOCLKernelL()->getForStmt()->getOMPFor().getReductionObjs();
	
	for (unsigned i=0; i<rVariables.size(); i++)
	{
		ValueDecl* d = rVariables[i];
		string type = getGlobalType(getCononicalType(d));
		string name = d->getNameAsString();

		if (i > 0)
			Out << ", ";

		Out << "__global "  << getOclVectorType(type,DEFAULT_VECTOR_SIZE,true) << "* input_" << name;
		Out << ", __global " << type << "* output_" << name;
	}

	Out << ") {\n";

	Out << "unsigned int tid = get_local_id(0);\n";
	Out << "unsigned int bid = get_group_id(0);\n";
	Out << "unsigned int gid = get_global_id(0);\n";
	Out << "unsigned int local_size = get_local_size(0);\n\n";

	for (unsigned i=0; i<rVariables.size(); i++)
	{
		ValueDecl* d = rVariables[i];
		string type = getGlobalType(getCononicalType(d));
		string name = d->getNameAsString();

		Out << "__local " << getOclVectorType(type,DEFAULT_VECTOR_SIZE, true) << " sdata_" << name << "[GROUP_SIZE];\n";
	}

	for (unsigned i=0; i<rVariables.size(); i++)
	{
		ValueDecl* d = rVariables[i];
		string name = d->getNameAsString();
		Out << "sdata_" << name << "[tid] = input_" << name << "[gid];\n";
	}

	Out << "barrier(CLK_LOCAL_MEM_FENCE);\n\n";

	Out << "for(unsigned int s = local_size / 2; s > 0; s >>= 1)\n";
	Out << "{\n";
	Out <<	"if(tid < s)\n";
	Out <<	"{\n";

	for (unsigned i=0; i<rVariables.size(); i++)
	{
		ValueDecl* d = rVariables[i];
		OMPReductionObj& obj = reductionObjs[i];

		string name = d->getNameAsString();
		Out << "	sdata_" << name << "[tid] " << getOpCodeFromString (obj.getOperatorCode()) << "= sdata_" << name << "[tid + s];\n";
	}
	Out << "}\n";
	Out << "barrier(CLK_LOCAL_MEM_FENCE);\n";
	Out << "}\n\n";

	Out << "if(tid == 0) {\n";

	for (unsigned i=0; i<rVariables.size(); i++)
	{
		ValueDecl* d = rVariables[i];
		OMPReductionObj& obj = reductionObjs[i];
		string name = d->getNameAsString();
		string sdataName = "sdata_" + name + "[0]";

		Out << "output_" << name << "[bid] = " << reductVectorType2Scalar(sdataName, getOpCodeFromString (obj.getOperatorCode()), DEFAULT_VECTOR_SIZE) << ";\n";
	}

	Out << "}\n";

	Out << "}\n\n";

	if (OCLCompilerOptions::EnableMLFeatureCollection)
	{
		Out << "#endif\n";
	}

	genMLFeatureScript(kernelName);
}
/*========================================================================
 *
 * Functions that are directly called by doIt() (END)
 *
 ========================================================================*/

void OpenCLReductionKernelGenerator::doIt()
{
	std::string kernel_name = OpenCLKernelName::getOpenCLKernelName(getOCLKernelL()->func);
	getOCLKernelL()->for_stmt->setKernelName(kernel_name);

	collectReducVarDecls();
	ReductionPreparePhase();
	ReductionFirstPhase();
	ReductionSecondPhase();
}
