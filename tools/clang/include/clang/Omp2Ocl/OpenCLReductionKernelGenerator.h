#ifndef __OPENCL_REDUCTION_KERNELGENERATOR_H__
#define __OPENCL_REDUCTION_KERNELGENERATOR_H__

#include "clang/Omp2Ocl/OpenCLKernelCodeGenerator.h"

using namespace clang;

namespace clang
{
	class OpenCLReductionKernelGenerator : public OpenCLKernelCodeGenerator
	{
		void collectReducVarDecls();
		void declareReductionVars();
		void genFirstPhaseArguments();
		void ReductionPreparePhase();
		void ReductionFirstPhase();
		void ReductionSecondPhase();
		void ReductionFirstPhaseWriteBack();

		vector<FunctionDecl*>& funcsNeed2Revised;	
		vector<ValueDecl*> rVariables;
		
	public:
		OpenCLReductionKernelGenerator (llvm::raw_ostream& O, ASTContext& C, OpenCLKernelLoop* L, 
				vector<FunctionDecl*>& funcNR, OpenCLDumpMLFeature* pf = NULL) : 
					clang::OpenCLKernelCodeGenerator(O, C, L, pf), funcsNeed2Revised(funcNR)
		{
			
		}

		virtual void doIt();
	};
}

#endif

