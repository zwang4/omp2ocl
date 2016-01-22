#ifndef __OPENCL_KERNEL_CODE_GENERATION_H__
#define __OPENCL_KERNEL_CODE_GENERATION_H__

#include "clang/AST/Decl.h"
#include "clang/AST/ASTContext.h"
#include "clang/Omp2Ocl/OpenCLKernel.h"
#include "clang/Omp2Ocl/OpenCLKernelLoop.h"
#include "clang/Omp2Ocl/OpenCLGlobalInfoContainer.h"
#include "clang/Omp2Ocl/OpenCLDumpMLFeature.h"
#include <vector>

using namespace clang;
using namespace std;

namespace clang
{
	class OpenCLKernelCodeGeneration
	{
		class GlobalMemoryObj
		{
			string type;
			string gtype;
			string name;
			unsigned dim;
			bool is_tgbf;
			public:
				GlobalMemoryObj(string name, string type, string gtype, unsigned dim, bool is_threadglobal)
				{
					this->type = type;
					this->name = name;
					this->gtype = gtype;
					this->dim = dim;	
					this->is_tgbf = is_threadglobal;
				}

				unsigned getDim() { return dim; }
				string getType() { return type; }
				string getName() { return name; }
				string getGType() { return gtype; }
				bool isTGBF() { return is_tgbf; } 
		};

		vector<ValueDecl*> genProto4RenamedFunc(FunctionDecl* D, RenamedFuncInfo& r, bool sc=false, bool recordTLSInput=false);
		void genDef4RenamedFunc(vector<FunctionDecl*>& candidateFuncs, vector<RenamedFuncInfo>& rnFuncs);
		FunctionDecl* pickFuncDeclByName(vector<FunctionDecl*>& candidateFuncs, string name);
		void genRenamedFuncProto(vector<FunctionDecl*>& candFuncs, vector<RenamedFuncInfo>& rnFuncs);
		void genFuncProto(FunctionDecl* D);
		void printGLInputParam(GlobalMemoryObj& g, string qualifier);

		void TLSCheckRoutines(unsigned int d);
		vector<ValueDecl*> funcLevelGPUTLSVars(FunctionDecl* D, RenamedFuncInfo& r, vector<ValueDecl*>& globalMemoryObjs, vector<unsigned>& globalMemObjIndexs, bool b);

		void optimisation();
		void postOptimisation();
		void genAMDMacros();
		void genNVIDIAMacros();
		void genArchDepHeaders();
		void genMacros();
		void genDataStructures();
		void genFuncs();
		void genKernels();
		void calculateThreadId();
		void genSpecRead(string type);
		void genSpecWrite(string type);

		llvm::raw_ostream& Out;
		ASTContext& Context;
		vector<OpenCLKernelLoop*> oclLoops;
		string kernelFile;
		OpenCLDumpMLFeature *pDMF;

	public:
		OpenCLKernelCodeGeneration(llvm::raw_ostream& O, ASTContext& Ctx, string KF) : Out(O), Context(Ctx), kernelFile(KF)
		{
			oclLoops = OpenCLGlobalInfoContainer::getOclLoops();
			pDMF = NULL;
		}

		virtual void doIt();
	};
}

#endif

