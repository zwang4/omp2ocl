#ifndef __OPENCL_KERNEL_CODE_GENERATOR_H__
#define __OPENCL_KERNEL_CODE_GENERATOR_H__

/*
 * This class generates the normal OpenCL Kernels (i.e. non-reduction kernels)
 *
 */

#include "clang/Omp2Ocl/OpenCLKernelLoop.h"
#include "clang/Omp2Ocl/OpenCLDumpMLFeature.h"

using namespace clang;

namespace clang
{
	class OpenCLKernelCodeGenerator
	{
		class InputParam
		{
			ValueDecl* decl;
			string name;
			unsigned dim;
			string type;
			public:
				InputParam(ValueDecl* decl, string name, unsigned dim, string type)
				{
					this->decl = decl;
					this->name = name;
					this->dim = dim;
					this->type = type;
				}
		
				ValueDecl* getDecl() { return decl; }
				string getName() { return name; }
				unsigned getDim() { return dim; }
				string getType() { return type; }			
		};

		OpenCLKernelLoop* loop;
		ASTContext& Context;
		OpenCLDumpMLFeature* pDMF; //used for dumping ML features
		vector<ValueDecl*> _localD;		
		vector<InputParam> _globalInputParams;

		bool isAnOpenCLOpenCLNDRangeVar(string& kernel_name);

		bool shouldDeclaredAsKernelArgu(PLoopParam& P);
		void printNDRangeVars();
		void declarePrivateVariables();
		void declareCopyInBuffers();
		void genCopyInCode();
		void genMultIterLoopHeader();
		void addLocalD(ValueDecl* d);
		void declLocalVars();
		vector<OpenCLTLSBufferAccess> scanTLSVariables();
		vector<OpenCLTLSBufferAccess> act_tls;
		bool isInActTls(string name);
		void scanGlobalWriteBufs();
		public:
			llvm::raw_ostream& Out;

			OpenCLKernelCodeGenerator(llvm::raw_ostream& O, ASTContext& C, 
				OpenCLKernelLoop* L, OpenCLDumpMLFeature* pf = NULL) :	
					loop(L), Context(C), Out(O)
			{
				pDMF = pf;
			}
			
			OpenCLKernelLoop* getOCLKernelL()
			{
				return loop;
			}

			ASTContext& getContext()
			{
				return Context;	
			}

			virtual void doIt();

			unsigned int genKernelProto(string kernel_name, vector<string>& gputls_params);
			unsigned int  genKernelArguments(vector<string>& gputls_params);
			void declareVars();
			void genCommonRoutine();
			void genLocalVarCopyCode();
			void genMLFeatureScript(string kernel_name);
			string oclKernelPrefix();
			void genKernelBodyInfo();
			void genKernelBody(bool enableTLS);
			void DeclPrivateVars4GlobalObjs();

	};
}

#endif
