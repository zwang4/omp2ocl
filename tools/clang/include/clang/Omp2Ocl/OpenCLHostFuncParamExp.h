#ifndef __OPENCL_HOST_FUNC_PARAM_EXP_H__
#define __OPENCL_HOST_FUNC_PARAM_EXP_H__
#include "clang/Omp2Ocl/OpenCLKernel.h"
#include "clang/AST/Decl.h"
#include <vector>

using namespace clang;
using namespace std;

namespace clang
{
	class OpenCLExpFuncBuf
	{
		ValueDecl* decl;
		string bname;

		public:
			OpenCLExpFuncBuf(ValueDecl* d, string buf_name)
			{
				decl = d;
				bname = buf_name;	
			}

			string getName() { return decl->getName(); }
			string getBufName() { return bname; }
			ValueDecl* getDecl() { return decl; }
	};

	class OpenCLHostFuncParamExp {
		ASTContext& Context;
		llvm::raw_ostream& Out;
		vector<OpenCLExpFuncBuf> exp_buf_name;
		public:
			OpenCLHostFuncParamExp(ASTContext& Context, llvm::raw_ostream& Out);
			void VisitCallArg(CallExpr* Call, unsigned i);
			void VisitFunctionParam(FunctionDecl* D, const FunctionProtoType* FT);
			bool shouldCallArgPassedWithOclBuffer(CallExpr* Call, unsigned i);
			vector<OpenCLExpFuncBuf> getExpBufName() {return exp_buf_name;}
	
	};
}


#endif
