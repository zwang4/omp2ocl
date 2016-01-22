#ifndef __OPENCL_KERNEL_CODE_H__
#define __OPENCL_KERNEL_CODE_H__

#include "clang/AST/ASTContext.h"
#include "clang/Lex/Pragma.h"
#include "clang/AST/DeclPrinter.h"

using namespace clang;

namespace clang
{
	class OpenCLKernelCode
	{
	private:
		llvm::raw_ostream& Out;
		ASTContext& Context;
		OMPThreadPrivate& otp;
		vector<RecordDecl*>& recordDecls;
		vector<TypedefDecl*>& typeDefs;
		string KernelFile;
	
		void processOpts();
	public:
		OpenCLKernelCode(llvm::raw_ostream& O, ASTContext& C, OMPThreadPrivate ot, string KF)
		: Out(O), Context(C), otp(ot), recordDecls(DeclPrinter::recordDecls), typeDefs(DeclPrinter::typeDefs),
		  KernelFile(KF)
		{
			
		}

		virtual void optimisation();
		virtual void codeGeneration();
		virtual void doIt();
	};
}

#endif
