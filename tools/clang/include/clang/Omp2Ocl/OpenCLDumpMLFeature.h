#ifndef __OPENCLDUMPMLFEATURE_H__
#define __OPENCLDUMPMLFEATURE_H__

#include "clang/AST/Stmt.h"
#include "clang/AST/StmtPicker.h"

using namespace std;
using namespace clang;

namespace clang
{

	class OpenCLDumpMLFeature
	{
		llvm::raw_fd_ostream* PO;
		llvm::raw_string_ostream* OS;
		string KF;
		string buf;
		vector<DeclRefExpr*> declareVars;
		vector<DeclRefExpr*> unResolveVars;

		void addUnresolveVar(DeclRefExpr* e);	
		void generateHeader(llvm::raw_ostream& Out);
		
		public:
			OpenCLDumpMLFeature(string KernelFile) : KF(KernelFile)
			{
				OS = new llvm::raw_string_ostream(buf);
			}

			~OpenCLDumpMLFeature()
			{
				flush();
				delete OS;
			}

			void generateScriptForKernel(string kernel_name, ASTContext& Context, Stmt* Kernel);
			void generateScriptForKernel(string kernel_name);
			void flush();
			void doIt(llvm::raw_ostream& Out)
			{
				generateHeader(Out);
				flush();
				Out << buf;
			}
	};
}

#endif
