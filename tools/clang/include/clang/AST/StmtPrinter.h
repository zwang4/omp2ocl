#ifndef __STMTPRINTER_H__
#define __STMTPRINTER_H__

#include "clang/AST/StmtVisitor.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/PrettyPrinter.h"
#include "llvm/Support/Format.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/Omp2Ocl/OpenCLKernelSchedule.h"
#include "clang/Basic/SourceManager.h"

#include <fstream>
#include<iostream>
#include<string>
#include<stdio.h>
#include <vector>

using namespace clang;
using namespace std;

namespace  clang {
	class OpenCLKernelSchedule;

	class StmtPrinter : public StmtVisitor<StmtPrinter> {
		llvm::raw_ostream &OS;
		llvm::raw_fd_ostream* fOpenCL;
		ASTContext &Context;
		unsigned IndentLevel;
		clang::PrinterHelper* Helper;
		PrintingPolicy Policy;
		bool isInParallelLoop;
		OpenCLKernelSchedule* ops;
			
		public:
		static bool isCollectedCallee;

		StmtPrinter(llvm::raw_ostream &os, ASTContext &C, PrinterHelper* helper,
				const PrintingPolicy &Policy,
				unsigned Indentation = 0);
		~StmtPrinter();

		void newOpenCLCurrentLoop(ForStmt* forNode); 
		void PrintStmt(Stmt *S);
		void PrintStmt(Stmt *S, int SubIndent); 
		void PrintRawCompoundStmt(CompoundStmt *S);
		void PrintRawDecl(Decl *D);
		void PrintRawDeclStmt(DeclStmt *S);
		void PrintRawIfStmt(IfStmt *If);
		void PrintRawCXXCatchStmt(CXXCatchStmt *Catch);
		void PrintCallArgs(CallExpr *E);
		void PrintExpr(Expr *E);

		llvm::raw_ostream &Indent(int Delta = 0) {
			for (int i = 0, e = IndentLevel+Delta; i < e; ++i)
				OS << "  ";
			return OS;
		}

		void Visit(Stmt* S) {
			if (Helper && Helper->handledStmt(S,OS))
				return;
			else StmtVisitor<StmtPrinter>::Visit(S);
		}

		void VisitStmt(Stmt *Node) LLVM_ATTRIBUTE_UNUSED {
			Indent() << "<<unknown stmt type>>\n";
		}
		void VisitExpr(Expr *Node) LLVM_ATTRIBUTE_UNUSED {
			OS << "<<unknown expr type>>";
		}
		void VisitCXXNamedCastExpr(CXXNamedCastExpr *Node);

#define ABSTRACT_STMT(CLASS)
#define STMT(CLASS, PARENT) \
		void Visit##CLASS(CLASS *Node);
#include "clang/AST/StmtNodes.inc"
	};
}

#endif
