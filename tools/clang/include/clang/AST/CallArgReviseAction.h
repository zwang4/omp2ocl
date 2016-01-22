#ifndef __CALLARGREVISEACTION_H__
#define __CALLARGREVISEACTION_H__
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/PrettyPrinter.h"
#include "llvm/Support/Format.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
using namespace clang;

namespace  clang{
	class CallArgReviseAction : public StmtVisitor<CallArgReviseAction> {
		llvm::raw_ostream &OS;
		ASTContext &Context;
		unsigned IndentLevel;
		clang::PrinterHelper* Helper;
		PrintingPolicy Policy;

		vector<FunctionDecl*>& revisedFuncs;
		FunctionDecl* isInRevisedFunctions(DeclRefExpr* expr);
		void reviseCallArgs(CallExpr *Call, vector<FuncProtoExt> addedVariables);
		vector<DeclRefExpr*> expVariables;

		public:

		CallArgReviseAction(llvm::raw_ostream &os, ASTContext &C, PrinterHelper* helper,
				const PrintingPolicy &Policy,
				unsigned Indentation, vector<FunctionDecl*>& f)
			: OS(os), Context(C), IndentLevel(Indentation), Helper(helper), Policy(Policy), revisedFuncs(f)
		{
		}

		void PrintStmt(Stmt *S) {
			PrintStmt(S, 0);
		}


		void addExpVariable(DeclRefExpr* e)
		{
			string name  = e->getNameInfo().getAsString();

			for (vector<DeclRefExpr*>::iterator iter = expVariables.begin(); iter!=expVariables.end(); iter++)
			{
				if ((*iter)->getNameInfo().getAsString() == name)
					return;
			}
			expVariables.push_back(e);
		}

		vector<DeclRefExpr*>& getExpVariables()
		{
			return expVariables;
		}

		void PrintStmt(Stmt *S, int SubIndent) {
			IndentLevel += SubIndent;
			if (S && isa<Expr>(S)) {
				// If this is an expr used in a stmt context, indent and newline it.
				Indent();
				Visit(S);
				OS << ";\n";
			} else if (S) {
				Visit(S);
			} else {
			}
			IndentLevel -= SubIndent;
		}

		void PrintRawCompoundStmt(CompoundStmt *S);
		void PrintRawDecl(Decl *D);
		void PrintRawDeclStmt(DeclStmt *S);
		void PrintRawIfStmt(IfStmt *If);
		void PrintRawCXXCatchStmt(CXXCatchStmt *Catch);
		void PrintCallArgs(CallExpr *E);

		void PrintExpr(Expr *E) {
			if (E)
				Visit(E);
			else
				OS << "<null expr>";
		}

		llvm::raw_ostream &Indent(int Delta = 0) {
			for (int i = 0, e = IndentLevel+Delta; i < e; ++i)
				OS << "  ";
			return OS;
		}

		void Visit(Stmt* S) {
			StmtVisitor<CallArgReviseAction>::Visit(S);
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
