#ifndef __GLOBALCALLARGPICKER_H__
#define __GLOBALCALLARGPICKER_H__
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/PrettyPrinter.h"
#include "llvm/Support/Format.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/Omp2Ocl/OpenCLKernel.h"
#include <iostream>

using namespace std;
using namespace clang;

namespace clang{

	class GlobalCallArgPicker : public StmtVisitor<GlobalCallArgPicker> {
		llvm::raw_ostream &OS;
		ASTContext &Context;
		unsigned IndentLevel;
		clang::PrinterHelper* Helper;
		PrintingPolicy Policy;
		vector<OCLGlobalMemVar>& gv;
		vector<OMPThreadPrivateObject> threadPrivates;

		vector<CallArgInfoContainer*> calledFuncs;
		bool inCallArg;
		bool inArray;

		CallExpr* cce;
		DeclRefExpr* currentArray;
		CallArgInfoContainer* ca;
		CallArgInfo cai;

		vector<string> collectArraySubs(ArraySubscriptExpr* a);

		public:

		GlobalCallArgPicker(llvm::raw_ostream &os, ASTContext &C, PrinterHelper* helper,
				const PrintingPolicy &Policy, vector<OCLGlobalMemVar>& GV, vector<OMPThreadPrivateObject>& t, 
				unsigned Indentation = 0)
			: OS(os), Context(C), IndentLevel(Indentation), Helper(helper), Policy(Policy), gv(GV), threadPrivates(t), cai(false)
		{
			inCallArg = false;
			cce = NULL;
			inArray = false;
			currentArray = false;
			ca = NULL;
		}

		void PrintStmt(Stmt *S) {
			PrintStmt(S, 0);
		}

		void enableCallArgTrack()
		{
			inCallArg = true;
		}

		void disableCallArgTrack()
		{
			inCallArg = false;
			cce = NULL;
		}

		//Whther this thread private variable is 
		bool isAGMThreadPrivateVariable(string name)
		{
			for (vector<OMPThreadPrivateObject>::iterator iter = threadPrivates.begin(); iter != threadPrivates.end(); iter++)
			{
				if (iter->getVariable() == name)
				{
					return (iter->isUseGlobalMem());
				}
			}

			return false;
		}
	
		bool isAThreadPrivateVariable(string& name)
		{
			for (vector<OMPThreadPrivateObject>::iterator iter = threadPrivates.begin(); iter != threadPrivates.end(); iter++)
			{
				if (iter->getVariable() == name)
				{
					return true;
				}
			}

			return false;
		}

		//Whthere this is a threadprivate variable and is stored in the 
		//global memory
		bool isAGTPMemVar(string name)
		{
			if (isAGMThreadPrivateVariable(name))
			{
				return true;
			}

			for (unsigned i=0; i<gv.size(); i++)
			{
				if (name == gv[i].getNameAsString())
				{
					return gv[i].isGlobalThreadPrivate;
				}
			}

			return false;
		}

		bool isAGlobalMemory(DeclRefExpr* node)
		{
			string name = node->getNameInfo().getAsString();
		
			if (isAThreadPrivateVariable(name) && !isAGMThreadPrivateVariable(name))
			{
				return false;
			}

			for (unsigned i=0; i<gv.size(); i++)
			{
				if (name == gv[i].getNameAsString())
				{
					return true;
				}
			}

			return false;
		}

		vector<CallArgInfoContainer*>& getCalledFuncs() { return calledFuncs; }

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

		void addGlobalVariable(DeclRefExpr* expr);
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
			StmtVisitor<GlobalCallArgPicker>::Visit(S);
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
