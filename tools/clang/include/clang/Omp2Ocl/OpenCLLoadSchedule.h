#ifndef __OPENCLLOADSCHEDULE_H__
#define __OPENCLLOADSCHEDULE_H__

/*!
 * This schedules loads of __global variables so that
 * we can vectorise the load
 *
 *
 */

#include <iostream>
#include <vector>
#include <stack>
#include <list>
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/Omp2Ocl/OpenCLKernel.h"
#include "clang/Omp2Ocl/OpenCLRoutine.h"
#include "clang/Omp2Ocl/OpenCLCommon.h"
#include "clang/Omp2Ocl/OpenCLKernelLoop.h"
#include "clang/Omp2Ocl/OpenCLCodeOptimisation.h" 
#include "clang/Omp2Ocl/OpenCLRoutine.h"
#include "clang/AST/StmtPicker.h"

using namespace std;
using namespace clang;

namespace clang
{

	class OpenCLLoadSchedule : public StmtVisitor<OpenCLLoadSchedule>, OpenCLCodeOptimisation
	{
		void scanCandidates(Stmt* Kernel);
		arrayBaseInfo getArrayBaseInfo(ArraySubscriptExpr* Node);
		bool isReadOnly(DeclRefExpr* e);
		bool shouldPerformLoadOpt(DeclRefExpr* e);
		void printArrayInfo(ArrayIndex& ai);	
		void addSortByNameArrayAccess(ArrayIndex& ai);
		void sortArrayAccessByName();
		void scheduleLoads();
		void sort(vector<ArrayIndex>& AIs);
		bool hasFunctionCall(ArrayIndex& A);
		int isInArrayIndexs(vector<ArrayIndex>& PAIs, ArrayIndex& A);
		bool isTwoAccessStrIdentical(ArrayIndex& l, ArrayIndex& r);
		int  conSequence(vector<ArrayIndex>& AIs, unsigned i);
		bool isCont(ArrayIndex& first, ArrayIndex& second);
		int cont(const IndexStr& lhs, const IndexStr& rhs, int idx, int& borrow);
		void _vectorLoad(vector<ArrayIndex>& AIs);
		void VectoriseLoads(CompoundStmt *Node);
		bool isOneStepInc(vector<BinaryOperator*> lOps, vector<BinaryOperator*> rOps);
		bool getIntOp(BinaryOperator* bop, int& value);
		void performVLoad(vector<ArrayIndex>& AIs, unsigned begin, unsigned end);
		OCLCompoundVLoadDeclareInfo genVLoadInsts(ArrayIndex& A, unsigned begin, 
									unsigned vs, vector<ArrayIndex>& AIs, bool& useAble);
		void addVLoadInfo(OCLCompoundVLoadDeclareInfo v);

		private:
		string str_buf;
		llvm::raw_string_ostream OS;
		ASTContext& Context;
		unsigned IndentLevel;
		clang::PrinterHelper* Helper;
		PrintingPolicy Policy;
		vector<OCLRWSet> rwS;
		StmtPicker sp;
		stack<DeclRefExpr*> arrayDecls;
		stack<ArrayIndex> arrayInx;
		stack<vector<ArrayIndex> > arrayInfos;
		vector<ArrayAccessInfo> arrayAccessInfo;	
		vector<OCLCompoundVLoadDeclareInfo> CVIs;		

		public:
		OpenCLLoadSchedule(ASTContext& Ctx, OpenCLKernelLoop*l)
			: OpenCLCodeOptimisation(Ctx, l), OS(str_buf), Context(Ctx), Policy(Ctx.PrintingPolicy), sp(llvm::nulls(), Ctx, NULL, Ctx.PrintingPolicy)
		{
			Helper = NULL;	
			IndentLevel = 0;
		}


		void doIt();


		/////////////////////////////////////////////////////////////////////////
		//
		// Visiting funcs
		//
		/////////////////////////////////////////////////////////////////////////	
		void Visit(Stmt* S) {
			StmtVisitor<OpenCLLoadSchedule>::Visit(S);
		}
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
