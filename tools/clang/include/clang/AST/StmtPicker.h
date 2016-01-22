#ifndef __STMTPICKER_H__
#define __STMTPICKER_H__

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
	class ArrayInfo
	{
		public:
			DeclRefExpr* dn;
			vector<ArraySubscriptExpr*> subscriptions;
	};

	class ReductionScope
	{
	public:
		ForStmt* f;
		OMPReductionObj obj;
		
		ReductionScope(ForStmt* ft, OMPReductionObj& o)
		{
			f = ft;
			obj = o;
		}
	};

	class StmtPicker : public StmtVisitor<StmtPicker> {
		llvm::raw_ostream &OS;
		ASTContext &Context;
		unsigned IndentLevel;
		clang::PrinterHelper* Helper;
		PrintingPolicy Policy;

		vector<DeclRefExpr*> decls;
		vector<ArraySubscriptExpr*> arrays;
		vector<ArrayInfo*> aInfos;
		vector<OCLRWSet> rwS;
		vector<ReductionScope> rS;
		vector<OMPReductionObj> reducObjs;
		vector<ForStmt*> forStmts;
		vector<Stmt*> Stmts;
		vector<CallExpr*> callExprs;
		vector<BinaryOperator*> binOPs;

		bool isInArraySub;
		bool captureTheArrayName;
		bool writeLeft;
		bool catchedReduction;

		string reducVariable;
		ForStmt* currentLoop;
		ForStmt* prevLoop;

		ArrayInfo* ai;
	
		void searchReductionVars(BinaryOperator *Node, bool& setHere);

		void addRWSet(DeclRefExpr* r, bool write)
		{
			string name = r->getNameInfo().getAsString();

			for (vector<OCLRWSet>::iterator iter = rwS.begin(); iter != rwS.end(); iter++)
			{
				if (iter->decl->getNameAsString() == name)
				{
					if (!iter->isWrite && write)
					{
						iter->isWrite = write;
						return;
					}
				}
			}

			OCLRWSet od(r->getDecl(), write);
			rwS.push_back(od);
		}

		//Get the corresponding op code of the specific Token::Kind
		string getCorOpCode(string& kind)
		{
			string result;
			if (kind == "plus")
			{
				result = "+";	
			}
			else
			{
				cerr << "Unsupport reduction operator code: " << kind << endl;
			        exit(-1);	
			}

			return result;
		}

		bool skOCL;

		public:

		StmtPicker(llvm::raw_ostream &os, ASTContext &C, PrinterHelper* helper,
				const PrintingPolicy &Policy,
				unsigned Indentation = 0, bool skipOCLLoop=false)
			: OS(os), Context(C), IndentLevel(Indentation), Helper(helper), Policy(Policy)
		{
			writeLeft = false;
			isInArraySub = false;
			captureTheArrayName = false;
			ai = NULL;
			currentLoop = NULL;
			prevLoop = NULL;
			catchedReduction = false;
			skOCL = skipOCLLoop;
			if (skOCL)
			cerr << "SET TO TRUE" << endl;
		}

		vector<ForStmt*> getForStmts()
		{
			return forStmts;
		}

		vector<CallExpr*>& getCallExprs()
		{
			return callExprs;
		}

		vector<BinaryOperator*> getBinOps()
		{
			return binOPs;
		}

		void clearBinOps()
		{
			binOPs.clear();
		}

		//Reduction Variables
		void setReductionVariables(vector<OMPReductionObj>& rS)
		{
			this->reducObjs = rS;
		}

		//Get the reduction scope
		vector<ReductionScope>& getReductionScope() { 
			return rS; 
		}

		//Is this a reduction variable
		bool isAReductionVar(string name)
		{
			for (vector<OMPReductionObj>::iterator iter=reducObjs.begin(); iter != reducObjs.end(); iter++)
			{
				if (name == iter->getVariable())
				{
					return true;
				}
			}

			return false;	
		}

		//Is this statement for a reduction variable with a particular op code
		bool isAReductionOp(string name, string opCode, OMPReductionObj& obj)
		{
			for (vector<OMPReductionObj>::iterator iter=reducObjs.begin(); iter != reducObjs.end(); iter++)
			{
				if (name == iter->getVariable())
				{
					string op = getCorOpCode(iter->getOperatorCode());
					if (opCode == op)
					{
						obj.setVariable(name);
						obj.kind = iter->kind;

						return true;
					}
				}
			}

			return false;	
		}

		//Get the read-write set
		vector<OCLRWSet> getRWS() { return rwS; }

		void PrintStmt(Stmt *S) {
			PrintStmt(S, 0);
		}

		vector<ArrayInfo*>& getArrayInfo()
		{
			return aInfos;
		}

		vector<DeclRefExpr*>& getDecl() { 
			return decls; 
		}

		DeclRefExpr* getFirstDecl()
		{
			if (decls.size())
			{
				return decls[0];
			}

			return NULL;
		}

		vector<ArraySubscriptExpr*>& getArraySubExpr() { 
			return arrays; 
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

		void VisitForStmtHeader(ForStmt *Node);
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
			Stmts.push_back(S);
			StmtVisitor<StmtPicker>::Visit(S);
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
