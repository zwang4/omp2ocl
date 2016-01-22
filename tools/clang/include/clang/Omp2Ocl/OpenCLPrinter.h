#ifndef __OPENCLPRINTER_H__
#define __OPENCLPRINTER_H__
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/PrettyPrinter.h"
#include "llvm/Support/Format.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include <stack>
#include "clang/Omp2Ocl/OpenCLGlobalInfoContainer.h"
#include "clang/Omp2Ocl/OpenCLCompilerOptions.h"

using namespace clang;

namespace  clang{
	class OpenCLKernelLoop;

	class OpenCLPrinter : public StmtVisitor<OpenCLPrinter> {
		llvm::raw_ostream &OS;
		ASTContext &Context;
		unsigned IndentLevel;
		clang::PrinterHelper* Helper;
		PrintingPolicy Policy;
		//OpenCLKernelLoop* curLoop;

		bool isMetArraySub;
		bool isCapturedArray;
		bool isGlobalArray;
		Decl* arrayDecl;
		vector<OCLGlobalMemVar> globalMemoryVariables;
		bool bReduction;
		vector<OMPReductionObj> reductionObjs;
		vector<ValueDecl*> gWRBufs;
		vector<ValueDecl*> gWRLCBufs;

		void metArray() { 
			resetArray();
			isMetArraySub = true; 
		}
		void resetArray() { isMetArraySub = false; isCapturedArray= false; isGlobalArray=false;}
		void catpuredArrayDecl() { isCapturedArray=true; }
		void PrintVLoadDeclareInfor(CompoundStmt* Node);
		void ReductionPrefix(ForStmt* Node);
		void GenerateLoadCode(vector<ArrayIndex>& I, unsigned arraySize, unsigned idx, unsigned width, string declareName, string gtype);
		void PrintLoadCode(CompoundStmt* Node);
		bool hasVLoadString(ArraySubscriptExpr* Node, ValueDecl* D, string& replaceString);

		vector<OMPThreadPrivateObject> threadPrivates;
		stack<string> arrayTokens;
		stack<DeclRefExpr*> arrayDecls;
		stack<unsigned> arrayDims;
		bool useOrgiCallExpr;
		stack<CompoundStmt*> CNode;

		OpenCLKernelLoop* pKL;

		//TLS Support
		static vector<OpenCLTLSBufferAccess> tls_access;
		vector<OpenCLTLSBufferAccess> act_tls_access;
		void addTLSAccessObj(ArraySubscriptExpr *Node, DeclRefExpr* t, string access_seq, bool isWrite);
		bool isAGlobalWriteBuf(string name);
		stack<bool> tls_write_stack;
		bool tls_write;
		bool has_tls_ops;
		bool track_tls_access;
		bool shouldTLSTrack(string name, string access, unsigned long long s);

		unsigned long long statement_version;

		bool isDeclRefInNDRange(DeclRefExpr* e);
		bool track_all_write_bufs;
		bool OclSpecLoad(ArraySubscriptExpr* Node);
		bool OclSpecWrite(ArraySubscriptExpr* Node, string name);
		bool ShouldReplaceWithSpecWrite(ArraySubscriptExpr* Node);
		bool CastToArraySubscriptExpr(Expr* e, ArraySubscriptExpr* Node);
		OpenCLPrinter* interval_visit_p;
		bool interval_visit_p_ed;
		bool spec_read_write;
		bool isInReadSet(string access);
		bool isPrintFunc;
		public:

		OpenCLPrinter(llvm::raw_ostream &os, ASTContext &C, PrinterHelper* helper,
				const PrintingPolicy &Policy,
				unsigned Indentation=0, bool isReductionLoop=false, bool useOrgiCallExpr=false)
			: OS(os), Context(C), IndentLevel(Indentation), Helper(helper), Policy(Policy)
		{
			interval_visit_p = NULL;
			interval_visit_p_ed = false;
			resetArray();
			bReduction = isReductionLoop;
			this->useOrgiCallExpr = useOrgiCallExpr;
			threadPrivates=OpenCLGlobalInfoContainer::getThreadPrivateVars();
			tls_write = false;
			has_tls_ops = false;
			statement_version = 0;
			pKL = NULL;
			track_all_write_bufs = false;
			track_tls_access = true;
			if (OCLCompilerOptions::EnableGPUTLs)
			{
				spec_read_write = true;
			}
			else
			{
				spec_read_write = false;
			}

			isPrintFunc = false;
		}
	
		~OpenCLPrinter()
		{
			if (interval_visit_p)
			{
				delete interval_visit_p;
			}
		}

		void SetIsPrintFunc(bool flag) { isPrintFunc = flag; }
		void EnableSpecReadWrite() { spec_read_write  = true;}
		void DisableSpecReadWrite() { spec_read_write  = false;}
		void EnableVisitP() {interval_visit_p_ed = true;} 
		void DisableVisitP() {interval_visit_p_ed = false;} 
		void DisableTLSAccessTrack() { track_tls_access = false;}
		void GPUTLsTrackStmts();
		void OCLTLsTrackStmts(bool b);
		void setOpenCLKernelLoop(OpenCLKernelLoop* l) { pKL = l; }
		void printFuncBody(Stmt* body, bool printBrackets=true, bool track_all=false);
		
		vector<OpenCLTLSBufferAccess> getActTLSVec() { 
			return act_tls_access; 
		}

		void setReductionObj(vector<OMPReductionObj>& reduObjs)
		{
			reductionObjs = reduObjs;
		}

		bool hasTLSOps()
		{
			return has_tls_ops;
		}

#if 1
		//Whether this is a thread private variable and is passed as a __global memory object
		bool isAGMThreadPrivateVariable(string& name)
		{
			for (vector<OMPThreadPrivateObject>::iterator iter = threadPrivates.begin(); iter != threadPrivates.end(); iter++)
			{
				if (iter->getVariable() == name)
				{
					return iter->isUseGlobalMem(); 
				}
			}

			return false;
		}

		bool isAThreadPrivateVariable(string& name)
		{
			for (vector<OMPThreadPrivateObject>::iterator iter = threadPrivates.begin(); iter != threadPrivates.end(); iter++)
			{
				if (iter->getVariable() == name)
					return true;
			}

			return false;
		}

		//Whther the variable is a thread private buffer and is stored
		//in the global memory
		bool isAGTPVariable(string name)
		{
			for (unsigned i=0; i<globalMemoryVariables.size(); i++)
			{
				if (globalMemoryVariables[i].getNameAsString() == name)
				{
					return globalMemoryVariables[i].isGlobalThreadPrivate;			
				}
			}

			return false;
		}

		//Whether a varialbe is a global variable
		bool isAGlobalMemoryVariable(string name)
		{
			if (isAGMThreadPrivateVariable(name) && !isAGMThreadPrivateVariable(name))
			{
				return false;
			}

			if  (isAGMThreadPrivateVariable(name))
				return true;

			for (vector<OCLGlobalMemVar>::iterator iter = globalMemoryVariables.begin(); iter != globalMemoryVariables.end(); iter++)
			{
				if (name == iter->getNameAsString())
				{
					if (iter->isAThreadPrivateVar())
					{
						if (iter->isGlobalTPBuf())
							return true;
						else
						{
							return false;	
						}
					}

					return true;
				}
			}

			return false;
		}

		bool isAGlobalMemoryVariable(DeclRefExpr* v)
		{
			std::string name = v->getNameInfo().getAsString();
			return isAGlobalMemoryVariable(name);
		}

		void setGlobalMemoryVariables(vector<OCLGlobalMemVar>& gv)
		{
			globalMemoryVariables = gv;
		}

		void setGlobalWriteBufs(vector<ValueDecl*> gWRBufs)
		{
			this->gWRBufs = gWRBufs;
		}
		
		void setGlobalLCWriteBufs(vector<ValueDecl*> gWRBufs)
		{
			this->gWRLCBufs = gWRBufs;
		}

		void addAGlobalMemoryVariables(OCLGlobalMemVar v)
		{
			string name = v.getNameAsString();
			for (unsigned i=0; i<globalMemoryVariables.size(); i++)
			{
				if (name == globalMemoryVariables[i].getNameAsString())
				{
					//update it
					globalMemoryVariables[i] = v;
					return;
				}
			}

			globalMemoryVariables.push_back(v);
		}

		void setGlobalMemoryVariables(vector<FuncProtoExt>& gvs)
		{
			for (unsigned i=0; i<gvs.size(); i++)
			{
				globalMemoryVariables.push_back(OCLGlobalMemVar(gvs[i]));
			}
		}
#endif
		void PrintStmt(Stmt *S) {
			PrintStmt(S, 0);
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

			statement_version++;
			IndentLevel -= SubIndent;

			//tls_access.clear();
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
			StmtVisitor<OpenCLPrinter>::Visit(S);
		}

		void VisitStmt(Stmt *Node) LLVM_ATTRIBUTE_UNUSED {
			Indent() << "<<unknown stmt type>>\n";
		}
		void VisitExpr(Expr *Node) LLVM_ATTRIBUTE_UNUSED {
			OS << "<<unknown expr type>>";
		}
		void VisitCXXNamedCastExpr(CXXNamedCastExpr *Node);
		void VisitForHeader(ForStmt *Node);
#define ABSTRACT_STMT(CLASS)
#define STMT(CLASS, PARENT) \
		void Visit##CLASS(CLASS *Node);
#include "clang/AST/StmtNodes.inc"
	};
}


#endif
