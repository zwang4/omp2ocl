#ifndef __OPENCLHOSTPRINTER_H__
#define __OPENCLHOSTPRINTER_H__
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/PrettyPrinter.h"
#include "llvm/Support/Format.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/Omp2Ocl/OpenCLHostPrinter.h"
#include "clang/Omp2Ocl/OpenCLKernel.h"
#include "clang/Omp2Ocl/OpenCLCompilerOptions.h"
#include "clang/Omp2Ocl/OpenCLGPUTLSHostCode.h"

using namespace clang;

namespace  clang{
	class OpenCLHostPrinter : public StmtVisitor<OpenCLHostPrinter> {
		llvm::raw_ostream &OS;
		ASTContext &Context;
		unsigned IndentLevel;
		clang::PrinterHelper* Helper;
		PrintingPolicy Policy;
		bool traceLVar;

		//OpenCLKernelLoop* curLoop;
		unsigned int getLineNumber(SourceLocation Loc);
		const char* getFileName(SourceLocation Loc);
		void generateKernelArgu(ForStmt* Node, string kernel, OpenCLInputArgu& arg, unsigned& i, vector<OpenCLNDRangeVar> GV, string specified_name="");
		void generateWorkSize(string kernel, vector<OpenCLNDRangeVar>& GV, vector<OMPMultIterIndex>& multIterIndex, bool hasGTPV, bool shoulHandleMULT=true);
		void generateReadSet(vector<OpenCLInputArgu>& inputArgs, ForStmt* Node);
		void generateWriteSet(vector<OpenCLInputArgu>& inputArgs, ForStmt* Node);
		void generateHostWriteSet(vector<OpenCLInputArgu>& inputArgs);
		void printForSerialVersion(ForStmt* Node);
		void ReductionFirstPhase(ForStmt *Node);
		void ReductionLoop(ForStmt *Node);
		void PrintReductionBufferSize(ForStmt* Node);
		void ReductionSecondPhase(ForStmt *Node);
		void ReductionFinalCPUStage(ForStmt* Node);
		void startKernel(string kernel, vector<OpenCLNDRangeVar>& GV, bool genGWS = true);
		void generatGTPBuffer(string kernel, OpenCLInputArgu& arg, vector<OpenCLNDRangeVar> GV);
		bool hasGThPrivate(vector<OpenCLInputArgu>& inputArgs);
		void printReductionGroupBound(vector<OpenCLNDRangeVar>::iterator& iter, unsigned i);
		//void declareReductionBufferSize(ForStmt* Node);	
		void releaseGTPBuffer(string kernel, OpenCLInputArgu& arg);
		void releaseGTPBuffers(string kernel, vector<OpenCLInputArgu>& inputArgs);
		vector<VarDecl*> stmtDecls; //This is used to track the variables that are declared within a statement
		vector<FunctionLevelOCLBuffer> funcOCLBuffers;
		void declareOCLBufferForLocalVars(vector<VarDecl*>& Ds);
		void prepareOCLBuffersForLocalVars(Stmt* S);
		void declareOCLBufferForLocalVars(vector<ValueDecl*>& Ds);
		
	
		void addFunctionLevelOCLBufferObj(ValueDecl* dc);
		bool isInFunctionLevelOCLBuffer(FunctionDecl* D, ValueDecl* expr);
		bool b_KernelMacro;
		bool b_UserWSGMacro;
		bool b_ProfilingMacro;

		bool isFunctionBody;

		//GPU TLS
		OpenCLGPUTLSHostCode* gpu_tls_handler;
		void genGPUTLSBuffer(ForStmt*& Node, string& kernel, unsigned int &k);
	public:

		OpenCLHostPrinter(llvm::raw_ostream &os, ASTContext &C, PrinterHelper* helper,
				const PrintingPolicy &Policy,
				unsigned Indentation = 0, bool traceLV=true, bool isFB=false)
			: OS(os), Context(C), IndentLevel(Indentation), Helper(helper), Policy(Policy), traceLVar(traceLV), isFunctionBody(isFB)
		{
			if (OCLCompilerOptions::EnableMLFeatureCollection)
			{
				b_KernelMacro = true;
				b_UserWSGMacro = true;
				b_ProfilingMacro = true;
			}
			else
			{
				b_KernelMacro = false;
				b_UserWSGMacro = false;
				b_ProfilingMacro = false;
			}

			if (OCLCompilerOptions::EnableDebugCG)
			{
				b_KernelMacro = true;
			}

			gpu_tls_handler = NULL;
		}

		vector<FunctionLevelOCLBuffer>& getFuncLevelOCLBuffers()
		{
			return funcOCLBuffers;	
		}

		void PrintStmt(Stmt *S) {
			PrintStmt(S, 0);
		}

		void setGPUTLSHandler(OpenCLGPUTLSHostCode* handler)
		{
			gpu_tls_handler = handler;
		}

		void resetGPUTLSHandler()
		{
			gpu_tls_handler = NULL;
		}

		void PrintStmt(Stmt *S, int SubIndent) {
			IndentLevel += SubIndent;
			if (S && isa<Expr>(S)) {
				//This is a call expr, I have to declare local before a call expr
				if (isa<CallExpr>(S))
				{
					prepareOCLBuffersForLocalVars(S);
				}
				Visit(S);
				Indent();
				OS << ";\n";
				// If this is an expr used in a stmt context, indent and newline it.
				//ZHENG Declare ocl_buffers for array variables
				if (!dyn_cast<DeclStmt>(S))
				{
					prepareOCLBuffersForLocalVars(S);
				}
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
			StmtVisitor<OpenCLHostPrinter>::Visit(S);
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
