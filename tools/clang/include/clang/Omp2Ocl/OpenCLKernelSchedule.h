#ifndef __OPENCLKERNELSCHEDULE_H__
#define __OPENCLKERNELSCHEDULE_H__
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/PrettyPrinter.h"
#include "llvm/Support/Format.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/Omp2Ocl/OpenCLPrinter.h"
#include "clang/Omp2Ocl/OpenCLNDRangeVarPicker.h"
#include "clang/AST/DeclPrinter.h"
#include "clang/AST/CallArgReviseAction.h"
#include "clang/AST/GlobalCallArgPicker.h"
#include "clang/AST/StmtPicker.h"
#include "clang/Omp2Ocl/OpenCLKernel.h"
#include "clang/Omp2Ocl/OpenCLKernelLoop.h"
#include "clang/Omp2Ocl/OpenCLDumpMLFeature.h"

#include <fstream>
#include<iostream>
#include<string>
#include<stdio.h>
#include <vector>

using namespace clang;
using namespace std;

namespace clang{
	class OpenCLKernelSchedule {
		private:
			unsigned IndentLevel;
			//std::fstream* fOpenCL;
			llvm::raw_fd_ostream* fOpenCL;
			static vector<OpenCLKernelLoop*> openclLoops;
			static vector<DeclRefExpr*> collectedFunctions;
			static vector<string> oclKernelNames;

			ASTContext& Context;
			vector<RenamedFuncInfo> RenamedFuncs;

			//Statically declared private variables
			OMPThreadPrivate otp;
			OpenCLKernelLoop* curLoop;

			bool isAnOpenCLOpenCLNDRangeVar(OpenCLKernelLoop* curLoop, string& name);

			bool isPerfectNestLoop(ForStmt* loop);
			void ReductionKernelExitRoutine(OpenCLKernelLoop* curLoop);
			void GeneratePrivateVariables(OpenCLKernelLoop* loop);
			LoopIndex* getLoopIndex(ForStmt* Node);
			Stmt* ScheduleLoops(OpenCLKernelLoop* curLoop, vector<SwapLoopInfo>& innerLoops, bool& swaped);
			void CollectGlobalInputParameters(OpenCLKernelLoop* curLoop);
			void _generateKernel(OpenCLKernelLoop* curLoop, vector<FunctionDecl*>&  funcsNeed2Revised);
			string _generateCommandRoutine();
			void genrerateCalledFunctions(llvm::raw_ostream& O, vector<FunctionDecl*>& functions, vector<RenamedFuncInfo>& RenamedFuncs);	
			void collectCallees(vector<DeclRefExpr*>& functions, vector<FunctionDecl*>& functionDefs);
			void generateFuncPrototype(llvm::raw_ostream& Out, FunctionDecl* D);
			bool holistOpenCLNDRangeVarInFunction(FunctionDecl* func);
			void reviseFunctionWithOpenCLNDRangeVar(FunctionDecl* D, vector<DeclRefExpr*> globalVariables);
			bool findCall2GlobalBuffer(Stmt* Body, vector<OCLGlobalMemVar>& globalMemoryVariables, vector<OMPThreadPrivateObject>& threadPrivates, vector<FunctionDecl*>& funcsNeed2Revised);
			void reviseCalledArgs(vector<FunctionDecl*>& functionDefs, Stmt* E, OpenCLKernelLoop* topF);
			bool isExpVariableAlreadyInParameterList(string name, vector<string>& ParamList);
			string getReNameFuncName(CallArgInfoContainer* cArg, vector<FunctionDecl*>& funcsNeed2Revised);
			void scanLoop(OpenCLKernelLoop* loop, vector<FunctionDecl*>& funcsNeed2Revised);
			vector<FunctionDecl*> collectCandidateFunc(vector<FunctionDecl*>& functionDefs);
			vector<FunctionDecl*> findExpendedFunc(vector<FunctionDecl*>& candidateFuncs);
			vector<FunctionDecl*> generateFuncRoutines(string& buf);
			void generatePrototypeForRenamedFunc(llvm::raw_ostream& Out, vector<FunctionDecl*>& candidateFuncs, vector<RenamedFuncInfo>& rnFuncs);
			void generateDef4RenamedFunc(llvm::raw_ostream& Out, vector<FunctionDecl*>& candidateFuncs, vector<RenamedFuncInfo>& rnFuncs);
			FunctionDecl* PickFuncDeclByName(vector<FunctionDecl*>& candidateFuncs, string name);
			string genProtoType4RenamedFunc(FunctionDecl* D, RenamedFuncInfo& r);
			void ScanThreadPrivate(OpenCLKernelLoop* loop);
			void _generateMultIterLoopHeader(llvm::raw_ostream& Out, OpenCLKernelLoop* loop);

			//Non primitive structure
			void ScanNonPrimitiveType(OpenCLKernelLoop* loop, vector<DeclRefExpr*>& decls);
			TypedefDecl* getTypeDefRef(string ty);

			ForStmt* LoopInCContext;
			bool isTArraySubExpr;
			bool isMetTheArrayDecl;
			vector<RecordDecl*>& recordDecls;
			vector<RecordDecl*> usedRDs;
			vector<QualType> qtypes;
			vector<TypedefDecl*>& typeDefs;
			vector<TypedefDecl*> usedDefs;
			string KernelFile;
			OpenCLDumpMLFeature* pDMF; //For extracting machine learning features

		public:
			OpenCLKernelSchedule(ASTContext& C, OMPThreadPrivate ot, string KF) : Context(C), otp(ot), recordDecls(DeclPrinter::recordDecls), typeDefs(DeclPrinter::typeDefs), KernelFile(KF)
		{
			NullCurLoop();	
			isTArraySubExpr = false;
			pDMF = new OpenCLDumpMLFeature(KF);
		}


			OpenCLKernelSchedule(ASTContext& C) : Context(C), recordDecls(DeclPrinter::recordDecls), typeDefs(DeclPrinter::typeDefs)
		{
			NullCurLoop();	
			isTArraySubExpr = false;
		}

			void GenerateOpenCLLoopKernel();


			static vector<OpenCLKernelLoop*>& getOpenCLKernelLoops()
			{
				return openclLoops;	
			}

			OpenCLKernelLoop* getCurLoop() { 
				return curLoop; 
			}

			void addCurLoop()
			{
				openclLoops.push_back(curLoop);
			}

			void NullCurLoop() { 
				curLoop = NULL; 
				LoopInCContext=NULL; 
			} 

			void setCContextLoop(ForStmt* for_stmt) { 
				LoopInCContext = for_stmt; 
			}

			ForStmt* getCContextLoop() { 
				return LoopInCContext; 
			}

			void initArraySubRecord(ArraySubscriptExpr* Node);
			void forLoopEndRoutine();
			void recordArraySub(DeclRefExpr* e);
			void setArrayBase(DeclRefExpr* base);
			DeclRefExpr* getArrayBase();
			void enableTrackingArraySubExpr(); 
			void metArrayDecl(DeclRefExpr* expr); 
			void addCollectedFunction(DeclRefExpr* expr);

			bool isMetArrayDecl() { return isMetTheArrayDecl; }
			void disableTrackingArraySubExpr(); 
			bool isTrackingArraySubExpr() { return isTArraySubExpr; }

			//OpenCL related functions
			void newOpenCLCurrentLoop(ForStmt* forNode, const FunctionDecl* func); 
			void setOpenCLOut(llvm::raw_fd_ostream* opencl);
			string OpenCLIndent(unsigned int level=0);
			unsigned int getLineNumber(SourceLocation Loc);
			const char* getFileName(SourceLocation Loc);
			static vector<OpenCLKernelLoop*> getOpenCLLoops() { return openclLoops; }
			static vector<string> getOCLKernelNames() { return oclKernelNames; }
	};
}

#endif

