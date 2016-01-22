/*!
 *
 * This file generates the host-side code to load the OpenCL Kernels
 *
 */

#ifndef __OPENCLHOSTCODE_H__
#define __OPENCLHOSTCODE_H__
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ASTContext.h"
#include "llvm/Support/Format.h"
#include "clang/Omp2Ocl/OpenCLGlobalInfoContainer.h"

#include <fstream>
#include<iostream>
#include<string>
#include<stdio.h>
#include <vector>

using namespace clang;
using namespace std;

namespace clang{
	class OpenCLKernelLoop;
	class ForStmt;

	class OpenCLHostCode {
		ASTContext& Context;
		llvm::raw_fd_ostream& Out;

		//Global thread private variables
		OMPThreadPrivate otp;
		vector<OpenCLKernelLoop*> OpenCLLoops;
		vector<string> alreadyDeclaredCLMemVars;
		vector<string> oclKernelNames;
		vector<string> headFiles;
		string entryFile;
		string oclKernelFile;
		vector<OCLGlobalMemVar> globalMemoryObjs;
		vector<CopyInBuffer> copyInObjs;
		string hostFileStriped;

		string generateOpenCLMemoryObjs(OpenCLKernelLoop*  ocls, vector<string>& s);
		bool isAreadyDeclCLMemVar(string name);
		void generateHeadFiles();
		void generateInitCode();
		void declareOCLKenels(llvm::raw_ostream& Out);
		void declarMLRecordVars(llvm::raw_ostream& Out);
		void bufferCreationRoutine();
		void bufferReleaseRoutine();
		void finalBufferSync();

		void scanCopyInObjs();
		void addCopyInObj(CopyInBuffer& d);
		bool isACopyInObj(string& name);
		string generateOpenCLReductionObjs(OpenCLKernelLoop* loop);
		string generateOpenCLGTPObjs(OpenCLKernelLoop* loop);
		string releaseOpenCLGTPObjs(OpenCLKernelLoop* loop);
		string releaseOpenCLReductionObjs(OpenCLKernelLoop* loop);
		string generateFuncLevelObjs(OpenCLKernelLoop* loop, vector<string>& s);
		string releaseFuncLevelObjs(OpenCLKernelLoop* loop, vector<string>& s);

		void generateDumpProfileInfo();
		void flushOclBufers();
		string flushFuncLevelBuffer(OpenCLKernelLoop* loop, vector<string>& buffers);
		void generateOclDef();
		void MLFeatureRoutine();
		void generateGDStructs(llvm::raw_ostream& O);
		void ResetMLFeatureRoutine();
		void genGWSScript();
		void TurnMLFeatureRoutine();
		void MLRoutines();

		void generateTLSLogArrays(llvm::raw_fd_ostream& Out);
		void generateTLSLogBuffers();
		public:
		OpenCLHostCode(ASTContext& C, llvm::raw_fd_ostream& O, OMPThreadPrivate& ot, 
				vector<string> ock, vector<string>& hf, string entryF,string oclKernel, string hostS) : 
			Context(C), Out(O), otp(ot), OpenCLLoops(OpenCLGlobalInfoContainer::getOclLoops()), oclKernelNames(ock), headFiles(hf), 
			entryFile(entryF), oclKernelFile(oclKernel), hostFileStriped(hostS)
		{

		}

		void generateHostSideCode();
		static OpenCLKernelLoop* isAnOpenCLLoop(ForStmt* for_stmt);
		void generateHostSideOCLKernelCode(OpenCLKernelLoop* kl);
	};	
}
#endif

