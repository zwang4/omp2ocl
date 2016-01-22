#ifndef __OPENCLREDUCTIONKERNEL_H__
#define __OPENCLREDUCTIONKERNEL_H__

namespace clang
{
	class OpenCLReductionKernel
	{
		static void ReductionPreparePhase(OpenCLKernelLoop* loop);
		static void ReductionFirstPhase(OpenCLKernelLoop* loop, vector<FunctionDecl*>& funcNeed2Revised);
		static void ReductionSecondPhase(OpenCLKernelLoop* loop);

		public:
		static doIt(OpenCLKernelLoop* loop, vector<FunctionDecl*>& funcsNeed2Revised);
	};

}

#endif
