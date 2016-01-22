#ifndef __OPENCLLOOPINTERCHANGE_H__
#define __OPENCLLOOPINTERCHANGE_H__
#include <iostream>
#include <vector>
#include "clang/Omp2Ocl/OpenCLKernel.h"
#include "clang/Omp2Ocl/OpenCLCommon.h"
#include "clang/Omp2Ocl/OpenCLKernelLoop.h"
#include "clang/Omp2Ocl/OpenCLCodeOptimisation.h" 

using namespace std;
using namespace clang;

namespace clang
{
	class OpenCLLoopInterChange : public OpenCLCodeOptimisation
	{
		vector<SwapLoopInfo> innerLoops;
		vector<SwapLoopInfo> SwapableLoops;
		bool swaped;

		bool isInLoopIndex(DeclRefExpr* expr, vector<LoopIndex*>& swapableIndex);
		ForStmt* whichLoop(vector<SwapLoopInfo>& SwapableLoops, DeclRefExpr* expr, unsigned& oi);
		unsigned int howManyLoopIndexUse(vector<LoopIndex*>& swapableIndex, ArraySubVariable* a);
		void InterChangeLoops(unsigned l);
		void InsertInnerLoops(vector<SwapLoopInfo>& newSwapLoops, vector<SwapLoopInfo>& SwapableLoops);
		bool CheckSanityofSwapIndexs(OMPParallelDepth& pd, vector<OMPSwapIndex>& swapIndexs);
		void userDefInterChange();
		void VanillaInterChange();

		public:
			OpenCLLoopInterChange(ASTContext& C, OpenCLKernelLoop* l)
			:  OpenCLCodeOptimisation(C, l)
			 {
				swaped = false;	 	
			 }
			
			void doIt();
	};
	
}


#endif
