#ifndef __OPENCLKERNELLOOP_H__
#define __OPENCLKERNELLOOP_H__
#include "clang/Omp2Ocl/OpenCLKernel.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Expr.h"
#include <vector>

using namespace clang;
using namespace std;

namespace clang
{
	class SwapLoopInfo
	{
		public:
			ForStmt* for_stmt;
			unsigned oi;
			SwapLoopInfo(ForStmt* ft, unsigned origi)
			{
				for_stmt = ft;
				oi =origi;
			}

			ForStmt* getForStmt()
			{
				return for_stmt;
			}
	};


	class OpenCLKernelLoop {

		vector<SwapLoopInfo> SwapableLoops;
		static vector<DeclRefExpr*> functions;
		bool optimised;		
		vector<OpenCLNDRangeVar> gV;
		bool useDefLocalWorkGroup;
		vector<ValueDecl*> _globalWriteBufs; /*Container for __global buffers that are written in the ocl kernel*/
		vector<ValueDecl*> _globalLCWriteBufs; /*Container for __global buffers that are written in the ocl kernel; This vector only records variables that are defined within the function body;*/
		public:
			ForStmt *for_stmt;
			FunctionDecl* func;
			vector<PLoopParam> params;
			vector<ForStmt*> subLoops;
			vector<Decl*> innerDecls;
			vector<OCLGlobalMemVar> globalMemoryVariables;
			vector<ArraySubVariable*> arraySubVs;
			static vector<OMPThreadPrivateObject> threadPrivates;
			bool swaped;
			Stmt* Kernel;
			bool bReductionLoop;
			vector<SwapLoopInfo> innerLoops;
			vector<CopyInBuffer> copyInBuffers;
			bool hasGlobalCopyInBuf;
			bool isUseDefLocalWorkGroup();
			void setUseDefLocalWorkGroup();
			
			//Read-write set
			vector<OCLRWSet> RWSet;

			/**
			 * This records the array accessing indexs in the inner most loop kernel
			 *
			 */
			ArraySubVariable* arrayV;
			LoopIndex* l;

			OpenCLKernelLoop(ForStmt* f);
			void addCopyInBuffer(ValueDecl* d, bool isGlobalBuf);
			bool hasOptimised();
			void setOptimised();			

			static vector<DeclRefExpr*>& getFunctions()
			{
				return functions;
			}

			vector<ValueDecl*> getGlobalWriteBufs()
			{
				return for_stmt->getGlobalWriteBufs();
			}

			bool isEnableTLSAutoCheck();

			void addGlobalWriteBuf(ValueDecl* d)
			{
				for_stmt->addGlobalWriteBuf(d);
			}

			void addGlobalLCWriteBuf(ValueDecl* d)
			{
				for_stmt->addGlobalLCWriteBuf(d);
			}

			vector<ValueDecl*> getGlobalLCWriteBufs()
			{
				return for_stmt->getGlobalLCWriteBufs();
			}

			vector<OCLGlobalMemVar>& getGlobalMemVars()
			{
				return globalMemoryVariables;
			}

			vector<PLoopParam>& getParams()
			{
				return params;
			}

			vector<CopyInBuffer>& getCopyInBuffers()
			{
				return copyInBuffers;
			}

			void setInnerLoops(vector<SwapLoopInfo>& innerLoops)
			{
				this->innerLoops = innerLoops;
			}

			vector<SwapLoopInfo>& getInnerLoops()
			{
				return innerLoops;
			}

			OMPFor& getOMPFor()
			{
				return for_stmt->getOMPFor();
			}	

			ForStmt* getForStmt()
			{
				return for_stmt;
			}

			FunctionDecl* getFunc()
			{
				return func;
			}

			void setRWSet(vector<OCLRWSet> rwSet)
			{
				RWSet = rwSet;
			}

			vector<OCLRWSet>& getRWSet()
			{
				return RWSet;
			}

			vector<SwapLoopInfo>& getSwapLoops()
			{
				return SwapableLoops;
			}

			void setSwapLoops(vector<SwapLoopInfo>& SLs)
			{
				SwapableLoops = SLs;
			}

			void setKernel(Stmt* Kernel)
			{
				this->Kernel = Kernel;
			}

			Stmt* getKernel()
			{
				return Kernel;
			}

			unsigned int getLineNumber(ASTContext& Ctx);
			bool isVariableBeenWrited(DeclRefExpr* expr);

			bool isReductionLoop() { return bReductionLoop; }

			bool isACopyIn(string& name);
			void newArraySubVariable()
			{
				this->arrayV = new ArraySubVariable();
			}

			void pushCurrentArraySubV()
			{
				if (this->arrayV)
				{
					arraySubVs.push_back(arrayV);
					this->arrayV = NULL;
				}
			}


			//Is a variable in the write set
			bool isInWriteSet(string name)
			{
				return for_stmt->isWriteSet(name);
			}

			/**
			*/
			ArraySubVariable* getInnerArrayV()
			{
				return arrayV;
			}

			void addGlobalMemoryVar(OCLGlobalMemVar v);
			void addDeclParam(Decl* S);
			bool isInnerDecl(DeclRefExpr* df);
			void removeFromGMVarList(string name);
			bool isAGlobalMemoryVariable(DeclRefExpr* v);
			bool isInTLSVarList(ValueDecl* d);
			bool isInTLSVarList(string name);
			void addOpenCLNDRangeVar(OpenCLNDRangeVar g);
			vector<OpenCLNDRangeVar>& getOclLoopIndexs()
			{
				return gV;
			}

			bool isAOpenCLNDRangeVar(string name);
			//Is this a reduction variable
			bool isAReductionVariable(string name);
			void addFuncParam(DeclRefExpr* ref);
			void addParam(DeclRefExpr *ref);
			void addSubLoop(ForStmt* node);
	};

}

#endif
