#ifndef __OPENCLGLOBALINFOCONTATIONER_H__
#define __OPENCLGLOBALINFOCONTATIONER_H__
#include <vector>
#include "clang/AST/Decl.h"
#include "clang/Omp2Ocl/OpenCLKernel.h"
#include "clang/Omp2Ocl/OpenCLKernelLoop.h"

using namespace std;
using namespace clang;

namespace clang
{
	class RecordDecl;
	class TypedefDecl;

	class RenameFuncGPUTLS
	{
		ValueDecl* decl;
		string func_name;
		unsigned index;
		public:
		RenameFuncGPUTLS(string func, ValueDecl* d, unsigned i)
		{
			func_name = func;
			decl = d;
			index = i;
		}

		string getFuncName()
		{
			return func_name;
		}

		string getDeclName()
		{
			return decl->getName();
		}

		ValueDecl* getDecl() { return decl; }

		unsigned getIndex() { return index; }
	};

	class OpenCLGlobalInfoContainer
	{
		static vector<RecordDecl*> RDs;
		static vector<RenameFuncGPUTLS> rfGPUTLS;
		static vector<TypedefDecl*> TDs;
		static vector<FunctionDecl*> candidateFuncs;
		static vector<RenamedFuncInfo> rnFuncs;
		static vector<OMPThreadPrivateObject> threadPrivateVars;
		static vector<FunctionDecl*> revisedFuncs;
		static vector<FunctionDecl*> allRecordFuncs;
		static vector<FunctionDecl*> calleeFuncs;
		static vector<OpenCLKernelLoop*> oclLoops;
		static vector<Decl*> globalDecls;
		static vector<OCLLocalVar> oclLocalVars;
		static vector<ValueDecl*> _writeGlobalMemObjs; //Used for GPU TLS. __global buffers that are written.
		public:
		static void addRenameFuncGPUTLS(string func_name, ValueDecl* d, unsigned index);
		static vector<RenameFuncGPUTLS> getRenameFuncGPUTLs();
		static bool isFuncHasGPUTLSLog(string func_name);
		static void addRecordDecl(RecordDecl* rd);
		static void addTypedefDecl(TypedefDecl* TD);
		static vector<RecordDecl*>& getRecordDecls();
		static vector<TypedefDecl*>& getTypedefDecls();
		static vector<FunctionDecl*>& getCandidateFuncs();
		static vector<RenamedFuncInfo>& getRenameFuncs();
		static void addFuncDecl(FunctionDecl* D);
		static void addRenameFuncInfo(RenamedFuncInfo& info);
		static void addThreadPrivate(OMPThreadPrivateObject& obj);
		static vector<OMPThreadPrivateObject>& getThreadPrivateVars();
		static vector<FunctionDecl*> getRevisedFuncs();
		static void addRevisedFunc(FunctionDecl* D);
		static void addRecordFunc(FunctionDecl* D);
		static vector<FunctionDecl*>& getRecordFuncs();
		static void addCalleeFunc(FunctionDecl* D);
		static vector<FunctionDecl*>& getCalleeFuncs();
		static vector<OpenCLKernelLoop*>& getOclLoops();
		static void addOclLoop(OpenCLKernelLoop* loop);
		static void addGlobalDecl(Decl* D);
		static vector<Decl*> getGlobalDecls();
		static void addLocalMemVar(OCLLocalVar& v);
		static vector<OCLLocalVar>& getLocalMemVars();
		static void addwriteGlobalMemObj(ValueDecl* d);
		static vector<ValueDecl*> getwriteGlobalMemObjs() { return _writeGlobalMemObjs; }
		static string getAName4RenamedFunc(ASTContext& Context, CallArgInfoContainer* cArg, vector<FunctionDecl*>& funcsNeed2Revised);
	};
}

#endif

