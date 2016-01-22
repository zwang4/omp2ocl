#ifndef __OPENCLGLOBALINFOCONTATIONER_H__
#define __OPENCLGLOBALINFOCONTATIONER_H__
#include <vector>
#include "clang/AST/Decl.h"
#include "clang/Omp2Ocl/OpenCLKernel.h"

using namespace std;
using namespace clang;

namespace clang
{
	class RecordDecl;
	class TypedefDecl;

	class OpenCLGlobalInfoContainer
	{
		static vector<RecordDecl*> RDs;
		static vector<TypedefDecl*> TDs;
		static vector<FunctionDecl*> candidateFuncs;
		static vector<RenamedFuncInfo>& rnFuncs;
		
		public:
			static void addRecordDecl(RecordDecl* rd);
			static void addTypedefDecl(TypedefDecl* TD);
			static vector<RecordDecl*>& getRecordDecls();
			static vector<TypedefDecl*>& getTypedefDecls();
			static vector<FunctionDecl*>& getCandidateFuncs();
			static vector<RenamedFuncInfo>& getRenameFuncs();
			static void addFuncDecl(FunctionDecl* D);
			static void addRenameFuncInfo(RenamedFuncInfo& info);
	};
}

#endif

