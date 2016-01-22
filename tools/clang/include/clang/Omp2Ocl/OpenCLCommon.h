#ifndef __OPENCLCOMMON_H__
#define __OPENCLCOMMON_H__

#include "clang/AST/Type.h"
#include <vector>

#define VERSION_STRING   "V0.01 alpha"

#define OCL_MAX_LOOP_LEVEL 3

#define DEFAULT_GROUP_SIZE	128	
#define DEFAULT_VECTOR_SIZE	4

#define DEFAULT_ALIGN_SIZE	"16"

#define DEFAULT_COPYIN_TYPE	"__constant"

#define OCL_KERNEL_PREFIX		"__kernel void "

#define LOCAL_VAR_PASSIN_PREFIX "l_"

#define MAX_TLS_DIMENSIONS	1

//CopyIn macro
#define DEFAULT_LOAD_VECTOR 		4 
#define COPYIN_MULTI_FACTOR_NAME 	"__ocl_mult_factor"
#define COPYIN_ADD_OFFSET_NAME 		"__ocl_add_offset"
#define OCL_NEAREST_MULTI		"OCL_NEAREST_MULTD"
#define OCL_LCM				"oclLCM"
#define DYN_BUFFER_CHECK 		"DYN_BUFFER_CHECK"
#define DYN_PROGRAM_CHECK		"DYN_PROGRAM_CHECK" 
#define CREATE_FUNC_LEVEL_BUFFER 	"CREATE_FUNC_LEVEL_BUFFER"
#define CREATE_REDUCTION_STEP1_BUFFER 	"CREATE_REDUCTION_STEP1_BUFFER"
#define CREATE_REDUCTION_STEP2_BUFFER 	"CREATE_REDUCTION_STEP2_BUFFER"
#define DECLARE_LOCALVAR_OCL_BUFFER	"DECLARE_LOCALVAR_OCL_BUFFER"
#define RELEASE_LOCALVAR_OCL_BUFFERS    "RELEASE_LOCALVAR_OCL_BUFFERS"
#define PROFILE_LOCALVAR_OCL_BUFFERS	"PROFILE_LOCALVAR_OCL_BUFFERS"
#define SYNC_LOCALVAR_OCL_BUFFERS	"SYNC_LOCALVAR_OCL_BUFFERS"
#define BUFFER_REUSED_RATIO 	2
#define VLOAD_CHECK

using namespace std;

namespace clang
{
	class OCLGlobalMemVar;
	class VarDecl;
	class FunctionDecl;
	class LoopIndex;
	class ForStmt;
	class OCLGlobalMemVar;
	class OMPThreadPrivateObject;
	class ValueDecl;

	static const unsigned PARALLEL_LOOP_DEPTH = OCL_MAX_LOOP_LEVEL;
	static const int ARCH_AMD = 0x123;
	static const int ARCH_NVIDIA = 0x231;
	static const int ARCH_CPU = 0x321;

	//Store info for variables used for VLoad
	class VLoadVarInfo
	{
		unsigned int c;
		string name;
		public:
			VLoadVarInfo(string name)
			{
				c = 0;
				this->name = name;
			}

			string getDeclareName();
			string getName() { return name; }
	};

	class OCLCommon
	{
		static int arch;
		static vector<VLoadVarInfo> vInfo;
		static void defaultCPUSetting();	
		
		public:
		static unsigned getMemAlignSize(string type);
		static string getVLoadVariableName(string name);
		static void setArch(string arch);
		static int getArch();
		static string getArchString();
		static FunctionDecl* CurrentVisitFunction;
		static RecordDecl* getRecordDecl(const QualType& type);
		static vector<OCLGlobalMemVar> globalMemoryObjs;
		//Whether this object is used as a __global memory in the OpenCL Kernel
		static bool isAGlobalMemObj(string name);
		static bool isAGlobalMemObj(VarDecl* d);
		static bool isOMP2OpenCLBuiltInRoutine(string name);
		static LoopIndex* getLoopIndex(ForStmt* Node);
		static bool isAThreadPrivateVariable(string& name, vector<OMPThreadPrivateObject>& threadPrivates, bool& isGlobal);
		static bool isAGlobalMemThreadPrivateVar(string& name);
		static bool isAGlobalMemThreadPrivateVar(ValueDecl* D);
		static bool isAThreadPrivateVariable(string& name);
		static bool isAThreadPrivateVariable(ValueDecl* name);
		static bool isAGTPVariable(string name, vector<OCLGlobalMemVar>& globalMemoryVariables, vector<OMPThreadPrivateObject>& threadPrivates);
		static bool isAGTPVariable(string name, vector<OMPThreadPrivateObject>& threadPrivates);
		static SourceLocation getPrivateVLoc(string& name, vector<OMPThreadPrivateObject>& threadPrivates);
		static unsigned int getLineNumber(ASTContext& Context, SourceLocation& Loc);
		static const char* getFileName(ASTContext& Context, SourceLocation& Loc);
		static bool isAOMPPrivateVariable(string name, ForStmt* for_stmt);
		static bool isAPerfectNestedLoop(ForStmt* for_stmt);
	};

}

#endif
