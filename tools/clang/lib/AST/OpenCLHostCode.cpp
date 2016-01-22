#include "clang/Omp2Ocl/OpenCLHostCode.h"
#include "clang/Omp2Ocl/OpenCLKernelSchedule.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclPrinter.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/Omp2Ocl/OpenCLRoutine.h"
#include "clang/Omp2Ocl/OpenCLDeclVisitor.h"
#include "clang/Omp2Ocl/OpenCLKernelLoop.h"
#include "clang/Omp2Ocl/OpenCLCommon.h"
#include "clang/Omp2Ocl/OpenCLGlobalInfoContainer.h"
#include "clang/Omp2Ocl/OpenCLGWSTuning.h"
#include "clang/Omp2Ocl/OpenCLCompilerOptions.h"
#include "clang/Omp2Ocl/OpenCLGPUTLSHostCode.h"

static QualType getDeclType(Decl* D) {
	if (TypedefDecl* TDD = dyn_cast<TypedefDecl>(D))
		return TDD->getUnderlyingType();
	if (ValueDecl* VD = dyn_cast<ValueDecl>(D))
		return VD->getType();
	return QualType();
}

static QualType GetBaseType(QualType T) {
	// FIXME: This should be on the Type class!
	QualType BaseType = T;
	while (!BaseType->isSpecifierType()) {
		if (isa<TypedefType>(BaseType))
			break;
		else if (const PointerType* PTy = BaseType->getAs<PointerType>())
			BaseType = PTy->getPointeeType();
		else if (const ArrayType* ATy = dyn_cast<ArrayType>(BaseType))
			BaseType = ATy->getElementType();
		else if (const FunctionType* FTy = BaseType->getAs<FunctionType>())
			BaseType = FTy->getResultType();
		else if (const VectorType *VTy = BaseType->getAs<VectorType>())
			BaseType = VTy->getElementType();
		else
			assert(0 && "Unknown declarator!");
	}
	return BaseType;
}

void OpenCLHostCode::generateGDStructs(llvm::raw_ostream& Out)
{
	vector<Decl*> Ds = OpenCLGlobalInfoContainer::getGlobalDecls();
	llvm::SmallVector<Decl*, 2> Decls;

	for (unsigned i=0; i<Ds.size(); i++)
	{
		Decl* D = Ds[i];

		if (isa<ObjCIvarDecl>(D))
			continue;

		QualType CurDeclType = getDeclType(D);
		if (!Decls.empty() && !CurDeclType.isNull()) {
			QualType BaseType = GetBaseType(CurDeclType);
			if (!BaseType.isNull() && isa<TagType>(BaseType) &&
					cast<TagType>(BaseType)->getDecl() == Decls[0]) {
				Decls.push_back(D);
				continue;
			}
		}
	}

	Decl::printGroup(Decls.data(), Decls.size(), Out, Context.PrintingPolicy, 0);
}

void OpenCLHostCode::generateTLSLogBuffers()
{
	vector<ValueDecl*> gMemObjs = OpenCLGlobalInfoContainer::getwriteGlobalMemObjs();

	if (OCLCompilerOptions::EnableGPUTLs && gMemObjs.size())
	{
		Out << "\n	//------------------------------------------\n";
		Out << "	// GPU TLS wr/rd buffers (BEGIN)\n";
		Out << "	//------------------------------------------\n";
		for (unsigned int i=0; i<gMemObjs.size(); i++)
		{
			string name = gMemObjs[i]->getName();
			string type = getCononicalType(gMemObjs[i]);
			vector<unsigned> dims = getArrayDef(type);
			string ar="(";
			unsigned sd = dims.size() - 1;
			for (unsigned j=0; j<dims.size(); j++)
			{
				ar = ar + uint2String(dims[j]);
				if (j < sd)
					ar = ar + " * ";
			}

			ar = ar + ")";

			Out << "rd_oclb_" << name << " = oclCreateBuffer(rd_log_" << name << ", " << ar << "* sizeof(int));\n"; 
			Out << "wr_oclb_" << name << " = oclCreateBuffer(wr_log_" << name << ", " << ar << " * sizeof(int));\n"; 
			Out << "oclHostWrites(rd_oclb_" << name << ");\n";
			Out << "oclHostWrites(wr_oclb_" << name << ");\n";
			Out << "DYN_BUFFER_CHECK(rd_oclb_" << name << ", -1);\n";
			Out << "DYN_BUFFER_CHECK(wr_oclb_" << name << ", -1);\n";
		}

		Out << "__oclb_gpu_tls_conflict_flag = oclCreateBuffer(&gpu_tls_conflict_flag, 1 * sizeof(int));\n";

		Out << "\n	//------------------------------------------\n";
		Out << "	// GPU TLS wr/rd buffers (END)\n";
		Out << "	//------------------------------------------\n";
	}
}

//Generate tls rd/wr arrays to track conflict for GPU TLS
void OpenCLHostCode::generateTLSLogArrays(llvm::raw_fd_ostream& Out)
{
	vector<ValueDecl*> gMemObjs = OpenCLGlobalInfoContainer::getwriteGlobalMemObjs();

	if (OCLCompilerOptions::EnableGPUTLs && gMemObjs.size())
	{
		Out << "\n//---------------------------------------------------------------------------\n";
		Out << "// GPU TLS wr/rd logs (BEGIN)\n";
		Out << "//---------------------------------------------------------------------------\n";
		
		for (unsigned int i=0; i<gMemObjs.size(); i++)
		{
			string name = gMemObjs[i]->getName();
			string type = getCononicalType(gMemObjs[i]);
			vector<unsigned> dims = getArrayDef(type);
			string ar="";

			for (unsigned j=0; j<dims.size(); j++)
			{
				ar = ar + "[" + uint2String(dims[j]) + "]";
			}

			Out << "static int rd_log_" << name << ar << ";\n";
			Out << "static int wr_log_" << name << ar << ";\n";
			Out << "static int tls_clear_" << name << " = 1;\n";
		}

		Out << "static int gpu_tls_conflict_flag = 0; \n";

		for (unsigned int i=0; i<gMemObjs.size(); i++)
		{
			string name = gMemObjs[i]->getName();
			string type = getCononicalType(gMemObjs[i]);
			vector<unsigned> dims = getArrayDef(type);
			string ar="";

			for (unsigned j=0; j<dims.size(); j++)
			{
				ar = ar + "[" + uint2String(dims[j]) + "]";
			}

			Out << "static ocl_buffer* rd_oclb_" << name << ";\n";
			Out << "static ocl_buffer* wr_oclb_" << name << ";\n";
		}

		Out << "static ocl_buffer* __oclb_gpu_tls_conflict_flag;\n";

		Out << "\n//---------------------------------------------------------------------------\n";
		Out << "// GPU TLS wr/rd logs (END)\n";
		Out << "//---------------------------------------------------------------------------\n";
	}
}

void OpenCLHostCode::generateOclDef()
{
	string errfile;
	llvm::raw_fd_ostream Out("ocldef.h", errfile);

	Out << "#include \"ocl_runtime.h\"\n";
	Out << "#ifndef __OCL_DEF_H__\n";
	Out << "#define __OCL_DEF_H__\n\n";
	Out << "#ifndef likely\n#define likely(x)       	__builtin_expect((x),1)\n#endif\n";
	Out << "#ifndef unlikely\n#define unlikely(x)       	__builtin_expect((x),0)\n#endif\n\n";
	Out << "#define " << OCL_NEAREST_MULTI << "(a, n) { do { if (a < n) {a = 1;} else {if (a % n) a = (a / n) + 1; else a = a / n;}  } while(0); }\n";

	if (OCLCompilerOptions::EnableGPUTLs && OCLCompilerOptions::StrictTLSChecking && !OCLCompilerOptions::OclTLSMechanism)
	{
		Out << "#define __RUN_CHECKING_KERNEL__\n";
	}

	Out << "#ifdef DEBUG\n";
	Out << "#define " << DYN_BUFFER_CHECK << "(__name__,__line__) {\\\n";
	Out << "if (unlikely (!__name__))\\\n";
	Out << "{\\\n";
	Out << "	fprintf (stderr,\\\n";
	Out << "			\"Failed to create the ocl buffer for %s at line: %d\\n\", #__name__, __line__);\\\n";
	Out << "	exit (-1);\\\n";
	Out <<"}\\\n";
	Out << "}\n";;
	Out << "#else\n";
	Out << "	#define " << DYN_BUFFER_CHECK << "(__name__,__line__) {}\n";
	Out << "#endif\n";
	Out << "\n";


	Out << "#ifdef DEBUG\n";
	Out << "#define " << DYN_PROGRAM_CHECK << "(__name__) {\\\n";
	Out << "if (unlikely (!__name__))\\\n";
	Out << "{\\\n";
	Out << "	fprintf (stderr,\\\n";
	Out << "			\"Failed to create the ocl kernel handle for %s \\n\", #__name__);\\\n";
	Out << "	exit (-1);\\\n";
	Out <<"}\\\n";
	Out << "}\n";;
	Out << "#else\n";
	Out << "	#define " << DYN_PROGRAM_CHECK << "(__name__) {}\n";
	Out << "#endif\n";
	Out << "\n";

	Out << "#define " << CREATE_FUNC_LEVEL_BUFFER << "(__ocl_buffer_name_,__p_buf_name__,__buffer_name__,__buffer_size__,__type__) {\\\n";
	Out << "	if (!__ocl_buffer_##__buffer_name__) {\\\n";
	Out << "	if ((__type__*)__p_buf_name__ != (__type__*)__buffer_name__)\\\n";
	Out << "	{\\\n";
	Out << "		__ocl_buffer_name_ =\\\n";
	Out << "			oclCreateBuffer (__buffer_name__, __buffer_size__ * sizeof (__type__));\\\n";
	Out << "			DYN_BUFFER_CHECK (__ocl_buffer_name_, -1);\\\n";
	Out << "			__p_buf_name__ = (__type__ *)__buffer_name__;\\\n";
	Out << "		__oclLVarBufferList_t* p = malloc(sizeof(__oclLVarBufferList_t));\\\n";
	Out << "			DYN_BUFFER_CHECK(p,__LINE__);\\\n";
	Out << "			p->buf = __ocl_buffer_name_;\\\n";
	Out << "			p->next = __ocl_lvar_buf_header;\\\n";
	Out << "			__ocl_lvar_buf_header = p;\\\n";
	Out << "	}\\\n";
	Out << "	__ocl_buffer_##__buffer_name__ = __ocl_buffer_name_;\\\n";
	Out << "	}\\\n";
	Out << "}\n\n";

	Out << "#define " << CREATE_REDUCTION_STEP1_BUFFER << "(__buffer_size_keeper__,__buffer_size__,__ocl_buffer_handle__,__type__) {\\\n";
	Out << "	        if (__buffer_size_keeper__ < __buffer_size__)\\\n";
	Out << "	        {\\\n";
	Out << "			                if (__buffer_size_keeper__ > 0)\\\n";
	Out << "			                {\\\n";
	Out << "						                        oclSync ();\\\n";
	Out << "						                        oclReleaseBuffer (__ocl_buffer_handle__);\\\n";
	Out << "						                }\\\n";
	Out << "			                __ocl_buffer_handle__ =\\\n";
	Out << "			                        oclCreateBuffer (NULL, sizeof (__type__) * __buffer_size__);\\\n";
	Out << "			                DYN_BUFFER_CHECK (__ocl_buffer_handle__, -1);\\\n";
	Out << "			                __buffer_size_keeper__ = __buffer_size__;\\\n";
	Out << "			        }\\\n";
	Out << "}\n\n";


	Out << "#define " << CREATE_REDUCTION_STEP2_BUFFER << "(__buffer_size_keeper__,__buffer_size__,__aligned_size__,__ocl_buffer_handle__,__buffer__,__type__) {\\\n";
	Out << "if (__buffer_size_keeper__ < sizeof (__type__) * __buffer_size__)\\\n";
	Out << "{\\\n";
	Out << "	if (__buffer_size_keeper__ > 0)\\\n";
	Out << "	{\\\n";
	Out << "		oclSync ();\\\n";
	Out << "			oclReleaseBuffer (__ocl_buffer_handle__);\\\n";
	Out << "			free (__buffer__);\\\n";
	Out << "	}\\\n";
	Out << "	posix_memalign ((void **) &__buffer__, __aligned_size__,\\\n";
	Out << "			sizeof (__type__) * __buffer_size__);\\\n";
	Out << "		DYN_BUFFER_CHECK (__buffer__, -1);\\\n";
	Out << "		__ocl_buffer_handle__ =\\\n";
	Out << "		oclCreateBuffer (__buffer__,\\\n";
	Out << "				sizeof (__type__) * __buffer_size__);\\\n";
	Out << "		DYN_BUFFER_CHECK (__ocl_buffer_handle__, -1);\\\n";
	Out << "		__buffer_size_keeper__ = sizeof (__type__) * __buffer_size__;\\\n";
	Out << "}\\\n";
	Out << "}\n\n";


	Out << "#define " << DECLARE_LOCALVAR_OCL_BUFFER << "(__variable_name__,__type__,__size__) \\\n";
	Out << "	ocl_buffer * __ocl_buffer_##__variable_name__;\\\n";
	Out << "	__ocl_buffer_##__variable_name__ = oclCreateBuffer(__variable_name__, sizeof(__type__)*__size__);\\\n";
	Out << "	DYN_BUFFER_CHECK(__ocl_buffer_##__variable_name__,-1);\\\n";
	Out << "	{\\\n";
	Out << "		__oclLVarBufferList_t* p = malloc(sizeof(__oclLVarBufferList_t));\\\n";
	Out << "			DYN_BUFFER_CHECK(p,__LINE__);\\\n";
	Out << "			p->buf = __ocl_buffer_##__variable_name__;\\\n";
	Out << "			p->next = __ocl_lvar_buf_header;\\\n";
	Out << "			__ocl_lvar_buf_header = p;\\\n";
	Out << "	}\n\n";


	Out << "#define "<< RELEASE_LOCALVAR_OCL_BUFFERS << "() {\\\n";
	Out << "	      	__oclLVarBufferList_t* header = __ocl_lvar_buf_header;\\\n";
	Out << "	        while (header)\\\n";
	Out << "	        {\\\n";
	Out << "		                __oclLVarBufferList_t* p = header;\\\n";
	Out << "			        header = header->next;\\\n";
	Out << "			        oclReleaseBuffer(p->buf);\\\n";
	Out << "			        free(p);\\\n";
	Out << "		}\\\n";
	Out << "}\n\n";


	Out << "#ifdef PROFILING\n";
	Out << "#define "<< PROFILE_LOCALVAR_OCL_BUFFERS << "(__buffer__,__prof__) {\\\n";
	Out << "	      	__oclLVarBufferList_t* header = __ocl_lvar_buf_header;\\\n";
	Out << "	        while (header)\\\n";
	Out << "	        {\\\n";
	Out << "		                __oclLVarBufferList_t* p = header;\\\n";
	Out << "			        header = header->next;\\\n";
	Out << "				__buffer__ += oclDumpBufferProfiling (p->buf, __prof__);\\\n";
	Out << "		}\\\n";
	Out << "}\n";
	Out << "#endif\n\n";

	Out << "#define "<< SYNC_LOCALVAR_OCL_BUFFERS << "() {\\\n";
	Out << "	      	__oclLVarBufferList_t* header = __ocl_lvar_buf_header;\\\n";
	Out << "	        while (header)\\\n";
	Out << "	        {\\\n";
	Out << "		                __oclLVarBufferList_t* p = header;\\\n";
	Out << "			        header = header->next;\\\n";
	Out << "				oclHostWrites(p->buf);\\\n";
	Out << "		}\\\n";
	Out << "}\n";

	Out << "#define REDUCTION_STEP1_MULT_NDRANGE() \\\n";
	Out << "      	size_t __ocl_buf_size = __ocl_act_buf_size;\\\n";
	Out << "            /*make sure the buffer length is multipled by (GROUP_SIZE * VECTOR_SIZE)*/\\\n";
	Out << "            unsigned mulFactor = (GROUP_SIZE * 4);\\\n";
	Out << "            __ocl_buf_size =\\\n";
	Out << "                (__ocl_buf_size <\\\n";
	Out << "				                  mulFactor) ? mulFactor : __ocl_buf_size;\\\n";
	Out << "            __ocl_buf_size =\\\n";
	Out << "                ((__ocl_buf_size / mulFactor) * mulFactor);\\\n";
	Out << "            if (__ocl_buf_size < __ocl_act_buf_size) {\\\n";
	Out << "				                __ocl_buf_size += mulFactor;\\\n";
	Out << "		    }\n";


	Out << "\n";

	Out << "#define CREATE_THREAD_PRIVATE_BUF(__buf__,__ocl_buf__,__type__,__size__,__align_size__) {\\\n";
	Out << "        size_t buf_size = sizeof(__type__) * (__size__);\\\n";
	Out << "        if (__buf__##_length < buf_size) {\\\n";
	Out << "	                if (__buf__) {\\\n";
	Out << "	                        oclSync();\\\n";
	Out << "	                        oclReleaseBuffer(__ocl_buf__);\\\n";
	Out << "	                        free(__buf__);\\\n";
	Out << "	                }\\\n";
	Out << "	                posix_memalign((void **)\\\n";
	Out << "                                &__buf__, __align_size__, __size__);\\\n";
	Out << "			                DYN_BUFFER_CHECK(__buf__, __LINE__);\\\n";
	Out << "			                __ocl_buf__ = oclCreateBuffer(__buf__, buf_size);\\\n";
	Out << "				                DYN_BUFFER_CHECK(__buf__, __LINE__);\\\n";
	Out << "			                __buf__##_length = buf_size;\\\n";
	Out << "			        }\\\n;";
	Out << "}\n";


	Out << "#define GROUP_SIZE	" << DEFAULT_GROUP_SIZE << "\n";
	Out << "#define DEFAULT_ALIGN_SIZE " << DEFAULT_ALIGN_SIZE << "\n";
	Out << "//#define OCL_RELEASE_GTP_BUFFERS_IMMEDIATE /* if defined, __global threadprivate buffers will be released immediately after each use. This may somehow alleviate the memory pressure */\n\n";

	if (OCLCompilerOptions::EnableDebugCG)
	{
		for (vector<string>::iterator iter = oclKernelNames.begin(); iter != oclKernelNames.end(); iter++)
		{
			Out << "#define ENABLE_OCL_KERNEL_" << (*iter) << " \n";
		}
	}	

	Out << "\n";


	Out << "typedef struct __oclLVarBufferList\n";
	Out << "{\n";
	Out << "	        ocl_buffer* buf;\n";
	Out << "		        struct __oclLVarBufferList* next;\n";
	Out << "}__oclLVarBufferList_t;\n\n";

	Out << "static __oclLVarBufferList_t* __ocl_lvar_buf_header = NULL;\n\n";
	Out << "static ocl_program *__ocl_program;\n\n";

	Out << "/** global data structures in : " << entryFile << " (BEGIN)*/\n";
	generateGDStructs(Out);
	Out << "/** global data structures in : " << entryFile << " (END)*/\n";

	declareOCLKenels(Out);

	vector<string> flbuffers;
	Out << "//OCL BUFFERS (BEGIN)\n";
	for (vector<OpenCLKernelLoop*>::iterator iter = OpenCLLoops.begin(); iter != OpenCLLoops.end(); iter++)
	{
		Out << generateOpenCLMemoryObjs((*iter), flbuffers);		
	}	

	Out << "//OCL BUFFERS (END)\n";
	Out << "\n";

	Out << "static void init_ocl_runtime();\n";
	Out << "static void create_ocl_buffers();\n";
	Out << "static void release_ocl_buffers();\n";
	Out << "static void sync_ocl_buffers(); \n";
	Out << "static void flush_ocl_buffers(); \n";
	
	if (OCLCompilerOptions::EnableGPUTLs)
	{
		Out << "static void ocl_gputls_checking();\n";
	}

	if (OCLCompilerOptions::GenProfilingFunc)
	{
		Out << "#ifdef PROFILING\n static void dump_profiling();\n #endif\n";
	}

	generateTLSLogArrays(Out);
	declarMLRecordVars(Out);

	Out << "#endif\n";	
	Out.flush();
	Out.close();
}

/*!
 * This method generates host 
 *
 */
void OpenCLHostCode::generateHostSideCode()
{
	generateOclDef();	
	generateHeadFiles();

	//Record the global memory objects, so that OpenCLDecVisitor and OpenCLHostPrinter can use it
	OCLCommon::globalMemoryObjs = globalMemoryObjs;
	OpenCLDeclVisitor ov(Out, Context, Context.PrintingPolicy, 0, entryFile);
	ov.VisitTranslationUnitDecl(Context.getTranslationUnitDecl());

	Out << "//---------------------------------------------------------------------------\n";
	Out << "//OCL related routines (BEGIN)\n";
	Out << "//---------------------------------------------------------------------------\n\n";
	generateInitCode();
	bufferCreationRoutine();
	finalBufferSync();
	bufferReleaseRoutine();
	flushOclBufers();
	OpenCLGPUTLSHostCode::checkConflictFlag(Out);
	if (OCLCompilerOptions::EnableMLFeatureCollection)
	{
		MLRoutines();
	}
	
	if (OCLCompilerOptions::GenProfilingFunc)
	{
		generateDumpProfileInfo();
	}
	Out << "//---------------------------------------------------------------------------\n";
	Out << "//OCL related routines (END)\n";
	Out << "//---------------------------------------------------------------------------\n\n";

	if (OCLCompilerOptions::EnableDebugCG)
	{
		OpenCLGWSTuning gwT(OpenCLGlobalInfoContainer::getOclLoops(), entryFile + ".gws.py");
		gwT.doIt();	
	}
}

void OpenCLHostCode::generateHeadFiles()
{
	for (vector<string>::iterator iter = headFiles.begin(); iter != headFiles.end(); iter++)
	{
		Out << "#include \"" << (*iter) << "\"\n";
	}
	Out << "#include \"ocldef.h\"\n";

	Out << "\n";
}

/*
 * Checking whether this loop is an OpenCL Kernel candidate
 *
 */
OpenCLKernelLoop* OpenCLHostCode::isAnOpenCLLoop(ForStmt* for_stmt)
{
	assert(for_stmt && "NULL forstmt");
	if (for_stmt->isParallelForLoop())
	{
		vector<OpenCLKernelLoop*>& ocl = OpenCLGlobalInfoContainer::getOclLoops();
		for (vector<OpenCLKernelLoop*>::iterator iter = ocl.begin(); iter != ocl.end(); iter++)
		{
			if ((*iter)->for_stmt == for_stmt)
				return (*iter);
		}
	}
	return NULL;
}

bool OpenCLHostCode::isAreadyDeclCLMemVar(string name)
{
	for (vector<string>::iterator iter = alreadyDeclaredCLMemVars.begin(); iter != alreadyDeclaredCLMemVars.end(); iter++)
	{
		if (*iter == name)
			return true;
	}

	return false;
}

//aggregate copyin objs
void OpenCLHostCode::scanCopyInObjs()
{
	for (unsigned i=0; i<OpenCLLoops.size(); i++)
	{
		OpenCLKernelLoop* l = OpenCLLoops[i];
		vector<CopyInBuffer>& copyInBuffers = l->getCopyInBuffers();

		for (unsigned j=0; j<copyInBuffers.size(); j++)
		{
			addCopyInObj(copyInBuffers[j]);
		}
	}
}

void OpenCLHostCode::addCopyInObj(CopyInBuffer& d)
{
	string name = d.d->getNameAsString();

	for (unsigned i=0; i<copyInObjs.size(); i++)
	{
		CopyInBuffer db = copyInObjs[i];
		if (db.d->getNameAsString() == name)
			return;
	}

	copyInObjs.push_back(d);
}

bool OpenCLHostCode::isACopyInObj(string& name)
{
	for (unsigned i=0; i<copyInObjs.size(); i++)
	{
		CopyInBuffer d = copyInObjs[i];
		if (d.d->getNameAsString() == name)
			return true;
	}

	return false;
}

string OpenCLHostCode::generateFuncLevelObjs(OpenCLKernelLoop* loop, vector<string>& buffers)
{
	string buf;
	llvm::raw_string_ostream OS(buf);

	FunctionDecl* D = loop->func;
	vector<OCLGlobalMemVar>& memVars = D->getFunctionLevelOCLBuffers();

	for (unsigned i=0; i<memVars.size(); i++)
	{
		OCLGlobalMemVar& var = memVars[i];

		string type = getGlobalType(getCononicalType(var.getDecl()));
		string name = var.getNameAsString();
		string buffer_name = "__ocl_buffer_" + name + "_" + D->getNameAsString();
		string p_name = "__ocl_p_" + name + "_" + D->getNameAsString();

		bool found = false;
		for (unsigned j=0; j<buffers.size(); j++)
		{
			if (buffers[j] == buffer_name)
			{
				found = true;
				break;
			}
		}

		if (found)
			continue;

		OS << "static ocl_buffer *" << buffer_name << " = NULL;\n";
		OS << "static " << type << " *" << p_name << " = NULL;\n";
		buffers.push_back(buffer_name);
	}

	OS.flush();

	return buf;
}

string OpenCLHostCode::generateOpenCLGTPObjs(OpenCLKernelLoop* loop)
	//Gnerate thread private variables whose locations are in __global memorstring OpenCLHostCode::generateOpenCLGTPObjs(OpenCLKernelLoop* loop)
{
	vector<OpenCLInputArgu>& inputArgs = loop->for_stmt->getInputArgs();
	string kernel = loop->for_stmt->getKernelName();

	string buf;
	llvm::raw_string_ostream OS(buf);

	for (unsigned i=0; i<inputArgs.size(); i++)
	{
		OpenCLInputArgu arg = inputArgs[i];
		if (arg.isAGlobalThMemVar())
		{
			string name = arg.getNameAsString();
			string pointerName = "__ocl_th_" + arg.getNameAsString() + "_" + kernel;
			string buffer_name = "__ocl_buffer_" + arg.getNameAsString() + "_" + kernel;
			string sizeofType =  arg.getGType();
			string buffer_size_name = pointerName + "_length";

			OS << " static " << sizeofType << " *" << pointerName + " = NULL;\n";
			OS << "	static ocl_buffer *" << buffer_name << " = NULL;\n";
			OS << "	static unsigned " << buffer_size_name << " = 0;\n";
			OS << "\n";	

		}
	}

	OS.flush();

	return buf; 
}

string OpenCLHostCode::generateOpenCLReductionObjs(OpenCLKernelLoop* loop)
{
	ForStmt* Node = loop->for_stmt;
	vector<ValueDecl*>& reducObjs = Node->getReductionVariables();

	string buf;
	llvm::raw_string_ostream OS(buf);

	for (unsigned i=0; i<reducObjs.size(); i++)
	{
		ValueDecl* d = reducObjs[i];
		string type = getGlobalType(d->getType().getAsString());
		string name = d->getNameAsString();

		OS << "static ocl_buffer* __ocl_buffer_" << name << "_" << Node->getKernelName() << " = NULL;\n";
		OS << "static unsigned __ocl_buffer_" << name << "_" << Node->getKernelName() << "_size = 0;\n";	

		string output_buffer_name = "__ocl_output_" + name + "_" + Node->getKernelName();
		string output_ocl_buffer_name  = "__ocl_output_buffer_" + name + "_" + Node->getKernelName();
		string output_buffer_size = output_buffer_name + "_size";

		OS << "static " << type << " *" << output_buffer_name << " = NULL;\n";
		OS << "static ocl_buffer* " << output_ocl_buffer_name << " = NULL;\n";
		OS << "static unsigned " << output_buffer_size << " = 0;\n";

	}

	OS.flush();

	return buf;
}

string OpenCLHostCode::generateOpenCLMemoryObjs(OpenCLKernelLoop* loop, vector<string>& FLBuffers)
{
	string result = "";	
	string name;

	assert(loop && "Invalid OpenCLKernelLoop");
	assert(loop->for_stmt && "Invalid OpenCL for stmt");

	for (vector<OCLGlobalMemVar>::iterator iter = loop->globalMemoryVariables.begin(); iter != loop->globalMemoryVariables.end(); iter++)
	{
		name = (iter)->getNameAsString();

		if (!iter->canbeDeclareAsGlobal())
		{
			string declaredName = name + "_" + loop->for_stmt->getKernelName();
			iter->setDeclaredName(declaredName);
			name = declaredName;
		}

		if (iter->isFuncLevel())
			continue;

		string type = iter->getType().getAsString();

		if (isAreadyDeclCLMemVar(name))
			continue;

		unsigned dim = getArrayDimension(type);
		if (dim)
		{
			result = result + "static ocl_buffer *__ocl_buffer_" + name + ";\n";
			alreadyDeclaredCLMemVars.push_back(name);
			globalMemoryObjs.push_back((*iter));
		}
	}

	result = result + generateOpenCLGTPObjs(loop) + generateOpenCLReductionObjs(loop) + generateFuncLevelObjs(loop, FLBuffers);

	return result;
}

//Declare ML features related vars
void OpenCLHostCode::declarMLRecordVars(llvm::raw_ostream& Out)
{
	if (OCLCompilerOptions::EnableMLFeatureCollection)
	{
		Out << "#ifndef DUMP_ML_FEATURES\n";
		Out << "#define DUMP_ML_FEATURES \n";
		Out << "#endif\n";

		Out << "#ifdef DUMP_ML_FEATURES\n";
		for (vector<string>::iterator iter = oclKernelNames.begin(); iter != oclKernelNames.end(); iter++)
		{
			Out << "unsigned long long __stats_" << (*iter) << " = 0;\n";
			Out << "unsigned long long __kernel_" << (*iter) << " = 0;\n";
		}

		Out << "static int __ocl_enable_ml_dump = 1;\n";
		Out << "static void reset_ml_features();\n";
		Out << "static void dump_ml_features();\n";
		Out << "static void enable_ml_record();\n";
		Out << "static void disable_ml_record();\n";
		Out << "static int is_enable_ml_record();\n";
		Out << "#endif\n";

	}
}

//*Declare opencl kernels/
void OpenCLHostCode::declareOCLKenels(llvm::raw_ostream& Out)
{
	Out << "//OCL KERNELS (BEGIN)\n";
	for (vector<string>::iterator iter = oclKernelNames.begin(); iter != oclKernelNames.end(); iter++)
	{
		Out << "static ocl_kernel *__ocl_" << (*iter) << ";\n";
	}	
	
	OpenCLGPUTLSHostCode::printCheckingKernelHandles(Out);
	
	Out << "//OCL KERNELS (END)\n\n";
}

//ML routines
void OpenCLHostCode::MLRoutines()
{
	Out << "#ifdef DUMP_ML_FEATURES\n";
	ResetMLFeatureRoutine();
	MLFeatureRoutine();
	TurnMLFeatureRoutine();
	Out << "#endif\n";
}

//Enable/disable ml dump
void OpenCLHostCode::TurnMLFeatureRoutine()
{
	Out << "static void enable_ml_record() {\n";
	Out << "__ocl_enable_ml_dump = 1;\n";
	Out << "}\n";

	Out << "static int is_enable_ml_record() {\n";
	Out << "return (__ocl_enable_ml_dump == 1);\n";
	Out << "}\n";

	Out << "static void disable_ml_record() {\n";
	Out << "__ocl_enable_ml_dump = 0;\n";
	Out << "}\n";
}

//Generate a fuction to reset the ml features to 0
void OpenCLHostCode::ResetMLFeatureRoutine()
{
	Out << "static void reset_ml_features() {\n";

	for (vector<string>::iterator iter = oclKernelNames.begin(); iter != oclKernelNames.end(); iter++)
	{
		Out << "__stats_" << (*iter) << " = 0;\n";	
		Out << "__kernel_" << (*iter) << " = 0;\n";	
	}

	Out << "}\n";
}

//Dump the collect ml features to a file
void OpenCLHostCode::MLFeatureRoutine()
{
	Out << "static void dump_ml_features() {\n";
	Out << "FILE* fp = fopen(\"ml_global_size.CLASS\", \"w\");\n";
	Out << "FILE* fpk = fopen(\"ml_kernels.CLASS\", \"w\");\n";

	for (vector<string>::iterator iter = oclKernelNames.begin(); iter != oclKernelNames.end(); iter++)
	{
		Out << "fprintf(fp, \"" << (*iter) << ": %llu\\n\", __stats_" << (*iter) <<  ");\n";
	}

	for (vector<string>::iterator iter = oclKernelNames.begin(); iter != oclKernelNames.end(); iter++)
	{
		Out << "fprintf(fpk, \"" << (*iter) << ": %llu\\n\", __kernel_" << (*iter) <<  ");\n";
	}

	Out << "fclose(fp);\n";
	Out << "fclose(fpk);\n";
	Out << "}\n";
}

void OpenCLHostCode::generateInitCode()
{
	string indent = "	";
	Out << "static void init_ocl_runtime() {\n";
	Out << indent << "int err;\n\n";
	Out << indent << "if (unlikely(err = oclInit(\"" << OCLCommon::getArchString() << "\", 0))) {\n";
	Out << indent << "	fprintf (stderr, \"Failed to init ocl runtime:%d.\\n\", err);\n";
	Out << indent << "	exit(err);\n";
	Out << indent << "}\n\n";

	Out << indent << "__ocl_program = oclBuildProgram(\"" << oclKernelFile << "\");\n";
	Out << indent << "if (unlikely(!__ocl_program)) {\n";
	Out << indent << "	fprintf (stderr, \"Failed to build the program:%d.\\n\", err);\n";
	Out << indent << "	exit(err);\n";
	Out << indent << "}\n\n";


	for (vector<string>::iterator iter = oclKernelNames.begin(); iter != oclKernelNames.end(); iter++)
	{
		Out << indent << "__ocl_" << (*iter) << " = oclCreateKernel(__ocl_program, \"" << (*iter) << "\");\n";
		Out << indent << DYN_PROGRAM_CHECK << "(__ocl_" << (*iter) << ");\n";
	}

	OpenCLGPUTLSHostCode::buildCheckingKernelHandles(Out);

	Out << indent << "create_ocl_buffers();\n";	

	Out << "}\n\n";
}

void OpenCLHostCode::bufferCreationRoutine()
{
	string indent = "	";

	Out << "static void create_ocl_buffers() {\n";

	for (vector<OCLGlobalMemVar>::iterator iter = globalMemoryObjs.begin(); iter != globalMemoryObjs.end(); iter++)
	{
		if (iter->canbeDeclareAsGlobal())
		{
			ValueDecl* d = iter->getDecl();
			createOCLBuffer(Out, d, iter->getNameAsString());
		}
	}

	generateTLSLogBuffers();

	Out << "}\n\n";

}

string OpenCLHostCode::releaseOpenCLGTPObjs(OpenCLKernelLoop* loop)
{
	vector<OpenCLInputArgu>& inputArgs = loop->for_stmt->getInputArgs();
	string kernel = loop->for_stmt->getKernelName();

	string buf;
	llvm::raw_string_ostream OS(buf);

	for (unsigned i=0; i<inputArgs.size(); i++)
	{
		OpenCLInputArgu arg = inputArgs[i];
		if (arg.isAGlobalThMemVar())
		{
			string name = arg.getNameAsString();
			string pointerName = "__ocl_th_" + arg.getNameAsString() + "_" + kernel;
			string buffer_name = "__ocl_buffer_" + arg.getNameAsString() + "_" + kernel;
			string sizeofType =  arg.getGType();
			string buffer_size_name = pointerName + "_length";

			OS << "	if (" << pointerName << ") {\n";
			releaseOCLBuffer(OS, buffer_name);
			OS << "	free(" << pointerName << ");\n";
			OS << buffer_size_name << " = 0;\n";
			OS << "	}\n";	
		}
	}

	OS.flush();

	return buf; 
}


string OpenCLHostCode::releaseOpenCLReductionObjs(OpenCLKernelLoop* loop)
{
	ForStmt* Node = loop->for_stmt;

	string buf;
	llvm::raw_string_ostream OS(buf);

	vector<ValueDecl*>& reducObjs = Node->getReductionVariables();

	for (unsigned i=0; i<reducObjs.size(); i++)
	{
		ValueDecl* d = reducObjs[i];
		string type = getGlobalType(d->getType().getAsString());
		string name = d->getNameAsString();
		string size_name = "__ocl_buffer_" + name + "_" + Node->getKernelName() + "_size";
		string buffer_name = "__ocl_buffer_" + name + "_" + Node->getKernelName();

		OS << "	if (" << size_name << " > 0) {\n";
		OS << " 	oclReleaseBuffer(" << buffer_name << ");\n";
		OS << size_name << " = 0;\n";
		OS << "	}\n";

		string output_buffer_name = "__ocl_output_" + name + "_" + Node->getKernelName();
		string output_ocl_buffer_name  = "__ocl_output_buffer_" + name + "_" + Node->getKernelName();
		string output_buffer_size = output_buffer_name + "_size";

		OS << "if (" << output_buffer_size << " > 0) {\n";
		OS << "	oclReleaseBuffer(" << output_ocl_buffer_name << ");\n";
		OS << "	free(" << output_buffer_name << ");\n";
		OS << output_buffer_size << " = 0;\n";
		OS << "}\n";
	}

	OS.flush();

	return buf; 
}


string OpenCLHostCode::releaseFuncLevelObjs(OpenCLKernelLoop* loop, vector<string>& buffers)
{
	string buf;
	llvm::raw_string_ostream OS(buf);

	FunctionDecl* D = loop->func;
	vector<OCLGlobalMemVar>& memVars = D->getFunctionLevelOCLBuffers();

	for (unsigned i=0; i<memVars.size(); i++)
	{
		OCLGlobalMemVar& var = memVars[i];

		string name = var.getNameAsString();
		string buffer_name = "__ocl_buffer_" + name + "_" + D->getNameAsString();
		string p_name = "__ocl_p_" + name + "_" + D->getNameAsString();

		bool found = false;
		for (unsigned j=0; j<buffers.size(); j++)
		{
			if (buffers[j] == buffer_name)
			{
				found = true;
				break;
			}
		}

		if (found)
			continue;

		OS << "if(" << buffer_name << ") {\n";
		OS << "	oclReleaseBuffer(" << buffer_name << ");\n";
		OS << p_name << " = NULL;\n";
		OS << buffer_name << " = NULL;\n";
		OS << "}\n";

		buffers.push_back(buffer_name);
	}

	OS.flush();

	return buf;
}

void OpenCLHostCode::bufferReleaseRoutine()
{
	Out << "static void release_ocl_buffers() {\n";

	for (vector<OCLGlobalMemVar>::iterator iter = globalMemoryObjs.begin(); iter != globalMemoryObjs.end(); iter++)
	{
		ValueDecl* d = (iter)->getDecl();
		if (iter->isFuncLevel())
			continue;

		releaseOCLBuffer(Out, d, iter->getNameAsString());
	}

	for (vector<OpenCLKernelLoop*>::iterator iter = OpenCLLoops.begin(); iter != OpenCLLoops.end(); iter++)
	{
		Out << releaseOpenCLGTPObjs((*iter));
	}


	for (vector<OpenCLKernelLoop*>::iterator iter = OpenCLLoops.begin(); iter != OpenCLLoops.end(); iter++)
	{
		Out << releaseOpenCLReductionObjs((*iter));
	}

#if 0
	vector<string> FLBuffers;
	for (vector<OpenCLKernelLoop*>::iterator iter = OpenCLLoops.begin(); iter != OpenCLLoops.end(); iter++)
	{
		Out << releaseFuncLevelObjs((*iter), FLBuffers);
	}
#endif
	Out << RELEASE_LOCALVAR_OCL_BUFFERS << "();\n";
	Out << "}\n\n";

}

string OpenCLHostCode::flushFuncLevelBuffer(OpenCLKernelLoop* loop, vector<string>& buffers)
{
	string buf;
	llvm::raw_string_ostream OS(buf);

	FunctionDecl* D = loop->func;
	vector<OCLGlobalMemVar>& memVars = D->getFunctionLevelOCLBuffers();

	for (unsigned i=0; i<memVars.size(); i++)
	{
		OCLGlobalMemVar& var = memVars[i];

		string name = var.getNameAsString();
		string buffer_name = "__ocl_buffer_" + name + "_" + D->getNameAsString();

		bool found = false;
		for (unsigned j=0; j<buffers.size(); j++)
		{
			if (buffers[j] == buffer_name)
			{
				found = true;
				break;
			}
		}

		if (found)
			continue;
		OS << "if(" << buffer_name << ") {\n";
		OS << " oclHostWrites(" << buffer_name << ");\n";
		OS << "}\n";

		buffers.push_back(buffer_name);
	}

	OS.flush();

	return buf;
}


void OpenCLHostCode::flushOclBufers()
{
	string indent = "	";

	Out << "static void flush_ocl_buffers() {\n";

	for (vector<OCLGlobalMemVar>::iterator iter = globalMemoryObjs.begin(); iter != globalMemoryObjs.end(); iter++)
	{
		string name = iter->getNameAsString();

		Out << indent << " oclHostWrites(__ocl_buffer_" << iter->getNameAsString() << ");\n";
	}

#if 1
	vector<string> FLBuffers;	
	for (vector<OpenCLKernelLoop*>::iterator iter = OpenCLLoops.begin(); iter != OpenCLLoops.end(); iter++)
	{
		Out << flushFuncLevelBuffer((*iter), FLBuffers);
	}
#endif
	Out << "//SYNC_LOCALVAR_OCL_BUFFERS();\n";

	Out << "	oclSync();\n";
	Out << "}\n\n";
}


void OpenCLHostCode::finalBufferSync()
{
	string indent = "	";

	Out << "static void sync_ocl_buffers() {\n";

	for (vector<OCLGlobalMemVar>::iterator iter = globalMemoryObjs.begin(); iter != globalMemoryObjs.end(); iter++)
	{
		ValueDecl* d = iter->getDecl();
		string name = d->getNameAsString();

		if (isACopyInObj(name))
			continue;


		Out << indent << " oclHostWrites(__ocl_buffer_" << iter->getNameAsString() << ");\n";
	}

	Out << "//SYNC_LOCALVAR_OCL_BUFFERS();\n";

	Out << "	oclSync();\n";
	Out << "}\n\n";
}


void OpenCLHostCode::generateHostSideOCLKernelCode(OpenCLKernelLoop* kl)
{

}

void OpenCLHostCode::generateDumpProfileInfo()
{
	Out << "#ifdef PROFILING\n";
	Out << "static void dump_profiling() {\n";
	Out << "FILE *prof = fopen(\"profiling-" << hostFileStriped << "\", \"w\");\n";
	Out << "float kernel = 0.0f, buffer = 0.0f;\n";

	Out << "\n";

	for (vector<string>::iterator iter = oclKernelNames.begin(); iter != oclKernelNames.end(); iter++)
	{
		Out << "kernel +=  oclDumpKernelProfiling (__ocl_" << (*iter) << ", prof);\n";
	}


	Out << "\n";
	for (vector<OCLGlobalMemVar>::iterator iter = globalMemoryObjs.begin(); iter != globalMemoryObjs.end(); iter++)
	{
		ValueDecl* d = iter->getDecl();
		string name = d->getNameAsString();

		Out << "buffer += oclDumpBufferProfiling (__ocl_buffer_" << name << ", prof);\n";
	}

	Out << "PROFILE_LOCALVAR_OCL_BUFFERS(buffer,prof);\n";	

	Out << "\n";
	Out << "fprintf(stderr, \"-- kernel: %.3fms\\n\", kernel);\n";
	Out << "fprintf(stderr, \"-- buffer: %.3fms\\n\", buffer);\n";

	Out << "fclose(prof);\n";

	Out << "}\n";
	Out << "#endif\n\n";
}
