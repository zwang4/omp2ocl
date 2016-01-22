#include "clang/Omp2Ocl/OpenCLGPUTLSHostCode.h"
#include "llvm/Support/Format.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/StmtPicker.h"
#include "clang/Omp2Ocl/OpenCLRoutine.h"
#include <algorithm>
#include <deque>
#include <math.h>

using namespace std;

vector<GPUTLSLogContainerFD> OpenCLGPUTLSHostCode::tls_FDC;
vector<GPUTLSLogContainer> OpenCLGPUTLSHostCode::tls_log_container;

void OpenCLGPUTLSHostCode::genTLSKernelCall(llvm::raw_ostream &OS, vector<ValueDecl*>& globalLCWriteBufs, vector<ValueDecl*>& globalWriteBufs)
{
	if (globalWriteBufs.size() > 0)
	{
		OS << "	//---------------------------------------\n";
		OS << "	// GPU TLS Checking (BEGIN)\n";
		OS << "	//---------------------------------------\n";
		OS << "	{\n";	
#if MULTI_D_TLS_CHECKING
		//Currently I only generate checking for global write buffers
		for (unsigned i=0; i<globalWriteBufs.size(); i++)
		{
			ValueDecl* d = globalWriteBufs[i];

			string name = d->getName();
			string type = getCononicalType(d);
			int dim = getArrayDimension(type);
			vector<unsigned> dims = getArrayDef(type);
			
			OS << "//Checking " << name << "\n";

			OS << "{\n";
			OS << "size_t __ocl_gws_" << name << "[" << dim << "] = {";
			for (int k=dim-1; k>=0; k--)
			{
				if (k < dim-1 )
					OS << ",";
				OS << dims[k];
			}
			OS << "};\n";

			//OS << "gpu_tls_conflict_flag=0;\n";
			//OS << "oclHostWrite(gpu_tls_conflict_flag);\n";

			OS << "if (tls_clear_" << name << "){\n";
			OS << "	oclHostWrites(rd_oclb_" << name << ");\n";
			OS << "	oclHostWrites(wr_oclb_" << name << ");\n";
			OS << " tls_clear_" << name << "=0;\n";
			OS << "}\n";

			string kernel_handle = "__ocl_tls_" + uint2String(dim) + "D_checking";
			int k = 0;
			for (k=0; k<dim; k++)
			{
				OS << "oclSetKernelArg(" << kernel_handle << "," << k << ",sizeof(int),&__ocl_gws_" << name << "[" << k << "]);\n";
			}

			OS << "oclSetKernelArgBuffer(" << kernel_handle << "," << k << "," << "rd_oclb_" << name << ");\n";
			k++;
			OS << "oclSetKernelArgBuffer(" << kernel_handle << "," << k << "," << "wr_oclb_" << name << ");\n";
			k++;
			OS << "oclSetKernelArgBuffer(" << kernel_handle << "," << k << "," << "__oclb_gpu_tls_conflict_flag);\n";

			OS << "oclDevWrites(wr_oclb_" << name << ");\n";
			OS << "oclDevWrites(rd_oclb_" << name << ");\n";
			OS << "oclDevWrites(__oclb_gpu_tls_conflict_flag);\n";

			OS << "oclRunKernel(" << kernel_handle << ", " << ((dim <= OCL_MAX_LOOP_LEVEL) ? dim : OCL_MAX_LOOP_LEVEL )<< ", __ocl_gws_" << name << ");\n";

			OS << "}\n";
		}
#else
		for (unsigned i=0; i<globalWriteBufs.size(); i++)
		{
			ValueDecl* d = globalWriteBufs[i];

			string name = d->getName();
			string type = getCononicalType(d);
			int dim = getArrayDimension(type);
			vector<unsigned> dims = getArrayDef(type);
			
			OS << "#ifdef __RUN_CHECKING_KERNEL__\n";

			OS << "//Checking " << name << "\n";

			OS << "{\n";
			OS << "size_t __ocl_gws_" << name << " = (";
			for (int k=dim-1; k>=0; k--)
			{
				if (k < dim-1 )
					OS << " * ";
				OS << dims[k];
			}
			OS << ");\n";

			//OS << "gpu_tls_conflict_flag=0;\n";
			//OS << "oclHostWrite(gpu_tls_conflict_flag);\n";
			
			//OS << "if (tls_clear_" << name << "){\n";
			//OS << "	oclHostWrites(rd_oclb_" << name << ");\n";
			//OS << "	oclHostWrites(wr_oclb_" << name << ");\n";
			//OS << " tls_clear_" << name << "=0;\n";
			//OS << "}\n";

			string kernel_handle = "__ocl_tls_1D_checking";
			int k = 0;
			OS << "oclSetKernelArg(" << kernel_handle << "," << k << ",sizeof(unsigned),&__ocl_gws_" << name << ");\n";
			k++;
			OS << "oclSetKernelArgBuffer(" << kernel_handle << "," << k << "," << "rd_oclb_" << name << ");\n";
			k++;
			OS << "oclSetKernelArgBuffer(" << kernel_handle << "," << k << "," << "wr_oclb_" << name << ");\n";
			k++;
			OS << "oclSetKernelArgBuffer(" << kernel_handle << "," << k << "," << "__oclb_gpu_tls_conflict_flag);\n";

			OS << "oclDevWrites(wr_oclb_" << name << ");\n";
			OS << "oclDevWrites(rd_oclb_" << name << ");\n";
			OS << "oclDevWrites(__oclb_gpu_tls_conflict_flag);\n";

			OS << "oclRunKernel(" << kernel_handle << ", 1 , &__ocl_gws_" << name << ");\n";

			OS << "}\n";
			
			OS << "#endif\n";
		}

#endif
		if (OCLCompilerOptions::StrictTLSChecking)
		{
			OS << "\n\n";
			OS << " oclHostReads(__oclb_gpu_tls_conflict_flag);\n";
			OS << " oclSync();\n";
			OS << "#ifdef __DUMP_TLS_CONFLICT__\n";
			OS << " if (gpu_tls_conflict_flag) {\n";
			OS << "    fprintf(stderr, \"conflict detected.\\n\");\n";
			OS << " }\n";
			OS << "#endif\n";
		}

		OS << " }\n";
		OS << "	//---------------------------------------\n";
		OS << "	// GPU TLS Checking (END)\n";
		OS << "	//---------------------------------------\n";
	}
}

void OpenCLGPUTLSHostCode::genTLSKernelCallLoopLevel(llvm::raw_ostream &OS, ForStmt* for_stmt)
{
	vector<ValueDecl*> pgwb = for_stmt->getGlobalWriteBufs();
	vector<ValueDecl*> lwb = for_stmt->getGlobalWriteBufs();
	vector<OpenCLTLSBufferAccess> act_tls_vec = for_stmt->getTLSCheckingVec();
	vector<ValueDecl*> gwb;

	for (unsigned i=0; i<pgwb.size(); i++)
	{
		string name = pgwb[i]->getName();
		for (unsigned j=0; j<act_tls_vec.size(); j++)
		{
			if (name == act_tls_vec[j].getName())
			{
				gwb.push_back( pgwb[i] );
			}
		}

	}

	genTLSKernelCall(OS, gwb, lwb);
}

void OpenCLGPUTLSHostCode::checkConflictFlag(llvm::raw_ostream &OS)
{
	if (OCLCompilerOptions::EnableGPUTLs)
	{
		OS << "void ocl_gputls_checking() {\n";

		if (OCLCompilerOptions::TLSCheckAtProgramEnd)
		{
			genTLSCheckingKernelCodeAllFuncs(OS);
		}
		
		OS << "oclHostReads(__oclb_gpu_tls_conflict_flag);\n";
		OS << "oclSync();\n";
		OS << "if (gpu_tls_conflict_flag){\n";
		OS << " fprintf(stderr, \"Found conflict.\\n\");\n";
		OS << " }else{\n";
		OS << " fprintf(stdout, \"No conflict.\\n\");\n";
		OS << "	}\n";
		OS << "}\n";
	}
}

void OpenCLGPUTLSHostCode::printCheckingKernelHandles(llvm::raw_ostream &OS)
{
	if (OCLCompilerOptions::EnableGPUTLs)
	{
		for (unsigned d=1; d<=MAX_TLS_DIMENSIONS; d++)
		{
			OS << "static ocl_kernel *__ocl_tls_" << d << "D_checking;\n";
		}
	}

}

void OpenCLGPUTLSHostCode::buildCheckingKernelHandles(llvm::raw_ostream &OS)
{
	if (OCLCompilerOptions::EnableGPUTLs && !OCLCompilerOptions::OclTLSMechanism)
	{
		for (unsigned d=1; d<=MAX_TLS_DIMENSIONS; d++)
		{
			OS << "__ocl_tls_"<< d << "D_checking = oclCreateKernel(__ocl_program, \"" << "TLS_Checking_" << d << "D" << "\");\n";
			OS << DYN_PROGRAM_CHECK << "(__ocl_tls_" << d << "D_checking" << ");\n";
		}
	}
}


static bool isInBufs(vector<ValueDecl*>& gWB, ValueDecl* d)
{
	string name = d->getName();
	for (unsigned i=0; i<gWB.size(); i++)
	{
		if (gWB[i]->getName() == name)
		{
			return true;
		}
	}

	return false;
}

//
// Generate call to the TLS Checking kernels (all)
//
void OpenCLGPUTLSHostCode::genTLSCheckingKernelCodeAllFuncs(llvm::raw_ostream &OS)
{
	vector<ValueDecl*> globalWriteBufs;
	vector<ValueDecl*> globalLCBufs;

	for (unsigned i=0; i<tls_FDC.size(); i++)
	{
		for (unsigned j=0; j<tls_FDC[i]._globalLCWriteBufs.size(); j++)
		{
			if (!isInBufs(globalLCBufs, tls_FDC[i]._globalLCWriteBufs[j]))
			{
				globalLCBufs.push_back(tls_FDC[i]._globalLCWriteBufs[j]);
			}
		} 
		
		for (unsigned j=0; j<tls_FDC[i]._globalWriteBufs.size(); j++)
		{
			if (!isInBufs(globalWriteBufs, tls_FDC[i]._globalWriteBufs[j]))
			{
				globalWriteBufs.push_back(tls_FDC[i]._globalWriteBufs[j]);
			}
		} 
	}

	genTLSKernelCall(OS, globalLCBufs, globalWriteBufs);
}

//
// Generate call to the TLS Checking kernels
//
void OpenCLGPUTLSHostCode::genTLSCheckingKernelCode(llvm::raw_ostream &OS)
{
	if (OCLCompilerOptions::TLSCheckAtProgramEnd) return;

	const FunctionDecl* FD = OCLCommon::CurrentVisitFunction;

	for (unsigned i=0; i<tls_FDC.size(); i++)
	{
		if (tls_FDC[i].F == FD)
		{
			genTLSKernelCall(OS, tls_FDC[i]._globalLCWriteBufs, tls_FDC[i]._globalWriteBufs);
			//tls_FDC[i]._globalLCWriteBufs.clear();
			//tls_FDC[i]._globalWriteBufs.clear();
		}
	}
}

bool OpenCLGPUTLSHostCode::isInVector(vector<ValueDecl*>& vec, ValueDecl* v)
{
	string name = v->getName();
	for (unsigned i=0; i<vec.size(); i++)
	{
		if (name == vec[i]->getName())
			return true;
	}

	return false;
}

void OpenCLGPUTLSHostCode::insertGPUTLsLog(ForStmt* for_stmt)
{
	const FunctionDecl* FD = OCLCommon::CurrentVisitFunction;
	if (!FD) return;

	tls_log_container.push_back(GPUTLSLogContainer(for_stmt, FD));
	vector<ValueDecl*> pgwb = for_stmt->getGlobalWriteBufs();
	vector<OpenCLTLSBufferAccess> act_tls_vec = for_stmt->getTLSCheckingVec();
	vector<ValueDecl*> gwb;

	int fg_i = -1;

	for (unsigned i=0; i<tls_FDC.size(); i++)
	{
		if (tls_FDC[i].F == FD)
		{
			fg_i = i;
			break;
		}
	}

	if (fg_i < 0)
	{
		tls_FDC.push_back(GPUTLSLogContainerFD(FD));
		fg_i = tls_FDC.size()-1;
	}

	GPUTLSLogContainerFD& fg = tls_FDC[fg_i];

	//refine, only checking a buffer that is actually used to record tls
	for (unsigned i=0; i<pgwb.size(); i++)
	{
		string name = pgwb[i]->getName();
		for (unsigned j=0; j<act_tls_vec.size(); j++)
		{
			if (name == act_tls_vec[j].getName())
			{
				gwb.push_back( pgwb[i] );
			}
		}

	}

	for (unsigned i=0; i<gwb.size(); i++)
	{
		if (isInVector(fg._globalWriteBufs, gwb[i]))
			continue;

		fg._globalWriteBufs.push_back(gwb[i]);
	}

	vector<ValueDecl*> lwb = for_stmt->getGlobalLCWriteBufs();

	for (unsigned i=0; i<lwb.size(); i++)
	{
		if (isInVector(fg._globalLCWriteBufs, lwb[i]))
			continue;
		fg._globalLCWriteBufs.push_back(lwb[i]);
	}
}

void OpenCLGPUTLSHostCode::VisitForStmt(ForStmt *Node) {
	if (Node->getGlobalWriteBufs().size() ||
			Node->getGlobalLCWriteBufs().size())
	{
		insertGPUTLsLog(Node);
	}	

	Indent() << "for (";
	if (Node->getInit()) {
		if (DeclStmt *DS = dyn_cast<DeclStmt>(Node->getInit()))
			PrintRawDeclStmt(DS);
		else
			PrintExpr(cast<Expr>(Node->getInit()));
	}
	OS << ";";
	if (Node->getCond()) {
		OS << " ";
		PrintExpr(Node->getCond());
	}
	OS << ";";
	if (Node->getInc()) {
		OS << " ";
		PrintExpr(Node->getInc());
	}
	OS << ") ";

	if (CompoundStmt *CS = dyn_cast<CompoundStmt>(Node->getBody())) {
		PrintRawCompoundStmt(CS);
		OS << "\n";
	} else {
		OS << "\n";
		PrintStmt(Node->getBody());
	}
}
