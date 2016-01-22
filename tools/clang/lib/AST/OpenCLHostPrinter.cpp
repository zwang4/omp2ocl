//===--- OpenCLHostPrinter.cpp - Printing implementation for Stmt ASTs ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Stmt::dumpPretty/Stmt::printPretty methods, which
// pretty print the AST back out to C code.
//
//===----------------------------------------------------------------------===//

#include "clang/Omp2Ocl/OpenCLHostPrinter.h"
#include "clang/Omp2Ocl/OpenCLRoutine.h"
#include "clang/AST/GlobalCallArgPicker.h"
#include "clang/AST/StmtPrinter.h"
#include "clang/Omp2Ocl/OpenCLCommon.h"
#include "clang/Omp2Ocl/OpenCLHostFuncParamExp.h"
#include "clang/Omp2Ocl/OpenCLHostSerialPrinter.h"
#include "clang/Omp2Ocl/OpenCLCompilerOptions.h"
using namespace clang;

//===----------------------------------------------------------------------===//
// OpenCLHostPrinter Visitor
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Stmt printing methods.
//===----------------------------------------------------------------------===//

bool OpenCLHostPrinter::isInFunctionLevelOCLBuffer(FunctionDecl* D, ValueDecl* expr)
{
	string name = expr->getNameAsString();
	for (unsigned i=0; i<funcOCLBuffers.size(); i++)
	{
		if (funcOCLBuffers[i].D == D)
		{
			if (funcOCLBuffers[i].E->getNameAsString() == name)
				return true;
		}
	}

	return false;
}


void OpenCLHostPrinter::addFunctionLevelOCLBufferObj(ValueDecl* dc)
{
	bool isLocal = !(dc->isDefinedOutsideFunctionOrMethod());
	if (isLocal)
	{
		FunctionDecl* D = OCLCommon::CurrentVisitFunction;
		if (!isInFunctionLevelOCLBuffer(D, dc))
		{
			bool found = false;
			for (unsigned i=0; i<D->getNumParams(); i++)
			{
				ParmVarDecl* PV = D->getParamDecl(i);
				if (PV->getNameAsString() == dc->getNameAsString())
				{
					found = true;
					break;
				}
			}

			if (!found)
			{
				funcOCLBuffers.push_back(FunctionLevelOCLBuffer(D, dc));
			}
		}
	}
}


void OpenCLHostPrinter::prepareOCLBuffersForLocalVars(Stmt* S)
{
	if (!traceLVar)
		return;

	OpenCLHostPrinter hp(llvm::nulls(), Context, NULL, Context.PrintingPolicy, 0, false);
	hp.PrintStmt(S);

	vector<FunctionLevelOCLBuffer>& oclBufs = hp.getFuncLevelOCLBuffers();
	vector<ValueDecl*> stmtDs;

	for (unsigned i=0; i<oclBufs.size(); i++)
	{
		FunctionDecl* FD = oclBufs[i].D;
		if (!(FD->isFLObjDeclare(oclBufs[i].E)))
		{
			FD->addDeclaredFLObj(oclBufs[i].E);
			stmtDs.push_back(oclBufs[i].E);
		}
	}	

	declareOCLBufferForLocalVars(stmtDs);
}

#if 1
void OpenCLHostPrinter::declareOCLBufferForLocalVars(vector<ValueDecl*>& Ds)
{
	for (unsigned i=0; i<Ds.size(); i++)
	{
		ValueDecl* d = Ds[i];
		unsigned dim = getArrayDimension(d);
		if (dim)
		{
			string type = getCononicalType(d);
			vector<unsigned> dims = getArrayDef(type);

			if (dims.size())
			{	
#if 0
				OS << "DECLARE_LOCALVAR_OCL_BUFFER(" << d->getNameAsString() << ", ";
				OS << getGlobalType(type) << ",";

				OS << "(";
				for (unsigned j=0; j<dims.size(); j++)
				{
					if (j > 0)
						OS << " * ";
					OS << dims[j];
				}
				OS << ")";
				OS << ");\n";
#endif
			}
			else
			{
				OS << "DECLARE_LOCALVAR_OCL_BUFFER(" << d->getNameAsString() << ", ";
				OS << getGlobalType(type) << ",(";
				OS << d->getNameAsString() << "_len";
				OS << "));\n";

			}
		}
	}

	if (Ds.size())
		OS << "\n";
	Ds.clear();
}
#endif

void OpenCLHostPrinter::declareOCLBufferForLocalVars(vector<VarDecl*>& Ds)
{
	for (unsigned i=0; i<Ds.size(); i++)
	{
		VarDecl* d = Ds[i];
		if (d->isLocalVarDecl())
		{
			unsigned dim = getArrayDimension(d);
			if (dim)
			{
				string type = getCononicalType(d);
				vector<unsigned> dims = getArrayDef(type);

				if (dims.size())
				{	
					OS << "DECLARE_LOCALVAR_OCL_BUFFER(" << d->getNameAsString() << ", ";
					OS << getGlobalType(type) << ",";

					OS << "(";
					for (unsigned j=0; j<dims.size(); j++)
					{
						if (j > 0)
							OS << " * ";
						OS << dims[j];
					}
					OS << ")";
					OS << ");\n";
				}
			}
		}
	}
	Ds.clear();

}

/// PrintRawCompoundStmt - Print a compound stmt without indenting the {, and
/// with no newline after the }.
void OpenCLHostPrinter::PrintRawCompoundStmt(CompoundStmt *Node) {
	bool function_body_here = false;
	if (isFunctionBody)
	{
		function_body_here = true;
		isFunctionBody = false;
	}
	OS << "{\n";
	for (CompoundStmt::body_iterator I = Node->body_begin(), E = Node->body_end();
			I != E; ++I)
		PrintStmt(*I);

	if (function_body_here)
	{
		if (gpu_tls_handler &&  OCLCompilerOptions::EnableGPUTLs && !OCLCompilerOptions::StrictTLSChecking && ! OCLCompilerOptions::OclTLSMechanism)
		{
			gpu_tls_handler->genTLSCheckingKernelCode(OS);
		}	
	}

	Indent() << "}";
}

void OpenCLHostPrinter::PrintRawDecl(Decl *D) {
	if (dyn_cast<VarDecl>(D))
	{
		VarDecl* dd = dyn_cast<VarDecl>(D);
		stmtDecls.push_back(dd);
	}
	D->print(OS, Policy, IndentLevel);
}

void OpenCLHostPrinter::PrintRawDeclStmt(DeclStmt *S) {
	DeclStmt::decl_iterator Begin = S->decl_begin(), End = S->decl_end();
	llvm::SmallVector<Decl*, 2> Decls;
	for ( ; Begin != End; ++Begin)
	{
		Decls.push_back(*Begin);
		//ZHENG: This is for tracking the variables that are declared by a statement
		Decl* d = (*Begin);
		if (dyn_cast<VarDecl>(d))
		{
			VarDecl* dd = dyn_cast<VarDecl>(d);
			stmtDecls.push_back(dd);
		}
	}

	Decl::printGroup(Decls.data(), Decls.size(), OS, Policy, IndentLevel);
}

void OpenCLHostPrinter::VisitNullStmt(NullStmt *Node) {
	Indent() << ";\n";
}

void OpenCLHostPrinter::VisitDeclStmt(DeclStmt *Node) {
	Indent();
	PrintRawDeclStmt(Node);
	OS << ";\n";
	//ZHENG Declare ocl_buffers for array variables
	declareOCLBufferForLocalVars(stmtDecls);
}

void OpenCLHostPrinter::VisitCompoundStmt(CompoundStmt *Node) {
	Indent();
	PrintRawCompoundStmt(Node);
	OS << "\n";
}

void OpenCLHostPrinter::VisitCaseStmt(CaseStmt *Node) {
	Indent(-1) << "case ";
	PrintExpr(Node->getLHS());
	if (Node->getRHS()) {
		OS << " ... ";
		PrintExpr(Node->getRHS());
	}
	OS << ":\n";

	PrintStmt(Node->getSubStmt(), 0);
}

void OpenCLHostPrinter::VisitDefaultStmt(DefaultStmt *Node) {
	Indent(-1) << "default:\n";
	PrintStmt(Node->getSubStmt(), 0);
}

void OpenCLHostPrinter::VisitLabelStmt(LabelStmt *Node) {
	Indent(-1) << Node->getName() << ":\n";
	PrintStmt(Node->getSubStmt(), 0);
}

void OpenCLHostPrinter::PrintRawIfStmt(IfStmt *If) {
	OS << "if (";
	PrintExpr(If->getCond());
	OS << ')';

	if (CompoundStmt *CS = dyn_cast<CompoundStmt>(If->getThen())) {
		OS << ' ';
		PrintRawCompoundStmt(CS);
		OS << (If->getElse() ? ' ' : '\n');
	} else {
		OS << '\n';
		PrintStmt(If->getThen());
		if (If->getElse()) Indent();
	}

	if (Stmt *Else = If->getElse()) {
		OS << "else";

		if (CompoundStmt *CS = dyn_cast<CompoundStmt>(Else)) {
			OS << ' ';
			PrintRawCompoundStmt(CS);
			OS << '\n';
		} else if (IfStmt *ElseIf = dyn_cast<IfStmt>(Else)) {
			OS << ' ';
			PrintRawIfStmt(ElseIf);
		} else {
			OS << '\n';
			PrintStmt(If->getElse());
		}
	}
}

void OpenCLHostPrinter::VisitIfStmt(IfStmt *If) {
	Indent();
	PrintRawIfStmt(If);
}

void OpenCLHostPrinter::VisitSwitchStmt(SwitchStmt *Node) {
	Indent() << "switch (";
	PrintExpr(Node->getCond());
	OS << ")";

	// Pretty print compoundstmt bodies (very common).
	if (CompoundStmt *CS = dyn_cast<CompoundStmt>(Node->getBody())) {
		OS << " ";
		PrintRawCompoundStmt(CS);
		OS << "\n";
	} else {
		OS << "\n";
		PrintStmt(Node->getBody());
	}
}

void OpenCLHostPrinter::VisitWhileStmt(WhileStmt *Node) {
	Indent() << "while (";
	PrintExpr(Node->getCond());
	OS << ")\n";
	PrintStmt(Node->getBody());
}

void OpenCLHostPrinter::VisitDoStmt(DoStmt *Node) {
	Indent() << "do ";
	if (CompoundStmt *CS = dyn_cast<CompoundStmt>(Node->getBody())) {
		PrintRawCompoundStmt(CS);
		OS << " ";
	} else {
		OS << "\n";
		PrintStmt(Node->getBody());
		Indent();
	}

	OS << "while (";
	PrintExpr(Node->getCond());
	OS << ");\n";
}

unsigned int OpenCLHostPrinter::getLineNumber(SourceLocation Loc)
{
	SourceManager &SM = Context.getSourceManager();
	PresumedLoc PLoc = SM.getPresumedLoc(Loc);
	return PLoc.isValid()? PLoc.getLine() : 0;
}


const char* OpenCLHostPrinter::getFileName(SourceLocation Loc)
{
	SourceManager &SM = Context.getSourceManager();
	PresumedLoc PLoc = SM.getPresumedLoc(Loc);
	return PLoc.getFilename();
}

void OpenCLHostPrinter::releaseGTPBuffer(string kernel, OpenCLInputArgu& arg)
{
	string pointerName = "__ocl_th_" + arg.getNameAsString() + "_" + kernel;
	string buffer_name = "__ocl_buffer_" + arg.getNameAsString() + "_" + kernel;
	string alignSize = DEFAULT_ALIGN_SIZE;
	string sizeofType =  arg.getGType();
	string buffer_size_name = pointerName + "_length";

	OS << "if (" << buffer_size_name << ") {\n";
	releaseOCLBuffer(OS, buffer_name);
	OS << "free(" << pointerName << ");\n";
	OS << pointerName << " = NULL;\n";
	OS << buffer_name << " = NULL;\n";
	OS << buffer_size_name << " = 0;\n";
	OS << "}\n";
}

void OpenCLHostPrinter::releaseGTPBuffers(string kernel, vector<OpenCLInputArgu>& inputArgs)
{
	bool found = false;
	for (unsigned i=0; i<inputArgs.size(); i++)
	{
		if (inputArgs[i].isAGlobalThMemVar())
		{
			found = true;
		}
	}

	if (found)
	{
		OS << "#ifdef OCL_RELEASE_GTP_BUFFERS_IMMEDIATE \n";
		OS << "oclSync();\n";
		for (unsigned i=0; i<inputArgs.size(); i++)
		{
			if (inputArgs[i].isAGlobalThMemVar())
			{
				releaseGTPBuffer(kernel, inputArgs[i]);
			}
		}
		OS << "#endif\n";
	}
}


void OpenCLHostPrinter::generatGTPBuffer(string kernel, OpenCLInputArgu& arg, vector<OpenCLNDRangeVar> GV)
{
	assert(arg.isAGlobalThMemVar() && "OpenCLHostPrinter: This is not a global threadprivate var!");

	vector<unsigned> arrayDefs = getArrayDef(arg.getType());
	unsigned size=arrayDefs[0];
	for (unsigned i=1; i<arrayDefs.size(); i++)
	{
		size = size * arrayDefs[i];
	}
	char buf[64];
	snprintf(buf, 64, "%u", size);

	string sizeofMult = buf;
	sizeofMult = "(" + sizeofMult + "* _ocl_thread_num)";

	string pointerName = "__ocl_th_" + arg.getNameAsString() + "_" + kernel;
	string buffer_name = "__ocl_buffer_" + arg.getNameAsString() + "_" + kernel;
	string alignSize = DEFAULT_ALIGN_SIZE;
	string sizeofType =  arg.getGType();
	string buffer_size = "sizeof(" + sizeofType + ") * " + sizeofMult;
	string buffer_size_name = pointerName + "_length";
	string l_buffer_size = "__ocl_bs_" +  arg.getNameAsString() + "_" + kernel;
#if 0
	OS << "\n// Prepare buffer for " << pointerName << " (START)\n";
	OS << "size_t " << l_buffer_size << " = " << buffer_size << ";\n";
	OS << "if (" << buffer_size_name << " < " << l_buffer_size << ") {\n";
	OS << "	if (" << pointerName << ") {\n";
	OS << "oclSync();\n";
	releaseOCLBuffer(OS, buffer_name);
	OS << "	free(" << buffer_name << ");\n";
	OS << " }\n";

	OS << generateAlignCode(pointerName, alignSize, l_buffer_size);
	createOCLBuffer(OS, pointerName, buffer_name, l_buffer_size);
	OS << buffer_size_name << " = " << l_buffer_size << ";\n";

	OS << "}\n";
	OS << "// Prepare buffer for " << pointerName << " (END)\n";
	OS << "\n";
#endif
	OS << "CREATE_THREAD_PRIVATE_BUF(" << pointerName << "," << buffer_name << "," << sizeofType << "," << sizeofMult << ", DEFAULT_ALIGN_SIZE);\n";
}

//Kernel arguments
void OpenCLHostPrinter::generateKernelArgu(ForStmt* Node, string kernel, OpenCLInputArgu& arg, unsigned& i, vector<OpenCLNDRangeVar> GV, string specified_name)
{
	string argName;
	string name = arg.getNameAsString();
	string type = arg.getGType();

	addFunctionLevelOCLBufferObj(arg.decl);

	if (specified_name.length())
	{
		name = specified_name;
	}

	if (arg.isBuffer)
	{
		argName = "__ocl_buffer_" + name;
#if 0
		if (arg.getFuncDecl())
		{
			if (arg.isFunclLevel())
			{
				argName = argName + "_" + arg.getFuncName();
			}
		}
#endif
	}
	else
	{
		argName = name;
	}

	if (arg.isAGlobalThMemVar())
	{
		//use getKernelName() instead of kernel here, because
		//kernel may be renamed by a reduction phase
		generatGTPBuffer(Node->getKernelName(), arg, GV);
		argName = argName + "_" + Node->getKernelName();
	}


	if (arg.isBuffer)
	{
		OS << "	oclSetKernelArgBuffer(";
	}
	else
	{
		OS << "	oclSetKernelArg(";
	}

	OS << "__ocl_" << kernel << ", " << i << ", ";


	if (arg.isBuffer)
	{
		OS << argName;
	}
	else
	{
		OS << "sizeof(" << type << "), &" << argName;
	}

	OS << ");\n";

	//PRINT OUT THE ORIGINAL BUFFER
	if (arg.isAGlobalThMemVar())
	{
		if (arg.isBuffer && arg.isCopyIn)
		{
			i++;
			argName = "__ocl_buffer_" + name;
			OS << "	oclSetKernelArgBuffer(" << "__ocl_" << kernel << ", " << i << ", ";
			OS << argName << ");\n";
		}

	}
}

void OpenCLHostPrinter::startKernel(string kernel, vector<OpenCLNDRangeVar>& GV, bool genCusGWS)
{
	if (genCusGWS && b_UserWSGMacro)
	{
		OS << "#ifndef USE_DEFINED_WGS\n";
		OS << "	oclRunKernel (__ocl_" << kernel << ", " << GV.size() << ", _ocl_gws);\n";
		OS << "#else\n";
		OS << " size_t __ocl_ls[" << GV.size() << "] = {";
		for (unsigned i=0; i<GV.size(); i++)
		{
			if (i > 0)
			{
				OS << ",";
			}
			OS << "WGS_" << kernel << "_" << i;
		}
		OS << "};\n";
		OS << "	oclRunKernelL(__ocl_" << kernel << ", " << GV.size() << ", _ocl_gws, __ocl_ls," << GV.size() << ");\n";
		OS << "#endif\n";
	}
	else
	{
		OS << "	oclRunKernel (__ocl_" << kernel << ", " << GV.size() << ", _ocl_gws);\n";
	}
}

/*!
 * This generates the work size (i, j, k) for OpenCL Kernels
 *
 */
void OpenCLHostPrinter::generateWorkSize(string kernel, vector<OpenCLNDRangeVar>& GV, vector<OMPMultIterIndex>& multIterIndex, bool hasGTPV, bool handleMLTI)
{
	OS << "	size_t _ocl_gws[" << GV.size() << "];\n";

	unsigned i = 0;
	for (vector<OpenCLNDRangeVar>::iterator iter = GV.begin(); iter != GV.end(); iter++)
	{
		string addition;

		assert(iter->Cond && "Invalid condition for the loop");
		BinaryOperator* op = dyn_cast<BinaryOperator>(iter->Cond);
		assert(op && "The loop bound is not binary operator");

		string opcode = iter->cond_opcode_str;
		Stmt* bound = op->getRHS();

		if ( (opcode == "<=") || (opcode == ">="))
		{
			addition = " + 1";
		}

		if ( (opcode == ">=") || (opcode == ">"))
		{
			OS << "//This swaps a decremental loop\n";
		}

		OS << "_ocl_gws[" << i << "] = (";
		Visit(bound);
		OS << ") - (";
		Visit(iter->Init);
		OS << ")" << addition + ";\n";

		if (iter->hasIncremental)
		{
			OS << OCL_NEAREST_MULTI << "(_ocl_gws[" << i << "], (size_t)" << iter->increment << ");\n";
		}
		i++;
	}

	if (handleMLTI)
	{
		OS << "\n";
		for (unsigned j=0; j<multIterIndex.size(); j++)
		{
			unsigned ii;
			int work_item = -1;
			for (ii=0; ii<GV.size(); ii++)
			{
				if (GV[ii].orig_loop_index == j)
				{
					string s = multIterIndex[GV[ii].orig_loop_index].getVariable();
					work_item = atoi( s.c_str());
					break;
				}
			}

			if (work_item > 1)
			{
				string macroName = kernel + "_MULT_ITER_" + uint2String(j);
				OS << "#ifndef " << macroName << "\n";
				OS << "	#define " << macroName << " " << multIterIndex[j].getVariable() << "\n";
				OS << "#endif\n";	
			}
		}

		OS << "\n";

		for (unsigned j=0; j<multIterIndex.size(); j++)
		{
			unsigned ii;
			int work_item = -1;
			for (ii=0; ii<GV.size(); ii++)
			{
				if (GV[ii].orig_loop_index == j)
				{
					string s = multIterIndex[GV[ii].orig_loop_index].getVariable();
					work_item = atoi( s.c_str());
					break;
				}
			}

			if (work_item > 1)
			{
				string macroName = kernel + "_MULT_ITER_" + uint2String(j);
				OS << OCL_NEAREST_MULTI << "(_ocl_gws[" << ii << "], " << macroName << ");\n";
			}
		}
	}

	OS << "\n";

	OS << "oclGetWorkSize(" << GV.size() << ",_ocl_gws, NULL);\n";

	if (hasGTPV)
	{
		OS << "size_t _ocl_thread_num = (";
		for (unsigned i=0; i<GV.size(); i++)
		{
			if (i > 0)
				OS << "*";
			OS << "_ocl_gws[" << i << "]";
		}

		OS << ");\n";
	}

	if (OCLCompilerOptions::EnableMLFeatureCollection)
	{
		OS << "#ifdef DUMP_ML_FEATURES\n";
		OS << "if (is_enable_ml_record()) {\n";
		OS << "__kernel_" << kernel << "++;\n";
		OS << "__stats_" << kernel << " += ";
		for (unsigned i=0; i<GV.size(); i++)
		{
			if (i > 0) OS << "*";
			OS << "_ocl_gws[" << i << "]";
		}
		OS << ";\n";
		OS << "}\n";
		OS << "#endif\n";
	}
}

void OpenCLHostPrinter::generateReadSet(vector<OpenCLInputArgu>& inputArgs, ForStmt* Node)
{
	for (unsigned i=0; i<inputArgs.size(); i++)
	{
		if (inputArgs[i].isNeedReadSync())
		{
			string name = inputArgs[i].decl->getNameAsString();
			if (!Node->isWriteSet(name) )
			{
				OS << "	oclDevReads(__ocl_buffer_" << name;
#if 0
				if (inputArgs[i].getFuncDecl())
				{	
					if (inputArgs[i].isFunclLevel())
						OS << "_" << inputArgs[i].getFuncName();
				}
#endif

				OS << ");\n";
			}
			else
				if (inputArgs[i].isCopyIn)
				{
					OS << "	oclDevReads(__ocl_buffer_" << name << ");\n";
				}
		}
	}	
}

void OpenCLHostPrinter::generateWriteSet(vector<OpenCLInputArgu>& inputArgs, ForStmt* Node)
{
	for (unsigned i=0; i<inputArgs.size(); i++)
	{
		if (inputArgs[i].isNeedWriteSync())
		{
			string name = inputArgs[i].decl->getNameAsString();
			if (Node->isWriteSet(name))
			{
				OS << "	oclDevWrites(__ocl_buffer_" << name;
#if 0
				if (inputArgs[i].getFuncDecl())
				{
					if (inputArgs[i].isFunclLevel())
						OS << "_" <<inputArgs[i].getFuncName();
				}
#endif
				OS << ");\n";
			}
		}
	}	
}


void OpenCLHostPrinter::generateHostWriteSet(vector<OpenCLInputArgu>& inputArgs)
{
	for (unsigned i=0; i<inputArgs.size(); i++)
	{
		if (inputArgs[i].isNeedWriteSync())
		{
			string name = inputArgs[i].decl->getNameAsString();
			OS << " if (__ocl_buffer_" << name << ") ";
			OS << "	oclHostWrites(__ocl_buffer_" << name;
#if 0
			if (inputArgs[i].getFuncDecl())
			{
				if (inputArgs[i].isFunclLevel())
					OS << "_" <<inputArgs[i].getFuncName();
			}
#endif
			OS << ");\n";
		}
	}

	OS << "oclSync();\n";	
	OS << "\n";
}

void OpenCLHostPrinter::printForSerialVersion(ForStmt* Node)
{
	OS << "#else\n";

	vector<OpenCLInputArgu>& inputArgs = Node->getInputArgs();
	generateHostWriteSet(inputArgs);

	Node->getOMPFor().print(OS);
	OS << "\n";

	OpenCLHostSerialPrinter serial(OS, Context, Helper, Policy, 0);
	serial.Visit(Node);

	OS << "#endif\n";
}

void OpenCLHostPrinter::printReductionGroupBound(vector<OpenCLNDRangeVar>::iterator& iter, unsigned i)
{
	string addition;

	assert(iter->Cond && "Invalid condition for the loop");
	BinaryOperator* op = dyn_cast<BinaryOperator>(iter->Cond);
	assert(op && "The loop bound is not binary operator");

	string opcode = BinaryOperator::getOpcodeStr(op->getOpcode());

	if (!((opcode == "<=") || (opcode == "<")))
	{
		cerr << "Warning: opcode at line: " << getLineNumber(op->getOperatorLoc()) << " is " << opcode << endl;
	}

	if (opcode == "<=")
	{
		addition = " + 1";
	}

	OS << "((";
	Visit(op->getRHS());
	OS << ") - (";
	Visit(iter->Init);
	OS << ")" << addition + ")";

}

void OpenCLHostPrinter::PrintReductionBufferSize(ForStmt* Node)
{
	vector<OpenCLNDRangeVar> GV = Node->getLoopIndex();

	unsigned i = 0;
	OS << "(";
	for (vector<OpenCLNDRangeVar>::iterator iter = GV.begin(); iter != GV.end(); iter++)
	{
		if (i > 0)
			OS << " * ";
		OS << "_ocl_gws[" << i << "]";

		i++;
	}

	OS << ");\n";
}

void OpenCLHostPrinter::ReductionSecondPhase(ForStmt *Node)
{
	vector<ValueDecl*>& reducObjs = Node->getReductionVariables();
	vector<OpenCLNDRangeVar> GV = Node->getLoopIndex();

	OS << "//Reduction Step 2\n";
	OS << "unsigned __ocl_num_block = __ocl_buf_size / (GROUP_SIZE * " << DEFAULT_VECTOR_SIZE << "); /*Vectorisation by a factor of " << DEFAULT_VECTOR_SIZE << "*/\n";	

	//Prepare reduction buffers	
	for (unsigned i=0; i<reducObjs.size(); i++)
	{
		ValueDecl* d = reducObjs[i];
		string type = getGlobalType(d->getType().getAsString());
		string name = d->getNameAsString();

		//	OS << type << " *__ocl_output_" << name << ";\n";
		string output_buffer_name = "__ocl_output_" + name + "_" + Node->getKernelName();
		string output_ocl_buffer_name  = "__ocl_output_buffer_" + name + "_" + Node->getKernelName();
		string output_buffer_size = output_buffer_name + "_size";

		OS << CREATE_REDUCTION_STEP2_BUFFER << "(" << output_buffer_size << ", __ocl_num_block, " << DEFAULT_ALIGN_SIZE << ", " << output_ocl_buffer_name << ", " << output_buffer_name << ", " << type << ");\n";
	}

	unsigned ii=0;
	string kernelName = Node->getKernelName() + "_reduction_step2";

	for (unsigned i=0; i<reducObjs.size(); i++)
	{
		ValueDecl* d = reducObjs[i];
		string type = getGlobalType(d->getType().getAsString());
		string name = d->getNameAsString();

		string input_buffer_name = "__ocl_buffer_" + name + "_" + Node->getKernelName();
		string output_ocl_buffer_name  = "__ocl_output_buffer_" + name + "_" + Node->getKernelName();

		OS << "	oclSetKernelArgBuffer(";
		OS << "__ocl_" << kernelName << ", " << ii << ", ";
		OS << input_buffer_name;
		OS << ");\n";
		ii++;


		OS << "	oclSetKernelArgBuffer(";
		OS << "__ocl_" << kernelName << ", " << ii << ", ";
		OS << output_ocl_buffer_name;
		OS << ");\n";
		ii++;

	}

	OS << "\n";


	for (unsigned i=0; i<reducObjs.size(); i++)
	{
		ValueDecl* d = reducObjs[i];
		string type = getGlobalType(d->getType().getAsString());
		string name = d->getNameAsString();

		string input_buffer_name = "__ocl_buffer_" + name + "_" + Node->getKernelName();
		string output_ocl_buffer_name  = "__ocl_output_buffer_" + name + "_" + Node->getKernelName();

		OS << "oclDevWrites(" << output_ocl_buffer_name << ");\n";
	}

	OS << "\n";
	OS << "size_t __ocl_globalThreads[] = {__ocl_buf_size / " << DEFAULT_VECTOR_SIZE << " }; /* Each work item performs " << DEFAULT_VECTOR_SIZE << " reductions*/\n";
	OS << "size_t __ocl_localThreads[] = {GROUP_SIZE};\n";
	OS << "\n";

	OS << "oclRunKernelL(__ocl_" << kernelName << ", 1, __ocl_globalThreads, __ocl_localThreads);\n";

	if (OCLCompilerOptions::EnableMLFeatureCollection)
	{
		OS << "#ifdef DUMP_ML_FEATURES\n";
		OS << "__stats_" << kernelName << " += __ocl_globalThreads[0];\n";
		OS << "__kernel_" << kernelName << "++;\n";
		OS << "#endif\n";
	}

	OS << "\n";
}

bool OpenCLHostPrinter::hasGThPrivate(vector<OpenCLInputArgu>& inputArgs)
{
	for (unsigned i=0; i<inputArgs.size(); i++)
	{
		if (inputArgs[i].isAGlobalThMemVar())
			return true;
	}

	return false;
}

//
//Generate kernel arguments for GPU TLS threads
void OpenCLHostPrinter::genGPUTLSBuffer(ForStmt*& Node, string& kernel, unsigned int &k)
{
	vector<ValueDecl*> globalWBufs = Node->getGlobalWriteBufs();
	vector<ValueDecl*> globalLCWBufs = Node->getGlobalLCWriteBufs();

	for (unsigned i=0; i<globalWBufs.size(); i++)
	{
		string name = globalWBufs[i]->getName();
		if (Node->isInTLSCeheckingVec(name))
		{
			OS << "oclSetKernelArgBuffer(__ocl_" << kernel << ", " << k++ << ", rd_oclb_" << name << ");\n";
			OS << "oclSetKernelArgBuffer(__ocl_" << kernel << ", " << k++ << ", wr_oclb_" << name << ");\n";
		}
	}

	for (unsigned i=0; i<globalLCWBufs.size(); i++)
	{
		string name = globalLCWBufs[i]->getName();
		if (Node->isInTLSCeheckingVec(name))
		{
			OS << "oclSetKernelArgBuffer(__ocl_" << kernel << ", " << k++ << ", rd_oclb_" << name << ");\n";
			OS << "oclSetKernelArgBuffer(__ocl_" << kernel << ", " << k++ << ", wr_oclb_" << name << ");\n";
		}
	}

	if (OCLCompilerOptions::OclTLSMechanism)
	{
		OS << "oclSetKernelArgBuffer(__ocl_" << kernel << "," << k++ << "," << "__oclb_gpu_tls_conflict_flag);\n";
	}
}

//The first step of reduction
void OpenCLHostPrinter::ReductionFirstPhase(ForStmt *Node)
{
	vector<ValueDecl*>& reducObjs = Node->getReductionVariables();
	vector<OpenCLNDRangeVar> GV = Node->getLoopIndex();
	//vector<OMPMultIterIndex>& omi = Node->getOMPFor().getMultIterIndex();
	vector<OpenCLInputArgu>& inputArgs = Node->getInputArgs();
	string kernelName = Node->getKernelName() + "_reduction_step1";

	OS << "	//------------------------------------------\n";
	OS << "	//Reduction step 1\n";
	OS << "	//------------------------------------------\n";
	generateWorkSize(kernelName, GV, Node->getOMPFor().getMultIterIndex(), hasGThPrivate(inputArgs), true);

	OS << "	size_t __ocl_act_buf_size = ";
	PrintReductionBufferSize(Node);
	OS << "REDUCTION_STEP1_MULT_NDRANGE();\n";

	//Prepare reduction buffers	
	for (unsigned i=0; i<reducObjs.size(); i++)
	{
		ValueDecl* d = reducObjs[i];
		string type = getGlobalType(d->getType().getAsString());
		string name = d->getNameAsString();

		string size_name = "__ocl_buffer_" + name + "_" + Node->getKernelName() + "_size";
		string buffer_name = "__ocl_buffer_" + name + "_" + Node->getKernelName();

		OS << "//Prepare buffer for the reduction variable: " << name << "\n";
		OS << CREATE_REDUCTION_STEP1_BUFFER << "(" << size_name << ", __ocl_buf_size, " <<  buffer_name << ", " << type << ");\n";
	}

	OS << "\n";

	vector<OpenCLInputArgu> reducArgs;
	unsigned ii=0;

	for (unsigned j=0; j<reducObjs.size(); j++)
	{
		reducArgs.push_back(OpenCLInputArgu(reducObjs[j], true, false, false, NULL));
	}

	OS << "	//------------------------------------------\n";
	OS << "	//OpenCL kernel arguments (BEGIN) \n";
	OS << "	//------------------------------------------\n";
	//Spawn buffer preparation routine
	{
		unsigned ip = 0;
		string pKernelName = Node->getKernelName() + "_reduction_step0";

		OS << "//init the round-up buffer spaces so that I can apply vectorisation on the second step\n";
		OS << "if (__ocl_buf_size > __ocl_act_buf_size) {\n";

		for (unsigned j=0; j<reducArgs.size(); j++)
		{
			string buffer_name = reducArgs[j].getNameAsString() + "_" + Node->getKernelName();
			generateKernelArgu(Node, pKernelName, reducArgs[j], ip, GV, buffer_name);
			ip++;
		}

		OS << "unsigned int __ocl_buffer_offset = __ocl_buf_size - __ocl_act_buf_size;\n";
		OS << "	oclSetKernelArg(__ocl_" << pKernelName << ", " << ip << ", sizeof(unsigned int), &__ocl_act_buf_size);\n";
		ip++;
		OS << "	oclSetKernelArg(__ocl_" << pKernelName << ", " << ip << ", sizeof(unsigned int), &__ocl_buffer_offset);\n";
		ip++;

		OS << "\n";

		OS << " size_t __offset_work_size[1] = { __ocl_buffer_offset };\n";

		OS << "	oclRunKernel (__ocl_" << pKernelName << ", 1, __offset_work_size);\n";
		if (OCLCompilerOptions::EnableMLFeatureCollection)
		{
			OS << "#ifdef DUMP_ML_FEATURES\n";
			OS << "__stats_" << pKernelName << " += __ocl_buffer_offset;\n";
			OS << "__kernel_" << pKernelName << " += __ocl_buffer_offset;\n";
			OS << "#endif\n";
		}
		OS << "}\n\n";
	}

	//NOTICE: SHOULDN'T USE ii to index, because ii can be changed in generateKernelArgu
	//GV stores the OpenCL get_global_id() indexs
	unsigned index = 0;
	for (ii=0; ii<inputArgs.size(); ii++)
	{
		generateKernelArgu(Node, kernelName, inputArgs[ii], index, GV);
		index++;
	}

	OS << "	//------------------------------------------\n";
	OS << "	//OpenCL kernel arguments (BEGIN) \n";
	OS << "	//------------------------------------------\n";

	//OpenCL Indexs
	for (vector<OpenCLNDRangeVar>::iterator iter = GV.begin(); iter != GV.end(); iter++)
	{
		if (iter->hasIncremental && !iter->isIncInt)
		{
			OS << "	 " << iter->type + " __ocl_" + iter->variable + "_inc = ";
			OS << iter->increment;
			OS << ";\n";

			OS << "	oclSetKernelArg(__ocl_" << kernelName << ", " << index << ", sizeof(" << iter->type << "), &" << "__ocl_" + iter->variable + "_inc);\n";
			index++;
		}

		if (iter->Cond && !iter->isCondInt)
		{
			OS << "	 " << iter->type + " __ocl_" + iter->variable + "_bound = ";

			BinaryOperator* op = dyn_cast<BinaryOperator>(iter->Cond);
			assert(op && "The loop bound is not binary operator");

			Visit(op->getRHS());
			OS << ";\n";

			OS << "	oclSetKernelArg(__ocl_" << kernelName << ", " << index << ", sizeof(" << iter->type << "), &" << "__ocl_" + iter->variable + "_bound);\n";
			index++;
		}
	}


	for (unsigned j=0; j<reducArgs.size(); j++)
	{
		string buffer_name = reducArgs[j].getNameAsString() + "_" + Node->getKernelName();
		generateKernelArgu(Node, kernelName, reducArgs[j], index, GV, buffer_name);
		index++;
	}

	OS << "	//------------------------------------------\n";
	OS << "	//OpenCL kernel arguments (END) \n";
	OS << "	//------------------------------------------\n\n";

	OS << "	//------------------------------------------\n";
	OS << "	//OpenCL kernel arguments (END) \n";
	OS << "	//------------------------------------------\n";
	OS << "\n";

	OS << "	//------------------------------------------\n";
	OS << "	//Write set (BEGIN) \n";
	OS << "	//------------------------------------------\n";
	generateWriteSet(inputArgs, Node);
	OS << "	//------------------------------------------\n";
	OS << "	//Write set (END) \n";
	OS << "	//------------------------------------------\n";

#if 0
	for (unsigned i=0; i<reducObjs.size(); i++)
	{
		ValueDecl* d = reducObjs[i];
		string name = d->getNameAsString();
		string buffer_name = "__ocl_buffer_" + name + "_" + Node->getKernelName();
		OS << "oclDevWrites(" << buffer_name << ");\n";
	}
#endif
	OS << "	//------------------------------------------\n";
	OS << "	//Read only buffers (BEGIN) \n";
	OS << "	//------------------------------------------\n";
	generateReadSet(inputArgs, Node);
	OS << "	//------------------------------------------\n";
	OS << "	//Read only buffers (END) \n";
	OS << "	//------------------------------------------\n\n";

	startKernel(kernelName, GV, false);

	releaseGTPBuffers(Node->getKernelName(), inputArgs);
	OS << "\n";
}

void OpenCLHostPrinter::ReductionFinalCPUStage(ForStmt* Node)
{
	vector<ValueDecl*>& reducObjs = Node->getReductionVariables();
	vector<OMPReductionObj>& reductionObjs = Node->getOMPFor().getReductionObjs();

	assert(reducObjs.size() ==reductionObjs.size() && "OS of boundery for reduction objects");

	OS << "//Do the final reduction part on the CPU\n";

	for (unsigned i = 0; i < reducObjs.size(); i++)
	{
		ValueDecl* d = reducObjs[i];

		string type = getGlobalType(d->getType().getAsString());
		string name = d->getNameAsString();
		string output_ocl_buffer_name  = "__ocl_output_buffer_" + name + "_" + Node->getKernelName();

		OS << "oclHostReads(" << output_ocl_buffer_name << ");\n";
	}

	OS << "oclSync();\n";
	OS << "\n";

	OS << "for (unsigned __ocl_i=0; __ocl_i < __ocl_num_block; __ocl_i++) {\n";

	for (unsigned i = 0; i < reducObjs.size(); i++)
	{
		ValueDecl* d = reducObjs[i];

		OMPReductionObj& obj = reductionObjs[i];
		string name = d->getNameAsString();
		string output_buffer_name = "__ocl_output_" + name + "_" + Node->getKernelName();

		OS << name << " = " << name << getOpCodeFromString (obj.getOperatorCode());
		OS << output_buffer_name << "[__ocl_i];\n";
	}

	OS << "}\n";

	OS << "\n";
}

void OpenCLHostPrinter::ReductionLoop(ForStmt *Node)
{
	SourceLocation loc = Node->getForLoc();
	string kernelName = Node->getKernelName();

	if (b_KernelMacro)
	{
		OS << "#if defined(ENABLE_OCL_KERNEL_" << kernelName + "_reduction_step1) && defined(ENABLE_OCL_KERNEL_" << kernelName + "_reduction_step2)\n";
	}
	OS << "	//--------------------------------------------------------------\n";
	OS << " //Loop defined at line "<< getLineNumber(loc) << " of " << getFileName(loc) << "\n";
	OS << "	//--------------------------------------------------------------\n";
	OS << "{\n";

	ReductionFirstPhase(Node);
	OS << "\n";

	ReductionSecondPhase(Node);

	OS << "\n";

	ReductionFinalCPUStage(Node);

	OS << "}\n";
}

void OpenCLHostPrinter::VisitForStmt(ForStmt *Node) {

	if (Node->isParallelForLoop())
	{
		if (Node->hasReductionVariable())
		{
			ReductionLoop(Node);
		}
		else
		{
			SourceLocation loc = Node->getForLoc();
			string kernelName = Node->getKernelName();
			vector<OpenCLNDRangeVar> GV = Node->getLoopIndex();

			if (b_KernelMacro)
			{
				OS << "#ifdef ENABLE_OCL_KERNEL_" << kernelName << "\n";
			}

			OS << "	//--------------------------------------------------------------\n";
			OS << " //Loop defined at line "<< getLineNumber(loc) << " of " << getFileName(loc) << "\n";
			OS << "	//--------------------------------------------------------------\n";

			vector<OpenCLInputArgu>& inputArgs = Node->getInputArgs();
			unsigned i=0;

			OS << "{\n";

			OS << "	//------------------------------------------\n";
			OS << "	//OpenCL kernel arguments (BEGIN) \n";
			OS << "	//------------------------------------------\n";
			generateWorkSize(kernelName, GV, Node->getOMPFor().getMultIterIndex(), hasGThPrivate(inputArgs));

			unsigned index = 0;
			for (i=0; i<inputArgs.size(); i++)
			{
				generateKernelArgu(Node, kernelName, inputArgs[i], index, GV);
				index++;
			}

			//Readset
			for (vector<OpenCLNDRangeVar>::iterator iter = GV.begin(); iter != GV.end(); iter++)
			{
				if (iter->hasIncremental && !iter->isIncInt)
				{
					OS << "	 " << iter->type + " __ocl_" + iter->variable + "_inc = ";
					OS << iter->increment;
					OS << ";\n";

					OS << "	oclSetKernelArg(__ocl_" << kernelName << ", " << index << ", sizeof(" << iter->type << "), &" << "__ocl_" + iter->variable + "_inc);\n";
					index++;
				}

				if (iter->Cond && !iter->isCondInt)
				{
					OS << "	 " << iter->type + " __ocl_" + iter->variable + "_bound = ";

					BinaryOperator* op = dyn_cast<BinaryOperator>(iter->Cond);
					assert(op && "The loop bound is not binary operator");

					Visit(op->getRHS());
					OS << ";\n";

					OS << "	oclSetKernelArg(__ocl_" << Node->getKernelName() << ", " << index << ", sizeof(" << iter->type << "), &" << "__ocl_" + iter->variable + "_bound);\n";
					index++;
				}
			}

			genGPUTLSBuffer(Node, kernelName, index);

			OS << "	//------------------------------------------\n";
			OS << "	//OpenCL kernel arguments (END) \n";
			OS << "	//------------------------------------------\n";
			OS << "\n";
			OS << "	//------------------------------------------\n";
			OS << "	//Write set (BEGIN) \n";
			OS << "	//------------------------------------------\n";
			generateWriteSet(inputArgs, Node);
			OS << "	//------------------------------------------\n";
			OS << "	//Write set (END) \n";
			OS << "	//------------------------------------------\n";

			OS << "	//------------------------------------------\n";
			OS << "	//Read only variables (BEGIN) \n";
			OS << "	//------------------------------------------\n";
			generateReadSet(inputArgs, Node);
			OS << "	//------------------------------------------\n";
			OS << "	//Read only variables (END) \n";
			OS << "	//------------------------------------------\n";
			OS << "\n";

			startKernel(kernelName, GV);

			OS << " #ifdef __STRICT_SYNC__\n";
			OS << " oclSync();\n";
			OS << " #endif\n";

			releaseGTPBuffers(Node->getKernelName(), inputArgs);

			//Check GPU TLS at the end of a for loop
			if (OCLCompilerOptions::EnableGPUTLs && OCLCompilerOptions::StrictTLSChecking)
			{
				OpenCLGPUTLSHostCode::genTLSKernelCallLoopLevel(OS, Node);
			}	
			
			OS << "}\n";


			OS << "\n";
		}

		if (b_KernelMacro)
		{
			printForSerialVersion(Node);
		}

		OS << "\n";
	}
	else
	{
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
}

void OpenCLHostPrinter::VisitGotoStmt(GotoStmt *Node) {
	Indent() << "goto " << Node->getLabel()->getName() << ";\n";
}

void OpenCLHostPrinter::VisitIndirectGotoStmt(IndirectGotoStmt *Node) {
	Indent() << "goto *";
	PrintExpr(Node->getTarget());
	OS << ";\n";
}

void OpenCLHostPrinter::VisitContinueStmt(ContinueStmt *Node) {
	Indent() << "continue;\n";
}

void OpenCLHostPrinter::VisitBreakStmt(BreakStmt *Node) {
	Indent() << "break;\n";
}


void OpenCLHostPrinter::VisitReturnStmt(ReturnStmt *Node) {
	Indent() << "return";
	if (Node->getRetValue()) {
		OS << " ";
		PrintExpr(Node->getRetValue());
	}
	OS << ";\n";
}


void OpenCLHostPrinter::VisitAsmStmt(AsmStmt *Node) {
	Indent() << "__asm__ ";

	if (Node->isVolatile())
		OS << "__volatile__";

	OS << "(";
	VisitStringLiteral(Node->getAsmString());

	// OSputs
	if (Node->getNumOutputs() != 0 || Node->getNumInputs() != 0 ||
			Node->getNumClobbers() != 0)
		OS << " : ";

	for (unsigned i = 0, e = Node->getNumOutputs(); i != e; ++i) {
		if (i != 0)
			OS << ", ";

		if (!Node->getOutputName(i).empty()) {
			OS << '[';
			OS << Node->getOutputName(i);
			OS << "] ";
		}

		VisitStringLiteral(Node->getOutputConstraintLiteral(i));
		OS << " ";
		Visit(Node->getOutputExpr(i));
	}

	// Inputs
	if (Node->getNumInputs() != 0 || Node->getNumClobbers() != 0)
		OS << " : ";

	for (unsigned i = 0, e = Node->getNumInputs(); i != e; ++i) {
		if (i != 0)
			OS << ", ";

		if (!Node->getInputName(i).empty()) {
			OS << '[';
			OS << Node->getInputName(i);
			OS << "] ";
		}

		VisitStringLiteral(Node->getInputConstraintLiteral(i));
		OS << " ";
		Visit(Node->getInputExpr(i));
	}

	// Clobbers
	if (Node->getNumClobbers() != 0)
		OS << " : ";

	for (unsigned i = 0, e = Node->getNumClobbers(); i != e; ++i) {
		if (i != 0)
			OS << ", ";

		VisitStringLiteral(Node->getClobber(i));
	}

	OS << ");\n";
}

void OpenCLHostPrinter::VisitObjCAtTryStmt(ObjCAtTryStmt *Node) {
	Indent() << "@try";
	if (CompoundStmt *TS = dyn_cast<CompoundStmt>(Node->getTryBody())) {
		PrintRawCompoundStmt(TS);
		OS << "\n";
	}

	for (unsigned I = 0, N = Node->getNumCatchStmts(); I != N; ++I) {
		ObjCAtCatchStmt *catchStmt = Node->getCatchStmt(I);
		Indent() << "@catch(";
		if (catchStmt->getCatchParamDecl()) {
			if (Decl *DS = catchStmt->getCatchParamDecl())
				PrintRawDecl(DS);
		}
		OS << ")";
		if (CompoundStmt *CS = dyn_cast<CompoundStmt>(catchStmt->getCatchBody())) {
			PrintRawCompoundStmt(CS);
			OS << "\n";
		}
	}

	if (ObjCAtFinallyStmt *FS = static_cast<ObjCAtFinallyStmt *>(
				Node->getFinallyStmt())) {
		Indent() << "@finally";
		PrintRawCompoundStmt(dyn_cast<CompoundStmt>(FS->getFinallyBody()));
		OS << "\n";
	}
}

void OpenCLHostPrinter::VisitObjCAtFinallyStmt(ObjCAtFinallyStmt *Node) {
}

void OpenCLHostPrinter::VisitObjCAtCatchStmt (ObjCAtCatchStmt *Node) {
	Indent() << "@catch (...) { /* todo */ } \n";
}

void OpenCLHostPrinter::VisitObjCAtThrowStmt(ObjCAtThrowStmt *Node) {
	Indent() << "@throw";
	if (Node->getThrowExpr()) {
		OS << " ";
		PrintExpr(Node->getThrowExpr());
	}
	OS << ";\n";
}

void OpenCLHostPrinter::VisitObjCAtSynchronizedStmt(ObjCAtSynchronizedStmt *Node) {
	Indent() << "@synchronized (";
	PrintExpr(Node->getSynchExpr());
	OS << ")";
	PrintRawCompoundStmt(Node->getSynchBody());
	OS << "\n";
}

void OpenCLHostPrinter::PrintRawCXXCatchStmt(CXXCatchStmt *Node) {
	OS << "catch (";
	if (Decl *ExDecl = Node->getExceptionDecl())
		PrintRawDecl(ExDecl);
	else
		OS << "...";
	OS << ") ";
	PrintRawCompoundStmt(cast<CompoundStmt>(Node->getHandlerBlock()));
}

void OpenCLHostPrinter::VisitCXXCatchStmt(CXXCatchStmt *Node) {
	Indent();
	PrintRawCXXCatchStmt(Node);
	OS << "\n";
}

void OpenCLHostPrinter::VisitCXXTryStmt(CXXTryStmt *Node) {
	Indent() << "try ";
	PrintRawCompoundStmt(Node->getTryBlock());
	for (unsigned i = 0, e = Node->getNumHandlers(); i < e; ++i) {
		OS << " ";
		PrintRawCXXCatchStmt(Node->getHandler(i));
	}
	OS << "\n";
}

//===----------------------------------------------------------------------===//
//  Expr printing methods.
//===----------------------------------------------------------------------===//

void OpenCLHostPrinter::VisitDeclRefExpr(DeclRefExpr *Node) {
	if (NestedNameSpecifier *Qualifier = Node->getQualifier())
		Qualifier->print(OS, Policy);
	OS << Node->getNameInfo();
	if (Node->hasExplicitTemplateArgs())
		OS << TemplateSpecializationType::PrintTemplateArgumentList(
				Node->getTemplateArgs(),
				Node->getNumTemplateArgs(),
				Policy);  

}

void OpenCLHostPrinter::VisitDependentScopeDeclRefExpr(
		DependentScopeDeclRefExpr *Node) {
	if (NestedNameSpecifier *Qualifier = Node->getQualifier())
		Qualifier->print(OS, Policy);
	OS << Node->getNameInfo();
	if (Node->hasExplicitTemplateArgs())
		OS << TemplateSpecializationType::PrintTemplateArgumentList(
				Node->getTemplateArgs(),
				Node->getNumTemplateArgs(),
				Policy);
}

void OpenCLHostPrinter::VisitUnresolvedLookupExpr(UnresolvedLookupExpr *Node) {
	if (Node->getQualifier())
		Node->getQualifier()->print(OS, Policy);
	OS << Node->getNameInfo();
	if (Node->hasExplicitTemplateArgs())
		OS << TemplateSpecializationType::PrintTemplateArgumentList(
				Node->getTemplateArgs(),
				Node->getNumTemplateArgs(),
				Policy);
}

void OpenCLHostPrinter::VisitObjCIvarRefExpr(ObjCIvarRefExpr *Node) {
	if (Node->getBase()) {
		PrintExpr(Node->getBase());
		OS << (Node->isArrow() ? "->" : ".");
	}
	OS << Node->getDecl();
}

void OpenCLHostPrinter::VisitObjCPropertyRefExpr(ObjCPropertyRefExpr *Node) {
	if (Node->isSuperReceiver())
		OS << "super.";
	else if (Node->getBase()) {
		PrintExpr(Node->getBase());
		OS << ".";
	}

	if (Node->isImplicitProperty())
		OS << Node->getImplicitPropertyGetter()->getSelector().getAsString();
	else
		OS << Node->getExplicitProperty()->getName();
}

void OpenCLHostPrinter::VisitPredefinedExpr(PredefinedExpr *Node) {
	switch (Node->getIdentType()) {
		default:
			assert(0 && "unknown case");
		case PredefinedExpr::Func:
			OS << "__func__";
			break;
		case PredefinedExpr::Function:
			OS << "__FUNCTION__";
			break;
		case PredefinedExpr::PrettyFunction:
			OS << "__PRETTY_FUNCTION__";
			break;
	}
}

void OpenCLHostPrinter::VisitCharacterLiteral(CharacterLiteral *Node) {
	unsigned value = Node->getValue();
	if (Node->isWide())
		OS << "L";
	switch (value) {
		case '\\':
			OS << "'\\\\'";
			break;
		case '\'':
			OS << "'\\''";
			break;
		case '\a':
			// TODO: K&R: the meaning of '\\a' is different in traditional C
			OS << "'\\a'";
			break;
		case '\b':
			OS << "'\\b'";
			break;
			// Nonstandard escape sequence.
			/*case '\e':
			  OS << "'\\e'";
			  break;*/
		case '\f':
			OS << "'\\f'";
			break;
		case '\n':
			OS << "'\\n'";
			break;
		case '\r':
			OS << "'\\r'";
			break;
		case '\t':
			OS << "'\\t'";
			break;
		case '\v':
			OS << "'\\v'";
			break;
		default:
			if (value < 256 && isprint(value)) {
				OS << "'" << (char)value << "'";
			} else if (value < 256) {
				OS << "'\\x" << llvm::format("%x", value) << "'";
			} else {
				// FIXME what to really do here?
				OS << value;
			}
	}
}

void OpenCLHostPrinter::VisitIntegerLiteral(IntegerLiteral *Node) {
	bool isSigned = Node->getType()->isSignedIntegerType();
	OS << Node->getValue().toString(10, isSigned);

	// Emit suffixes.  Integer literals are always a builtin integer type.
	switch (Node->getType()->getAs<BuiltinType>()->getKind()) {
		default: assert(0 && "Unexpected type for integer literal!");
		case BuiltinType::Int:       break; // no suffix.
		case BuiltinType::UInt:      OS << 'U'; break;
		case BuiltinType::Long:      OS << 'L'; break;
		case BuiltinType::ULong:     OS << "UL"; break;
		case BuiltinType::LongLong:  OS << "LL"; break;
		case BuiltinType::ULongLong: OS << "ULL"; break;
	}
}
void OpenCLHostPrinter::VisitFloatingLiteral(FloatingLiteral *Node) {
	// FIXME: print value more precisely.
	//OS << Node->getValueAsApproximateDouble();
	OS << Node->getLexString();
}

void OpenCLHostPrinter::VisitImaginaryLiteral(ImaginaryLiteral *Node) {
	PrintExpr(Node->getSubExpr());
	OS << "i";
}

void OpenCLHostPrinter::VisitStringLiteral(StringLiteral *Str) {
	if (Str->isWide()) OS << 'L';
	OS << '"';

	// FIXME: this doesn't print wstrings right.
	llvm::StringRef StrData = Str->getString();
	for (llvm::StringRef::iterator I = StrData.begin(), E = StrData.end(); 
			I != E; ++I) {
		unsigned char Char = *I;

		switch (Char) {
			default:
				if (isprint(Char))
					OS << (char)Char;
				else  // OSput anything hard as an octal escape.
					OS << '\\'
						<< (char)('0'+ ((Char >> 6) & 7))
						<< (char)('0'+ ((Char >> 3) & 7))
						<< (char)('0'+ ((Char >> 0) & 7));
				break;
				// Handle some common non-printable cases to make dumps prettier.
			case '\\': OS << "\\\\"; break;
			case '"': OS << "\\\""; break;
			case '\n': OS << "\\n"; break;
			case '\t': OS << "\\t"; break;
			case '\a': OS << "\\a"; break;
			case '\b': OS << "\\b"; break;
		}
	}
	OS << '"';
}
void OpenCLHostPrinter::VisitParenExpr(ParenExpr *Node) {
	OS << "(";
	PrintExpr(Node->getSubExpr());
	OS << ")";
}
void OpenCLHostPrinter::VisitUnaryOperator(UnaryOperator *Node) {
	if (!Node->isPostfix()) {
		OS << UnaryOperator::getOpcodeStr(Node->getOpcode());

		// Print a space if this is an "identifier operator" like __real, or if
		// it might be concatenated incorrectly like '+'.
		switch (Node->getOpcode()) {
			default: break;
			case UO_Real:
			case UO_Imag:
			case UO_Extension:
					 OS << ' ';
					 break;
			case UO_Plus:
			case UO_Minus:
					 if (isa<UnaryOperator>(Node->getSubExpr()))
						 OS << ' ';
					 break;
		}
	}
	PrintExpr(Node->getSubExpr());

	if (Node->isPostfix())
		OS << UnaryOperator::getOpcodeStr(Node->getOpcode());
}

void OpenCLHostPrinter::VisitOffsetOfExpr(OffsetOfExpr *Node) {
	OS << "__builtin_offsetof(";
	OS << Node->getTypeSourceInfo()->getType().getAsString(Policy) << ", ";
	bool PrintedSomething = false;
	for (unsigned i = 0, n = Node->getNumComponents(); i < n; ++i) {
		OffsetOfExpr::OffsetOfNode ON = Node->getComponent(i);
		if (ON.getKind() == OffsetOfExpr::OffsetOfNode::Array) {
			// Array node
			OS << "[";
			PrintExpr(Node->getIndexExpr(ON.getArrayExprIndex()));
			OS << "]";
			PrintedSomething = true;
			continue;
		}

		// Skip implicit base indirections.
		if (ON.getKind() == OffsetOfExpr::OffsetOfNode::Base)
			continue;

		// Field or identifier node.
		IdentifierInfo *Id = ON.getFieldName();
		if (!Id)
			continue;

		if (PrintedSomething)
			OS << ".";
		else
			PrintedSomething = true;
		OS << Id->getName();    
	}
	OS << ")";
}

void OpenCLHostPrinter::VisitUnaryExprOrTypeTraitExpr(UnaryExprOrTypeTraitExpr *Node){
	switch(Node->getKind()) {
		case UETT_SizeOf:
			OS << "sizeof";
			break;
		case UETT_AlignOf:
			OS << "__alignof";
			break;
		case UETT_VecStep:
			OS << "vec_step";
			break;
	}
	if (Node->isArgumentType())
		OS << "(" << Node->getArgumentType().getAsString(Policy) << ")";
	else {
		OS << " ";
		PrintExpr(Node->getArgumentExpr());
	}
}

void OpenCLHostPrinter::VisitArraySubscriptExpr(ArraySubscriptExpr *Node) {
	PrintExpr(Node->getLHS());

	OS << "[";

	PrintExpr(Node->getRHS());

	OS << "]";
}

void OpenCLHostPrinter::PrintCallArgs(CallExpr *Call) {
	vector<unsigned> gIds;
	vector<CallArgInfo> gCs;
	CallArgInfoContainer* cac = Call->getOffsetedArgument();

	if (cac)
	{
		gCs = cac->gCallArgs;
		for (unsigned i=0; i<gCs.size(); i++)
		{
			if (gCs[i].isPointerAccess)
			{
				gIds.push_back(i);
			}
		}
	}

	for (unsigned i = 0, e = Call->getNumArgs(); i != e; ++i) {
		if (isa<CXXDefaultArgExpr>(Call->getArg(i))) {
			// Don't print any defaulted arguments
			break;
		}

		if (i) OS << ", ";

		bool found=false;
		for (unsigned j=0; j<gIds.size(); j++)
		{
			if (gIds[i] == i)
			{
				found = true;
				break;
			}
		}

		//This should be passed as a pointer to the global memory object
		if (found)
		{
			OS << gCs[i].name;
		}
		else
		{
			OpenCLHostFuncParamExp pe (Context, OS);
			//		string string_buf;
			//		llvm::raw_string_ostream O(string_buf);
			OpenCLHostFuncParamExp pc (Context, llvm::nulls());
			if (pc.shouldCallArgPassedWithOclBuffer(Call, i))
			{
				StmtPicker sp(llvm::nulls(), Context, NULL, Context.PrintingPolicy);
				Expr* ee = Call->getArg(i);
				sp.Visit(ee);
				DeclRefExpr* dc = sp.getFirstDecl();

				if (dc)
				{
					addFunctionLevelOCLBufferObj(dc->getDecl());
				}
			}

			pe.VisitCallArg(Call, i);
			//PrintExpr(Call->getArg(i));
		}
	}

	//Print offset arguments
	for (unsigned j=0; j<gIds.size(); j++)
	{
		unsigned index = gIds[j];
		OS << ", " << gCs[index].access_offset;
	}
}

void OpenCLHostPrinter::VisitCallExpr(CallExpr *Call) {

	vector<DeclRefExpr*>& addV = Call->getExpendedVariables();
	unsigned numArgs = Call->getNumArgs();

	if (Call->isRevised())
	{
		unsigned actualNum = numArgs - addV.size();
		Call->setNumArgs(Context, actualNum);
	}

	PrintExpr(Call->getCallee());

	OS << "(";
	PrintCallArgs(Call);
	OS << ")";

	if (Call->isRevised())
	{
		Call->setNumArgs(Context, numArgs);
	}
}
void OpenCLHostPrinter::VisitMemberExpr(MemberExpr *Node) {
	// FIXME: Suppress printing implicit bases (like "this")
	PrintExpr(Node->getBase());
	if (FieldDecl *FD = dyn_cast<FieldDecl>(Node->getMemberDecl()))
		if (FD->isAnonymousStructOrUnion())
			return;
	OS << (Node->isArrow() ? "->" : ".");
	if (NestedNameSpecifier *Qualifier = Node->getQualifier())
		Qualifier->print(OS, Policy);

	OS << Node->getMemberNameInfo();

	if (Node->hasExplicitTemplateArgs())
		OS << TemplateSpecializationType::PrintTemplateArgumentList(
				Node->getTemplateArgs(),
				Node->getNumTemplateArgs(),
				Policy);
}
void OpenCLHostPrinter::VisitObjCIsaExpr(ObjCIsaExpr *Node) {
	PrintExpr(Node->getBase());
	OS << (Node->isArrow() ? "->isa" : ".isa");
}

void OpenCLHostPrinter::VisitExtVectorElementExpr(ExtVectorElementExpr *Node) {
	PrintExpr(Node->getBase());
	OS << ".";
	OS << Node->getAccessor().getName();
}
void OpenCLHostPrinter::VisitCStyleCastExpr(CStyleCastExpr *Node) {
	OS << "(" << Node->getType().getAsString(Policy) << ")";
	PrintExpr(Node->getSubExpr());
}
void OpenCLHostPrinter::VisitCompoundLiteralExpr(CompoundLiteralExpr *Node) {
	OS << "(" << Node->getType().getAsString(Policy) << ")";
	PrintExpr(Node->getInitializer());
}
void OpenCLHostPrinter::VisitImplicitCastExpr(ImplicitCastExpr *Node) {
	// No need to print anything, simply forward to the sub expression.
	PrintExpr(Node->getSubExpr());
}
void OpenCLHostPrinter::VisitBinaryOperator(BinaryOperator *Node) {
	PrintExpr(Node->getLHS());
	OS << " " << BinaryOperator::getOpcodeStr(Node->getOpcode()) << " ";
	PrintExpr(Node->getRHS());
}
void OpenCLHostPrinter::VisitCompoundAssignOperator(CompoundAssignOperator *Node) {
	PrintExpr(Node->getLHS());
	OS << " " << BinaryOperator::getOpcodeStr(Node->getOpcode()) << " ";
	PrintExpr(Node->getRHS());
}
void OpenCLHostPrinter::VisitConditionalOperator(ConditionalOperator *Node) {
	PrintExpr(Node->getCond());
	OS << " ? ";
	PrintExpr(Node->getLHS());
	OS << " : ";
	PrintExpr(Node->getRHS());
}

// GNU extensions.

void
OpenCLHostPrinter::VisitBinaryConditionalOperator(BinaryConditionalOperator *Node) {
	PrintExpr(Node->getCommon());
	OS << " ?: ";
	PrintExpr(Node->getFalseExpr());
}
void OpenCLHostPrinter::VisitAddrLabelExpr(AddrLabelExpr *Node) {
	OS << "&&" << Node->getLabel()->getName();
}

void OpenCLHostPrinter::VisitStmtExpr(StmtExpr *E) {
	OS << "(";
	PrintRawCompoundStmt(E->getSubStmt());
	OS << ")";
}

void OpenCLHostPrinter::VisitChooseExpr(ChooseExpr *Node) {
	OS << "__builtin_choose_expr(";
	PrintExpr(Node->getCond());
	OS << ", ";
	PrintExpr(Node->getLHS());
	OS << ", ";
	PrintExpr(Node->getRHS());
	OS << ")";
}

void OpenCLHostPrinter::VisitGNUNullExpr(GNUNullExpr *) {
	OS << "__null";
}

void OpenCLHostPrinter::VisitShuffleVectorExpr(ShuffleVectorExpr *Node) {
	OS << "__builtin_shufflevector(";
	for (unsigned i = 0, e = Node->getNumSubExprs(); i != e; ++i) {
		if (i) OS << ", ";
		PrintExpr(Node->getExpr(i));
	}
	OS << ")";
}

void OpenCLHostPrinter::VisitInitListExpr(InitListExpr* Node) {
	if (Node->getSyntacticForm()) {
		Visit(Node->getSyntacticForm());
		return;
	}

	OS << "{ ";
	for (unsigned i = 0, e = Node->getNumInits(); i != e; ++i) {
		if (i) OS << ", ";
		if (Node->getInit(i))
			PrintExpr(Node->getInit(i));
		else
			OS << "0";
	}
	OS << " }";
}

void OpenCLHostPrinter::VisitParenListExpr(ParenListExpr* Node) {
	OS << "( ";
	for (unsigned i = 0, e = Node->getNumExprs(); i != e; ++i) {
		if (i) OS << ", ";
		PrintExpr(Node->getExpr(i));
	}
	OS << " )";
}

void OpenCLHostPrinter::VisitDesignatedInitExpr(DesignatedInitExpr *Node) {
	for (DesignatedInitExpr::designators_iterator D = Node->designators_begin(),
			DEnd = Node->designators_end();
			D != DEnd; ++D) {
		if (D->isFieldDesignator()) {
			if (D->getDotLoc().isInvalid())
				OS << D->getFieldName()->getName() << ":";
			else
				OS << "." << D->getFieldName()->getName();
		} else {
			OS << "[";
			if (D->isArrayDesignator()) {
				PrintExpr(Node->getArrayIndex(*D));
			} else {
				PrintExpr(Node->getArrayRangeStart(*D));
				OS << " ... ";
				PrintExpr(Node->getArrayRangeEnd(*D));
			}
			OS << "]";
		}
	}

	OS << " = ";
	PrintExpr(Node->getInit());
}

void OpenCLHostPrinter::VisitImplicitValueInitExpr(ImplicitValueInitExpr *Node) {
	if (Policy.LangOpts.CPlusPlus)
		OS << "/*implicit*/" << Node->getType().getAsString(Policy) << "()";
	else {
		OS << "/*implicit*/(" << Node->getType().getAsString(Policy) << ")";
		if (Node->getType()->isRecordType())
			OS << "{}";
		else
			OS << 0;
	}
}

void OpenCLHostPrinter::VisitVAArgExpr(VAArgExpr *Node) {
	OS << "__builtin_va_arg(";
	PrintExpr(Node->getSubExpr());
	OS << ", ";
	OS << Node->getType().getAsString(Policy);
	OS << ")";
}

// C++
void OpenCLHostPrinter::VisitCXXOperatorCallExpr(CXXOperatorCallExpr *Node) {
	const char *OpStrings[NUM_OVERLOADED_OPERATORS] = {
		"",
#define OVERLOADED_OPERATOR(Name,Spelling,Token,Unary,Binary,MemberOnly) \
		Spelling,
#include "clang/Basic/OperatorKinds.def"
	};

	OverloadedOperatorKind Kind = Node->getOperator();
	if (Kind == OO_PlusPlus || Kind == OO_MinusMinus) {
		if (Node->getNumArgs() == 1) {
			OS << OpStrings[Kind] << ' ';
			PrintExpr(Node->getArg(0));
		} else {
			PrintExpr(Node->getArg(0));
			OS << ' ' << OpStrings[Kind];
		}
	} else if (Kind == OO_Call) {
		PrintExpr(Node->getArg(0));
		OS << '(';
		for (unsigned ArgIdx = 1; ArgIdx < Node->getNumArgs(); ++ArgIdx) {
			if (ArgIdx > 1)
				OS << ", ";
			if (!isa<CXXDefaultArgExpr>(Node->getArg(ArgIdx)))
				PrintExpr(Node->getArg(ArgIdx));
		}
		OS << ')';
	} else if (Kind == OO_Subscript) {
		PrintExpr(Node->getArg(0));
		OS << '[';
		PrintExpr(Node->getArg(1));
		OS << ']';
	} else if (Node->getNumArgs() == 1) {
		OS << OpStrings[Kind] << ' ';
		PrintExpr(Node->getArg(0));
	} else if (Node->getNumArgs() == 2) {
		PrintExpr(Node->getArg(0));
		OS << ' ' << OpStrings[Kind] << ' ';
		PrintExpr(Node->getArg(1));
	} else {
		assert(false && "unknown overloaded operator");
	}
}

void OpenCLHostPrinter::VisitCXXMemberCallExpr(CXXMemberCallExpr *Node) {
	VisitCallExpr(cast<CallExpr>(Node));
}

void OpenCLHostPrinter::VisitCUDAKernelCallExpr(CUDAKernelCallExpr *Node) {
	PrintExpr(Node->getCallee());
	OS << "<<<";
	PrintCallArgs(Node->getConfig());
	OS << ">>>(";
	PrintCallArgs(Node);
	OS << ")";
}

void OpenCLHostPrinter::VisitCXXNamedCastExpr(CXXNamedCastExpr *Node) {
	OS << Node->getCastName() << '<';
	OS << Node->getTypeAsWritten().getAsString(Policy) << ">(";
	PrintExpr(Node->getSubExpr());
	OS << ")";
}

void OpenCLHostPrinter::VisitCXXStaticCastExpr(CXXStaticCastExpr *Node) {
	VisitCXXNamedCastExpr(Node);
}

void OpenCLHostPrinter::VisitCXXDynamicCastExpr(CXXDynamicCastExpr *Node) {
	VisitCXXNamedCastExpr(Node);
}

void OpenCLHostPrinter::VisitCXXReinterpretCastExpr(CXXReinterpretCastExpr *Node) {
	VisitCXXNamedCastExpr(Node);
}

void OpenCLHostPrinter::VisitCXXConstCastExpr(CXXConstCastExpr *Node) {
	VisitCXXNamedCastExpr(Node);
}

void OpenCLHostPrinter::VisitCXXTypeidExpr(CXXTypeidExpr *Node) {
	OS << "typeid(";
	if (Node->isTypeOperand()) {
		OS << Node->getTypeOperand().getAsString(Policy);
	} else {
		PrintExpr(Node->getExprOperand());
	}
	OS << ")";
}

void OpenCLHostPrinter::VisitCXXUuidofExpr(CXXUuidofExpr *Node) {
	OS << "__uuidof(";
	if (Node->isTypeOperand()) {
		OS << Node->getTypeOperand().getAsString(Policy);
	} else {
		PrintExpr(Node->getExprOperand());
	}
	OS << ")";
}

void OpenCLHostPrinter::VisitCXXBoolLiteralExpr(CXXBoolLiteralExpr *Node) {
	OS << (Node->getValue() ? "true" : "false");
}

void OpenCLHostPrinter::VisitCXXNullPtrLiteralExpr(CXXNullPtrLiteralExpr *Node) {
	OS << "nullptr";
}

void OpenCLHostPrinter::VisitCXXThisExpr(CXXThisExpr *Node) {
	OS << "this";
}

void OpenCLHostPrinter::VisitCXXThrowExpr(CXXThrowExpr *Node) {
	if (Node->getSubExpr() == 0)
		OS << "throw";
	else {
		OS << "throw ";
		PrintExpr(Node->getSubExpr());
	}
}

void OpenCLHostPrinter::VisitCXXDefaultArgExpr(CXXDefaultArgExpr *Node) {
	// Nothing to print: we picked up the default argument
}

void OpenCLHostPrinter::VisitCXXFunctionalCastExpr(CXXFunctionalCastExpr *Node) {
	OS << Node->getType().getAsString(Policy);
	OS << "(";
	PrintExpr(Node->getSubExpr());
	OS << ")";
}

void OpenCLHostPrinter::VisitCXXBindTemporaryExpr(CXXBindTemporaryExpr *Node) {
	PrintExpr(Node->getSubExpr());
}

void OpenCLHostPrinter::VisitCXXTemporaryObjectExpr(CXXTemporaryObjectExpr *Node) {
	OS << Node->getType().getAsString(Policy);
	OS << "(";
	for (CXXTemporaryObjectExpr::arg_iterator Arg = Node->arg_begin(),
			ArgEnd = Node->arg_end();
			Arg != ArgEnd; ++Arg) {
		if (Arg != Node->arg_begin())
			OS << ", ";
		PrintExpr(*Arg);
	}
	OS << ")";
}

void OpenCLHostPrinter::VisitCXXScalarValueInitExpr(CXXScalarValueInitExpr *Node) {
	if (TypeSourceInfo *TSInfo = Node->getTypeSourceInfo())
		OS << TSInfo->getType().getAsString(Policy) << "()";
	else
		OS << Node->getType().getAsString(Policy) << "()";
}

void OpenCLHostPrinter::VisitCXXNewExpr(CXXNewExpr *E) {
}

void OpenCLHostPrinter::VisitCXXDeleteExpr(CXXDeleteExpr *E) {
	if (E->isGlobalDelete())
		OS << "::";
	OS << "delete ";
	if (E->isArrayForm())
		OS << "[] ";
	PrintExpr(E->getArgument());
}

void OpenCLHostPrinter::VisitCXXPseudoDestructorExpr(CXXPseudoDestructorExpr *E) {
	PrintExpr(E->getBase());
	if (E->isArrow())
		OS << "->";
	else
		OS << '.';
	if (E->getQualifier())
		E->getQualifier()->print(OS, Policy);

	std::string TypeS;
	if (IdentifierInfo *II = E->getDestroyedTypeIdentifier())
		OS << II->getName();
	else
		E->getDestroyedType().getAsStringInternal(TypeS, Policy);
	OS << TypeS;
}

void OpenCLHostPrinter::VisitCXXConstructExpr(CXXConstructExpr *E) {
	for (unsigned i = 0, e = E->getNumArgs(); i != e; ++i) {
		if (isa<CXXDefaultArgExpr>(E->getArg(i))) {
			// Don't print any defaulted arguments
			break;
		}

		if (i) OS << ", ";
		PrintExpr(E->getArg(i));
	}
}

void OpenCLHostPrinter::VisitExprWithCleanups(ExprWithCleanups *E) {
	// Just forward to the sub expression.
	PrintExpr(E->getSubExpr());
}

void
OpenCLHostPrinter::VisitCXXUnresolvedConstructExpr(
		CXXUnresolvedConstructExpr *Node) {
	OS << Node->getTypeAsWritten().getAsString(Policy);
	OS << "(";
	for (CXXUnresolvedConstructExpr::arg_iterator Arg = Node->arg_begin(),
			ArgEnd = Node->arg_end();
			Arg != ArgEnd; ++Arg) {
		if (Arg != Node->arg_begin())
			OS << ", ";
		PrintExpr(*Arg);
	}
	OS << ")";
}

void OpenCLHostPrinter::VisitCXXDependentScopeMemberExpr(
		CXXDependentScopeMemberExpr *Node) {
	if (!Node->isImplicitAccess()) {
		PrintExpr(Node->getBase());
		OS << (Node->isArrow() ? "->" : ".");
	}
	if (NestedNameSpecifier *Qualifier = Node->getQualifier())
		Qualifier->print(OS, Policy);
	else if (Node->hasExplicitTemplateArgs())
		// FIXME: Track use of "template" keyword explicitly?
		OS << "template ";

	OS << Node->getMemberNameInfo();

	if (Node->hasExplicitTemplateArgs()) {
		OS << TemplateSpecializationType::PrintTemplateArgumentList(
				Node->getTemplateArgs(),
				Node->getNumTemplateArgs(),
				Policy);
	}
}

void OpenCLHostPrinter::VisitUnresolvedMemberExpr(UnresolvedMemberExpr *Node) {
	if (!Node->isImplicitAccess()) {
		PrintExpr(Node->getBase());
		OS << (Node->isArrow() ? "->" : ".");
	}
	if (NestedNameSpecifier *Qualifier = Node->getQualifier())
		Qualifier->print(OS, Policy);

	// FIXME: this might originally have been written with 'template'

	OS << Node->getMemberNameInfo();

	if (Node->hasExplicitTemplateArgs()) {
		OS << TemplateSpecializationType::PrintTemplateArgumentList(
				Node->getTemplateArgs(),
				Node->getNumTemplateArgs(),
				Policy);
	}
}

void OpenCLHostPrinter::VisitCXXNoexceptExpr(CXXNoexceptExpr *E) {
	OS << "noexcept(";
	PrintExpr(E->getOperand());
	OS << ")";
}

void OpenCLHostPrinter::VisitPackExpansionExpr(PackExpansionExpr *E) {
	PrintExpr(E->getPattern());
	OS << "...";
}

void OpenCLHostPrinter::VisitSizeOfPackExpr(SizeOfPackExpr *E) {
	OS << "sizeof...(" << E->getPack()->getNameAsString() << ")";
}

void OpenCLHostPrinter::VisitSubstNonTypeTemplateParmPackExpr(
		SubstNonTypeTemplateParmPackExpr *Node) {
	OS << Node->getParameterPack()->getNameAsString();
}

// Obj-C

void OpenCLHostPrinter::VisitObjCStringLiteral(ObjCStringLiteral *Node) {
	OS << "@";
	VisitStringLiteral(Node->getString());
}

void OpenCLHostPrinter::VisitObjCEncodeExpr(ObjCEncodeExpr *Node) {
	OS << "@encode(" << Node->getEncodedType().getAsString(Policy) << ')';
}

void OpenCLHostPrinter::VisitObjCSelectorExpr(ObjCSelectorExpr *Node) {
	OS << "@selector(" << Node->getSelector().getAsString() << ')';
}

void OpenCLHostPrinter::VisitObjCProtocolExpr(ObjCProtocolExpr *Node) {
	OS << "@protocol(" << Node->getProtocol() << ')';
}

void OpenCLHostPrinter::VisitObjCMessageExpr(ObjCMessageExpr *Mess) {
	OS << "[";
	switch (Mess->getReceiverKind()) {
		case ObjCMessageExpr::Instance:
			PrintExpr(Mess->getInstanceReceiver());
			break;

		case ObjCMessageExpr::Class:
			OS << Mess->getClassReceiver().getAsString(Policy);
			break;

		case ObjCMessageExpr::SuperInstance:
		case ObjCMessageExpr::SuperClass:
			OS << "Super";
			break;
	}

	OS << ' ';
	Selector selector = Mess->getSelector();
	if (selector.isUnarySelector()) {
		OS << selector.getNameForSlot(0);
	} else {
		for (unsigned i = 0, e = Mess->getNumArgs(); i != e; ++i) {
			if (i < selector.getNumArgs()) {
				if (i > 0) OS << ' ';
				if (selector.getIdentifierInfoForSlot(i))
					OS << selector.getIdentifierInfoForSlot(i)->getName() << ':';
				else
					OS << ":";
			}
			else OS << ", "; // Handle variadic methods.

			PrintExpr(Mess->getArg(i));
		}
	}
	OS << "]";
}


void OpenCLHostPrinter::VisitBlockExpr(BlockExpr *Node) {
	BlockDecl *BD = Node->getBlockDecl();
	OS << "^";

	const FunctionType *AFT = Node->getFunctionType();

	if (isa<FunctionNoProtoType>(AFT)) {
		OS << "()";
	} else if (!BD->param_empty() || cast<FunctionProtoType>(AFT)->isVariadic()) {
		OS << '(';
		std::string ParamStr;
		for (BlockDecl::param_iterator AI = BD->param_begin(),
				E = BD->param_end(); AI != E; ++AI) {
			if (AI != BD->param_begin()) OS << ", ";
			ParamStr = (*AI)->getNameAsString();
			(*AI)->getType().getAsStringInternal(ParamStr, Policy);
			OS << ParamStr;
		}

		const FunctionProtoType *FT = cast<FunctionProtoType>(AFT);
		if (FT->isVariadic()) {
			if (!BD->param_empty()) OS << ", ";
			OS << "...";
		}
		OS << ')';
	}
}

void OpenCLHostPrinter::VisitBlockDeclRefExpr(BlockDeclRefExpr *Node) {
	OS << Node->getDecl();
}

void OpenCLHostPrinter::VisitOpaqueValueExpr(OpaqueValueExpr *Node) {}

//===----------------------------------------------------------------------===//
// Stmt method implementations
//===----------------------------------------------------------------------===//
#if 0
void Stmt::dumpPretty(ASTContext& Context) const {
	printPretty(llvm::errs(), Context, 0,
			PrintingPolicy(Context.getLangOptions()));
}

void Stmt::printPretty(llvm::raw_ostream &OS, ASTContext& Context,
		PrinterHelper* Helper,
		const PrintingPolicy &Policy,
		unsigned Indentation) const 
{
	if (this == 0) {
		OS << "<NULL>";
		return;
	}

	if (Policy.Dump && &Context) {
		dump(OS, Context.getSourceManager());
		return;
	}

	OpenCLHostPrinter P(OS, Context, Helper, Policy, Indentation);
	P.Visit(const_cast<Stmt*>(this));
}
#endif

//===----------------------------------------------------------------------===//
// PrinterHelper
//===----------------------------------------------------------------------===//

// Implement virtual destructor.

void OpenCLHostPrinter::VisitBinaryTypeTraitExpr(clang::BinaryTypeTraitExpr*)
{

}

void OpenCLHostPrinter::VisitObjCForCollectionStmt(clang::ObjCForCollectionStmt*)
{

}

void OpenCLHostPrinter::VisitUnaryTypeTraitExpr(clang::UnaryTypeTraitExpr*)
{

}

void OpenCLHostPrinter::VisitOclFlush(clang::OclFlush*) {
	OS << "flush_ocl_buffers();\n";
}

void OpenCLHostPrinter::VisitOclHostFlush(clang::OclHostFlush* f) {
	vector<string> vars = f->getVars();
	for (unsigned i=0; i<vars.size(); i++)
	{
		OS << "oclHostWrites(__ocl_buffer_" << vars[i] << ");\n";
	}
}

void OpenCLHostPrinter::VisitOclInit(clang::OclInit*) {
	OS << "init_ocl_runtime();\n";
}
void OpenCLHostPrinter::VisitOclTerm(clang::OclTerm*) {
	OS << "release_ocl_buffers();\n";
}

void OpenCLHostPrinter::VisitOclSync(clang::OclSync*) {
	OS << "sync_ocl_buffers();\n";
}

void OpenCLHostPrinter::VisitOclResetMLStmt(clang::OclResetMLStmt*) {
	if (OCLCompilerOptions::EnableMLFeatureCollection)
	{
		OS << "#ifdef DUMP_ML_FEATURES\n";
		OS << "  reset_ml_features();\n";
		OS << "#endif\n";
	}
}

void OpenCLHostPrinter::VisitOclDisableMLRecordStmt(clang::OclDisableMLRecordStmt*) {
	if (OCLCompilerOptions::EnableMLFeatureCollection)
	{
		OS << "#ifdef DUMP_ML_FEATURES\n";
		OS << "  disable_ml_record();\n";
		OS << "#endif\n";
	}
}

void OpenCLHostPrinter::VisitOclEnableMLRecordStmt(clang::OclEnableMLRecordStmt*) {
	if (OCLCompilerOptions::EnableMLFeatureCollection)
	{
		OS << "#ifdef DUMP_ML_FEATURES\n";
		OS << "  enable_ml_record();\n";
		OS << "#endif\n";
	}
}

void OpenCLHostPrinter::VisitOclDumpMLFStmt(clang::OclDumpMLFStmt*) {
	if (OCLCompilerOptions::EnableMLFeatureCollection)
	{
		OS << "#ifdef DUMP_ML_FEATURES\n";
		OS << "  dump_ml_features();\n";
		OS << "#endif\n";
	}
}


void OpenCLHostPrinter::VisitOclStartProfile(clang::OclStartProfile*) {
	OS << "#ifdef PROFILING\noclStartProfiling();\n#endif\n";
}

void OpenCLHostPrinter::VisitOclDumpProfile(clang::OclDumpProfile*) {
	OS << "#ifdef PROFILING\ndump_profiling();\n#endif\n";
}
void OpenCLHostPrinter::VisitOclStopProfile(clang::OclStopProfile*) {
	OS << "#ifdef PROFILING\noclStopProfiling();\n#endif\n";
}

void OpenCLHostPrinter::VisitOclHostRead(clang::OclHostRead* f) {
	vector<string> vars = f->getVars();
	for (unsigned i=0; i<vars.size(); i++)
	{
		OS << "oclHostReads(__ocl_buffer_" << vars[i] << ");\n";
	}
}

void OpenCLHostPrinter::VisitOclDevRead(clang::OclDevRead* f) {
	vector<string> vars = f->getVars();
	for (unsigned i=0; i<vars.size(); i++)
	{
		OS << "oclDevReads(__ocl_buffer_" << vars[i] << ");\n";
	}

}

void OpenCLHostPrinter::VisitOclDevWrite(clang::OclDevWrite* f) {
	vector<string> vars = f->getVars();
	for (unsigned i=0; i<vars.size(); i++)
	{
		OS << "oclDevWrites(__ocl_buffer_" << vars[i] << ");\n";
	}
}

