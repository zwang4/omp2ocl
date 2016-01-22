//===--- OpenCLPrinter.cpp - Printing implementation for Stmt ASTs ----------===//
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

#include "clang/Omp2Ocl/OpenCLRoutine.h"
#include "clang/Omp2Ocl/OpenCLPrinter.h"
#include "clang/AST/GlobalCallArgPicker.h"
#include "clang/AST/StmtPrinter.h"
#include "clang/Omp2Ocl/OpenCLCommon.h"
#include "clang/Omp2Ocl/OpenCLBinaryOperatorPrinter.h"

using namespace clang;

static bool printBracket = true;
vector<OpenCLTLSBufferAccess> OpenCLPrinter::tls_access;

//===----------------------------------------------------------------------===//
// OpenCLPrinter Visitor
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Stmt printing methods.
//===----------------------------------------------------------------------===//

bool OpenCLPrinter::isInReadSet(string access)
{
	for (unsigned i=0; i<tls_access.size(); i++)
	{
		if ((access == tls_access[i].getAccess()) 
				&& (tls_access[i].isWritten() == false))
		{
			return true;
		}
	}

	return false;
}


bool OpenCLPrinter::isAGlobalWriteBuf(string name)
{
	for (unsigned i=0; i<gWRBufs.size(); i++)
	{
		string n = gWRBufs[i]->getName();
		if (gWRBufs[i]->getName() == name)
		{
			return true;
		}
	}

	for (unsigned i=0; i<gWRLCBufs.size(); i++)
	{
		if (gWRLCBufs[i]->getName() == name)
		{
			return true;
		}
	}

	return false;
}

//Is there exist a read to a buffer that has been written
// Used for GPU tLS
bool OpenCLPrinter::shouldTLSTrack(string name, string access, unsigned long long stmt_ver)
{
	bool flag = false;

	if (track_all_write_bufs)
	{
		return true;
	}

	for (unsigned i=0; i<tls_access.size(); i++)
	{
		if (tls_access[i].getName() == name)
		{
			if (tls_access[i].hasBinaryOps())
			{
				flag = true;
				break;
			}

			//Check read not in the same statement
			//if (tls_access[i].getStmtVer() != stmt_ver)
			{
				if ((access == tls_access[i].getAccess()) 
						&& (tls_access[i].isWritten() == false))
				{
					flag = true;
					break;
				}
			}
		}		
	}

	return flag;
}

void OpenCLPrinter::OCLTLsTrackStmts(bool read)
{
}


void OpenCLPrinter::GPUTLsTrackStmts()
{
	//Insert GPU TLS Support Code
	if (OCLCompilerOptions::EnableGPUTLs)
	{
		if ((gWRBufs.size() > 0 )|| (gWRLCBufs.size() > 0))
		{
			vector<string> candidates;
			vector<string> access_pattern;
			for (unsigned i=0; i<tls_access.size(); i++)
			{
				if (tls_access[i].isWritten())
				{
					string name = tls_access[i].getName();
					if (shouldTLSTrack(name, tls_access[i].getAccess(), tls_access[i].getStmtVer()))
					{
						candidates.push_back(name);
						access_pattern.push_back(tls_access[i].getAccess());
					}
				}
			}


			if (candidates.size())
			{
				OS << " //-------------------------------------------\n";
				OS << "	// GPU TLS logs (BEGIN) \n";
				OS << " //-------------------------------------------\n";
				vector<string> rd_access;
				for (unsigned i=0; i<tls_access.size(); i++)
				{
					string name = tls_access[i].getName();
					string access = tls_access[i].getAccess();
					bool found = false;

					for (unsigned k=0; k<candidates.size(); k++)
					{
						if (candidates[k] == name && access_pattern[k] == access)
						{
							found = true;
							break;
						}
					}

					if (found)
					{
						act_tls_access.push_back(tls_access[i]);
						if (tls_access[i].isWritten())
						{
							OS << "atom_inc(&wr_log_" << tls_access[i].getAccess() << ");\n";
						}
						else
						{
							rd_access.push_back(tls_access[i].getAccess());
						}
					}
				}

				for (unsigned i=0; i<rd_access.size(); i++)
				{
					OS << "rd_log_" << rd_access[i] << " = 1;\n";
				}

				OS << " //-------------------------------------------\n";
				OS << "	// GPU TLS logs (END)\n";
				OS << " //-------------------------------------------\n";

				has_tls_ops = true;
			}
		}
	}

	tls_access.clear();
}

bool OpenCLPrinter::isDeclRefInNDRange(DeclRefExpr* d)
{
	if (pKL)
	{
		string name = d->getNameInfo().getAsString();
		vector<OpenCLNDRangeVar> GV = pKL->getOclLoopIndexs();
		for (unsigned i=0; i<GV.size(); i++)
		{
			if (GV[i].getName() == name)
			{
				return true;	
			}
		}
	}

	return false;
}

//GPU TLS SUPPORT
//Record access to array
void OpenCLPrinter::addTLSAccessObj(ArraySubscriptExpr *Node, DeclRefExpr* t, string access_seq,  bool isWrite)
{
	if (!track_tls_access)
	{
		return;
	}

	string name = t->getNameInfo().getAsString();

	if (pKL)
	{
		//this loop is set to don't do tls check by the user
		if (!pKL->getOMPFor().isTLSCheck())
			return;
	}

	if (!isAGlobalWriteBuf(name))
		return;

	bool should_skip = false;


	StmtPicker op(llvm::nulls(), Context, NULL, Context.PrintingPolicy, -4);
	op.Visit(Node);

	vector<BinaryOperator*> binOps = op.getBinOps();
	//Do some simple check first
	//if all the array subscriptions are global indexs, it will be fine.
	//Make sure the array is not accessed by function calls
	if (op.getCallExprs().size() == 0)
	{
		if (binOps.size() == 0)
		{
			vector<DeclRefExpr*>& ds = op.getDecl();
			bool found = false;
			for (unsigned i=0; i<ds.size(); i++)
			{
				if (!isDeclRefInNDRange(ds[i]))
				{
					found = true;
					break;
				}		
			}

			if (!found)
			{
				should_skip = true;
			}

		}
	}

	if (should_skip)
	{
		return;
	}

	for (unsigned i=0; i<tls_access.size(); i++)
	{
		string tls_name = tls_access[i].getName();
		string tls_access_seq = tls_access[i].getAccess();
		bool isW = tls_access[i].isWritten();

		if ((tls_name == name) &&
				(tls_access_seq == access_seq) &&
				(isW == isWrite) 
		   )
		{
			if (isWrite)
			{
				tls_access[i].IncTime();
			}

			return;
		}
	}

	tls_access.push_back(OpenCLTLSBufferAccess(binOps, name, access_seq, statement_version, isWrite));
}




/// PrintRawCompoundStmt - Print a compound stmt without indenting the {, and
/// with no newline after the }.
//
//

void OpenCLPrinter::printFuncBody(Stmt* body, bool printB, bool t)
{
	track_all_write_bufs = t;
	if (printB)
	{
		PrintStmt(body);	
	}
	else
	{
		printBracket = false;
		PrintStmt(body);
		printBracket = true;
	}
}

//Print out VLoad Information
void OpenCLPrinter::PrintVLoadDeclareInfor(CompoundStmt* Node)
{
	vector<OCLCompoundVLoadDeclareInfo>& CVIs = Node->getVLoadInfo();

	if (CVIs.size())
	{
		SourceLocation Loc = Node->getLBracLoc();
		unsigned ln = OCLCommon::getLineNumber(Context, Loc);
		OS << "	//-------------------------------------------\n";
		OS << "		//Declare prefetching Buffers (BEGIN) : " << ln << "\n";
		OS << "	//-------------------------------------------\n";

	}

	for (unsigned i=0; i<CVIs.size(); i++)
	{
		vector<vector<ArrayIndex> >& AIs = CVIs[i].getAIs();
		unsigned vs = CVIs[i].getVWidth();
		if (AIs.size())
		{
			ArrayIndex& A = AIs[0][0];
			ValueDecl* D = A.getDecl();
			string ctype = getCononicalType(D);	
			string gtype = getGlobalType(ctype);
			string declareName = OCLCommon::getVLoadVariableName(A.getName());
			A.setVDeclareName(declareName);

			for (unsigned j=1; j<AIs.size(); j++)
			{
				AIs[j][0].setVDeclareName(declareName);
			}

			OS << gtype;

			if (vs > 1)
			{
				OS << uint2String(vs);
			}

			if (AIs.size() > 1)
			{
				OS << " " << declareName << "[" << AIs.size() << "];\n"; 			
			}
			else
			{
				OS << " " << declareName << ";\n";
			}
		}
	}

	if (CVIs.size())
	{
		OS << "	//-------------------------------------------\n";
		OS << "	//Declare prefetching buffers (END)\n";
		OS << "	//-------------------------------------------\n";
	}
}

//Generate VLoad Code
void OpenCLPrinter::GenerateLoadCode(vector<ArrayIndex>& Is, unsigned arraySize, unsigned idx, 
		unsigned width, string declareName, string gtype)
{
	ArrayIndex& I = Is[0];
	if (width > 1)
	{
		unsigned MAS = OCLCommon::getMemAlignSize(gtype);
		string vname = I.getName();
		string pname = "p_" + declareName + "_" + uint2String(idx);

		if (OCLCompilerOptions::UseArrayLinearize)
		{
			OS << "__global " << gtype << "* "<< pname << "= " << vname;
			OS << " + " << I.getOffsetStr() << ";\n";
		}else
		{
			OS << "__global " << gtype << "* "<< pname << "= (" << "__global " << gtype << "*) &";
			OS << I.getOffsetStr() << ";\n";
		}
#ifdef VLOAD_CHECK
		OS << "if ((unsigned long)" << pname << " % " << MAS << " == 0)\n{\n";
		OS << declareName;
		if (arraySize > 1)
		{
			OS << "[" << idx << "]";
		}

		OS << " = vload" << width << "(0, " << pname << ");\n";
		OS << "}\n";
		OS << "else {\n";		
#endif
		for (unsigned i=0; i<Is.size(); i++)
		{
			OS << declareName;
			if (arraySize > 1)
			{
				OS << "[" << idx << "]";
			}	
			OS <<"." << getVStructureName(width, i);
			OS << " = " << pname << "[0];\n";
			OS << pname << "++;\n";
		}

#ifdef VLOAD_CHECK
		OS << "}\n";
#endif
#if 0
		OS << declareName;
		if (arraySize > 1)
		{
			OS << "[" << idx << "]";
		}
		OS << " = vload" << width << "(0, " << pname << ");\n";
#endif
	}
	else //sequential load e.g. width == 1
	{
		if (arraySize > 1)
		{
			OS << declareName << "[" << idx << "] ";
		}
		else
		{
			OS << declareName;
		}

		OS << " = " << I.getOrgLStr() << ";\n";
	}
}

//Print Load Code
void OpenCLPrinter::PrintLoadCode(CompoundStmt* Node)
{
	vector<OCLCompoundVLoadDeclareInfo>& CVIs = Node->getVLoadInfo();

	if (CVIs.size())
	{
		SourceLocation Loc = Node->getLBracLoc();
		unsigned ln = OCLCommon::getLineNumber(Context, Loc);
		OS << "	//-------------------------------------------\n";
		OS << "	//Prefetching (BEGIN) : " << ln << "\n";
		OS << " //Candidates:\n";
		for (unsigned i=0; i<CVIs.size(); i++)
		{
			vector<vector<ArrayIndex> >& AIs = CVIs[i].getAIs();
			//CVIs is organised in such a way that
			//CVIs[i] is vload info for a buffer with the same name and the same VWidth
			//CVIs[i].AIs() is the vload info for the buffer with the same witdh (where AIs[i] 
			//represents different starting pointer
			//See: OpenCLLoadSchedule::addVLoadInfo
			if (AIs.size())
			{
				//Different starting pointers
				for (unsigned i=0; i<AIs.size(); i++)
				{
					vector<ArrayIndex>& A = AIs[i];
					ValueDecl* D = A[0].getDecl();
					string ctype = getCononicalType(D);	
					string gtype = getGlobalType(ctype);
					//get the Vectorlised Declare Name -- this is defined when declared the variable
					string declareName = A[0].getVDeclareName();

					for (unsigned j=0; j<A.size(); j++)
					{
						OS << "	//	";
						OS << getStringExpr(Context, A[j].getASENode()); 
						OS << "\n";
					}
				}
			}

		}
		OS << "	//-------------------------------------------\n";
	}
	for (unsigned i=0; i<CVIs.size(); i++)
	{
		vector<vector<ArrayIndex> >& AIs = CVIs[i].getAIs();
		unsigned vs = CVIs[i].getVWidth();

		//CVIs is organised in such a way that
		//CVIs[i] is vload info for a buffer with the same name and the same VWidth
		//CVIs[i].AIs() is the vload info for the buffer with the same witdh (where AIs[i] 
		//represents different starting pointer
		//See: OpenCLLoadSchedule::addVLoadInfo
		if (AIs.size())
		{
			//Different starting pointers
			for (unsigned i=0; i<AIs.size(); i++)
			{
				vector<ArrayIndex>& A = AIs[i];
				ValueDecl* D = A[0].getDecl();
				string ctype = getCononicalType(D);	
				string gtype = getGlobalType(ctype);
				//get the Vectorlised Declare Name -- this is defined when declared the variable
				string declareName = A[0].getVDeclareName();

				for (unsigned j=0; j<A.size(); j++)
				{
					A[j].setLIndexs(i, j);

					//Get the normal __global load str	
					ArraySubscriptExpr* Node = A[j].getASENode();
					OCLGlobalMemVar g(A[j].getDecl(), true, false, false);
					string str;
					llvm::raw_string_ostream O(str);
					OpenCLPrinter opv(O, Context, NULL, Policy);
					opv.addAGlobalMemoryVariables(g);
					opv.VisitArraySubscriptExpr(Node);
					O.flush();

					if (OCLCompilerOptions::UseArrayLinearize) {
						string ss;
						//FIXME: THIS IS URGLY.
						//u[CALC_IDX_INX(2,2,3,i,j,k)] - > CALC_IDX_INX(2,2,3,i,j,k)	
						unsigned k=0;
						for (; k<str.length(); k++)
						{
							if (str[k] == '[')
							{
								break;
							}
						}

						k++;
						if (k < str.length())
						{
							for (; k<str.length()-1;k++)
							{
								ss = ss + str[k];
							}
						}
						A[j].setOffsetStr(ss);
					}
					else
					{	
						A[j].setOffsetStr(str);
					}
					A[j].setOrgLStr(str);
					A[j].setVDeclareName(declareName);
					A[j].setWidth(vs);
				}

				GenerateLoadCode(A, AIs.size(), i, vs, declareName, gtype);
			} // for (unsigned i=0; i<AIs.size(); i++)
		}
	}

	if (CVIs.size())
	{
		OS << "	//-------------------------------------------\n";
		OS << "	//Prefetching (END)\n";
		OS << "	//-------------------------------------------\n\n";
	}
}

bool OpenCLPrinter::ShouldReplaceWithSpecWrite(ArraySubscriptExpr* Node)
{
	return false;
}

bool OpenCLPrinter::OclSpecWrite(ArraySubscriptExpr* Node, string node_name)
{
	string access_pattern= getStringExpr(Context, Node);
	string type = getGlobalType(Node->getType().getAsString());

	OS << "spec_write_" << type << "(&" << access_pattern << ",&wr_log_" << access_pattern << ",";
	OS << "&rd_log_" << access_pattern << "," << "tls_thread_id" << ", tls_validflag,";

	return false;
}

bool OpenCLPrinter::OclSpecLoad(ArraySubscriptExpr* Node)
{
	string access_pattern = getStringExpr(Context, Node);
	string type = getGlobalType(Node->getType().getAsString());

	OS << "spec_read_" << type << "(&" << access_pattern << ",&wr_log_" << access_pattern << ",";
	OS << "&rd_log_" << access_pattern << "," << "tls_thread_id" << ", tls_validflag)";

	return false;
}

bool OpenCLPrinter::CastToArraySubscriptExpr(Expr* e, ArraySubscriptExpr* Node)
{
	Expr *expr = e;
	ImplicitCastExpr* icast = dyn_cast<ImplicitCastExpr>(expr);

	while(icast)
	{
		expr = icast->getSubExpr();
		icast = dyn_cast<ImplicitCastExpr>(expr);
	}

	Node = dyn_cast<ArraySubscriptExpr>(expr);
	if (Node)
	{
		return true;
	}

	return false;
}

//Check whether I can use a VLoad access instead a load to a global buffer
bool OpenCLPrinter::hasVLoadString(ArraySubscriptExpr* Node, ValueDecl* D, string& replaceString)
{
	if (CNode.empty())
	{
		return false;
	}

	vector<OCLCompoundVLoadDeclareInfo>& CVIs = CNode.top()->getVLoadInfo();

	replaceString = "";

	string str;
	llvm::raw_string_ostream O(str);
	OCLGlobalMemVar g(D, true, false, false);
	OpenCLPrinter opv(O, Context, NULL, Policy);
	opv.addAGlobalMemoryVariables(g);
	opv.VisitArraySubscriptExpr(Node);
	O.flush();

	//cerr << "STR=" << str << endl;

	//See OpenCLLoadSchedule::addVLoadInfo for the origanisation of 
	//CVIs
	for (unsigned i=0; i<CVIs.size(); i++)
	{
		vector<vector<ArrayIndex> >& AIs = CVIs[i].getAIs();

		for (unsigned k=0; k<AIs.size(); k++)
		{
			vector<ArrayIndex>& A = AIs[k];
			for (unsigned j=0; j<A.size(); j++)
			{
				if (A[j].getOrgLStr() == str)
				{
					llvm::raw_string_ostream O(replaceString);
					O << A[j].getVDeclareName();

					if (AIs.size() > 1)
					{
						O << "[" << k << "]";
					}

					if (A[j].getWidth() > 1)
					{
						O << "." << getVStructureName(A[j].getWidth(), j);
					}

					O.flush();
					return true;
				}
			}	
		}	
	}

	return false;
}

void OpenCLPrinter::PrintRawCompoundStmt(CompoundStmt *Node) {
	bool setHere = false;
	bool visit_set_here = false;	

	if (!printBracket)
	{
		setHere = true;
		printBracket = true;
	}
	else
	{
		OS << "{\n";
	}

	//Our gpu tls scheme is enable
	if (OCLCompilerOptions::OclTLSMechanism)
	{
		if (!interval_visit_p_ed)
		{
			visit_set_here = true;

			if (!interval_visit_p)
			{
				interval_visit_p = new OpenCLPrinter(llvm::nulls(), Context, Helper, Policy);
				interval_visit_p->DisableSpecReadWrite();
			}

			interval_visit_p_ed  = true;
			interval_visit_p->EnableVisitP();
			interval_visit_p->DisableTLSAccessTrack();
			interval_visit_p->Visit(Node);
		}
	}

	CNode.push(Node);
	PrintVLoadDeclareInfor(Node);
	PrintLoadCode(Node);

	for (CompoundStmt::body_iterator I = Node->body_begin(), E = Node->body_end();
			I != E; ++I)
		PrintStmt(*I);

	//Scott's scheme
	if (!OCLCompilerOptions::OclTLSMechanism && OCLCompilerOptions::EnableGPUTLs && spec_read_write)
	{
		GPUTLsTrackStmts();
	}

	if (!setHere)
	{
		Indent() << "}";
	}


	if (CNode.size())
	{
		CNode.pop();
	}

	if (visit_set_here)
	{
		interval_visit_p->DisableVisitP();
		interval_visit_p_ed  = false;
	}
}

void OpenCLPrinter::PrintRawDecl(Decl *D) {
	D->print(OS, Policy, IndentLevel);
}

void OpenCLPrinter::PrintRawDeclStmt(DeclStmt *S) {
	DeclStmt::decl_iterator Begin = S->decl_begin(), End = S->decl_end();
	llvm::SmallVector<Decl*, 2> Decls;
	for ( ; Begin != End; ++Begin)
		Decls.push_back(*Begin);

	Decl::printGroup(Decls.data(), Decls.size(), OS, Policy, IndentLevel);
}

void OpenCLPrinter::VisitNullStmt(NullStmt *Node) {
	Indent() << ";\n";
}

void OpenCLPrinter::VisitDeclStmt(DeclStmt *Node) {
	Indent();
	PrintRawDeclStmt(Node);
	OS << ";\n";
}

void OpenCLPrinter::VisitCompoundStmt(CompoundStmt *Node) {
	Indent();
	PrintRawCompoundStmt(Node);
	OS << "\n";
}

void OpenCLPrinter::VisitCaseStmt(CaseStmt *Node) {
	Indent(-1) << "case ";
	PrintExpr(Node->getLHS());
	if (Node->getRHS()) {
		OS << " ... ";
		PrintExpr(Node->getRHS());
	}
	OS << ":\n";

	PrintStmt(Node->getSubStmt(), 0);
}

void OpenCLPrinter::VisitDefaultStmt(DefaultStmt *Node) {
	Indent(-1) << "default:\n";
	PrintStmt(Node->getSubStmt(), 0);
}

void OpenCLPrinter::VisitLabelStmt(LabelStmt *Node) {
	Indent(-1) << Node->getName() << ":\n";
	PrintStmt(Node->getSubStmt(), 0);
}

void OpenCLPrinter::PrintRawIfStmt(IfStmt *If) {
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

void OpenCLPrinter::VisitIfStmt(IfStmt *If) {
	Indent();
	PrintRawIfStmt(If);
}

void OpenCLPrinter::VisitSwitchStmt(SwitchStmt *Node) {
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

void OpenCLPrinter::VisitWhileStmt(WhileStmt *Node) {
	Indent() << "while (";
	PrintExpr(Node->getCond());
	OS << ")\n";
	PrintStmt(Node->getBody());
}

void OpenCLPrinter::VisitDoStmt(DoStmt *Node) {
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

void OpenCLPrinter::VisitForHeader(ForStmt *Node)
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
}

//Generate the prefix for reduction operations
void OpenCLPrinter::ReductionPrefix(ForStmt* Node)
{
	//FIXME: Need to handle compund statments here
	Stmt* body = Node->getBody();
	vector<DeclRefExpr*> Ds = getDeclRefExprs(Context, body);
	vector<string> gvs;

	for (unsigned i = 0; i < Ds.size(); i++)
	{
		string name = Ds[i]->getNameInfo().getAsString();
		string type = getGlobalType(Ds[i]->getType().getAsString());

		if (isAGlobalMemoryVariable(Ds[i]))
		{
			bool found = false;
			for (unsigned j=0; j<gvs.size(); j++)
			{
				if (gvs[j] == name)
				{
					found = true;
					break;
				}
			}

			if (!found)
			{
				OS << "__global " << type << "* __orgi_g_" << name << " = " << name << ";\n";
				gvs.push_back(name);
			}
		}
	}
}

void OpenCLPrinter::VisitForStmt(ForStmt *Node) {
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

void OpenCLPrinter::VisitGotoStmt(GotoStmt *Node) {
	Indent() << "goto " << Node->getLabel()->getName() << ";\n";
}

void OpenCLPrinter::VisitIndirectGotoStmt(IndirectGotoStmt *Node) {
	Indent() << "goto *";
	PrintExpr(Node->getTarget());
	OS << ";\n";
}

void OpenCLPrinter::VisitContinueStmt(ContinueStmt *Node) {
	Indent() << "continue;\n";
}

void OpenCLPrinter::VisitBreakStmt(BreakStmt *Node) {
	Indent() << "break;\n";
}


void OpenCLPrinter::VisitReturnStmt(ReturnStmt *Node) {
	Indent() << "return";
	if (Node->getRetValue()) {
		OS << " ";
		PrintExpr(Node->getRetValue());
	}
	OS << ";\n";
}


void OpenCLPrinter::VisitAsmStmt(AsmStmt *Node) {
	Indent() << "asm ";

	if (Node->isVolatile())
		OS << "volatile ";

	OS << "(";
	VisitStringLiteral(Node->getAsmString());

	// Outputs
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

void OpenCLPrinter::VisitObjCAtTryStmt(ObjCAtTryStmt *Node) {
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

void OpenCLPrinter::VisitObjCAtFinallyStmt(ObjCAtFinallyStmt *Node) {
}

void OpenCLPrinter::VisitObjCAtCatchStmt (ObjCAtCatchStmt *Node) {
	Indent() << "@catch (...) { /* todo */ } \n";
}

void OpenCLPrinter::VisitObjCAtThrowStmt(ObjCAtThrowStmt *Node) {
	Indent() << "@throw";
	if (Node->getThrowExpr()) {
		OS << " ";
		PrintExpr(Node->getThrowExpr());
	}
	OS << ";\n";
}

void OpenCLPrinter::VisitObjCAtSynchronizedStmt(ObjCAtSynchronizedStmt *Node) {
	Indent() << "@synchronized (";
	PrintExpr(Node->getSynchExpr());
	OS << ")";
	PrintRawCompoundStmt(Node->getSynchBody());
	OS << "\n";
}

void OpenCLPrinter::PrintRawCXXCatchStmt(CXXCatchStmt *Node) {
	OS << "catch (";
	if (Decl *ExDecl = Node->getExceptionDecl())
		PrintRawDecl(ExDecl);
	else
		OS << "...";
	OS << ") ";
	PrintRawCompoundStmt(cast<CompoundStmt>(Node->getHandlerBlock()));
}

void OpenCLPrinter::VisitCXXCatchStmt(CXXCatchStmt *Node) {
	Indent();
	PrintRawCXXCatchStmt(Node);
	OS << "\n";
}

void OpenCLPrinter::VisitCXXTryStmt(CXXTryStmt *Node) {
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

void OpenCLPrinter::VisitDeclRefExpr(DeclRefExpr *Node) {
	if (NestedNameSpecifier *Qualifier = Node->getQualifier())
		Qualifier->print(OS, Policy);
	OS << Node->getNameInfo();
	if (Node->hasExplicitTemplateArgs())
		OS << TemplateSpecializationType::PrintTemplateArgumentList(
				Node->getTemplateArgs(),
				Node->getNumTemplateArgs(),
				Policy);  
}

void OpenCLPrinter::VisitDependentScopeDeclRefExpr(
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

void OpenCLPrinter::VisitUnresolvedLookupExpr(UnresolvedLookupExpr *Node) {
	if (Node->getQualifier())
		Node->getQualifier()->print(OS, Policy);
	OS << Node->getNameInfo();
	if (Node->hasExplicitTemplateArgs())
		OS << TemplateSpecializationType::PrintTemplateArgumentList(
				Node->getTemplateArgs(),
				Node->getNumTemplateArgs(),
				Policy);
}

void OpenCLPrinter::VisitObjCIvarRefExpr(ObjCIvarRefExpr *Node) {
	if (Node->getBase()) {
		PrintExpr(Node->getBase());
		OS << (Node->isArrow() ? "->" : ".");
	}
	OS << Node->getDecl();
}

void OpenCLPrinter::VisitObjCPropertyRefExpr(ObjCPropertyRefExpr *Node) {
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

void OpenCLPrinter::VisitPredefinedExpr(PredefinedExpr *Node) {
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

void OpenCLPrinter::VisitCharacterLiteral(CharacterLiteral *Node) {
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

void OpenCLPrinter::VisitIntegerLiteral(IntegerLiteral *Node) {
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
void OpenCLPrinter::VisitFloatingLiteral(FloatingLiteral *Node) {
	// FIXME: print value more precisely.
	OS << Node->getLexString();
	//OS << Node->getValueAsApproximateDouble();
}

void OpenCLPrinter::VisitImaginaryLiteral(ImaginaryLiteral *Node) {
	PrintExpr(Node->getSubExpr());
	OS << "i";
}

void OpenCLPrinter::VisitStringLiteral(StringLiteral *Str) {
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
				else  // Output anything hard as an octal escape.
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
void OpenCLPrinter::VisitParenExpr(ParenExpr *Node) {
	OS << "(";
	PrintExpr(Node->getSubExpr());
	OS << ")";
}
void OpenCLPrinter::VisitUnaryOperator(UnaryOperator *Node) {
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

void OpenCLPrinter::VisitOffsetOfExpr(OffsetOfExpr *Node) {
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

void OpenCLPrinter::VisitUnaryExprOrTypeTraitExpr(UnaryExprOrTypeTraitExpr *Node){
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

/*
 * This is the key function in handling and translating global memory access
 *
 */
void OpenCLPrinter::VisitArraySubscriptExpr(ArraySubscriptExpr *Node) {

	Expr* base = Node->getBase();
	string base_string = getStringStmt(Context, base);
	arraySubInfo ai = getSubScripts(base_string);
	DeclRefExpr* t = NULL;

	StmtPicker op(llvm::nulls(), Context, NULL, Context.PrintingPolicy, -4);
	op.Visit(base);
	vector<DeclRefExpr*>& ds = op.getDecl();

	if (OCLCompilerOptions::OclTLSMechanism && spec_read_write)
	{
		if (isAGlobalWriteBuf(ai.name) /*&& isInReadSet(getStringStmt(Context, Node))*/)
		{
			OclSpecLoad(Node);
			return;
		}	
	}

	if (ds.size())
	{
		t = ds[0];
	}

	bool isGlobalBuffer = false;
	bool isTGBuffer = false; 
	bool firstMet = false;
	unsigned dim = 0;
	bool catcheBaseHere = false;

	if (t)
	{
		isGlobalBuffer = isAGlobalMemoryVariable(ai.name);
		isTGBuffer = OCLCommon::isAGlobalMemThreadPrivateVar(t->getDecl()) | isAGTPVariable(ai.name);
	}

	if (arrayDecls.size())
	{
		DeclRefExpr* top = arrayDecls.top();
		if (top != t)
		{
			firstMet = true;
			arrayDecls.push(t);
		}
	}
	else
	{
		firstMet = true;
		arrayDecls.push(t);
	}

	if (firstMet)
	{
		//For GPU TLS
		if (isAGlobalWriteBuf(t->getNameInfo().getAsString()))
		{
			string access_seq = getStringStmt(Context, Node);
			//Write to a global buffer
			if (tls_write)
			{
				addTLSAccessObj(Node, t, access_seq, tls_write);			
				tls_write = false;
			}
			else
			{
				addTLSAccessObj(Node, t, access_seq, tls_write);			
			}
		}
	}

	string offset_string;
	string arrayAccPrefix;

	if (isGlobalBuffer)
	{
		if (firstMet)
		{
			//Check whether I can replace it with an access to a local vload array
			string r;
			if (hasVLoadString(Node, t->getDecl(), r))
			{
				OS << r;
				OS << " /*" << getStringStmt(Context, Node) << "*/ ";
				return;
			}

			//Check whether I should replace it with a spec read or write
		}

		if (OCLCompilerOptions::UseArrayLinearize)
		{
			//Check if we reach the real base
			//for A[1][2][3] a real base is A
			if (base_string.find('[') == string::npos)
			{
				Expr* expr = base;
				ImplicitCastExpr* icast = dyn_cast<ImplicitCastExpr>(expr);

				while(icast)
				{
					expr = icast->getSubExpr();
					icast = dyn_cast<ImplicitCastExpr>(expr);
				}

				DeclRefExpr* decl = dyn_cast<DeclRefExpr>(expr);

				if (decl)
				{
					ValueDecl* d = decl->getDecl();
					catcheBaseHere = true;

					string type = getCononicalType(d);
					dim = getArrayDimension(type);
					offset_string = d->getOffsetString();

					arrayDims.push(dim);

					if (dim > 1)
					{
						string array_decl;
						vector<unsigned> defs = getArrayDef(type);

						for (unsigned int i=0; i<defs.size(); i++)
						{
							if (i > 0)
								array_decl = array_decl + ",";

							char buf[64];
							snprintf(buf, 64, "%u", defs[i]);

							array_decl = array_decl + buf;
						}	

						char dbuf[64];
						snprintf(dbuf, 64, "%u", dim);
						arrayAccPrefix = "CALC_";
						arrayAccPrefix = arrayAccPrefix + dbuf;
						arrayAccPrefix = arrayAccPrefix + "D_IDX(" + array_decl;
					}
				}
				else
				{
					cerr << "I could not cast " << base_string << " to DeclRefExpr " << endl;
					exit(-1);
				}
			}
		}
	}

	PrintExpr(Node->getLHS());

	if (OCLCompilerOptions::UseArrayLinearize)
	{
		if (isGlobalBuffer)
		{
			dim = arrayDims.top();

			if (catcheBaseHere)
			{
				if (offset_string.length())
					OS << offset_string << " + ";

				OS << "[";

				if (dim > 1)
				{
					OS << arrayAccPrefix;
				}	
			}


			if (dim > 1)
				OS << ", ";

			OS << "(";
		}
		else
		{
			OS << "[";
		}
	}
	else
	{
		OS << "[";
	}

	PrintExpr(Node->getRHS());

	if (OCLCompilerOptions::UseArrayLinearize)
	{
		if (isGlobalBuffer)
		{
			OS << ")";
			if (firstMet)
			{
				if (dim > 1)
					OS << ")";

				if (isTGBuffer)
				{
					OS << " * " << COPYIN_MULTI_FACTOR_NAME; 
					OS << " + " << COPYIN_ADD_OFFSET_NAME;
				}

				OS << "]";

				//Pop out current array base token
				//	arrayTokens.pop();
				arrayDecls.pop();
				arrayDims.pop();
			}
		}
		else
		{
			OS << "]";
		}
	}
	else
	{
		OS << "]";
		if (isGlobalBuffer)
		{
			if (firstMet)
			{
				if (isTGBuffer && OCLCompilerOptions::UseArrayLinearize)
				{
					OS << "[" << COPYIN_ADD_OFFSET_NAME << "]";
				}
				arrayDecls.pop();
			}
		}
	}
	//#else
	//	PrintExpr(Node->getLHS());
	//	OS << "[";
	//	PrintExpr(Node->getRHS());
	//	OS << "]";
}

//
// For __glboal arguments that are passed as pointers,
// special care is needed
//
void OpenCLPrinter::PrintCallArgs(CallExpr *Call) {
	if (OCLCompilerOptions::UseArrayLinearize) {
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
				PrintExpr(Call->getArg(i));
			}
		}

		//Print offset arguments
		for (unsigned j=0; j<gIds.size(); j++)
		{
			unsigned index = gIds[j];
			OS << ", " << gCs[index].access_offset;
		}
	}
	else
	{
		for (unsigned i = 0, e = Call->getNumArgs(); i != e; ++i) {
			if (isa<CXXDefaultArgExpr>(Call->getArg(i))) {
				// Don't print any defaulted arguments
				break;
			}

			if (i) OS << ", ";
			PrintExpr(Call->getArg(i));
		}
	}
	//There are threadprivate arguments whose locations 
	//is in in the global memory
	if (Call->hasGTPVar())
	{
		if (OCLCompilerOptions::UseArrayLinearize) {
			OS << ", " << COPYIN_MULTI_FACTOR_NAME;
		}
		OS << ", " << COPYIN_ADD_OFFSET_NAME;	
	}

}

void OpenCLPrinter::VisitCallExpr(CallExpr *Call) {
	//ZHENG: A CallExpr may be expended if the callee has been
	//expended
	vector<DeclRefExpr*>& addV = Call->getExpendedVariables();
	string newName = Call->getNewName();
	bool no_spec = false;

	//This callee has been renamed
	if (newName.length())
	{
		if (!spec_read_write && OCLCompilerOptions::EnableGPUTLs)
		{
			newName = newName + "_no_spec";
			no_spec = true;
		}

		OS << newName;
	}
	else
	{
		PrintExpr(Call->getCallee());
	}

	bool read_write = spec_read_write;
	if (no_spec)
	{
		spec_read_write = false;
	}


	OS << "(";
	PrintCallArgs(Call);

	//GPU TLS Checking
	if (OCLCompilerOptions::EnableGPUTLs)
	{
		if (OpenCLGlobalInfoContainer::isFuncHasGPUTLSLog(newName))
		{
			vector<RenameFuncGPUTLS> rn = OpenCLGlobalInfoContainer::getRenameFuncGPUTLs();
			vector<unsigned> ins;

			for (unsigned i=0; i<rn.size(); i++)
			{
				if (rn[i].getFuncName() == newName)
				{
					ins.push_back(rn[i].getIndex());
				}
			}

			for (unsigned i=0; i<ins.size(); i++)
			{
				OS << ", rd_log_";
				PrintExpr(Call->getArg(ins[i]));	
				OS << ", wr_log_";
				PrintExpr(Call->getArg(ins[i]));	

				StmtPicker op(llvm::nulls(), Context, NULL, Context.PrintingPolicy, -4);
				op.Visit(Call->getArg(ins[i]));
				DeclRefExpr* dr = op.getFirstDecl();

				if (dr)
				{
					string name = dr->getNameInfo().getAsString();
					string access_seq = getStringExpr(Context, Call->getArg(ins[i]));
					act_tls_access.push_back(OpenCLTLSBufferAccess(op.getBinOps(), name, access_seq, 0, true));
				}
			}
		}			
	}

	if (newName.length())
	{
		if (OCLCompilerOptions::OclTLSMechanism)
		{
			OS << ", tls_validflag, tls_thread_id";
		}
	}

	OS << ")";

	if (addV.size() > 0)
	{
		OS << "/*ARGEXP: ";
		for (unsigned int i=0; i<addV.size(); i++)
		{
			if (i > 0)
				OS << ",";

			OS << addV[i]->getNameInfo().getAsString();
		}

		OS << "*/";
	}

	spec_read_write = read_write;
}

void OpenCLPrinter::VisitMemberExpr(MemberExpr *Node) {
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
void OpenCLPrinter::VisitObjCIsaExpr(ObjCIsaExpr *Node) {
	PrintExpr(Node->getBase());
	OS << (Node->isArrow() ? "->isa" : ".isa");
}

void OpenCLPrinter::VisitExtVectorElementExpr(ExtVectorElementExpr *Node) {
	PrintExpr(Node->getBase());
	OS << ".";
	OS << Node->getAccessor().getName();
}
void OpenCLPrinter::VisitCStyleCastExpr(CStyleCastExpr *Node) {
	OS << "(" << Node->getType().getAsString(Policy) << ")";
	PrintExpr(Node->getSubExpr());
}
void OpenCLPrinter::VisitCompoundLiteralExpr(CompoundLiteralExpr *Node) {
	OS << "(" << Node->getType().getAsString(Policy) << ")";
	PrintExpr(Node->getInitializer());
}
void OpenCLPrinter::VisitImplicitCastExpr(ImplicitCastExpr *Node) {
	// No need to print anything, simply forward to the sub expression.
	PrintExpr(Node->getSubExpr());
}

void OpenCLPrinter::VisitBinaryOperator(BinaryOperator *Node) {
	OpenCLBinaryOperatorPrinter bop(OS, Context);
	bool is = bop.ShouldTransformBinaryOperator(Node);
	bool is_spec_write = false;

	string opstr = BinaryOperator::getOpcodeStr(Node->getOpcode());
	tls_write = false;

	//This is used to track write set
	if (opstr == "=")
	{
		tls_write = true;
	}

	tls_write_stack.push(tls_write);	

	if (tls_write_stack.size())
	{
		tls_write = tls_write_stack.top();
		tls_write_stack.pop();
	}

	//Our ocl scheme
	if (OCLCompilerOptions::OclTLSMechanism && opstr == "=" && spec_read_write)
	{
		Expr *e =Node->getLHS();
		ArraySubscriptExpr* ANode = NULL;
		CastToArraySubscriptExpr(e, ANode);
		ANode = dyn_cast<ArraySubscriptExpr>(e);

		if (ANode)
		{
			Expr* base = ANode->getBase();
			string base_string = getStringStmt(Context, base);
			arraySubInfo ai = getSubScripts(base_string);

			if (isAGlobalWriteBuf(ai.name) && (isPrintFunc || isInReadSet(getStringStmt(Context, ANode))))
			{
				OclSpecWrite(ANode, ai.name);
				OS << "(";
				is_spec_write = true;
			}
			else
			{
				is_spec_write = false;
			}
		}
	}

	if (!is_spec_write)
	{
		PrintExpr(Node->getLHS());
		OS << " " << opstr << " ";
	}

	if (is)
	{
		OS << "((";
	}

	tls_write_stack.push(false);

	PrintExpr(Node->getRHS());

	if (tls_write_stack.size())
	{
		tls_write = tls_write_stack.top();
		tls_write_stack.pop();
	}

	if (is)
	{
		OS << ") * " << COPYIN_MULTI_FACTOR_NAME << ")";
	}

	if (is_spec_write)
	{
		OS << "))";
	}
}
void OpenCLPrinter::VisitCompoundAssignOperator(CompoundAssignOperator *Node) {
	PrintExpr(Node->getLHS());
	OS << " " << BinaryOperator::getOpcodeStr(Node->getOpcode()) << " ";
	PrintExpr(Node->getRHS());
}

void OpenCLPrinter::VisitConditionalOperator(ConditionalOperator *Node) {
	PrintExpr(Node->getCond());
	OS << " ? ";
	PrintExpr(Node->getLHS());
	OS << " : ";
	PrintExpr(Node->getRHS());
}

// GNU extensions.

void
OpenCLPrinter::VisitBinaryConditionalOperator(BinaryConditionalOperator *Node) {
	PrintExpr(Node->getCommon());
	OS << " ?: ";
	PrintExpr(Node->getFalseExpr());
}
void OpenCLPrinter::VisitAddrLabelExpr(AddrLabelExpr *Node) {
	OS << "&&" << Node->getLabel()->getName();
}

void OpenCLPrinter::VisitStmtExpr(StmtExpr *E) {
	OS << "(";
	PrintRawCompoundStmt(E->getSubStmt());
	OS << ")";
}

void OpenCLPrinter::VisitChooseExpr(ChooseExpr *Node) {
	OS << "__builtin_choose_expr(";
	PrintExpr(Node->getCond());
	OS << ", ";
	PrintExpr(Node->getLHS());
	OS << ", ";
	PrintExpr(Node->getRHS());
	OS << ")";
}

void OpenCLPrinter::VisitGNUNullExpr(GNUNullExpr *) {
	OS << "__null";
}

void OpenCLPrinter::VisitShuffleVectorExpr(ShuffleVectorExpr *Node) {
	OS << "__builtin_shufflevector(";
	for (unsigned i = 0, e = Node->getNumSubExprs(); i != e; ++i) {
		if (i) OS << ", ";
		PrintExpr(Node->getExpr(i));
	}
	OS << ")";
}

void OpenCLPrinter::VisitInitListExpr(InitListExpr* Node) {
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

void OpenCLPrinter::VisitParenListExpr(ParenListExpr* Node) {
	OS << "( ";
	for (unsigned i = 0, e = Node->getNumExprs(); i != e; ++i) {
		if (i) OS << ", ";
		PrintExpr(Node->getExpr(i));
	}
	OS << " )";
}

void OpenCLPrinter::VisitDesignatedInitExpr(DesignatedInitExpr *Node) {
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

void OpenCLPrinter::VisitImplicitValueInitExpr(ImplicitValueInitExpr *Node) {
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

void OpenCLPrinter::VisitVAArgExpr(VAArgExpr *Node) {
	OS << "__builtin_va_arg(";
	PrintExpr(Node->getSubExpr());
	OS << ", ";
	OS << Node->getType().getAsString(Policy);
	OS << ")";
}

// C++
void OpenCLPrinter::VisitCXXOperatorCallExpr(CXXOperatorCallExpr *Node) {
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

void OpenCLPrinter::VisitCXXMemberCallExpr(CXXMemberCallExpr *Node) {
	VisitCallExpr(cast<CallExpr>(Node));
}

void OpenCLPrinter::VisitCUDAKernelCallExpr(CUDAKernelCallExpr *Node) {
	PrintExpr(Node->getCallee());
	OS << "<<<";
	PrintCallArgs(Node->getConfig());
	OS << ">>>(";
	PrintCallArgs(Node);
	OS << ")";
}

void OpenCLPrinter::VisitCXXNamedCastExpr(CXXNamedCastExpr *Node) {
	OS << Node->getCastName() << '<';
	OS << Node->getTypeAsWritten().getAsString(Policy) << ">(";
	PrintExpr(Node->getSubExpr());
	OS << ")";
}

void OpenCLPrinter::VisitCXXStaticCastExpr(CXXStaticCastExpr *Node) {
	VisitCXXNamedCastExpr(Node);
}

void OpenCLPrinter::VisitCXXDynamicCastExpr(CXXDynamicCastExpr *Node) {
	VisitCXXNamedCastExpr(Node);
}

void OpenCLPrinter::VisitCXXReinterpretCastExpr(CXXReinterpretCastExpr *Node) {
	VisitCXXNamedCastExpr(Node);
}

void OpenCLPrinter::VisitCXXConstCastExpr(CXXConstCastExpr *Node) {
	VisitCXXNamedCastExpr(Node);
}

void OpenCLPrinter::VisitCXXTypeidExpr(CXXTypeidExpr *Node) {
	OS << "typeid(";
	if (Node->isTypeOperand()) {
		OS << Node->getTypeOperand().getAsString(Policy);
	} else {
		PrintExpr(Node->getExprOperand());
	}
	OS << ")";
}

void OpenCLPrinter::VisitCXXUuidofExpr(CXXUuidofExpr *Node) {
	OS << "__uuidof(";
	if (Node->isTypeOperand()) {
		OS << Node->getTypeOperand().getAsString(Policy);
	} else {
		PrintExpr(Node->getExprOperand());
	}
	OS << ")";
}

void OpenCLPrinter::VisitCXXBoolLiteralExpr(CXXBoolLiteralExpr *Node) {
	OS << (Node->getValue() ? "true" : "false");
}

void OpenCLPrinter::VisitCXXNullPtrLiteralExpr(CXXNullPtrLiteralExpr *Node) {
	OS << "nullptr";
}

void OpenCLPrinter::VisitCXXThisExpr(CXXThisExpr *Node) {
	OS << "this";
}

void OpenCLPrinter::VisitCXXThrowExpr(CXXThrowExpr *Node) {
	if (Node->getSubExpr() == 0)
		OS << "throw";
	else {
		OS << "throw ";
		PrintExpr(Node->getSubExpr());
	}
}

void OpenCLPrinter::VisitCXXDefaultArgExpr(CXXDefaultArgExpr *Node) {
	// Nothing to print: we picked up the default argument
}

void OpenCLPrinter::VisitCXXFunctionalCastExpr(CXXFunctionalCastExpr *Node) {
	OS << Node->getType().getAsString(Policy);
	OS << "(";
	PrintExpr(Node->getSubExpr());
	OS << ")";
}

void OpenCLPrinter::VisitCXXBindTemporaryExpr(CXXBindTemporaryExpr *Node) {
	PrintExpr(Node->getSubExpr());
}

void OpenCLPrinter::VisitCXXTemporaryObjectExpr(CXXTemporaryObjectExpr *Node) {
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

void OpenCLPrinter::VisitCXXScalarValueInitExpr(CXXScalarValueInitExpr *Node) {
	if (TypeSourceInfo *TSInfo = Node->getTypeSourceInfo())
		OS << TSInfo->getType().getAsString(Policy) << "()";
	else
		OS << Node->getType().getAsString(Policy) << "()";
}

void OpenCLPrinter::VisitCXXNewExpr(CXXNewExpr *E) {
}

void OpenCLPrinter::VisitCXXDeleteExpr(CXXDeleteExpr *E) {
	if (E->isGlobalDelete())
		OS << "::";
	OS << "delete ";
	if (E->isArrayForm())
		OS << "[] ";
	PrintExpr(E->getArgument());
}

void OpenCLPrinter::VisitCXXPseudoDestructorExpr(CXXPseudoDestructorExpr *E) {
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

void OpenCLPrinter::VisitCXXConstructExpr(CXXConstructExpr *E) {
	for (unsigned i = 0, e = E->getNumArgs(); i != e; ++i) {
		if (isa<CXXDefaultArgExpr>(E->getArg(i))) {
			// Don't print any defaulted arguments
			break;
		}

		if (i) OS << ", ";
		PrintExpr(E->getArg(i));
	}
}

void OpenCLPrinter::VisitExprWithCleanups(ExprWithCleanups *E) {
	// Just forward to the sub expression.
	PrintExpr(E->getSubExpr());
}

void
OpenCLPrinter::VisitCXXUnresolvedConstructExpr(
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

void OpenCLPrinter::VisitCXXDependentScopeMemberExpr(
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

void OpenCLPrinter::VisitUnresolvedMemberExpr(UnresolvedMemberExpr *Node) {
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

void OpenCLPrinter::VisitCXXNoexceptExpr(CXXNoexceptExpr *E) {
	OS << "noexcept(";
	PrintExpr(E->getOperand());
	OS << ")";
}

void OpenCLPrinter::VisitPackExpansionExpr(PackExpansionExpr *E) {
	PrintExpr(E->getPattern());
	OS << "...";
}

void OpenCLPrinter::VisitSizeOfPackExpr(SizeOfPackExpr *E) {
	OS << "sizeof...(" << E->getPack()->getNameAsString() << ")";
}

void OpenCLPrinter::VisitSubstNonTypeTemplateParmPackExpr(
		SubstNonTypeTemplateParmPackExpr *Node) {
	OS << Node->getParameterPack()->getNameAsString();
}

// Obj-C

void OpenCLPrinter::VisitObjCStringLiteral(ObjCStringLiteral *Node) {
	OS << "@";
	VisitStringLiteral(Node->getString());
}

void OpenCLPrinter::VisitObjCEncodeExpr(ObjCEncodeExpr *Node) {
	OS << "@encode(" << Node->getEncodedType().getAsString(Policy) << ')';
}

void OpenCLPrinter::VisitObjCSelectorExpr(ObjCSelectorExpr *Node) {
	OS << "@selector(" << Node->getSelector().getAsString() << ')';
}

void OpenCLPrinter::VisitObjCProtocolExpr(ObjCProtocolExpr *Node) {
	OS << "@protocol(" << Node->getProtocol() << ')';
}

void OpenCLPrinter::VisitObjCMessageExpr(ObjCMessageExpr *Mess) {
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


void OpenCLPrinter::VisitBlockExpr(BlockExpr *Node) {
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

void OpenCLPrinter::VisitBlockDeclRefExpr(BlockDeclRefExpr *Node) {
	OS << Node->getDecl();
}

void OpenCLPrinter::VisitOpaqueValueExpr(OpaqueValueExpr *Node) {}

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

	OpenCLPrinter P(OS, Context, Helper, Policy, Indentation);
	P.Visit(const_cast<Stmt*>(this));
}
#endif

//===----------------------------------------------------------------------===//
// PrinterHelper
//===----------------------------------------------------------------------===//

// Implement virtual destructor.

void OpenCLPrinter::VisitBinaryTypeTraitExpr(clang::BinaryTypeTraitExpr*)
{

}

void OpenCLPrinter::VisitObjCForCollectionStmt(clang::ObjCForCollectionStmt*)
{

}

void OpenCLPrinter::VisitUnaryTypeTraitExpr(clang::UnaryTypeTraitExpr*)
{

}

//OMP2OCL
void OpenCLPrinter::VisitOclFlush(clang::OclFlush*) {}
void OpenCLPrinter::VisitOclHostFlush(clang::OclHostFlush*) {}
void OpenCLPrinter::VisitOclInit(clang::OclInit*) {}
void OpenCLPrinter::VisitOclTerm(clang::OclTerm*) {}
void OpenCLPrinter::VisitOclSync(clang::OclSync*) {}
void OpenCLPrinter::VisitOclResetMLStmt(clang::OclResetMLStmt*) {}
void OpenCLPrinter::VisitOclEnableMLRecordStmt(clang::OclEnableMLRecordStmt*) {}
void OpenCLPrinter::VisitOclDisableMLRecordStmt(clang::OclDisableMLRecordStmt*) {}
void OpenCLPrinter::VisitOclDumpMLFStmt(clang::OclDumpMLFStmt*){}
void OpenCLPrinter::VisitOclStartProfile(clang::OclStartProfile*) {}
void OpenCLPrinter::VisitOclDumpProfile(clang::OclDumpProfile*) {}
void OpenCLPrinter::VisitOclStopProfile(clang::OclStopProfile*) {}
void OpenCLPrinter::VisitOclHostRead(clang::OclHostRead*) {}
void OpenCLPrinter::VisitOclDevRead(clang::OclDevRead*) {}
void OpenCLPrinter::VisitOclDevWrite(clang::OclDevWrite*) {}
