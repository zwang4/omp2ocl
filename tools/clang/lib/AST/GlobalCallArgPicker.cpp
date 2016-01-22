//===--- GlobalCallArgPicker.cpp - Printing implementation for Stmt ASTs ----------===//

#include "clang/AST/GlobalCallArgPicker.h"
#include "clang/Omp2Ocl/OpenCLRoutine.h"
#include "clang/Omp2Ocl/OpenCLCompilerOptions.h"
#include "clang/AST/StmtPicker.h"
#include "clang/Basic/SourceManager.h"

using namespace clang;

//===----------------------------------------------------------------------===//
// GlobalCallArgPicker Visitor
//===----------------------------------------------------------------------===//

//Zheng: this will traverse the body of a callee function to discover any
//global variables that are passed as pointers...
//For a global memory buffers that is passed as pointers, we need to expand
//the call arugments of the callee functions

/// PrintRawCompoundStmt - Print a compound stmt without indenting the {, and
/// with no newline after the }.
void GlobalCallArgPicker::PrintRawCompoundStmt(CompoundStmt *Node) {
	OS << "{\n";
	for (CompoundStmt::body_iterator I = Node->body_begin(), E = Node->body_end();
			I != E; ++I)
		PrintStmt(*I);

	Indent() << "}";
}

void GlobalCallArgPicker::PrintRawDecl(Decl *D) {
	D->print(OS, Policy, IndentLevel);
}

void GlobalCallArgPicker::PrintRawDeclStmt(DeclStmt *S) {
	DeclStmt::decl_iterator Begin = S->decl_begin(), End = S->decl_end();
	llvm::SmallVector<Decl*, 2> Decls;
	for ( ; Begin != End; ++Begin)
		Decls.push_back(*Begin);

	Decl::printGroup(Decls.data(), Decls.size(), OS, Policy, IndentLevel);
}

void GlobalCallArgPicker::VisitNullStmt(NullStmt *Node) {
	Indent() << ";\n";
}

void GlobalCallArgPicker::VisitDeclStmt(DeclStmt *Node) {
	Indent();
	PrintRawDeclStmt(Node);
	OS << ";\n";
}

void GlobalCallArgPicker::VisitCompoundStmt(CompoundStmt *Node) {
	Indent();
	PrintRawCompoundStmt(Node);
	OS << "\n";
}

void GlobalCallArgPicker::VisitCaseStmt(CaseStmt *Node) {
	Indent(-1) << "case ";
	PrintExpr(Node->getLHS());
	if (Node->getRHS()) {
		OS << " ... ";
		PrintExpr(Node->getRHS());
	}
	OS << ":\n";

	PrintStmt(Node->getSubStmt(), 0);
}

void GlobalCallArgPicker::VisitDefaultStmt(DefaultStmt *Node) {
	Indent(-1) << "default:\n";
	PrintStmt(Node->getSubStmt(), 0);
}

void GlobalCallArgPicker::VisitLabelStmt(LabelStmt *Node) {
	Indent(-1) << Node->getName() << ":\n";
	PrintStmt(Node->getSubStmt(), 0);
}

void GlobalCallArgPicker::PrintRawIfStmt(IfStmt *If) {
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

void GlobalCallArgPicker::VisitIfStmt(IfStmt *If) {
	Indent();
	PrintRawIfStmt(If);
}

void GlobalCallArgPicker::VisitSwitchStmt(SwitchStmt *Node) {
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

void GlobalCallArgPicker::VisitWhileStmt(WhileStmt *Node) {
	Indent() << "while (";
	PrintExpr(Node->getCond());
	OS << ")\n";
	PrintStmt(Node->getBody());
}

void GlobalCallArgPicker::VisitDoStmt(DoStmt *Node) {
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

void GlobalCallArgPicker::VisitForStmt(ForStmt *Node) {
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

void GlobalCallArgPicker::VisitGotoStmt(GotoStmt *Node) {
	Indent() << "goto " << Node->getLabel()->getName() << ";\n";
}

void GlobalCallArgPicker::VisitIndirectGotoStmt(IndirectGotoStmt *Node) {
	Indent() << "goto *";
	PrintExpr(Node->getTarget());
	OS << ";\n";
}

void GlobalCallArgPicker::VisitContinueStmt(ContinueStmt *Node) {
	Indent() << "continue;\n";
}

void GlobalCallArgPicker::VisitBreakStmt(BreakStmt *Node) {
	Indent() << "break;\n";
}


void GlobalCallArgPicker::VisitReturnStmt(ReturnStmt *Node) {
	Indent() << "return";
	if (Node->getRetValue()) {
		OS << " ";
		PrintExpr(Node->getRetValue());
	}
	OS << ";\n";
}


void GlobalCallArgPicker::VisitAsmStmt(AsmStmt *Node) {
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

void GlobalCallArgPicker::VisitObjCAtTryStmt(ObjCAtTryStmt *Node) {
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

void GlobalCallArgPicker::VisitObjCAtFinallyStmt(ObjCAtFinallyStmt *Node) {
}

void GlobalCallArgPicker::VisitObjCAtCatchStmt (ObjCAtCatchStmt *Node) {
	Indent() << "@catch (...) { /* todo */ } \n";
}

void GlobalCallArgPicker::VisitObjCAtThrowStmt(ObjCAtThrowStmt *Node) {
	Indent() << "@throw";
	if (Node->getThrowExpr()) {
		OS << " ";
		PrintExpr(Node->getThrowExpr());
	}
	OS << ";\n";
}

void GlobalCallArgPicker::VisitObjCAtSynchronizedStmt(ObjCAtSynchronizedStmt *Node) {
	Indent() << "@synchronized (";
	PrintExpr(Node->getSynchExpr());
	OS << ")";
	PrintRawCompoundStmt(Node->getSynchBody());
	OS << "\n";
}

void GlobalCallArgPicker::PrintRawCXXCatchStmt(CXXCatchStmt *Node) {
	OS << "catch (";
	if (Decl *ExDecl = Node->getExceptionDecl())
		PrintRawDecl(ExDecl);
	else
		OS << "...";
	OS << ") ";
	PrintRawCompoundStmt(cast<CompoundStmt>(Node->getHandlerBlock()));
}

void GlobalCallArgPicker::VisitCXXCatchStmt(CXXCatchStmt *Node) {
	Indent();
	PrintRawCXXCatchStmt(Node);
	OS << "\n";
}

void GlobalCallArgPicker::VisitCXXTryStmt(CXXTryStmt *Node) {
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

void GlobalCallArgPicker::VisitDeclRefExpr(DeclRefExpr *Node) {

	if (NestedNameSpecifier *Qualifier = Node->getQualifier())
		Qualifier->print(OS, Policy);
	OS << Node->getNameInfo();
	if (Node->hasExplicitTemplateArgs())
		OS << TemplateSpecializationType::PrintTemplateArgumentList(
				Node->getTemplateArgs(),
				Node->getNumTemplateArgs(),
				Policy);  
}

void GlobalCallArgPicker::VisitDependentScopeDeclRefExpr(
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

void GlobalCallArgPicker::VisitUnresolvedLookupExpr(UnresolvedLookupExpr *Node) {
	if (Node->getQualifier())
		Node->getQualifier()->print(OS, Policy);
	OS << Node->getNameInfo();
	if (Node->hasExplicitTemplateArgs())
		OS << TemplateSpecializationType::PrintTemplateArgumentList(
				Node->getTemplateArgs(),
				Node->getNumTemplateArgs(),
				Policy);
}

void GlobalCallArgPicker::VisitObjCIvarRefExpr(ObjCIvarRefExpr *Node) {
	if (Node->getBase()) {
		PrintExpr(Node->getBase());
		OS << (Node->isArrow() ? "->" : ".");
	}
	OS << Node->getDecl();
}

void GlobalCallArgPicker::VisitObjCPropertyRefExpr(ObjCPropertyRefExpr *Node) {
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

void GlobalCallArgPicker::VisitPredefinedExpr(PredefinedExpr *Node) {
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

void GlobalCallArgPicker::VisitCharacterLiteral(CharacterLiteral *Node) {
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

void GlobalCallArgPicker::VisitIntegerLiteral(IntegerLiteral *Node) {
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
void GlobalCallArgPicker::VisitFloatingLiteral(FloatingLiteral *Node) {
	// FIXME: print value more precisely.
	OS << Node->getValueAsApproximateDouble();
}

void GlobalCallArgPicker::VisitImaginaryLiteral(ImaginaryLiteral *Node) {
	PrintExpr(Node->getSubExpr());
	OS << "i";
}

void GlobalCallArgPicker::VisitStringLiteral(StringLiteral *Str) {
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
void GlobalCallArgPicker::VisitParenExpr(ParenExpr *Node) {
	OS << "(";
	PrintExpr(Node->getSubExpr());
	OS << ")";
}
void GlobalCallArgPicker::VisitUnaryOperator(UnaryOperator *Node) {
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

void GlobalCallArgPicker::VisitOffsetOfExpr(OffsetOfExpr *Node) {
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

void GlobalCallArgPicker::VisitUnaryExprOrTypeTraitExpr(UnaryExprOrTypeTraitExpr *Node){
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

//
//This will record the call argument if a global array is passed as a pointer
void GlobalCallArgPicker::VisitArraySubscriptExpr(ArraySubscriptExpr *Node) {

	//ZHENG: THIS MAY HAVE A PROBLEM WHEN THERE IS ARRAY
	//INDIRECT ACCESS AND THE INDIRECT ACCESS IS
	//A GLOBAL MEMORY VAR!
	if (inCallArg)
	{
		string string_buf;
		llvm::raw_string_ostream Out(string_buf);
		StmtPicker dr (Out, Context, NULL, Policy, 0);
		dr.VisitArraySubscriptExpr(Node);
		Out.flush();

		vector<DeclRefExpr*>& decls = dr.getDecl();
		DeclRefExpr* e = decls[0];

		//This is a global memory buffer
		if (isAGlobalMemory(e))
		{
			string type = getCononicalType(e->getDecl());
			unsigned dim = 	getArrayDimension(type);

			arraySubInfo ai = getSubScripts(string_buf);
			cai.name = ai.name;

			//This is a pointer to a global buffer
			if (ai.subExprs.size() < dim)
			{
				cai.isPointerAccess = true;
				vector<unsigned> arrDefs = getArrayDef(getCononicalType(e->getDecl()));

				string access_string="";
				for (unsigned ii=0; ii<ai.subExprs.size(); ii++)
				{
					if (ii > 0)
						access_string = access_string + " + ";
					access_string = access_string + calcArrayIndex(arrDefs, ii, ai.subExprs[ii]);
				}

				cai.access_offset = "(" + access_string + ")";	
			}
			else
			{
				cai.isPointerAccess = false;
				cai.isBuf = false;
			}

			if (!(ai.subExprs.size() <= dim))
			{
				cerr << "Something is wrong for " << string_buf << endl;
			}


			assert ((ai.subExprs.size() <= dim) && "Something is wrong when gather the array Subscriptions");
		}
		else
		{
			PrintExpr(Node->getRHS());
		}
	}
	else
	{
		PrintExpr(Node->getLHS());
		PrintExpr(Node->getRHS());
	}
}

void GlobalCallArgPicker::PrintCallArgs(CallExpr *Call) {
	for (unsigned i = 0, ce = Call->getNumArgs(); i != ce; ++i) {
		if (isa<CXXDefaultArgExpr>(Call->getArg(i))) {
			// Don't print any defaulted arguments
			break;
		}

		if (i) OS << ", ";
		SourceLocation Loc = Call->getExprLoc();
		SourceManager &SM = Context.getSourceManager();
		PresumedLoc PLoc = SM.getPresumedLoc(Loc);
		unsigned lc = PLoc.isValid()? PLoc.getLine() : 0;

		string string_buf;
		llvm::raw_string_ostream Out(string_buf);
		StmtPicker dr (Out, Context, NULL, Policy, 0);
		dr.Visit(Call->getArg(i));
		Out.flush();

		vector<DeclRefExpr*>& decls = dr.getDecl();
		DeclRefExpr* e = NULL;

		if (decls.size())
		{
			e = decls[0];
		}
		else
		{
			continue;
		}

		cai.reset();
		//cai.access_offset ="";
		cai.string_buf = string_buf;
		cai.index = i;
		cai.arg = e;
		cai.lc = lc;
		cai.isGlobalMemThreadPrivate = false;

		cai.name = e->getNameInfo().getAsString();
		cai.isPointerAccess = false;
		ca->curIndex = i;

		ValueDecl* d = e->getDecl();
		string type = getCononicalType(d); 	
		unsigned dim = getArrayDimension(type); 

		if (dim)
		{
			cai.isBuf = true;
		}


		//After this call the isBuf flag may be changed
		PrintExpr(Call->getArg(i));

		if (e && isAGlobalMemory(e))
		{
			cai.isGlobalMemThreadPrivate = isAGTPMemVar(e->getNameInfo().getAsString());

			if (cai.isBuf)
			{
				if (cai.isGlobalMemThreadPrivate)
				{
					ca->hasGlobalMemThreadPrivate = true;
				}
				ca->gCallArgs.push_back(cai);
			}
			else
			{
				//ZHENG FIXME: THIS ASSUME A DECLREFEXPR IS AN EXTENDED ARGUMENT
				Expr *e = Call->getArg(i);
				//It may be an ImplicitCastExpr cast as well!
				ImplicitCastExpr* icast = dyn_cast<ImplicitCastExpr>(e);

				while(icast)
				{
					e = icast->getSubExpr(); 
					icast = dyn_cast<ImplicitCastExpr>(e);
				}

				DeclRefExpr* re = dyn_cast<DeclRefExpr>(e);

				if (re)
				{
					string type = getCononicalType(re->getDecl()); 	
					unsigned dim = getArrayDimension(type); 

					//This is an array!
					if (dim)
					{
						cai.index = i;
						cai.isPointerAccess = false;
					
						if (cai.isGlobalMemThreadPrivate)
						{
							ca->hasGlobalMemThreadPrivate = true;
						}
						
						ca->gCallArgs.push_back(cai);
					}
				}
			}
		}

	}
}

void GlobalCallArgPicker::VisitCallExpr(CallExpr *Call) {
	CallExpr* previousCe = NULL;
	bool enableHere = false;
	CallArgInfoContainer* previousCa = NULL;

	PrintExpr(Call->getCallee());
	OS << "(";

	if (inCallArg)
	{
		previousCa = ca;
		previousCe = cce;
	}
	else
	{
		enableHere = true;
		enableCallArgTrack();
		cce = Call;
	}

	ca = new CallArgInfoContainer;
	ca->ce = Call;
	
	PrintCallArgs(Call);
	OS << ")";

	if (inCallArg)
	{
		calledFuncs.push_back(ca);
	}

	if (enableHere)
	{
		disableCallArgTrack();
	}
	else{
		if (inCallArg)
		{
			//restore the callArg
			cce = previousCe;
			ca = previousCa;
		}
	}
}
void GlobalCallArgPicker::VisitMemberExpr(MemberExpr *Node) {
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
void GlobalCallArgPicker::VisitObjCIsaExpr(ObjCIsaExpr *Node) {
	PrintExpr(Node->getBase());
	OS << (Node->isArrow() ? "->isa" : ".isa");
}

void GlobalCallArgPicker::VisitExtVectorElementExpr(ExtVectorElementExpr *Node) {
	PrintExpr(Node->getBase());
	OS << ".";
	OS << Node->getAccessor().getName();
}
void GlobalCallArgPicker::VisitCStyleCastExpr(CStyleCastExpr *Node) {
	OS << "(" << Node->getType().getAsString(Policy) << ")";
	PrintExpr(Node->getSubExpr());
}
void GlobalCallArgPicker::VisitCompoundLiteralExpr(CompoundLiteralExpr *Node) {
	OS << "(" << Node->getType().getAsString(Policy) << ")";
	PrintExpr(Node->getInitializer());
}
void GlobalCallArgPicker::VisitImplicitCastExpr(ImplicitCastExpr *Node) {
	// No need to print anything, simply forward to the sub expression.
	PrintExpr(Node->getSubExpr());
}
void GlobalCallArgPicker::VisitBinaryOperator(BinaryOperator *Node) {
	PrintExpr(Node->getLHS());
	OS << " " << BinaryOperator::getOpcodeStr(Node->getOpcode()) << " ";
	PrintExpr(Node->getRHS());
}
void GlobalCallArgPicker::VisitCompoundAssignOperator(CompoundAssignOperator *Node) {
	PrintExpr(Node->getLHS());
	OS << " " << BinaryOperator::getOpcodeStr(Node->getOpcode()) << " ";
	PrintExpr(Node->getRHS());
}
void GlobalCallArgPicker::VisitConditionalOperator(ConditionalOperator *Node) {
	PrintExpr(Node->getCond());
	OS << " ? ";
	PrintExpr(Node->getLHS());
	OS << " : ";
	PrintExpr(Node->getRHS());
}

// GNU extensions.

void
GlobalCallArgPicker::VisitBinaryConditionalOperator(BinaryConditionalOperator *Node) {
	PrintExpr(Node->getCommon());
	OS << " ?: ";
	PrintExpr(Node->getFalseExpr());
}
void GlobalCallArgPicker::VisitAddrLabelExpr(AddrLabelExpr *Node) {
	OS << "&&" << Node->getLabel()->getName();
}

void GlobalCallArgPicker::VisitStmtExpr(StmtExpr *E) {
	OS << "(";
	PrintRawCompoundStmt(E->getSubStmt());
	OS << ")";
}

void GlobalCallArgPicker::VisitChooseExpr(ChooseExpr *Node) {
	OS << "__builtin_choose_expr(";
	PrintExpr(Node->getCond());
	OS << ", ";
	PrintExpr(Node->getLHS());
	OS << ", ";
	PrintExpr(Node->getRHS());
	OS << ")";
}

void GlobalCallArgPicker::VisitGNUNullExpr(GNUNullExpr *) {
	OS << "__null";
}

void GlobalCallArgPicker::VisitShuffleVectorExpr(ShuffleVectorExpr *Node) {
	OS << "__builtin_shufflevector(";
	for (unsigned i = 0, e = Node->getNumSubExprs(); i != e; ++i) {
		if (i) OS << ", ";
		PrintExpr(Node->getExpr(i));
	}
	OS << ")";
}

void GlobalCallArgPicker::VisitInitListExpr(InitListExpr* Node) {
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

void GlobalCallArgPicker::VisitParenListExpr(ParenListExpr* Node) {
	OS << "( ";
	for (unsigned i = 0, e = Node->getNumExprs(); i != e; ++i) {
		if (i) OS << ", ";
		PrintExpr(Node->getExpr(i));
	}
	OS << " )";
}

void GlobalCallArgPicker::VisitDesignatedInitExpr(DesignatedInitExpr *Node) {
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

void GlobalCallArgPicker::VisitImplicitValueInitExpr(ImplicitValueInitExpr *Node) {
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

void GlobalCallArgPicker::VisitVAArgExpr(VAArgExpr *Node) {
	OS << "__builtin_va_arg(";
	PrintExpr(Node->getSubExpr());
	OS << ", ";
	OS << Node->getType().getAsString(Policy);
	OS << ")";
}

// C++
void GlobalCallArgPicker::VisitCXXOperatorCallExpr(CXXOperatorCallExpr *Node) {
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

void GlobalCallArgPicker::VisitCXXMemberCallExpr(CXXMemberCallExpr *Node) {
	VisitCallExpr(cast<CallExpr>(Node));
}

void GlobalCallArgPicker::VisitCUDAKernelCallExpr(CUDAKernelCallExpr *Node) {
	PrintExpr(Node->getCallee());
	OS << "<<<";
	PrintCallArgs(Node->getConfig());
	OS << ">>>(";
	PrintCallArgs(Node);
	OS << ")";
}

void GlobalCallArgPicker::VisitCXXNamedCastExpr(CXXNamedCastExpr *Node) {
	OS << Node->getCastName() << '<';
	OS << Node->getTypeAsWritten().getAsString(Policy) << ">(";
	PrintExpr(Node->getSubExpr());
	OS << ")";
}

void GlobalCallArgPicker::VisitCXXStaticCastExpr(CXXStaticCastExpr *Node) {
	VisitCXXNamedCastExpr(Node);
}

void GlobalCallArgPicker::VisitCXXDynamicCastExpr(CXXDynamicCastExpr *Node) {
	VisitCXXNamedCastExpr(Node);
}

void GlobalCallArgPicker::VisitCXXReinterpretCastExpr(CXXReinterpretCastExpr *Node) {
	VisitCXXNamedCastExpr(Node);
}

void GlobalCallArgPicker::VisitCXXConstCastExpr(CXXConstCastExpr *Node) {
	VisitCXXNamedCastExpr(Node);
}

void GlobalCallArgPicker::VisitCXXTypeidExpr(CXXTypeidExpr *Node) {
	OS << "typeid(";
	if (Node->isTypeOperand()) {
		OS << Node->getTypeOperand().getAsString(Policy);
	} else {
		PrintExpr(Node->getExprOperand());
	}
	OS << ")";
}

void GlobalCallArgPicker::VisitCXXUuidofExpr(CXXUuidofExpr *Node) {
	OS << "__uuidof(";
	if (Node->isTypeOperand()) {
		OS << Node->getTypeOperand().getAsString(Policy);
	} else {
		PrintExpr(Node->getExprOperand());
	}
	OS << ")";
}

void GlobalCallArgPicker::VisitCXXBoolLiteralExpr(CXXBoolLiteralExpr *Node) {
	OS << (Node->getValue() ? "true" : "false");
}

void GlobalCallArgPicker::VisitCXXNullPtrLiteralExpr(CXXNullPtrLiteralExpr *Node) {
	OS << "nullptr";
}

void GlobalCallArgPicker::VisitCXXThisExpr(CXXThisExpr *Node) {
	OS << "this";
}

void GlobalCallArgPicker::VisitCXXThrowExpr(CXXThrowExpr *Node) {
	if (Node->getSubExpr() == 0)
		OS << "throw";
	else {
		OS << "throw ";
		PrintExpr(Node->getSubExpr());
	}
}

void GlobalCallArgPicker::VisitCXXDefaultArgExpr(CXXDefaultArgExpr *Node) {
	// Nothing to print: we picked up the default argument
}

void GlobalCallArgPicker::VisitCXXFunctionalCastExpr(CXXFunctionalCastExpr *Node) {
	OS << Node->getType().getAsString(Policy);
	OS << "(";
	PrintExpr(Node->getSubExpr());
	OS << ")";
}

void GlobalCallArgPicker::VisitCXXBindTemporaryExpr(CXXBindTemporaryExpr *Node) {
	PrintExpr(Node->getSubExpr());
}

void GlobalCallArgPicker::VisitCXXTemporaryObjectExpr(CXXTemporaryObjectExpr *Node) {
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

void GlobalCallArgPicker::VisitCXXScalarValueInitExpr(CXXScalarValueInitExpr *Node) {
	if (TypeSourceInfo *TSInfo = Node->getTypeSourceInfo())
		OS << TSInfo->getType().getAsString(Policy) << "()";
	else
		OS << Node->getType().getAsString(Policy) << "()";
}

void GlobalCallArgPicker::VisitCXXNewExpr(CXXNewExpr *E) {
}

void GlobalCallArgPicker::VisitCXXDeleteExpr(CXXDeleteExpr *E) {
	if (E->isGlobalDelete())
		OS << "::";
	OS << "delete ";
	if (E->isArrayForm())
		OS << "[] ";
	PrintExpr(E->getArgument());
}

void GlobalCallArgPicker::VisitCXXPseudoDestructorExpr(CXXPseudoDestructorExpr *E) {
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

void GlobalCallArgPicker::VisitCXXConstructExpr(CXXConstructExpr *E) {
	for (unsigned i = 0, e = E->getNumArgs(); i != e; ++i) {
		if (isa<CXXDefaultArgExpr>(E->getArg(i))) {
			// Don't print any defaulted arguments
			break;
		}

		if (i) OS << ", ";
		PrintExpr(E->getArg(i));
	}
}

void GlobalCallArgPicker::VisitExprWithCleanups(ExprWithCleanups *E) {
	// Just forward to the sub expression.
	PrintExpr(E->getSubExpr());
}

void
GlobalCallArgPicker::VisitCXXUnresolvedConstructExpr(
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

void GlobalCallArgPicker::VisitCXXDependentScopeMemberExpr(
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

void GlobalCallArgPicker::VisitUnresolvedMemberExpr(UnresolvedMemberExpr *Node) {
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

void GlobalCallArgPicker::VisitCXXNoexceptExpr(CXXNoexceptExpr *E) {
	OS << "noexcept(";
	PrintExpr(E->getOperand());
	OS << ")";
}

void GlobalCallArgPicker::VisitPackExpansionExpr(PackExpansionExpr *E) {
	PrintExpr(E->getPattern());
	OS << "...";
}

void GlobalCallArgPicker::VisitSizeOfPackExpr(SizeOfPackExpr *E) {
	OS << "sizeof...(" << E->getPack()->getNameAsString() << ")";
}

void GlobalCallArgPicker::VisitSubstNonTypeTemplateParmPackExpr(
		SubstNonTypeTemplateParmPackExpr *Node) {
	OS << Node->getParameterPack()->getNameAsString();
}

// Obj-C

void GlobalCallArgPicker::VisitObjCStringLiteral(ObjCStringLiteral *Node) {
	OS << "@";
	VisitStringLiteral(Node->getString());
}

void GlobalCallArgPicker::VisitObjCEncodeExpr(ObjCEncodeExpr *Node) {
	OS << "@encode(" << Node->getEncodedType().getAsString(Policy) << ')';
}

void GlobalCallArgPicker::VisitObjCSelectorExpr(ObjCSelectorExpr *Node) {
	OS << "@selector(" << Node->getSelector().getAsString() << ')';
}

void GlobalCallArgPicker::VisitObjCProtocolExpr(ObjCProtocolExpr *Node) {
	OS << "@protocol(" << Node->getProtocol() << ')';
}

void GlobalCallArgPicker::VisitObjCMessageExpr(ObjCMessageExpr *Mess) {
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


void GlobalCallArgPicker::VisitBlockExpr(BlockExpr *Node) {
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

void GlobalCallArgPicker::VisitBlockDeclRefExpr(BlockDeclRefExpr *Node) {
	OS << Node->getDecl();
}

void GlobalCallArgPicker::VisitOpaqueValueExpr(OpaqueValueExpr *Node) {}

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

	GlobalCallArgPicker P(OS, Context, Helper, Policy, Indentation);
	P.Visit(const_cast<Stmt*>(this));
}
#endif

//===----------------------------------------------------------------------===//
// PrinterHelper
//===----------------------------------------------------------------------===//

// Implement virtual destructor.

void GlobalCallArgPicker::VisitBinaryTypeTraitExpr(clang::BinaryTypeTraitExpr*)
{

}

void GlobalCallArgPicker::VisitObjCForCollectionStmt(clang::ObjCForCollectionStmt*)
{

}

void GlobalCallArgPicker::VisitUnaryTypeTraitExpr(clang::UnaryTypeTraitExpr*)
{

}

//OMP2OCL
void GlobalCallArgPicker::VisitOclFlush(clang::OclFlush*) {}
void GlobalCallArgPicker::VisitOclHostFlush(clang::OclHostFlush*) {}
void GlobalCallArgPicker::VisitOclInit(clang::OclInit*) {}
void GlobalCallArgPicker::VisitOclTerm(clang::OclTerm*) {}
void GlobalCallArgPicker::VisitOclSync(clang::OclSync*) {}
void GlobalCallArgPicker::VisitOclResetMLStmt(clang::OclResetMLStmt*) {}
void GlobalCallArgPicker::VisitOclEnableMLRecordStmt(clang::OclEnableMLRecordStmt*) {}
void GlobalCallArgPicker::VisitOclDisableMLRecordStmt(clang::OclDisableMLRecordStmt*) {}
void GlobalCallArgPicker::VisitOclDumpMLFStmt(clang::OclDumpMLFStmt*) {}
void GlobalCallArgPicker::VisitOclStartProfile(clang::OclStartProfile*) {}
void GlobalCallArgPicker::VisitOclDumpProfile(clang::OclDumpProfile*) {}
void GlobalCallArgPicker::VisitOclStopProfile(clang::OclStopProfile*) {}
void GlobalCallArgPicker::VisitOclHostRead(clang::OclHostRead*) {}
void GlobalCallArgPicker::VisitOclDevRead(clang::OclDevRead*) {}
void GlobalCallArgPicker::VisitOclDevWrite(clang::OclDevWrite*) {}
