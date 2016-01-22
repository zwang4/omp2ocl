//===--- OpenCLGenericStmtVisitor.cpp - Printing implementation for Stmt ASTs ----------===//
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

#include "clang/Omp2Ocl/OpenCLKernelSchedule.h"
#include "clang/Omp2Ocl/OpenCLGenericStmtVisitor.h"
#include "clang/Omp2Ocl/OpenCLHostPrinter.h"
#include "clang/Omp2Ocl/OpenCLGlobalInfoContainer.h"

using namespace clang;

//===----------------------------------------------------------------------===//
// OpenCLGenericStmtVisitor Visitor
//===----------------------------------------------------------------------===//
OpenCLGenericStmtVisitor::OpenCLGenericStmtVisitor(llvm::raw_ostream &os, ASTContext &C, PrinterHelper* helper,
		const PrintingPolicy &Policy,
		unsigned Indentation)
: OS(os), Context(C), IndentLevel(Indentation), Helper(helper), Policy(Policy){
}

OpenCLGenericStmtVisitor::~OpenCLGenericStmtVisitor()
{
}


void OpenCLGenericStmtVisitor::PrintStmt(Stmt *S) {
	PrintStmt(S, Policy.Indentation);
}

void OpenCLGenericStmtVisitor::PrintStmt(Stmt *S, int SubIndent) {
	IndentLevel += SubIndent;
	if (S && isa<Expr>(S)) {
		// If this is an expr used in a stmt context, indent and newline it.
		Indent();
		Visit(S);
		OS << ";\n";
	} else if (S) {
		Visit(S);
	} else {
		Indent() << "<<<NULL STATEMENT>>>\n";
	}
	IndentLevel -= SubIndent;
}

void OpenCLGenericStmtVisitor::PrintExpr(Expr *E) {
	if (E)
		Visit(E);
	else
		OS << "<null expr>";
}

void OpenCLGenericStmtVisitor::VisitForStmt(ForStmt *Node) {
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



/// PrintRawCompoundStmt - Print a compound stmt without indenting the {, and
/// with no newline after the }.
void OpenCLGenericStmtVisitor::PrintRawCompoundStmt(CompoundStmt *Node) {
	OS << "{\n";
	for (CompoundStmt::body_iterator I = Node->body_begin(), E = Node->body_end();
			I != E; ++I)
	{
		PrintStmt(*I);
	}

	Indent() << "}";
}

void OpenCLGenericStmtVisitor::PrintRawDecl(Decl *D) {
	D->print(OS, Policy, IndentLevel);
}

void OpenCLGenericStmtVisitor::PrintRawDeclStmt(DeclStmt *S) {
	DeclStmt::decl_iterator Begin = S->decl_begin(), End = S->decl_end();
	llvm::SmallVector<Decl*, 2> Decls;
	for ( ; Begin != End; ++Begin)
		Decls.push_back(*Begin);

	Decl::printGroup(Decls.data(), Decls.size(), OS, Policy, IndentLevel);
}

void OpenCLGenericStmtVisitor::VisitNullStmt(NullStmt *Node) {
	Indent() << ";\n";
}

void OpenCLGenericStmtVisitor::VisitDeclStmt(DeclStmt *Node) {
	Indent();
	PrintRawDeclStmt(Node);
	OS << ";\n";
}

void OpenCLGenericStmtVisitor::VisitCompoundStmt(CompoundStmt *Node) {
	Indent();
	PrintRawCompoundStmt(Node);
	OS << "\n";
}

void OpenCLGenericStmtVisitor::VisitCaseStmt(CaseStmt *Node) {
	Indent(-1) << "case ";
	PrintExpr(Node->getLHS());
	if (Node->getRHS()) {
		OS << " ... ";
		PrintExpr(Node->getRHS());
	}
	OS << ":\n";

	PrintStmt(Node->getSubStmt(), 0);
}

void OpenCLGenericStmtVisitor::VisitDefaultStmt(DefaultStmt *Node) {
	Indent(-1) << "default:\n";
	PrintStmt(Node->getSubStmt(), 0);
}

void OpenCLGenericStmtVisitor::VisitLabelStmt(LabelStmt *Node) {
	Indent(-1) << Node->getName() << ":\n";
	PrintStmt(Node->getSubStmt(), 0);
}

void OpenCLGenericStmtVisitor::PrintRawIfStmt(IfStmt *If) {
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

void OpenCLGenericStmtVisitor::VisitIfStmt(IfStmt *If) {
	Indent();
	PrintRawIfStmt(If);
}

void OpenCLGenericStmtVisitor::VisitSwitchStmt(SwitchStmt *Node) {
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

void OpenCLGenericStmtVisitor::VisitWhileStmt(WhileStmt *Node) {
	Indent() << "while (";
	PrintExpr(Node->getCond());
	OS << ")\n";
	PrintStmt(Node->getBody());
}

void OpenCLGenericStmtVisitor::VisitDoStmt(DoStmt *Node) {
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




void OpenCLGenericStmtVisitor::VisitObjCForCollectionStmt(ObjCForCollectionStmt *Node) {
	Indent() << "for (";
	if (DeclStmt *DS = dyn_cast<DeclStmt>(Node->getElement()))
		PrintRawDeclStmt(DS);
	else
		PrintExpr(cast<Expr>(Node->getElement()));
	OS << " in ";
	PrintExpr(Node->getCollection());
	OS << ") ";

	if (CompoundStmt *CS = dyn_cast<CompoundStmt>(Node->getBody())) {
		PrintRawCompoundStmt(CS);
		OS << "\n";
	} else {
		OS << "\n";
		PrintStmt(Node->getBody());
	}
}

void OpenCLGenericStmtVisitor::VisitGotoStmt(GotoStmt *Node) {
	Indent() << "goto " << Node->getLabel()->getName() << ";\n";
}

void OpenCLGenericStmtVisitor::VisitIndirectGotoStmt(IndirectGotoStmt *Node) {
	Indent() << "goto *";
	PrintExpr(Node->getTarget());
	OS << ";\n";
}

void OpenCLGenericStmtVisitor::VisitContinueStmt(ContinueStmt *Node) {
	Indent() << "continue;\n";
}

void OpenCLGenericStmtVisitor::VisitBreakStmt(BreakStmt *Node) {
	Indent() << "break;\n";
}


void OpenCLGenericStmtVisitor::VisitReturnStmt(ReturnStmt *Node) {
	Indent() << "return";
	if (Node->getRetValue()) {
		OS << " ";
		PrintExpr(Node->getRetValue());
	}
	OS << ";\n";
}


void OpenCLGenericStmtVisitor::VisitAsmStmt(AsmStmt *Node) {
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

void OpenCLGenericStmtVisitor::VisitObjCAtTryStmt(ObjCAtTryStmt *Node) {
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

void OpenCLGenericStmtVisitor::VisitObjCAtFinallyStmt(ObjCAtFinallyStmt *Node) {
}

void OpenCLGenericStmtVisitor::VisitObjCAtCatchStmt (ObjCAtCatchStmt *Node) {
	Indent() << "@catch (...) { /* todo */ } \n";
}

void OpenCLGenericStmtVisitor::VisitObjCAtThrowStmt(ObjCAtThrowStmt *Node) {
	Indent() << "@throw";
	if (Node->getThrowExpr()) {
		OS << " ";
		PrintExpr(Node->getThrowExpr());
	}
	OS << ";\n";
}

void OpenCLGenericStmtVisitor::VisitObjCAtSynchronizedStmt(ObjCAtSynchronizedStmt *Node) {
	Indent() << "@synchronized (";
	PrintExpr(Node->getSynchExpr());
	OS << ")";
	PrintRawCompoundStmt(Node->getSynchBody());
	OS << "\n";
}

void OpenCLGenericStmtVisitor::PrintRawCXXCatchStmt(CXXCatchStmt *Node) {
	OS << "catch (";
	if (Decl *ExDecl = Node->getExceptionDecl())
		PrintRawDecl(ExDecl);
	else
		OS << "...";
	OS << ") ";
	PrintRawCompoundStmt(cast<CompoundStmt>(Node->getHandlerBlock()));
}

void OpenCLGenericStmtVisitor::VisitCXXCatchStmt(CXXCatchStmt *Node) {
	Indent();
	PrintRawCXXCatchStmt(Node);
	OS << "\n";
}

void OpenCLGenericStmtVisitor::VisitCXXTryStmt(CXXTryStmt *Node) {
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

void OpenCLGenericStmtVisitor::VisitDeclRefExpr(DeclRefExpr *Node) {
	if (NestedNameSpecifier *Qualifier = Node->getQualifier())
		Qualifier->print(OS, Policy);
	OS << Node->getNameInfo();
	if (Node->hasExplicitTemplateArgs())
		OS << TemplateSpecializationType::PrintTemplateArgumentList(
				Node->getTemplateArgs(),
				Node->getNumTemplateArgs(),
				Policy);  
}

void OpenCLGenericStmtVisitor::VisitDependentScopeDeclRefExpr(
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

void OpenCLGenericStmtVisitor::VisitUnresolvedLookupExpr(UnresolvedLookupExpr *Node) {
	if (Node->getQualifier())
		Node->getQualifier()->print(OS, Policy);
	OS << Node->getNameInfo();
	if (Node->hasExplicitTemplateArgs())
		OS << TemplateSpecializationType::PrintTemplateArgumentList(
				Node->getTemplateArgs(),
				Node->getNumTemplateArgs(),
				Policy);
}

void OpenCLGenericStmtVisitor::VisitObjCIvarRefExpr(ObjCIvarRefExpr *Node) {
	if (Node->getBase()) {
		PrintExpr(Node->getBase());
		OS << (Node->isArrow() ? "->" : ".");
	}
	OS << Node->getDecl();
}

void OpenCLGenericStmtVisitor::VisitObjCPropertyRefExpr(ObjCPropertyRefExpr *Node) {
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

void OpenCLGenericStmtVisitor::VisitPredefinedExpr(PredefinedExpr *Node) {
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

void OpenCLGenericStmtVisitor::VisitCharacterLiteral(CharacterLiteral *Node) {
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

void OpenCLGenericStmtVisitor::VisitIntegerLiteral(IntegerLiteral *Node) {
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
void OpenCLGenericStmtVisitor::VisitFloatingLiteral(FloatingLiteral *Node) {
	// FIXME: print value more precisely.
	//OS << Node->getValueAsApproximateDouble();
	OS << Node->getLexString();
		
}

void OpenCLGenericStmtVisitor::VisitImaginaryLiteral(ImaginaryLiteral *Node) {
	PrintExpr(Node->getSubExpr());
	OS << "i";
}

void OpenCLGenericStmtVisitor::VisitStringLiteral(StringLiteral *Str) {
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
void OpenCLGenericStmtVisitor::VisitParenExpr(ParenExpr *Node) {
	OS << "(";
	PrintExpr(Node->getSubExpr());
	OS << ")";
}
void OpenCLGenericStmtVisitor::VisitUnaryOperator(UnaryOperator *Node) {
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

void OpenCLGenericStmtVisitor::VisitOffsetOfExpr(OffsetOfExpr *Node) {
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

void OpenCLGenericStmtVisitor::VisitUnaryExprOrTypeTraitExpr(UnaryExprOrTypeTraitExpr *Node){
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


void OpenCLGenericStmtVisitor::VisitArraySubscriptExpr(ArraySubscriptExpr *Node) {
		PrintExpr(Node->getLHS());
		OS << "[";
		PrintExpr(Node->getRHS());
	OS << "]";
}

void OpenCLGenericStmtVisitor::PrintCallArgs(CallExpr *Call) {
	for (unsigned i = 0, e = Call->getNumArgs(); i != e; ++i) {
		if (isa<CXXDefaultArgExpr>(Call->getArg(i))) {
			// Don't print any defaulted arguments
			break;
		}

		if (i) OS << ", ";
		PrintExpr(Call->getArg(i));
	}
}

void OpenCLGenericStmtVisitor::VisitCallExpr(CallExpr *Call) {
	PrintExpr(Call->getCallee());
	OS << "(";
	PrintCallArgs(Call);
	OS << ")";
}
void OpenCLGenericStmtVisitor::VisitMemberExpr(MemberExpr *Node) {
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
void OpenCLGenericStmtVisitor::VisitObjCIsaExpr(ObjCIsaExpr *Node) {
	PrintExpr(Node->getBase());
	OS << (Node->isArrow() ? "->isa" : ".isa");
}

void OpenCLGenericStmtVisitor::VisitExtVectorElementExpr(ExtVectorElementExpr *Node) {
	PrintExpr(Node->getBase());
	OS << ".";
	OS << Node->getAccessor().getName();
}
void OpenCLGenericStmtVisitor::VisitCStyleCastExpr(CStyleCastExpr *Node) {
	OS << "(" << Node->getType().getAsString(Policy) << ")";
	PrintExpr(Node->getSubExpr());
}
void OpenCLGenericStmtVisitor::VisitCompoundLiteralExpr(CompoundLiteralExpr *Node) {
	OS << "(" << Node->getType().getAsString(Policy) << ")";
	PrintExpr(Node->getInitializer());
}
void OpenCLGenericStmtVisitor::VisitImplicitCastExpr(ImplicitCastExpr *Node) {
	// No need to print anything, simply forward to the sub expression.
	PrintExpr(Node->getSubExpr());
}
void OpenCLGenericStmtVisitor::VisitBinaryOperator(BinaryOperator *Node) {
	PrintExpr(Node->getLHS());
	OS << " " << BinaryOperator::getOpcodeStr(Node->getOpcode()) << " ";
	PrintExpr(Node->getRHS());
}
void OpenCLGenericStmtVisitor::VisitCompoundAssignOperator(CompoundAssignOperator *Node) {
	PrintExpr(Node->getLHS());
	OS << " " << BinaryOperator::getOpcodeStr(Node->getOpcode()) << " ";
	PrintExpr(Node->getRHS());
}
void OpenCLGenericStmtVisitor::VisitConditionalOperator(ConditionalOperator *Node) {
	PrintExpr(Node->getCond());
	OS << " ? ";
	PrintExpr(Node->getLHS());
	OS << " : ";
	PrintExpr(Node->getRHS());
}

// GNU extensions.

void
OpenCLGenericStmtVisitor::VisitBinaryConditionalOperator(BinaryConditionalOperator *Node) {
	PrintExpr(Node->getCommon());
	OS << " ?: ";
	PrintExpr(Node->getFalseExpr());
}
void OpenCLGenericStmtVisitor::VisitAddrLabelExpr(AddrLabelExpr *Node) {
	OS << "&&" << Node->getLabel()->getName();
}

void OpenCLGenericStmtVisitor::VisitStmtExpr(StmtExpr *E) {
	OS << "(";
	PrintRawCompoundStmt(E->getSubStmt());
	OS << ")";
}

void OpenCLGenericStmtVisitor::VisitChooseExpr(ChooseExpr *Node) {
	OS << "__builtin_choose_expr(";
	PrintExpr(Node->getCond());
	OS << ", ";
	PrintExpr(Node->getLHS());
	OS << ", ";
	PrintExpr(Node->getRHS());
	OS << ")";
}

void OpenCLGenericStmtVisitor::VisitGNUNullExpr(GNUNullExpr *) {
	OS << "__null";
}

void OpenCLGenericStmtVisitor::VisitShuffleVectorExpr(ShuffleVectorExpr *Node) {
	OS << "__builtin_shufflevector(";
	for (unsigned i = 0, e = Node->getNumSubExprs(); i != e; ++i) {
		if (i) OS << ", ";
		PrintExpr(Node->getExpr(i));
	}
	OS << ")";
}

void OpenCLGenericStmtVisitor::VisitInitListExpr(InitListExpr* Node) {
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

void OpenCLGenericStmtVisitor::VisitParenListExpr(ParenListExpr* Node) {
	OS << "( ";
	for (unsigned i = 0, e = Node->getNumExprs(); i != e; ++i) {
		if (i) OS << ", ";
		PrintExpr(Node->getExpr(i));
	}
	OS << " )";
}

void OpenCLGenericStmtVisitor::VisitDesignatedInitExpr(DesignatedInitExpr *Node) {
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

void OpenCLGenericStmtVisitor::VisitImplicitValueInitExpr(ImplicitValueInitExpr *Node) {
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

void OpenCLGenericStmtVisitor::VisitVAArgExpr(VAArgExpr *Node) {
	OS << "__builtin_va_arg(";
	PrintExpr(Node->getSubExpr());
	OS << ", ";
	OS << Node->getType().getAsString(Policy);
	OS << ")";
}

// C++
void OpenCLGenericStmtVisitor::VisitCXXOperatorCallExpr(CXXOperatorCallExpr *Node) {
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

void OpenCLGenericStmtVisitor::VisitCXXMemberCallExpr(CXXMemberCallExpr *Node) {
	VisitCallExpr(cast<CallExpr>(Node));
}

void OpenCLGenericStmtVisitor::VisitCUDAKernelCallExpr(CUDAKernelCallExpr *Node) {
	PrintExpr(Node->getCallee());
	OS << "<<<";
	PrintCallArgs(Node->getConfig());
	OS << ">>>(";
	PrintCallArgs(Node);
	OS << ")";
}

void OpenCLGenericStmtVisitor::VisitCXXNamedCastExpr(CXXNamedCastExpr *Node) {
	OS << Node->getCastName() << '<';
	OS << Node->getTypeAsWritten().getAsString(Policy) << ">(";
	PrintExpr(Node->getSubExpr());
	OS << ")";
}

void OpenCLGenericStmtVisitor::VisitCXXStaticCastExpr(CXXStaticCastExpr *Node) {
	VisitCXXNamedCastExpr(Node);
}

void OpenCLGenericStmtVisitor::VisitCXXDynamicCastExpr(CXXDynamicCastExpr *Node) {
	VisitCXXNamedCastExpr(Node);
}

void OpenCLGenericStmtVisitor::VisitCXXReinterpretCastExpr(CXXReinterpretCastExpr *Node) {
	VisitCXXNamedCastExpr(Node);
}

void OpenCLGenericStmtVisitor::VisitCXXConstCastExpr(CXXConstCastExpr *Node) {
	VisitCXXNamedCastExpr(Node);
}

void OpenCLGenericStmtVisitor::VisitCXXTypeidExpr(CXXTypeidExpr *Node) {
	OS << "typeid(";
	if (Node->isTypeOperand()) {
		OS << Node->getTypeOperand().getAsString(Policy);
	} else {
		PrintExpr(Node->getExprOperand());
	}
	OS << ")";
}

void OpenCLGenericStmtVisitor::VisitCXXUuidofExpr(CXXUuidofExpr *Node) {
	OS << "__uuidof(";
	if (Node->isTypeOperand()) {
		OS << Node->getTypeOperand().getAsString(Policy);
	} else {
		PrintExpr(Node->getExprOperand());
	}
	OS << ")";
}

void OpenCLGenericStmtVisitor::VisitCXXBoolLiteralExpr(CXXBoolLiteralExpr *Node) {
	OS << (Node->getValue() ? "true" : "false");
}

void OpenCLGenericStmtVisitor::VisitCXXNullPtrLiteralExpr(CXXNullPtrLiteralExpr *Node) {
	OS << "nullptr";
}

void OpenCLGenericStmtVisitor::VisitCXXThisExpr(CXXThisExpr *Node) {
	OS << "this";
}

void OpenCLGenericStmtVisitor::VisitCXXThrowExpr(CXXThrowExpr *Node) {
	if (Node->getSubExpr() == 0)
		OS << "throw";
	else {
		OS << "throw ";
		PrintExpr(Node->getSubExpr());
	}
}

void OpenCLGenericStmtVisitor::VisitCXXDefaultArgExpr(CXXDefaultArgExpr *Node) {
	// Nothing to print: we picked up the default argument
}

void OpenCLGenericStmtVisitor::VisitCXXFunctionalCastExpr(CXXFunctionalCastExpr *Node) {
	OS << Node->getType().getAsString(Policy);
	OS << "(";
	PrintExpr(Node->getSubExpr());
	OS << ")";
}

void OpenCLGenericStmtVisitor::VisitCXXBindTemporaryExpr(CXXBindTemporaryExpr *Node) {
	PrintExpr(Node->getSubExpr());
}

void OpenCLGenericStmtVisitor::VisitCXXTemporaryObjectExpr(CXXTemporaryObjectExpr *Node) {
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

void OpenCLGenericStmtVisitor::VisitCXXScalarValueInitExpr(CXXScalarValueInitExpr *Node) {
	if (TypeSourceInfo *TSInfo = Node->getTypeSourceInfo())
		OS << TSInfo->getType().getAsString(Policy) << "()";
	else
		OS << Node->getType().getAsString(Policy) << "()";
}

void OpenCLGenericStmtVisitor::VisitCXXNewExpr(CXXNewExpr *E) {
	if (E->isGlobalNew())
		OS << "::";
	OS << "new ";
	unsigned NumPlace = E->getNumPlacementArgs();
	if (NumPlace > 0) {
		OS << "(";
		PrintExpr(E->getPlacementArg(0));
		for (unsigned i = 1; i < NumPlace; ++i) {
			OS << ", ";
			PrintExpr(E->getPlacementArg(i));
		}
		OS << ") ";
	}
	if (E->isParenTypeId())
		OS << "(";
	std::string TypeS;
	if (Expr *Size = E->getArraySize()) {
		llvm::raw_string_ostream s(TypeS);
		OS << Size;
		s.flush();
		TypeS = "[" + TypeS + "]";
	}
	E->getAllocatedType().getAsStringInternal(TypeS, Policy);
	OS << TypeS;
	if (E->isParenTypeId())
		OS << ")";

	if (E->hasInitializer()) {
		OS << "(";
		unsigned NumCons = E->getNumConstructorArgs();
		if (NumCons > 0) {
			PrintExpr(E->getConstructorArg(0));
			for (unsigned i = 1; i < NumCons; ++i) {
				OS << ", ";
				PrintExpr(E->getConstructorArg(i));
			}
		}
		OS << ")";
	}
}

void OpenCLGenericStmtVisitor::VisitCXXDeleteExpr(CXXDeleteExpr *E) {
	if (E->isGlobalDelete())
		OS << "::";
	OS << "delete ";
	if (E->isArrayForm())
		OS << "[] ";
	PrintExpr(E->getArgument());
}

void OpenCLGenericStmtVisitor::VisitCXXPseudoDestructorExpr(CXXPseudoDestructorExpr *E) {
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

void OpenCLGenericStmtVisitor::VisitCXXConstructExpr(CXXConstructExpr *E) {
	for (unsigned i = 0, e = E->getNumArgs(); i != e; ++i) {
		if (isa<CXXDefaultArgExpr>(E->getArg(i))) {
			// Don't print any defaulted arguments
			break;
		}

		if (i) OS << ", ";
		PrintExpr(E->getArg(i));
	}
}

void OpenCLGenericStmtVisitor::VisitExprWithCleanups(ExprWithCleanups *E) {
	// Just forward to the sub expression.
	PrintExpr(E->getSubExpr());
}

void
OpenCLGenericStmtVisitor::VisitCXXUnresolvedConstructExpr(
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

void OpenCLGenericStmtVisitor::VisitCXXDependentScopeMemberExpr(
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

void OpenCLGenericStmtVisitor::VisitUnresolvedMemberExpr(UnresolvedMemberExpr *Node) {
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

static const char *getTypeTraitName(UnaryTypeTrait UTT) {
	switch (UTT) {
		default: llvm_unreachable("Unknown unary type trait");
		case UTT_HasNothrowAssign:      return "__has_nothrow_assign";
		case UTT_HasNothrowCopy:        return "__has_nothrow_copy";
		case UTT_HasNothrowConstructor: return "__has_nothrow_constructor";
		case UTT_HasTrivialAssign:      return "__has_trivial_assign";
		case UTT_HasTrivialCopy:        return "__has_trivial_copy";
		case UTT_HasTrivialConstructor: return "__has_trivial_constructor";
		case UTT_HasTrivialDestructor:  return "__has_trivial_destructor";
		case UTT_HasVirtualDestructor:  return "__has_virtual_destructor";
		case UTT_IsAbstract:            return "__is_abstract";
		case UTT_IsClass:               return "__is_class";
		case UTT_IsEmpty:               return "__is_empty";
		case UTT_IsEnum:                return "__is_enum";
		case UTT_IsPOD:                 return "__is_pod";
		case UTT_IsPolymorphic:         return "__is_polymorphic";
		case UTT_IsUnion:               return "__is_union";
	}
	return "";
}

static const char *getTypeTraitName(BinaryTypeTrait BTT) {
	switch (BTT) {
		case BTT_IsBaseOf:         return "__is_base_of";
		case BTT_TypeCompatible:   return "__builtin_types_compatible_p";
		case BTT_IsConvertibleTo:  return "__is_convertible_to";
	}
	return "";
}

void OpenCLGenericStmtVisitor::VisitUnaryTypeTraitExpr(UnaryTypeTraitExpr *E) {
	OS << getTypeTraitName(E->getTrait()) << "("
		<< E->getQueriedType().getAsString(Policy) << ")";
}

void OpenCLGenericStmtVisitor::VisitBinaryTypeTraitExpr(BinaryTypeTraitExpr *E) {
	OS << getTypeTraitName(E->getTrait()) << "("
		<< E->getLhsType().getAsString(Policy) << ","
		<< E->getRhsType().getAsString(Policy) << ")";
}

void OpenCLGenericStmtVisitor::VisitCXXNoexceptExpr(CXXNoexceptExpr *E) {
	OS << "noexcept(";
	PrintExpr(E->getOperand());
	OS << ")";
}

void OpenCLGenericStmtVisitor::VisitPackExpansionExpr(PackExpansionExpr *E) {
	PrintExpr(E->getPattern());
	OS << "...";
}

void OpenCLGenericStmtVisitor::VisitSizeOfPackExpr(SizeOfPackExpr *E) {
	OS << "sizeof...(" << E->getPack()->getNameAsString() << ")";
}

void OpenCLGenericStmtVisitor::VisitSubstNonTypeTemplateParmPackExpr(
		SubstNonTypeTemplateParmPackExpr *Node) {
	OS << Node->getParameterPack()->getNameAsString();
}

// Obj-C

void OpenCLGenericStmtVisitor::VisitObjCStringLiteral(ObjCStringLiteral *Node) {
	OS << "@";
	VisitStringLiteral(Node->getString());
}

void OpenCLGenericStmtVisitor::VisitObjCEncodeExpr(ObjCEncodeExpr *Node) {
	OS << "@encode(" << Node->getEncodedType().getAsString(Policy) << ')';
}

void OpenCLGenericStmtVisitor::VisitObjCSelectorExpr(ObjCSelectorExpr *Node) {
	OS << "@selector(" << Node->getSelector().getAsString() << ')';
}

void OpenCLGenericStmtVisitor::VisitObjCProtocolExpr(ObjCProtocolExpr *Node) {
	OS << "@protocol(" << Node->getProtocol() << ')';
}

void OpenCLGenericStmtVisitor::VisitObjCMessageExpr(ObjCMessageExpr *Mess) {
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


void OpenCLGenericStmtVisitor::VisitBlockExpr(BlockExpr *Node) {
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

void OpenCLGenericStmtVisitor::VisitBlockDeclRefExpr(BlockDeclRefExpr *Node) {
	OS << Node->getDecl();
}

void OpenCLGenericStmtVisitor::VisitOpaqueValueExpr(OpaqueValueExpr *Node) {}

//===----------------------------------------------------------------------===//
// PrinterHelper
//===----------------------------------------------------------------------===//

// Implement virtual destructor.
//OMP2OCL
void OpenCLGenericStmtVisitor::VisitOclFlush(clang::OclFlush*) {}
void OpenCLGenericStmtVisitor::VisitOclHostFlush(clang::OclHostFlush*) {}
void OpenCLGenericStmtVisitor::VisitOclInit(clang::OclInit*) {}
void OpenCLGenericStmtVisitor::VisitOclTerm(clang::OclTerm*) {}
void OpenCLGenericStmtVisitor::VisitOclSync(clang::OclSync*) {}
void OpenCLGenericStmtVisitor::VisitOclResetMLStmt(clang::OclResetMLStmt*) {}
void OpenCLGenericStmtVisitor::VisitOclEnableMLRecordStmt(clang::OclEnableMLRecordStmt*) {}
void OpenCLGenericStmtVisitor::VisitOclDisableMLRecordStmt(clang::OclDisableMLRecordStmt*) {}
void OpenCLGenericStmtVisitor::VisitOclDumpMLFStmt(clang::OclDumpMLFStmt*) {}

void OpenCLGenericStmtVisitor::VisitOclStartProfile(clang::OclStartProfile*) {}
void OpenCLGenericStmtVisitor::VisitOclDumpProfile(clang::OclDumpProfile*) {}
void OpenCLGenericStmtVisitor::VisitOclStopProfile(clang::OclStopProfile*) {}
void OpenCLGenericStmtVisitor::VisitOclHostRead(clang::OclHostRead*) {}
void OpenCLGenericStmtVisitor::VisitOclDevRead(clang::OclDevRead*) {}
void OpenCLGenericStmtVisitor::VisitOclDevWrite(clang::OclDevWrite*) {}
