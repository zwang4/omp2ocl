#include "clang/Omp2Ocl/OpenCLHostFuncParamExp.h"
#include "clang/Omp2Ocl/OpenCLRoutine.h"
#include "clang/AST/DeclPrinter.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/StmtPicker.h"
#include "clang/Omp2Ocl/OpenCLCommon.h"

OpenCLHostFuncParamExp::OpenCLHostFuncParamExp(ASTContext& C, llvm::raw_ostream& O) :
Context(C), Out(O)
{

}


bool OpenCLHostFuncParamExp::shouldCallArgPassedWithOclBuffer(CallExpr* Call, unsigned i)
{
	Expr* e = Call->getArg(i);
	Decl* Callee = Call->getCalleeDecl();

	while(dyn_cast<ImplicitCastExpr>(e))
	{
		ImplicitCastExpr* I = dyn_cast<ImplicitCastExpr>(e);
		e = I->getSubExpr();
	}

	FunctionDecl* D = dyn_cast<FunctionDecl>(Callee);
	bool hasBody = false;
	if (D)
	{
		hasBody = D->hasBody();
	}

	if (e)
	{
		StmtPicker sp(Out, Context, NULL, Context.PrintingPolicy);
		sp.Visit(e);
		DeclRefExpr* dc = sp.getFirstDecl();

		//Make sure this argument is not a return value of a callee function
		//e.g. a (b(1,2), c)
		
		if (hasBody && !dyn_cast<CallExpr>(e))
		{
			ParmVarDecl* decl = NULL;
			unsigned dim = 0;
			if (i < D->getNumParams())
			{
				decl = D->getParamDecl(i);
				dim  = getArrayDimension(decl);
			}

			unsigned argD = 0;
			if (dc)
			{
				argD = getArrayDimension(dc->getDecl());
			}
			vector<DeclRefExpr*>& decls = sp.getDecl();
			string SArg = getStringStmt(Context, e);
			
			StringLiteral* SL = dyn_cast<StringLiteral>(e);

			//FIXME: ZHENG, THIS IS VERY VERY URGELY
			if (SArg == "((void *)0)" || SArg == "NULL" || SL)
			{
				return false;
			}
			else
			{
				if (decls.size())
				{
					DeclRefExpr* dr = decls[0];
					bool isLocal = !(dr->getDecl()->isDefinedOutsideFunctionOrMethod());
					string name = dr->getNameInfo().getAsString();

					if (dim)
					{
						if (argD && (isLocal || OCLCommon::isAGlobalMemObj(name)))
						{
							return true;
						}
						else
						{
							return false;
						}
					}	
				}
			}
		}
	}

	return false;
}

void OpenCLHostFuncParamExp::VisitCallArg(CallExpr* Call, unsigned i)
{
	Expr* e = Call->getArg(i);
	Decl* Callee = Call->getCalleeDecl();

	while(dyn_cast<ImplicitCastExpr>(e))
	{
		ImplicitCastExpr* I = dyn_cast<ImplicitCastExpr>(e);
		e = I->getSubExpr();
	}

	FunctionDecl* D = dyn_cast<FunctionDecl>(Callee);
	bool hasBody = false;
	if (D)
	{
		hasBody = D->hasBody();
	}

	if (e)
	{
		StmtPicker sp(Out, Context, NULL, Context.PrintingPolicy);
		sp.Visit(e);
		DeclRefExpr* dc = sp.getFirstDecl();

		//Make sure this argument is not a return value of a callee function
		//e.g. a (b(1,2), c)
		
		if (hasBody && !dyn_cast<CallExpr>(e))
		{
			ParmVarDecl* decl = NULL;
			unsigned dim = 0;
			if (i < D->getNumParams())
			{
				decl = D->getParamDecl(i);
				dim  = getArrayDimension(decl);
			}
			unsigned argD = 0;
			if (dc)
			{
				argD = getArrayDimension(dc->getDecl());
			}
			vector<DeclRefExpr*>& decls = sp.getDecl();
			string SArg = getStringStmt(Context, e);
			
			StringLiteral* SL = dyn_cast<StringLiteral>(e);

			//FIXME: ZHENG, THIS IS VERY VERY URGELY
			if (SArg == "((void *)0)" || SArg == "NULL" || SL)
			{
				if (dim)
					Out << ", NULL";	
			}
			else
			{
				if (decls.size())
				{
					DeclRefExpr* dr = decls[0];
					bool isLocal = !(dr->getDecl()->isDefinedOutsideFunctionOrMethod());
					string name = dr->getNameInfo().getAsString();

					if (dim)
					{
						if (argD && (isLocal || OCLCommon::isAGlobalMemObj(name)))
						{
							Out << ", __ocl_buffer_" << name;
						}
						else
						{
							Out << ", NULL";
						}
					}	
				}
			}
		}
	}
}


/**
 * This revises the function prototype to perform the follow task:
 * a. For each passed in param, checking whether it is an array
 * b. For each array param, declaring an associated ocl_buffer
 *
 */
void OpenCLHostFuncParamExp::VisitFunctionParam(FunctionDecl* D, const FunctionProtoType* FT)
{
	DeclPrinter dp (Out, Context, Context.PrintingPolicy, 0);
	for (unsigned i=0, e=D->getNumParams(); i != e; ++i)
	{
		ParmVarDecl* d = D->getParamDecl(i);
		string type = getCononicalType(d); 
		unsigned dim = getArrayDimension(type);

		if (i)
			Out << ", ";

		dp.VisitParmVarDecl(d);

		if (dim)
		{
			string buf_name = "__ocl_buffer_" + d->getNameAsString();
			Out << ", ocl_buffer * " << buf_name;
			exp_buf_name.push_back(OpenCLExpFuncBuf(d, buf_name));
		}
	}

	if (FT)
	{	
		if (FT->isVariadic()) {
			if (D->getNumParams()) 
				Out << ", ";
			Out << "...";
		}
	}

	Out.flush();
}
