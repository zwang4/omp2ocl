#include "clang/Omp2Ocl/OpenCLBinaryOperatorPrinter.h"
#include "clang/Omp2Ocl/OpenCLRoutine.h"

bool OpenCLBinaryOperatorPrinter::ShouldPerformArithmTrans(BinaryOperator* Node)
{
	string op = BinaryOperator::getOpcodeStr(Node->getOpcode());

	if (op == "+" || op == "-" || op == "/" || op == "*")
	{
		return true;
	}

	return false;
}

//Check whether it is a threadprivate var that declared as a global 
//buffer
//Only threadprivate var that declared as a global buffer needed to 
//be performed arithmetic transformation
bool OpenCLBinaryOperatorPrinter::isGlobalThreadPrivateBuff(DeclRefExpr* e)
{
	ValueDecl* D = e->getDecl();
	unsigned dim = getArrayDimension(D);

	if (dim)
	{
		if (dyn_cast<VarDecl>(D))
		{
			VarDecl* d = dyn_cast<VarDecl>(D);
			if (d->isLocalVarDecl())
			{
				return false;
			}
		}

		if (!D->isDefinedOutsideFunctionOrMethod())
		{
			return false;
		}

		//This is a function parameter
		if (dyn_cast<ParmVarDecl>(D))
		{
			return false;
		}
		
		string name = D->getNameAsString();
		if (OCLCommon::isAGlobalMemThreadPrivateVar(name))
		{
			return true;	
		}
	}

	return false;
}

//This is used to update the arithmetic operation for a threadprivate var that
//is declared as __global
//
//For example:
//  static int x[1024];
//  #pragma omp threadprivate(x) __global
//
//  vranlc(2*NK, &t1, A, x-1);
//
bool OpenCLBinaryOperatorPrinter::ShouldTransformBinaryOperator(BinaryOperator* Node)
{
	Expr* E = Node->getLHS();
	string trans;

	//LHS should be an implicitCastExpr, becuase
	//I only care about array	
	if (dyn_cast<ImplicitCastExpr>(E))
	{
		ImplicitCastExpr* icast = dyn_cast<ImplicitCastExpr>(E);
		E = icast->getSubExpr();

		while(dyn_cast<ImplicitCastExpr>(E))
		{
			icast = dyn_cast<ImplicitCastExpr>(E);
			E = icast->getSubExpr();			
		}

		if (dyn_cast<DeclRefExpr>(E))
		{
			DeclRefExpr* d = dyn_cast<DeclRefExpr>(E);
			if (isGlobalThreadPrivateBuff(d))
			{
				if (ShouldPerformArithmTrans(Node))
				{
					return true;
				}
			}
		}

	}

	return false;
}
