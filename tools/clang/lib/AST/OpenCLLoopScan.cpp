#include "clang/Omp2Ocl/OpenCLLoopScan.h"
#include "clang/AST/StmtPicker.h"
#include "clang/Omp2Ocl/OpenCLCommon.h"
#include "clang/Omp2Ocl/OpenCLRoutine.h"
#include "clang/Omp2Ocl/OpenCLGlobalInfoContainer.h"
#include "clang/AST/CallArgReviseAction.h"
#include "clang/Omp2Ocl/OpenCLKernel.h"
#include "clang/AST/DeclPrinter.h"
#include "clang/AST/Decl.h"

vector<QualType> OpenCLLoopScan::qtypes;

#if 0
void OpenCLLoopScan::scanThreadPrivateVars(OpenCLKernelLoop* loop)
{	
	//dr visits the kernel to record the DeclRef information of variables
	StmtPicker dr (llvm::nulls(), Context, NULL, Context.PrintingPolicy, 0);
	dr.Visit(loop->getKernel());
	vector<DeclRefExpr*>& decls = dr.getDecl();

	for (vector<DeclRefExpr*>::iterator iter = decls.begin(); iter != decls.end(); iter++)
	{
		string name = (*iter)->getNameInfo().getAsString();
		ValueDecl* decl = (*iter)->getDecl();
		VarDecl* varDecl = dyn_cast<VarDecl>(decl);

		vector<OMPThreadPrivateObject>& priVars = OpenCLGlobalInfoContainer::getThreadPrivateVars();
		//I have met a threadprivate name that has the same name and it is 
		//not declared locally
		if (OCLCommon::isAThreadPrivateVariable(name, priVars))
		{
			bool pri=true;
			if (varDecl)
			{
				if (varDecl->isLocalVarDecl())
				{
					pri=false;
				}
			}

			if (pri)
			{
				//Do worry, the addPrivateVariable will check whether a variable
				//with the same name has already existed or not!
				OMPThreadPrivateObject obj(name, OCLCommon::getPrivateVLoc(name, priVars), OCLCommon::isAGTPVariable(name, priVars));
				loop->addThreadPrivateVariable(obj);
			}
		}
	}
}
#endif

void OpenCLLoopScan::reviseCalledArgs(OpenCLKernelLoop* loop)
{
	ForStmt *topF = loop->getForStmt();

	CallArgReviseAction ca(llvm::nulls(), Context, NULL, Context.PrintingPolicy, 0, revisedFuncs);
	ca.Visit(loop->getKernel());
	vector<DeclRefExpr*>& expV = ca.getExpVariables();

	/*
	 * Recorded the expended variables (if any) so that they will be added into 
	 * the OpenCL Kernel arguments.
	 *
	 */
	for (vector<DeclRefExpr*>::iterator iter = expV.begin(); iter != expV.end(); iter++)
	{
		string type = getCononicalType((*iter)->getDecl());

		string passInName = (*iter)->getNameInfo().getAsString();
		string localName = passInName;

		unsigned dim = getArrayDimension(type);
		//This maybe a global memory variable
		if (dim)
		{
			if (!topF->getOMPFor().isVariablePrivate(passInName))
			{
				loop->addParam((*iter));
			}
		}	

		ExpendedCallArg arg(passInName, localName, (*iter));
		topF->addedExpendedCallArg(arg);
	}	
}

void OpenCLLoopScan::collectGlobalInputArguments(OpenCLKernelLoop* loop)
{
	std::string name;
	vector<string> ParamList;
	vector<PLoopParam>& params = loop->getParams();

	vector<OMPThreadPrivateObject>& priVars = OpenCLGlobalInfoContainer::getThreadPrivateVars();

	for (vector<PLoopParam>::iterator iter = params.begin(); iter != params.end(); iter++)
	{
		name = iter->getName();

		//make sure this is not a private variable
		if ((loop->getForStmt()->getOMPFor().isVariablePrivate(name)) == true)
		{
			continue;
		}

		//This variable is declared inside the loop
		if (loop->isInnerDecl(iter->declRef))
		{
			continue;
		}

		if (loop->isAOpenCLNDRangeVar(name))
			continue;

		bool tp = OCLCommon::isAGTPVariable(name, priVars);

		//If it is an one dimensional array
		if (getArrayDimension(iter->getDecl()))
		{
			ValueDecl* d = iter->declRef->getDecl();
			bool isFL = d->isDefinedOutsideFunctionOrMethod();

			//Only threadprivate variables decleared in the __global memory will be treated as global memory
			if (OCLCommon::isAThreadPrivateVariable(name))
			{
				loop->addGlobalMemoryVar(OCLGlobalMemVar(d, tp, isFL, true));
			}
			else
			{
				loop->addGlobalMemoryVar(OCLGlobalMemVar(d, tp, isFL, false));
			}
		}
	}
}

TypedefDecl* OpenCLLoopScan::getTypeDefRef(string ty)
{
	vector<TypedefDecl*>& typeDefs = DeclPrinter::typeDefs;
	for (unsigned i=0; i<typeDefs.size(); i++)
	{
		TypedefDecl* D = typeDefs[i];
		if (D->getUnderlyingType().getAsString() == ty)
			return D;
	}

	return NULL;
}

/*!
 * scan non-primitve data types: i.e. typedef struct or struct
 *
 */
void OpenCLLoopScan::scanNonPrimitiveType(OpenCLKernelLoop* loop, vector<DeclRefExpr*>& decls)
{
	for (unsigned i=0; i<decls.size(); i++)
	{
		DeclRefExpr* d = decls[i];
		const QualType type = d->getType().getCanonicalType();
		string ty = getGlobalType(type.getAsString());

		//skip functions
		if (d->getDecl()->getKind() == Decl::Function)
			continue;

		if (!isOCLPremitiveType(ty))
		{
			bool found=false;
			for (unsigned j=0; j<qtypes.size(); j++)
			{
				if (getGlobalType(qtypes[j].getAsString()) == ty)
				{
					found = true;
					break;
				}
			}

			if (!found)
			{
				this->qtypes.push_back(type);
				RecordDecl* rd = OCLCommon::getRecordDecl(type);
				TypedefDecl* TD = getTypeDefRef(ty);

				if (rd)
				{
					OpenCLGlobalInfoContainer::addRecordDecl(rd);
				}

				if (TD)
				{
					OpenCLGlobalInfoContainer::addTypedefDecl(TD);
				}
			}	
		}	
	}	

}


static void addLoopParams(vector<DeclRefExpr*>& decls, OpenCLKernelLoop* loop)
{
	for (unsigned i=0; i<decls.size(); i++)
	{
		loop->addParam(decls[i]);
	}
}

//Scan Loop
void OpenCLLoopScan::scanLoop(OpenCLKernelLoop* loop)
{
//	scanThreadPrivateVars(loop);
	reviseCalledArgs(loop);
	collectGlobalInputArguments(loop);
	
	//dr visits the kernel to record the DeclRef information of variables
	StmtPicker dr (llvm::nulls(), Context, NULL, Context.PrintingPolicy, 0);
	dr.Visit(loop->getKernel());
	//FIXME: ALL THE DECLREF in getLHS() will be treated as write
	loop->getForStmt()->setRWS(dr.getRWS());
	loop->setRWSet(dr.getRWS());

	//Scan non-primitive types of structures
	scanNonPrimitiveType(loop, dr.getDecl());
	
	vector<OpenCLNDRangeVar>& gvs = loop->getOclLoopIndexs();
	for (unsigned i=0; i<gvs.size(); i++)
	{
		dr.Visit(gvs[i].Init);
	}

	vector<SwapLoopInfo>& innerLoops = loop->getInnerLoops();

	for (unsigned i=0; i<innerLoops.size(); i++)
	{
		dr.VisitForStmtHeader(innerLoops[i].for_stmt);
	}

	vector<DeclRefExpr*>& decls = dr.getDecl();
	addLoopParams(decls, loop);
}

vector<OpenCLKernelLoop*>& OpenCLLoopScan::getOclLoops()
{
	return oclLoops; 
}

void OpenCLLoopScan::addRevisedFunc(FunctionDecl* D)
{
	string name = D->getNameAsString();
	for (unsigned i=0; i<revisedFuncs.size(); i++)
	{
		FunctionDecl* d = revisedFuncs[i];
		if (name == d->getNameAsString())
		{
			return;
		}
	}

	revisedFuncs.push_back(D);
}

void OpenCLLoopScan::_do()
{
	for (unsigned i=0; i<oclLoops.size(); i++)
	{
		if (!oclLoops[i]->hasOptimised())
		{
			cerr << "Warning: no optimised pass for the loop" << endl;
		}

		scanLoop(oclLoops[i]);
		retriveOpenCLNDRangeVars(oclLoops[i]);
	}
}

void OpenCLLoopScan::retriveOpenCLNDRangeVar(OpenCLKernelLoop* loop, ForStmt* Node, unsigned int orig_index)
{
	if (!Node)
	{
		cerr << "NULL Node..." << endl;
	}

	LoopIndex *l = OCLCommon::getLoopIndex(Node);
	if (dyn_cast<DeclRefExpr>(l->variable))
	{
		DeclRefExpr *expr = (DeclRefExpr*) (l->variable);
		OpenCLNDRangeVar g(orig_index);
		g.variable = expr->getNameInfo().getAsString();
		g.type =getCononicalType(expr->getDecl()); 
		Stmt* init = Node->getInit();
		g.Cond = Node->getCond();
		g.Inc = Node->getInc();

		BinaryOperator *opInc = dyn_cast<BinaryOperator>(g.Inc);
		if (opInc)
		{
			Expr* rhs = opInc->getRHS();
			if (rhs)
			{
				if (dyn_cast<IntegerLiteral>(rhs))
				{
					g.isIncInt = true;
				}
				g.increment = getStringExpr(Context, rhs);
				g.hasIncremental = true;	
			}
			else
			{
				g.hasIncremental = false;	
			}
		}

		BinaryOperator* op = dyn_cast<BinaryOperator>(g.Cond);
		assert(op && "The loop bound is not binary operator");
	
		string opcode = BinaryOperator::getOpcodeStr(op->getOpcode());
		if ((opcode == "<=") || (opcode == "<"))
		{
			if (dyn_cast<BinaryOperator>(init))
			{
				BinaryOperator *bo = dyn_cast<BinaryOperator>(init);	
				g.Init = bo->getRHS();
			}
			else
			{
				cerr << "The loop init is not a BinaryOperator class" << endl;
				exit(-1);
			}

			if (g.Cond)
			{
				BinaryOperator* oc = dyn_cast<BinaryOperator>(g.Cond);
				if(oc)
				{
					//This is a constant integer value
					//I record it here so that I don't have to declare it as 
					//a global memory object
					if (dyn_cast<IntegerLiteral>(oc->getRHS()))
					{
						g.cond_string = getStringStmt(Context, g.Cond); 
						g.isCondInt = true;
					}

					g.cond_opcode_str = BinaryOperator::getOpcodeStr(oc->getOpcode());
				}
			}
		}
		else
		//e.g. for(i=8; i>=0; i--)
		if ((opcode == ">=") || (opcode == ">"))
		{
			Stmt* Cond = Node->getCond();	
			if (dyn_cast<BinaryOperator>(Cond))
			{
				BinaryOperator *bo = dyn_cast<BinaryOperator>(Cond);	
				g.Init = bo->getRHS();
				g.cond_string = getStringStmt(Context, bo->getLHS()) + "<=";
				g.cond_opcode_str = BinaryOperator::getOpcodeStr(bo->getOpcode());
			}

			Stmt* Init = Node->getInit();
			g.Cond = Init;
			
			if (Init)
			{
				BinaryOperator* oc = dyn_cast<BinaryOperator>(Init);
				if(oc)
				{
					if (dyn_cast<IntegerLiteral>(oc->getRHS()))
					{
						g.cond_string = g.cond_string + getStringStmt(Context, oc->getRHS());
						g.isCondInt = true;
					}

				}
				
			}
			
		}

		loop->addOpenCLNDRangeVar(g);
	}
	else
	{
		cerr << "The loop index is not a DeclRefExpr" << endl;
		exit(-1);
	}
}


//Collect the OCL NDRANGE info
void OpenCLLoopScan::retriveOpenCLNDRangeVars(OpenCLKernelLoop* loop)
{
	vector<SwapLoopInfo>& SLs = loop->getSwapLoops();
	for (unsigned int i=0; i<SLs.size(); i++)
	{
		retriveOpenCLNDRangeVar(loop, SLs[i].getForStmt(), i);	
	}
}

void OpenCLLoopScan::doIt()
{
	_do();
}
