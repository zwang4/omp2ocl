#include "clang/Omp2Ocl/OpenCLLoadSchedule.h"
#include "llvm/Support/Format.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/StmtPicker.h"
#include <algorithm>
#include <deque>
#include <math.h>

using namespace std;

static int getIntegerValue(const IntegerLiteral* IL)
{
	llvm::APInt I = IL->getValue();
	bool isSigned = IL->getType()->isSignedIntegerType();

	string ILS = IL->getValue().toString(10, isSigned);

	return atoi(ILS.c_str());
}

//is lhs larger than rhs
static bool IntegerCompare(const Expr* lhs, const Expr* rhs)
{
	const IntegerLiteral* LIL = dyn_cast<IntegerLiteral>(lhs);
	const IntegerLiteral* RIL = dyn_cast<IntegerLiteral>(rhs);

	assert(LIL && RIL && "Invalid IntegerLiteral");

	int li = getIntegerValue(LIL);
	int ri = getIntegerValue(RIL);

	return (li > ri) ? true : false;	
}

//Check if the expr is an integer expr
//if it is, check whether its value is zero
static bool isZeroInt(const Expr* E)
{
	const IntegerLiteral* EI = dyn_cast<IntegerLiteral>(E);

	if (EI)
	{
		return (getIntegerValue(EI) == 0);
	}

	return false;
}


//Check wheter the exper is an increasing operation
static bool isIncrease(int v, string ops)
{
	if (ops == "+")
	{
		if (v > 0)
		{
			return true;
		}
		else //add a negative integer
		{
			return false;
		}
	}			
	else
		if (ops == "-")
		{
			if (v > 0)
			{
				return false;
			}
			else //sub a negative integer, then it is considered as an
				//inscreasing operation
			{
				return true;
			}
		}
		else
			if (ops == "/")
			{
				if (v != 1)
				{
					return true;
				}
				else
				{
					return false;
				}
			}
			else
				if (ops == "*")
				{
					if (v > 0)
					{
						return true;
					}
					else
					{
						return false;
					}
				}

	return false;
}

//Can I convert an binaryoperator to a single integer value?
static bool collapse2Int(BinaryOperator* bop, int& value)
{
	Expr* lhs = bop->getLHS();
	Expr* rhs = bop->getRHS();
	int iLIL;
	int iRIL;

	assert(lhs && rhs && "Invalid rhs or lhs");

	IntegerLiteral* LIL = dyn_cast<IntegerLiteral>(lhs);
	IntegerLiteral* RIL = dyn_cast<IntegerLiteral>(rhs);
	string ops = BinaryOperator::getOpcodeStr(bop->getOpcode());

	//Both are integer values
	//Convert them to a single value
	if (LIL && RIL)
	{
		iLIL = getIntegerValue(LIL);
		iRIL = getIntegerValue(RIL);

		if (ops == "+")
		{
			value = iLIL + iRIL;
		}
		else
			if (ops == "-")
			{
				value = iLIL - iRIL;
			}
			else
				if (ops == "/")
				{
					value = iLIL / iRIL;
				}
				else
					if (ops == "*")
					{
						value = iLIL * iRIL;
					}

		return true;
	}

	return false;
}


static bool isIncrease(BinaryOperator* bop, int& value, bool& pureInt)
{
	pureInt = false;
	Expr* lhs = bop->getLHS();
	Expr* rhs = bop->getRHS();
	int iLIL;
	int iRIL;

	assert(lhs && rhs && "Invalid rhs or lhs");

	IntegerLiteral* LIL = dyn_cast<IntegerLiteral>(lhs);
	IntegerLiteral* RIL = dyn_cast<IntegerLiteral>(rhs);
	string ops = BinaryOperator::getOpcodeStr(bop->getOpcode());

	//Both are integer values
	//Convert them to a single value
	if (collapse2Int(bop, value))
	{
		pureInt = true;
	}
	else
		if (LIL || RIL)
		{
			if (LIL)
			{
				iLIL = getIntegerValue(LIL);

				if (iLIL == 0 && ops == "*")
				{
					pureInt = true;
					value = 0;

					return false;
				}

				return isIncrease(iLIL, ops);
			}
			else
				if (RIL)
				{
					iRIL = getIntegerValue(RIL);

					if (iRIL == 0 && ops == "*")
					{
						pureInt = true;
						value = 0;

						return false;
					}

					return isIncrease(iRIL, ops);
				}	
		}

	//Both symbolic, I cannot compare them
	return false;
}

//convert a string to int
static bool getIntFromStr(string str, int& value)
{
	for (unsigned i=0; i<str.length(); i++)
	{
		if (str[i] >= '0' && str[i] <= '9')
		{
			continue;
		}
		else
		{
			return false;
		}
	}

	value = atoi(str.c_str());

	return true;
}

//Compare two array accessing sequences that have symbolics
//return true if lExpr < rExpr
static bool symbolicCompare(ASTContext* pCtx, const IndexStr& l, const IndexStr& r)
{
	const string ls = l.getAsString();
	const string rs = r.getAsString();

	StmtPicker lop(llvm::nulls(), *pCtx, NULL, pCtx->PrintingPolicy, -4);
	lop.PrintExpr(const_cast<Expr*>(l.getExpr()));
	vector<BinaryOperator*> lOps = lop.getBinOps(); 	

	StmtPicker rop(llvm::nulls(), *pCtx, NULL, pCtx->PrintingPolicy, -4);
	rop.PrintExpr(const_cast<Expr*>(r.getExpr()));
	vector<BinaryOperator*> rOps = rop.getBinOps(); 	

	//FIXME: Currently, I can only handle simple cases where
	//there is only one binary operation (e.g. k + 1, not (k + 1) + m).
	if (lOps.size() > 1)
	{
		return true;
	}
	else	
		if (rOps.size() > 1)
		{
			return true;
		}

	//Left: x, right: x -/+///* 1
	if (lOps.size() == 0)
	{
		if (rOps.size())
		{
			BinaryOperator* bop = rOps[0];
			bool pureInt = false;
			int value = 0;

			//Both rExps is integer
			if (collapse2Int(bop, value))
			{
				int lv = 0;
				//Check if left can be converted to an integer or not
				bool b = getIntFromStr(ls, lv);
				if (b) //yes, it is converted to an integer
				{
					return (lv < value) ? true : false;
				}
			}
			else if (isIncrease(bop, value, pureInt) ) //rhs is increasing
			{
				return true;
			}
		}
	
		// default:
		return false;
	}
	else //lOps.size() == 1, e.g. left: x + 1
	{
		//Check whether LHS can be converted to an integer
		BinaryOperator* bop = lOps[0];
		bool pureInt = false;
		int value = 0;

		if (rOps.size() == 0)
		{
			if (collapse2Int(bop, value))
			{
				int rv = 0;
				//check whether rhs canbe coverted to an integer
				bool b = getIntFromStr(rs, rv);
				if (b)
				{
					return (value < rv) ? true : false;
				}
			}
			
			if (isIncrease(bop, value, pureInt))
			{
				return false;
			}
			
			//FIXED ME: this is a problem
			//e.g. left=n-1, right =n
			string left_ops = BinaryOperator::getOpcodeStr(bop->getOpcode());
			if (!isIncrease(bop, value, pureInt) && left_ops == "-")
			{
				return true;
			}

		}
		else //rOps.size() == 1, e.g. right x + n
		{
			BinaryOperator* rbop = rOps[0];
			bool rpureInt = false;
			int rvalue = 0;

			//lhs is an integer
			if (collapse2Int(bop, value))
			{
				//I can convert the rExpr to an integer as well
				if (collapse2Int(rbop, rvalue))
				{
					return (value < rvalue) ? true : false;
				}
			}

			if (isIncrease(rbop, rvalue, rpureInt) && !isIncrease(bop, value, pureInt))
			{
				return true;
			}
			
		} //else
	}

	return false;
}

//Check is l larger than r
//l and r are unsigned integers
//This is used to sort the access to a buffer
static bool isLarger(ASTContext* pCtx, const IndexStr& l, const IndexStr& r)
{
	const Expr* lExpr = l.getExpr();
	const Expr* rExpr = r.getExpr();
	const string ls = l.getAsString();
	const string rs = r.getAsString();

	if (dyn_cast<IntegerLiteral>(lExpr) && dyn_cast<IntegerLiteral>(rExpr))
	{
		return IntegerCompare(lExpr, rExpr);
	}
	else if (isZeroInt(lExpr)) //left: 0
	{
		return false;	
	}
	else if (isZeroInt(rExpr)) //right 0
	{
		return true;
	}	
	else if (ls == rs) //identical strings
	{
		return false;
	}

	return !symbolicCompare(pCtx, l, r);
}

//This is used for sorting array accessing sequences
static bool compare_array_str(ASTContext* pCtx, const list<IndexStr>& left, const list<IndexStr>& right)
{
	if ((left.size() == right.size())  && left.size() )
	{
		for (list<IndexStr>::const_iterator l_iter=left.begin(), r_iter=right.begin(); l_iter != left.end(), r_iter != right.end(); l_iter++, r_iter++)
		{
			if (isLarger(pCtx, (*l_iter), (*r_iter)))
			{
				return false;
			}
		}
	}

	return true;
}

//Sorting two array access
//Currently, if an array access is indirect access
//e.g. a[b[i]][j], then I cannot optimise it
bool ArrayIndex::operator< (const ArrayIndex& rhs) const
{
	const list<IndexStr>& this_acc_str = this->getAccessIdx();
	const list<IndexStr>& rhs_acc_str = rhs.getAccessIdx();

	return compare_array_str(this->getContext(), this_acc_str, rhs_acc_str);
}

//Get the read/write status of variables
void OpenCLLoadSchedule::scanCandidates(Stmt* Kernel)
{
	sp.Visit(Kernel);
	rwS = sp.getRWS();
}

//Is this variable read only
bool OpenCLLoadSchedule::isReadOnly(DeclRefExpr* e)
{
	for (unsigned i=0; i<rwS.size(); i++)
	{
		if (e->getDecl() == rwS[i].getDecl())
		{
			return (!rwS[i].isWriteVar());
		}
	}

	return false;
}

//Check whether the array access sequence has a function call
bool OpenCLLoadSchedule::hasFunctionCall(ArrayIndex& A)
{
	const list<IndexStr>& acc_str = A.getAccessIdx();
	for (list<IndexStr>::const_iterator iter=acc_str.begin(); iter != acc_str.end(); iter++)
	{
		const Expr* expr = iter->getExpr();
		if (expr)
		{
			StmtPicker op(llvm::nulls(), Context, NULL, Context.PrintingPolicy, -4);
			op.PrintExpr(const_cast<Expr*>(expr));
			if (op.getCallExprs().size())
			{
				return true;
			}
		}	
	}

	return false;
}

//Is two access sequence identical?
bool OpenCLLoadSchedule::isTwoAccessStrIdentical(ArrayIndex& l, ArrayIndex& r)
{
	const list<IndexStr>& l_acc_str = l.getAccessIdx();
	const list<IndexStr>& r_acc_str = r.getAccessIdx();

	if (l_acc_str.size() != r_acc_str.size())
	{
		return false;
	}

	for (list<IndexStr>::const_iterator l_iter=l_acc_str.begin(), r_iter=r_acc_str.begin(); l_iter != l_acc_str.end(); l_iter++, r_iter++)
	{
		if (l_iter->getAsString() != r_iter->getAsString())
		{
			return false;
		}
	}

	return true;
}

//Have I already recorded this access sequence?
int OpenCLLoadSchedule::isInArrayIndexs(vector<ArrayIndex>& PAIs, ArrayIndex& A)
{
	for (unsigned i=0; i<PAIs.size(); i++)
	{
		if (isTwoAccessStrIdentical(PAIs[i], A))
		{
			return i;
		}
	}

	return -1;
}

//Sorting array access patterns
void OpenCLLoadSchedule::sort(vector<ArrayIndex>& AIs)
{
	vector<ArrayIndex> PAIs;
	for (unsigned i=0; i<AIs.size(); i++)
	{
		if (!AIs[i].hasIndirectAcc() && !hasFunctionCall(AIs[i]))
		{
			int id = isInArrayIndexs(PAIs, AIs[i]);
			if (id < 0)
			{
				PAIs.push_back(AIs[i]);
				id = PAIs.size() - 1;
			}
	
			PAIs[id].incOccurance();	
			PAIs[id].addASE(AIs[i].getASENode());
		}
	}

	if (PAIs.size())
	{
		std::sort(PAIs.begin(), PAIs.end());
	#if 0
		cerr << "Sorting results:" << endl;
		for (unsigned i=0; i<PAIs.size(); i++)
		{
			printArrayInfo(PAIs[i]);	
		}
		cerr << endl;
	#endif
	}

	//Restore AIs
	AIs.clear();
	for (unsigned i=0; i<PAIs.size(); i++)
	{
		AIs.push_back(PAIs[i]);
	}
}

//Schedule loads within  arrayAccessInfo
void OpenCLLoadSchedule::scheduleLoads()
{
	if (arrayAccessInfo.size())
	{
		for (unsigned i=0; i<arrayAccessInfo.size(); i++)
		{
			sort(arrayAccessInfo[i].getAIs());	
		}
	}
}

void OpenCLLoadSchedule::doIt()
{
	Stmt* Kernel = getLoop()->getKernel();
	scanCandidates(Kernel);

	Visit(Kernel);
}

void OpenCLLoadSchedule::PrintStmt(Stmt *S) {
	PrintStmt(S, Policy.Indentation);
}

void OpenCLLoadSchedule::PrintStmt(Stmt *S, int SubIndent) {
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

void OpenCLLoadSchedule::PrintExpr(Expr *E) {
	if (E)
		Visit(E);
	else
		OS << "<null expr>";
}

void OpenCLLoadSchedule::VisitForStmt(ForStmt *Node) {
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

//Print array info
void OpenCLLoadSchedule::printArrayInfo(ArrayIndex& ai)
{
	//cerr << getStringExpr(Context, ai.Node) << ":";
	list<IndexStr> as = ai.getAccessIdx();
	cerr << ai.getName();

	for (list<IndexStr>::iterator iter=as.begin(); iter !=  as.end(); iter++)
	{
		cerr << "[" << iter->getAsString() << "]";
	}
}

//Add a array access statement to arrayAccessInfo
//If an entry with the same array name has already exists, add call addArrayIndex()
//Otherwise, create an entry and then call addArrayIndex()
void OpenCLLoadSchedule::addSortByNameArrayAccess(ArrayIndex& ai)
{
	string name = ai.getName();

	for (unsigned i=0; i<arrayAccessInfo.size(); i++)
	{
		if (arrayAccessInfo[i].getName() == name)
		{
			arrayAccessInfo[i].addArrayIndex(ai);
			return;
		}	
	}

	arrayAccessInfo.push_back(ArrayAccessInfo(ai.base->getDecl(), ai)); 
}

//Sort the access to a global buffer (that is a candidate for load optimisation) by name
//The sorting result is stored in arrayAccessInfo
void OpenCLLoadSchedule::sortArrayAccessByName()
{
	vector<ArrayIndex>& A = arrayInfos.top();
	for (unsigned i=0; i<A.size(); i++)
	{
		addSortByNameArrayAccess(A[i]);
	}
}

//Get an interget op from a BinaryOperator (e.g. k + 1)
//return false if both ops are symbolics
bool OpenCLLoadSchedule::getIntOp(BinaryOperator* bop, int& value)
{
	const IntegerLiteral* LIL = dyn_cast<IntegerLiteral>(bop->getLHS());
	const IntegerLiteral* RIL = dyn_cast<IntegerLiteral>(bop->getRHS());
	string ops = BinaryOperator::getOpcodeStr(bop->getOpcode());

	assert (!(LIL && RIL) && "Shouldn't fall to this path");

	if (LIL || RIL)
	{
		if (LIL)
		{
			value = getIntegerValue(LIL);
		}
		
		if (RIL)
		{
			value = getIntegerValue(RIL);
		}

		return true;
	}
	
	//both op of rhs are symbols (e.g k + m)
	//Return false because I cannot make a decision
	return false;
}

static const int ACC_EQUAL = 0;
static const int ACC_CONT  = 1;
static const int ACC_DCONT = 2;
static const int ACC_UNKNOWN = 3;


//if the right is a 1 step increase of the left
bool OpenCLLoadSchedule::isOneStepInc(vector<BinaryOperator*> lOps, vector<BinaryOperator*> rOps)
{
	int lv = -1;
	bool lbv = false;
	int rv = -1;
	bool rbv = false;

	if (lOps.size())
	{
		lbv = getIntOp(lOps[0], lv);	
	}
	
	if (rOps.size())
	{
		rbv = getIntOp(rOps[0], rv);
	}

	if (lbv)
	{
		string ops = BinaryOperator::getOpcodeStr(lOps[0]->getOpcode());

		if (ops == "-" || ops == "+")
		{
			if (lv > 0)
			{
				if (ops == "+")
				{
					//rOps is either k + m  or k, 
					//therefore, I should return false
					if (rbv && rOps.size())
					{
						string rops = BinaryOperator::getOpcodeStr(rOps[0]->getOpcode());

						if (rops == "+" && ((rv - lv) == 1))
						{
							return true;
						}
					}
				}

				if (ops == "-")
				{
					if (rOps.size())
					{
						string rops = BinaryOperator::getOpcodeStr(rOps[0]->getOpcode());
					
						//e.g. left: k-2, right: k-1
						//here: lv = 2, rv = 1	
						if (rbv && rops == "-")
						{
							if ( (rv - lv) == -1)
							{
								return true;
							}
						}
					}

					//e.g. left: k - 1, right k
					if (!rOps.size() && ((0 - lv) == -1) )
					{
							return true;
					}	
				}
			}
		
			//e.g. left: k	
			if (lv == 0)
			{
				if (rOps.size())	
				{
					string rops = BinaryOperator::getOpcodeStr(rOps[0]->getOpcode());
					
					//check if right is k + 1
					if (rops == "+" && (rv - 0) == 1)
					{
						return true;
					}	
				}
			}

			return false;
		}
		else
		{
			//I can only handle '-' and '+'
			return false;
		}

		return false;
	}//if lbv

	//If this branch is felt, 
	//It means !lbv -- left could be either k + m or k
	if (rbv)
	{
		//e.g. left is k
		if (!lOps.size())
		{
			if (rOps.size())
			{
				string rops = BinaryOperator::getOpcodeStr(rOps[0]->getOpcode());
				//check if right is k + 1
				if ( rops == "+" && ((rv - 0) == 1) )
				{
					return true;
				}
			}	
		}
	}

	return false;
}

//Does rhs is one step ahead of lhs
//e.g. if lhs is k, then rhs should be k + 1 
int OpenCLLoadSchedule::cont(const IndexStr& lhs, const IndexStr& rhs, 
		int idx, int& borrow)
{
	const IntegerLiteral* LIL = dyn_cast<IntegerLiteral>(lhs.getExpr());
	const IntegerLiteral* RIL = dyn_cast<IntegerLiteral>(rhs.getExpr());

	if (lhs.getAsString() == rhs.getAsString())
	{
		return ACC_EQUAL;
	}

	if (LIL && RIL)
	{
		int iLIL = getIntegerValue(LIL);	
		int iRIL = getIntegerValue(RIL);	

		if (((iLIL + 1) % idx) == iRIL)
		{
			if (borrow == 0)
			{
				//e.g. for definition int A[4][60]
				//when left A[2][59] and A[3][0] is compared
				//and iLIL=59, iRIL=0, I will treat them as CONT;
				if (iLIL > iRIL)
				{
					borrow = -1;
				}
				
				return ACC_CONT;
			}
			else
			{
				//e.g. left A[2][59], right A[3][0]
				//iLIL will be 2, and iRIL will 3
				//This will be treated as EQUAL because the last index of right borrow
				//-1 from 3
				//reset borrow
				borrow = 0;
				return ACC_EQUAL;
			}
		}
		
		if (iLIL == iRIL)
		{
			return ACC_EQUAL;	
		}
	
		return ACC_DCONT;
	}
	else
	{
		StmtPicker lop(llvm::nulls(), Context, NULL, Context.PrintingPolicy, -4);
		lop.PrintExpr(const_cast<Expr*>(lhs.getExpr()));
		vector<BinaryOperator*> lOps = lop.getBinOps(); 	

		StmtPicker rop(llvm::nulls(), Context, NULL, Context.PrintingPolicy, -4);
		rop.PrintExpr(const_cast<Expr*>(rhs.getExpr()));
		vector<BinaryOperator*> rOps = rop.getBinOps(); 	

		int iLIL = -1;
		bool lbi = false;
		int iRIL = -1;
		bool rbi = false;

		if (lOps.size())
		{
			lbi = collapse2Int(lOps[0], iLIL);
		}

		if (rOps.size())
		{
			rbi = collapse2Int(rOps[0], iRIL);
		}

		//Both can be collapsed to an integer value
		if (lbi && rbi)
		{
			int iLIL = getIntegerValue(LIL);	
			int iRIL = getIntegerValue(RIL);	

			if (((iLIL + 1) % idx) == iRIL)
			{
				if (borrow == 0)
				{
					if (iLIL > iRIL)
					{
						borrow = -1;
					}

					return ACC_CONT;
				}
				else
				{
					//e.g. left A[2][59], right A[3][0]
					//iLIL will be 2, and iRIL will 3
					//This will be treated as EQUAL because the last index of right borrow
					//-1 from 3
					//reset borrow
					borrow = 0;
					return ACC_EQUAL;
				}
			}

			if (iLIL == iRIL)
			{
				return ACC_EQUAL;	
			}

			return ACC_DCONT;
		}

		//one is a symbol and the other is an integer
		//therefore, I am not sure!
		if (lbi || rbi)
		{
			return ACC_UNKNOWN;
		}

		if (isOneStepInc(lOps, rOps))
		{
			return ACC_CONT;
		}
	} //if(! (LIL && RIL)

	return ACC_UNKNOWN;
}

//is first and second are two adjanct sequences?
bool OpenCLLoadSchedule::isCont(ArrayIndex& first, ArrayIndex& second)
{
	const list<IndexStr>& first_acc = first.getAccessIdx();
	const list<IndexStr>& second_acc = second.getAccessIdx();

	string firstType = getCononicalType(first.getDecl());
	vector<unsigned> first_ad = getArrayDef(firstType);

	if (first_acc.size() != second_acc.size() || first_acc.size() <= 0 || first_acc.size() != first_ad.size())
	{
		return false;
	}

	int i = first_acc.size() - 1;
	deque<IndexStr> vFirst;
	deque<IndexStr> vSecond;

	for (list<IndexStr>::const_iterator first_iter = first_acc.begin(), second_iter = second_acc.begin();
			first_iter != first_acc.end(); first_iter++, second_iter++)
	{
		vFirst.push_back(*first_iter);
		vSecond.push_back(*second_iter);
	}

	int borrow = 0;

	if (cont(vFirst[i], vSecond[i], first_ad[i], borrow) != ACC_CONT)
	{
		return false;
	}

	//Borrow is not allowed!
	if (borrow != 0 && first_ad.size() <= 1)
	{
		return false;
	}

	i--;
	while (i >= 0)
	{
		int b = cont(vFirst[i], vSecond[i], first_ad[i], borrow);

		if (b != ACC_EQUAL)
		{
	//		cerr << vFirst[i].getAsString() << ":" << vSecond[i].getAsString() << ": " << i << " [3. FAILED]" << endl;
			return false;
		}

		i--;
	}

	return true;
}

//Find the maximum continue sequence
int OpenCLLoadSchedule::conSequence(vector<ArrayIndex>& AIs, unsigned i)
{
	unsigned c = 1;

	for (; i < AIs.size() - 1; i++)
	{
		if (isCont(AIs[i], AIs[i+1]))
		{
			c++;
		}
		else
		{
			break;
		}
	}

	return c;
}

unsigned int roundPower2(unsigned i)
{
	int m = 0;

	while (i / 2)
	{
		m++;
		i = i / 2;
	}

	return pow(2, m);
}


//Genload instructions
OCLCompoundVLoadDeclareInfo OpenCLLoadSchedule::genVLoadInsts(ArrayIndex& A, unsigned begin, unsigned vs, vector<ArrayIndex>& AIs, bool& useAble)
{
	//string name = OCLCommon::getVLoadVariableName(A.getName());
	string name = A.getName();
	ValueDecl* d = A.getDecl();
	string t = getGlobalType(getCononicalType(d));
	vector<ArrayIndex> As;
	useAble = false;

	if (isOCLPremitiveType(t))
	{
//		cerr << "VLOAD" << vs << " : ";
		for (unsigned i=begin, j=0; j<vs; j++, i++)
		{
			As.push_back(AIs[i]);
//			cerr << getStringExpr(Context, AIs[i].getASENode()) << " ";
		}
//		cerr << endl;

		useAble = true;
	}

	return OCLCompoundVLoadDeclareInfo(name, vs, As);
}

void OpenCLLoadSchedule::addVLoadInfo(OCLCompoundVLoadDeclareInfo v)
{
	string dname = v.getDeclareName();
	unsigned w = v.getVWidth(); 

	for (unsigned i=0; i<CVIs.size(); i++)
	{
		if (CVIs[i].getDeclareName() == dname &&
				CVIs[i].getVWidth() == w)
		{
			vector< vector<ArrayIndex> >& AIs = v.getAIs();
			//Only push vector<ArrayIndex>
			CVIs[i].addAI(AIs[0]);
			return;
		}
	}

	CVIs.push_back(v);
}

//Perform vector loads
void OpenCLLoadSchedule::performVLoad(vector<ArrayIndex>& AIs, unsigned iBegin, unsigned iEnd)
{
	unsigned begin = iBegin;
	unsigned item = iEnd - iBegin;
	bool useAble;

	if (item >= DEFAULT_LOAD_VECTOR)
	{
		for (unsigned i=iBegin; i<iEnd; i+=DEFAULT_LOAD_VECTOR)
		{
			OCLCompoundVLoadDeclareInfo v = genVLoadInsts(AIs[i], i, DEFAULT_LOAD_VECTOR, AIs, useAble);
	
			if (useAble)
			{
				addVLoadInfo(v);
			}

			item -= DEFAULT_LOAD_VECTOR;
			if (item < DEFAULT_LOAD_VECTOR)
			{
				break;
			}
			
			begin += DEFAULT_LOAD_VECTOR + 1;
		}
	}

	//VLoad2
	while(item >= 2)
	{
		unsigned p = roundPower2(item);

		OCLCompoundVLoadDeclareInfo v = genVLoadInsts(AIs[begin], begin, p, AIs, useAble);
		if (useAble)
		{
			addVLoadInfo(v);
		}

		begin += (p + 1);
		item -= p;
	}

	//Sequential
	if ( (iEnd - iBegin) % 2)
	{

		OCLCompoundVLoadDeclareInfo v = genVLoadInsts(AIs[iEnd-1], iEnd-1, 1, AIs, useAble);
		if (useAble && AIs[iEnd-1].getOccurance() >= BUFFER_REUSED_RATIO)
		{
			addVLoadInfo(v);
		}
	}//if
}

//
void OpenCLLoadSchedule::_vectorLoad(vector<ArrayIndex>& AIs)
{
	int i = 0;
	unsigned c = 1;
	int num = AIs.size() - 1;
	
	while(i < num)
	{
		c = conSequence(AIs, i);
		
		if (c > 1)
		{
		#if 0
			SourceLocation Loc = AIs[i].getASENode()->getRBracketLoc();
			unsigned ln = OCLCommon::getLineNumber(Context, Loc);
			cerr << "*********************Catch one [" << c << "] : " << i << " : " << AIs.size() << " : LINENUMBER: " << ln << " NEXT: " << (i + c) << " ";
			for (unsigned j=i; j<i+c; j++)
			{
				printArrayInfo(AIs[j]);
				cerr << " ";
			}

			//if (i + c < AIs.size())
			//{
			//	cerr << "NEXT: " << i + c << ", ";
			//	printArrayInfo(AIs[i+c]);
			//}

			cerr << endl;
		#endif
			int end = i + c;
			if (end % 2)
			{
				end--;
			}

			performVLoad(AIs, i, end);
	
			i += c;
		}
		else
		{
			if (i < num)
			{
				bool useAble = false;
				//Sequential Loads
				//I will still outline them
				//This is only safe when VLoad is performed on read-only buffers
				OCLCompoundVLoadDeclareInfo v = genVLoadInsts(AIs[i], i, 1, AIs, useAble);
				if (useAble && AIs[i].getOccurance() >= BUFFER_REUSED_RATIO)
				{
					addVLoadInfo(v);
				}
			}
			i++;
		} // else
	} //while
}

//Vectorise Loads
void OpenCLLoadSchedule::VectoriseLoads(CompoundStmt *Node)
{
	sortArrayAccessByName();
	scheduleLoads();
	
	for (unsigned i=0; i<arrayAccessInfo.size(); i++)
	{
		_vectorLoad(arrayAccessInfo[i].getAIs());
	}
	
	//Record the Vectorisation info
	if (CVIs.size())
	{
		//SourceLocation Loc = Node->getLBracLoc();
		Node->addVLoadInfo(CVIs);
	}
}

/// PrintRawCompoundStmt - Print a compound stmt without indenting the {, and
/// with no newline after the }.
void OpenCLLoadSchedule::PrintRawCompoundStmt(CompoundStmt *Node) {
	OS << "{\n";

	vector<ArrayIndex> A;
	arrayInfos.push(A);

	CompoundStmt::body_iterator I, E;
		
	for (I = Node->body_begin(), E = Node->body_end();
			I != E; ++I)
	{
		PrintStmt(*I);
	}
	
	I = Node->body_begin();
	if (I != E)
	{
		if (!dyn_cast<ForStmt>(*I) && !dyn_cast<CompoundStmt>(*I))
		{
			VectoriseLoads(Node);
		}
	}
	
	arrayInfos.pop();
	CVIs.clear();
	arrayAccessInfo.clear();

	Indent() << "}";
}

void OpenCLLoadSchedule::PrintRawDecl(Decl *D) {
	D->print(OS, Policy, IndentLevel);
}

void OpenCLLoadSchedule::PrintRawDeclStmt(DeclStmt *S) {
	DeclStmt::decl_iterator Begin = S->decl_begin(), End = S->decl_end();
	llvm::SmallVector<Decl*, 2> Decls;
	for ( ; Begin != End; ++Begin)
		Decls.push_back(*Begin);

	Decl::printGroup(Decls.data(), Decls.size(), OS, Policy, IndentLevel);
}

void OpenCLLoadSchedule::VisitNullStmt(NullStmt *Node) {
	Indent() << ";\n";
}

void OpenCLLoadSchedule::VisitDeclStmt(DeclStmt *Node) {
	Indent();
	PrintRawDeclStmt(Node);
	OS << ";\n";
}

void OpenCLLoadSchedule::VisitCompoundStmt(CompoundStmt *Node) {
	Indent();
	PrintRawCompoundStmt(Node);
	OS << "\n";
}

void OpenCLLoadSchedule::VisitCaseStmt(CaseStmt *Node) {
	Indent(-1) << "case ";
	PrintExpr(Node->getLHS());
	if (Node->getRHS()) {
		OS << " ... ";
		PrintExpr(Node->getRHS());
	}
	OS << ":\n";

	PrintStmt(Node->getSubStmt(), 0);
}

void OpenCLLoadSchedule::VisitDefaultStmt(DefaultStmt *Node) {
	Indent(-1) << "default:\n";
	PrintStmt(Node->getSubStmt(), 0);
}

void OpenCLLoadSchedule::VisitLabelStmt(LabelStmt *Node) {
	Indent(-1) << Node->getName() << ":\n";
	PrintStmt(Node->getSubStmt(), 0);
}

void OpenCLLoadSchedule::PrintRawIfStmt(IfStmt *If) {
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

void OpenCLLoadSchedule::VisitIfStmt(IfStmt *If) {
	Indent();
	PrintRawIfStmt(If);
}

void OpenCLLoadSchedule::VisitSwitchStmt(SwitchStmt *Node) {
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

void OpenCLLoadSchedule::VisitWhileStmt(WhileStmt *Node) {
	Indent() << "while (";
	PrintExpr(Node->getCond());
	OS << ")\n";
	PrintStmt(Node->getBody());
}

void OpenCLLoadSchedule::VisitDoStmt(DoStmt *Node) {
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




void OpenCLLoadSchedule::VisitObjCForCollectionStmt(ObjCForCollectionStmt *Node) {
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

void OpenCLLoadSchedule::VisitGotoStmt(GotoStmt *Node) {
	Indent() << "goto " << Node->getLabel()->getName() << ";\n";
}

void OpenCLLoadSchedule::VisitIndirectGotoStmt(IndirectGotoStmt *Node) {
	Indent() << "goto *";
	PrintExpr(Node->getTarget());
	OS << ";\n";
}

void OpenCLLoadSchedule::VisitContinueStmt(ContinueStmt *Node) {
	Indent() << "continue;\n";
}

void OpenCLLoadSchedule::VisitBreakStmt(BreakStmt *Node) {
	Indent() << "break;\n";
}


void OpenCLLoadSchedule::VisitReturnStmt(ReturnStmt *Node) {
	Indent() << "return";
	if (Node->getRetValue()) {
		OS << " ";
		PrintExpr(Node->getRetValue());
	}
	OS << ";\n";
}


void OpenCLLoadSchedule::VisitAsmStmt(AsmStmt *Node) {
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

void OpenCLLoadSchedule::VisitObjCAtTryStmt(ObjCAtTryStmt *Node) {
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

void OpenCLLoadSchedule::VisitObjCAtFinallyStmt(ObjCAtFinallyStmt *Node) {
}

void OpenCLLoadSchedule::VisitObjCAtCatchStmt (ObjCAtCatchStmt *Node) {
	Indent() << "@catch (...) { /* todo */ } \n";
}

void OpenCLLoadSchedule::VisitObjCAtThrowStmt(ObjCAtThrowStmt *Node) {
	Indent() << "@throw";
	if (Node->getThrowExpr()) {
		OS << " ";
		PrintExpr(Node->getThrowExpr());
	}
	OS << ";\n";
}

void OpenCLLoadSchedule::VisitObjCAtSynchronizedStmt(ObjCAtSynchronizedStmt *Node) {
	Indent() << "@synchronized (";
	PrintExpr(Node->getSynchExpr());
	OS << ")";
	PrintRawCompoundStmt(Node->getSynchBody());
	OS << "\n";
}

void OpenCLLoadSchedule::PrintRawCXXCatchStmt(CXXCatchStmt *Node) {
	OS << "catch (";
	if (Decl *ExDecl = Node->getExceptionDecl())
		PrintRawDecl(ExDecl);
	else
		OS << "...";
	OS << ") ";
	PrintRawCompoundStmt(cast<CompoundStmt>(Node->getHandlerBlock()));
}

void OpenCLLoadSchedule::VisitCXXCatchStmt(CXXCatchStmt *Node) {
	Indent();
	PrintRawCXXCatchStmt(Node);
	OS << "\n";
}

void OpenCLLoadSchedule::VisitCXXTryStmt(CXXTryStmt *Node) {
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

void OpenCLLoadSchedule::VisitDeclRefExpr(DeclRefExpr *Node) {
	if (NestedNameSpecifier *Qualifier = Node->getQualifier())
		Qualifier->print(OS, Policy);
	OS << Node->getNameInfo();
	if (Node->hasExplicitTemplateArgs())
		OS << TemplateSpecializationType::PrintTemplateArgumentList(
				Node->getTemplateArgs(),
				Node->getNumTemplateArgs(),
				Policy);  
}

void OpenCLLoadSchedule::VisitDependentScopeDeclRefExpr(
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

void OpenCLLoadSchedule::VisitUnresolvedLookupExpr(UnresolvedLookupExpr *Node) {
	if (Node->getQualifier())
		Node->getQualifier()->print(OS, Policy);
	OS << Node->getNameInfo();
	if (Node->hasExplicitTemplateArgs())
		OS << TemplateSpecializationType::PrintTemplateArgumentList(
				Node->getTemplateArgs(),
				Node->getNumTemplateArgs(),
				Policy);
}

void OpenCLLoadSchedule::VisitObjCIvarRefExpr(ObjCIvarRefExpr *Node) {
	if (Node->getBase()) {
		PrintExpr(Node->getBase());
		OS << (Node->isArrow() ? "->" : ".");
	}
	OS << Node->getDecl();
}

void OpenCLLoadSchedule::VisitObjCPropertyRefExpr(ObjCPropertyRefExpr *Node) {
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

void OpenCLLoadSchedule::VisitPredefinedExpr(PredefinedExpr *Node) {
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

void OpenCLLoadSchedule::VisitCharacterLiteral(CharacterLiteral *Node) {
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

void OpenCLLoadSchedule::VisitIntegerLiteral(IntegerLiteral *Node) {
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
void OpenCLLoadSchedule::VisitFloatingLiteral(FloatingLiteral *Node) {
	// FIXME: print value more precisely.
	//OS << Node->getValueAsApproximateDouble();
	OS << Node->getLexString();

}

void OpenCLLoadSchedule::VisitImaginaryLiteral(ImaginaryLiteral *Node) {
	PrintExpr(Node->getSubExpr());
	OS << "i";
}

void OpenCLLoadSchedule::VisitStringLiteral(StringLiteral *Str) {
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
void OpenCLLoadSchedule::VisitParenExpr(ParenExpr *Node) {
	OS << "(";
	PrintExpr(Node->getSubExpr());
	OS << ")";
}
void OpenCLLoadSchedule::VisitUnaryOperator(UnaryOperator *Node) {
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

void OpenCLLoadSchedule::VisitOffsetOfExpr(OffsetOfExpr *Node) {
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

void OpenCLLoadSchedule::VisitUnaryExprOrTypeTraitExpr(UnaryExprOrTypeTraitExpr *Node){
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


//This is used to process an array
arrayBaseInfo OpenCLLoadSchedule::getArrayBaseInfo(ArraySubscriptExpr* Node)
{
	arrayBaseInfo info;
	Expr* base = Node->getBase();
	info.base_string = getStringStmt(Context, base);
	info.ai = getSubScripts(info.base_string);

	StmtPicker op(llvm::nulls(), Context, NULL, Context.PrintingPolicy, -4);
	op.Visit(base);
	vector<DeclRefExpr*>& ds = op.getDecl();

	if (ds.size())
	{
		info.t = ds[0];
	}

	info.isGlobalBuffer = OCLCommon::isAGlobalMemThreadPrivateVar(info.t->getDecl());	
	info.isTGBuffer = OCLCommon::isAThreadPrivateVariable(info.t->getDecl());	

	return info;
}

//Check whether we should perform load optimisation
//Current only a read only __global buffer can perform load optimisation
bool OpenCLLoadSchedule::shouldPerformLoadOpt(DeclRefExpr* e)
{
	if (OCLCommon::isAGlobalMemThreadPrivateVar(e->getDecl()))
	{
		return false;
	}

	if (OCLCommon::isAThreadPrivateVariable(e->getDecl()))
	{
		return false;
	}

	//Currently, only an array is read-only can be performed vload
	return isReadOnly(e);
}

//This is the heart of the class
void OpenCLLoadSchedule::VisitArraySubscriptExpr(ArraySubscriptExpr *Node) {

	arrayBaseInfo info = getArrayBaseInfo(Node);
	string& base_string = info.base_string;
	DeclRefExpr* t = info.t;
	bool firstMet = false;
	string offset_string;
	string arrayAccPrefix;

	assert(t && "Cannot find declare");

	//arrayDecls is a stack that keeps all the array seen from the first visit to an array
	//e.g. a[b[i]][j][k]
	// arrayDecls[0] = a; arrayDecls[1] = b;
	if (arrayDecls.size())
	{
		DeclRefExpr* top = arrayDecls.top();
		if (top != t)
		{
			firstMet = true;
		}
	}
	else
	{
		firstMet = true;
		OS.flush();
		str_buf = "";
	}

	if (firstMet)
	{
		arrayDecls.push(t);
		arrayInx.push(ArrayIndex(t, Node, &Context));
	}

	if (shouldPerformLoadOpt(t))
	{
		if (base_string.find('[') == string::npos)
		{
			Expr* expr = info.t;
			ImplicitCastExpr* icast = dyn_cast<ImplicitCastExpr>(expr);

			while(icast)
			{
				expr = icast->getSubExpr();
				icast = dyn_cast<ImplicitCastExpr>(expr);
			}

			DeclRefExpr* decl = dyn_cast<DeclRefExpr>(expr);

			if (decl)
			{
				assert(arrayInx.size() > 0 && "EMPTY ARRAY INDEX STACK!");
				//cerr << "THIS IS: " << decl->getNameInfo().getAsString() << endl;
			}
		}

	}


	PrintExpr(Node->getLHS());
	OS << "[";
	PrintExpr(Node->getRHS());
	OS << "]";

	if (arrayInx.size())
	{
		ArrayIndex& ai = arrayInx.top();
		string str = getStringExpr(Context, Node->getRHS());
		ai.addStrIndex(str, Node->getRHS());
	}

	if (firstMet)
	{
		ArrayIndex ai = arrayInx.top();

		//indirect access
		if(arrayInx.size() > 1)
		{
			arrayInx.pop();
			assert((arrayInx.size() >= 1) && "Index array should large than one");
			ArrayIndex& t = arrayInx.top();

			t.indexs.push_back(ai);
			t.hasIndirectAccess = true;
		}
		else if (arrayInx.size() == 1)
		{
			string name = t->getNameInfo().getAsString();
			if (shouldPerformLoadOpt(t))
			{
				ArrayIndex top = arrayInx.top();
				vector<ArrayIndex>& A = arrayInfos.top();
				A.push_back(top);
			}
			arrayInx.pop();
		}

		arrayDecls.pop();
	}
}

void OpenCLLoadSchedule::PrintCallArgs(CallExpr *Call) {
	unsigned cNum = Call->getNumArgs();
	for (unsigned i = 0, e = cNum; i != e; ++i) {
		if (isa<CXXDefaultArgExpr>(Call->getArg(i))) {
			// Don't print any defaulted arguments
			break;
		}

		if (i) OS << ", ";
		PrintExpr(Call->getArg(i));
	}

}

void OpenCLLoadSchedule::VisitCallExpr(CallExpr *Call) {
	PrintExpr(Call->getCallee());
	OS << "(";
	PrintCallArgs(Call);
	OS << ")";
}
void OpenCLLoadSchedule::VisitMemberExpr(MemberExpr *Node) {
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
void OpenCLLoadSchedule::VisitObjCIsaExpr(ObjCIsaExpr *Node) {
	PrintExpr(Node->getBase());
	OS << (Node->isArrow() ? "->isa" : ".isa");
}

void OpenCLLoadSchedule::VisitExtVectorElementExpr(ExtVectorElementExpr *Node) {
	PrintExpr(Node->getBase());
	OS << ".";
	OS << Node->getAccessor().getName();
}
void OpenCLLoadSchedule::VisitCStyleCastExpr(CStyleCastExpr *Node) {
	OS << "(" << Node->getType().getAsString(Policy) << ")";
	PrintExpr(Node->getSubExpr());
}
void OpenCLLoadSchedule::VisitCompoundLiteralExpr(CompoundLiteralExpr *Node) {
	OS << "(" << Node->getType().getAsString(Policy) << ")";
	PrintExpr(Node->getInitializer());
}
void OpenCLLoadSchedule::VisitImplicitCastExpr(ImplicitCastExpr *Node) {
	// No need to print anything, simply forward to the sub expression.
	PrintExpr(Node->getSubExpr());
}
void OpenCLLoadSchedule::VisitBinaryOperator(BinaryOperator *Node) {
	PrintExpr(Node->getLHS());
	OS << " " << BinaryOperator::getOpcodeStr(Node->getOpcode()) << " ";
	PrintExpr(Node->getRHS());
}
void OpenCLLoadSchedule::VisitCompoundAssignOperator(CompoundAssignOperator *Node) {
	PrintExpr(Node->getLHS());
	OS << " " << BinaryOperator::getOpcodeStr(Node->getOpcode()) << " ";
	PrintExpr(Node->getRHS());
}
void OpenCLLoadSchedule::VisitConditionalOperator(ConditionalOperator *Node) {
	PrintExpr(Node->getCond());
	OS << " ? ";
	PrintExpr(Node->getLHS());
	OS << " : ";
	PrintExpr(Node->getRHS());
}

// GNU extensions.

void
OpenCLLoadSchedule::VisitBinaryConditionalOperator(BinaryConditionalOperator *Node) {
	PrintExpr(Node->getCommon());
	OS << " ?: ";
	PrintExpr(Node->getFalseExpr());
}
void OpenCLLoadSchedule::VisitAddrLabelExpr(AddrLabelExpr *Node) {
	OS << "&&" << Node->getLabel()->getName();
}

void OpenCLLoadSchedule::VisitStmtExpr(StmtExpr *E) {
	OS << "(";
	PrintRawCompoundStmt(E->getSubStmt());
	OS << ")";
}

void OpenCLLoadSchedule::VisitChooseExpr(ChooseExpr *Node) {
	OS << "__builtin_choose_expr(";
	PrintExpr(Node->getCond());
	OS << ", ";
	PrintExpr(Node->getLHS());
	OS << ", ";
	PrintExpr(Node->getRHS());
	OS << ")";
}

void OpenCLLoadSchedule::VisitGNUNullExpr(GNUNullExpr *) {
	OS << "__null";
}

void OpenCLLoadSchedule::VisitShuffleVectorExpr(ShuffleVectorExpr *Node) {
	OS << "__builtin_shufflevector(";
	for (unsigned i = 0, e = Node->getNumSubExprs(); i != e; ++i) {
		if (i) OS << ", ";
		PrintExpr(Node->getExpr(i));
	}
	OS << ")";
}

void OpenCLLoadSchedule::VisitInitListExpr(InitListExpr* Node) {
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

void OpenCLLoadSchedule::VisitParenListExpr(ParenListExpr* Node) {
	OS << "( ";
	for (unsigned i = 0, e = Node->getNumExprs(); i != e; ++i) {
		if (i) OS << ", ";
		PrintExpr(Node->getExpr(i));
	}
	OS << " )";
}

void OpenCLLoadSchedule::VisitDesignatedInitExpr(DesignatedInitExpr *Node) {
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

void OpenCLLoadSchedule::VisitImplicitValueInitExpr(ImplicitValueInitExpr *Node) {
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

void OpenCLLoadSchedule::VisitVAArgExpr(VAArgExpr *Node) {
	OS << "__builtin_va_arg(";
	PrintExpr(Node->getSubExpr());
	OS << ", ";
	OS << Node->getType().getAsString(Policy);
	OS << ")";
}

// C++
void OpenCLLoadSchedule::VisitCXXOperatorCallExpr(CXXOperatorCallExpr *Node) {
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

void OpenCLLoadSchedule::VisitCXXMemberCallExpr(CXXMemberCallExpr *Node) {
	VisitCallExpr(cast<CallExpr>(Node));
}

void OpenCLLoadSchedule::VisitCUDAKernelCallExpr(CUDAKernelCallExpr *Node) {
	PrintExpr(Node->getCallee());
	OS << "<<<";
	PrintCallArgs(Node->getConfig());
	OS << ">>>(";
	PrintCallArgs(Node);
	OS << ")";
}

void OpenCLLoadSchedule::VisitCXXNamedCastExpr(CXXNamedCastExpr *Node) {
	OS << Node->getCastName() << '<';
	OS << Node->getTypeAsWritten().getAsString(Policy) << ">(";
	PrintExpr(Node->getSubExpr());
	OS << ")";
}

void OpenCLLoadSchedule::VisitCXXStaticCastExpr(CXXStaticCastExpr *Node) {
	VisitCXXNamedCastExpr(Node);
}

void OpenCLLoadSchedule::VisitCXXDynamicCastExpr(CXXDynamicCastExpr *Node) {
	VisitCXXNamedCastExpr(Node);
}

void OpenCLLoadSchedule::VisitCXXReinterpretCastExpr(CXXReinterpretCastExpr *Node) {
	VisitCXXNamedCastExpr(Node);
}

void OpenCLLoadSchedule::VisitCXXConstCastExpr(CXXConstCastExpr *Node) {
	VisitCXXNamedCastExpr(Node);
}

void OpenCLLoadSchedule::VisitCXXTypeidExpr(CXXTypeidExpr *Node) {
	OS << "typeid(";
	if (Node->isTypeOperand()) {
		OS << Node->getTypeOperand().getAsString(Policy);
	} else {
		PrintExpr(Node->getExprOperand());
	}
	OS << ")";
}

void OpenCLLoadSchedule::VisitCXXUuidofExpr(CXXUuidofExpr *Node) {
	OS << "__uuidof(";
	if (Node->isTypeOperand()) {
		OS << Node->getTypeOperand().getAsString(Policy);
	} else {
		PrintExpr(Node->getExprOperand());
	}
	OS << ")";
}

void OpenCLLoadSchedule::VisitCXXBoolLiteralExpr(CXXBoolLiteralExpr *Node) {
	OS << (Node->getValue() ? "true" : "false");
}

void OpenCLLoadSchedule::VisitCXXNullPtrLiteralExpr(CXXNullPtrLiteralExpr *Node) {
	OS << "nullptr";
}

void OpenCLLoadSchedule::VisitCXXThisExpr(CXXThisExpr *Node) {
	OS << "this";
}

void OpenCLLoadSchedule::VisitCXXThrowExpr(CXXThrowExpr *Node) {
	if (Node->getSubExpr() == 0)
		OS << "throw";
	else {
		OS << "throw ";
		PrintExpr(Node->getSubExpr());
	}
}

void OpenCLLoadSchedule::VisitCXXDefaultArgExpr(CXXDefaultArgExpr *Node) {
	// Nothing to print: we picked up the default argument
}

void OpenCLLoadSchedule::VisitCXXFunctionalCastExpr(CXXFunctionalCastExpr *Node) {
	OS << Node->getType().getAsString(Policy);
	OS << "(";
	PrintExpr(Node->getSubExpr());
	OS << ")";
}

void OpenCLLoadSchedule::VisitCXXBindTemporaryExpr(CXXBindTemporaryExpr *Node) {
	PrintExpr(Node->getSubExpr());
}

void OpenCLLoadSchedule::VisitCXXTemporaryObjectExpr(CXXTemporaryObjectExpr *Node) {
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

void OpenCLLoadSchedule::VisitCXXScalarValueInitExpr(CXXScalarValueInitExpr *Node) {
	if (TypeSourceInfo *TSInfo = Node->getTypeSourceInfo())
		OS << TSInfo->getType().getAsString(Policy) << "()";
	else
		OS << Node->getType().getAsString(Policy) << "()";
}

void OpenCLLoadSchedule::VisitCXXNewExpr(CXXNewExpr *E) {
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

void OpenCLLoadSchedule::VisitCXXDeleteExpr(CXXDeleteExpr *E) {
	if (E->isGlobalDelete())
		OS << "::";
	OS << "delete ";
	if (E->isArrayForm())
		OS << "[] ";
	PrintExpr(E->getArgument());
}

void OpenCLLoadSchedule::VisitCXXPseudoDestructorExpr(CXXPseudoDestructorExpr *E) {
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

void OpenCLLoadSchedule::VisitCXXConstructExpr(CXXConstructExpr *E) {
	for (unsigned i = 0, e = E->getNumArgs(); i != e; ++i) {
		if (isa<CXXDefaultArgExpr>(E->getArg(i))) {
			// Don't print any defaulted arguments
			break;
		}

		if (i) OS << ", ";
		PrintExpr(E->getArg(i));
	}
}

void OpenCLLoadSchedule::VisitExprWithCleanups(ExprWithCleanups *E) {
	// Just forward to the sub expression.
	PrintExpr(E->getSubExpr());
}

void
OpenCLLoadSchedule::VisitCXXUnresolvedConstructExpr(
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

void OpenCLLoadSchedule::VisitCXXDependentScopeMemberExpr(
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

void OpenCLLoadSchedule::VisitUnresolvedMemberExpr(UnresolvedMemberExpr *Node) {
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

void OpenCLLoadSchedule::VisitUnaryTypeTraitExpr(UnaryTypeTraitExpr *E) {
	OS << getTypeTraitName(E->getTrait()) << "("
		<< E->getQueriedType().getAsString(Policy) << ")";
}

void OpenCLLoadSchedule::VisitBinaryTypeTraitExpr(BinaryTypeTraitExpr *E) {
	OS << getTypeTraitName(E->getTrait()) << "("
		<< E->getLhsType().getAsString(Policy) << ","
		<< E->getRhsType().getAsString(Policy) << ")";
}

void OpenCLLoadSchedule::VisitCXXNoexceptExpr(CXXNoexceptExpr *E) {
	OS << "noexcept(";
	PrintExpr(E->getOperand());
	OS << ")";
}

void OpenCLLoadSchedule::VisitPackExpansionExpr(PackExpansionExpr *E) {
	PrintExpr(E->getPattern());
	OS << "...";
}

void OpenCLLoadSchedule::VisitSizeOfPackExpr(SizeOfPackExpr *E) {
	OS << "sizeof...(" << E->getPack()->getNameAsString() << ")";
}

void OpenCLLoadSchedule::VisitSubstNonTypeTemplateParmPackExpr(
		SubstNonTypeTemplateParmPackExpr *Node) {
}

// Obj-C

void OpenCLLoadSchedule::VisitObjCStringLiteral(ObjCStringLiteral *Node) {
	OS << "@";
	VisitStringLiteral(Node->getString());
}

void OpenCLLoadSchedule::VisitObjCEncodeExpr(ObjCEncodeExpr *Node) {
	OS << "@encode(" << Node->getEncodedType().getAsString(Policy) << ')';
}

void OpenCLLoadSchedule::VisitObjCSelectorExpr(ObjCSelectorExpr *Node) {
	OS << "@selector(" << Node->getSelector().getAsString() << ')';
}

void OpenCLLoadSchedule::VisitObjCProtocolExpr(ObjCProtocolExpr *Node) {
	OS << "@protocol(" << Node->getProtocol() << ')';
}

void OpenCLLoadSchedule::VisitObjCMessageExpr(ObjCMessageExpr *Mess) {
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


void OpenCLLoadSchedule::VisitBlockExpr(BlockExpr *Node) {
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

void OpenCLLoadSchedule::VisitBlockDeclRefExpr(BlockDeclRefExpr *Node) {
	OS << Node->getDecl();
}

void OpenCLLoadSchedule::VisitOpaqueValueExpr(OpaqueValueExpr *Node) {}

//===----------------------------------------------------------------------===//
// Stmt method implementations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PrinterHelper
//===----------------------------------------------------------------------===//

// Implement virtual destructor.
//OMP2OCL
void OpenCLLoadSchedule::VisitOclFlush(clang::OclFlush*) {}
void OpenCLLoadSchedule::VisitOclHostFlush(clang::OclHostFlush*) {}
void OpenCLLoadSchedule::VisitOclInit(clang::OclInit*) {}
void OpenCLLoadSchedule::VisitOclTerm(clang::OclTerm*) {}
void OpenCLLoadSchedule::VisitOclSync(clang::OclSync*) {}
void OpenCLLoadSchedule::VisitOclResetMLStmt(clang::OclResetMLStmt*) {}
void OpenCLLoadSchedule::VisitOclEnableMLRecordStmt(clang::OclEnableMLRecordStmt*) {}
void OpenCLLoadSchedule::VisitOclDisableMLRecordStmt(clang::OclDisableMLRecordStmt*) {}
void OpenCLLoadSchedule::VisitOclDumpMLFStmt(clang::OclDumpMLFStmt*) {}

void OpenCLLoadSchedule::VisitOclStartProfile(clang::OclStartProfile*) {}
void OpenCLLoadSchedule::VisitOclDumpProfile(clang::OclDumpProfile*) {}
void OpenCLLoadSchedule::VisitOclStopProfile(clang::OclStopProfile*) {}
void OpenCLLoadSchedule::VisitOclHostRead(clang::OclHostRead*) {}
void OpenCLLoadSchedule::VisitOclDevRead(clang::OclDevRead*) {}
void OpenCLLoadSchedule::VisitOclDevWrite(clang::OclDevWrite*) {}
