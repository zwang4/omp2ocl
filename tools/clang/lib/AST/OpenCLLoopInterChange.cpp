#include "clang/Omp2Ocl/OpenCLLoopInterChange.h"
#include "clang/Omp2Ocl/OpenCLKernelLoop.h"
#include "clang/Omp2Ocl/OpenCLCompilerOptions.h"
#include "clang/Omp2Ocl/OpenCLRoutine.h"

ForStmt* OpenCLLoopInterChange::whichLoop(vector<SwapLoopInfo>& SwapableLoops, DeclRefExpr* expr, unsigned& origin_i)
{
	unsigned i = 0;
	for (vector<SwapLoopInfo>::iterator iter = SwapableLoops.begin(); iter != SwapableLoops.end(); iter++)
	{
		ForStmt* for_stmt = iter->for_stmt;
		LoopIndex *l = OCLCommon::getLoopIndex(for_stmt);

		if (dyn_cast<DeclRefExpr>(l->variable))
		{
			DeclRefExpr *lexpr = (DeclRefExpr*) (l->variable);
			if (lexpr->getNameInfo().getAsString() == expr->getNameInfo().getAsString())
			{
				origin_i = i;
				return for_stmt;
			}
		}

		i++;
	}

	return NULL;
}

bool OpenCLLoopInterChange::isInLoopIndex(DeclRefExpr* expr, vector<LoopIndex*>& swapableIndex)
{
	string name = expr->getNameInfo().getAsString();
	for (vector<LoopIndex*>::iterator iter = swapableIndex.begin(); iter != swapableIndex.end(); iter++)
	{
		LoopIndex *l = (*iter);

		if (dyn_cast<DeclRefExpr>(l->variable))
		{
			DeclRefExpr *expr = (DeclRefExpr*) (l->variable);
			if (expr->getNameInfo().getAsString() == name)
			{
				return true;
			}
		}
	}

	return false;
}

unsigned int OpenCLLoopInterChange::howManyLoopIndexUse(vector<LoopIndex*>& swapableIndex, ArraySubVariable* a)
{
	unsigned int num = 0;

	for (vector<DeclRefExpr*>::iterator it = a->v.begin(); it != a->v.end(); it++)
	{
		if (isInLoopIndex(*it, swapableIndex))
		{
			num++;
		}
	}

	return num;
}

//Perform loop interchange
void OpenCLLoopInterChange::InterChangeLoops(unsigned int max_loop_level)
{
	unsigned int inputLoopNum = SwapableLoops.size();

	/*
	 * Collect swapable loops index
	 *
	 */
	vector<LoopIndex*> swapableIndexs;
	for (vector<SwapLoopInfo>::iterator iter = SwapableLoops.begin(); iter != SwapableLoops.end(); iter++)
	{
		ForStmt* for_stmt = iter->for_stmt;
		LoopIndex *l = OCLCommon::getLoopIndex(for_stmt);
		swapableIndexs.push_back(l);
	}

	ArraySubVariable* bestA = NULL;
	unsigned int bestNum=0;

	for (vector<ArraySubVariable*>::iterator iter = getLoop()->arraySubVs.begin(); iter != getLoop()->arraySubVs.end(); iter++)
	{
		ArraySubVariable* a = (*iter);
		unsigned int num = howManyLoopIndexUse(swapableIndexs, a);
		if (num > bestNum)
		{
			bestA = a;
			bestNum = num;
		}
	}

	vector<SwapLoopInfo> swapedLoops;
	//The index of the original loop sequence
	unsigned whichI;

	//Find the best array accessing sequence
	if (bestA)
	{
		if (bestA->v.size() > 0)
		{
			for (int i=bestA->v.size()-1; i>=0; i--)
			{
				if (swapedLoops.size()  >= max_loop_level)
				{
					break;
				}

				ForStmt* stmt = whichLoop(SwapableLoops, bestA->v[i], whichI);
				if (stmt != NULL)
				{
					swapedLoops.push_back(SwapLoopInfo(stmt, whichI));
				}		
			}
		}
		swaped = true;
	}
	else
	{
		unsigned loopNum = (SwapableLoops.size() > max_loop_level) ? max_loop_level : SwapableLoops.size();
		for (unsigned i=0; i<loopNum; i++)
		{
			swapedLoops.push_back(SwapableLoops[i]);	
		}
	}

	//Add the remaining loops back to swapedLoops
	if ((swapedLoops.size() != SwapableLoops.size()) && (swapedLoops.size() < max_loop_level))
	{
		for (unsigned i=0; i<SwapableLoops.size(); i++)
		{
			ForStmt* f = SwapableLoops[i].for_stmt;
			bool found = false;
			unsigned j;
			for (j=0; j<swapedLoops.size(); j++)
			{
				if (swapedLoops[j].for_stmt == f)
				{
					found = true;
					break;
				}
			}

			if (!found)
			{
				swapedLoops.push_back(SwapLoopInfo(f, j));
			}
		}
	}

	InsertInnerLoops(swapedLoops, SwapableLoops);

	for (unsigned i=0; i<swapableIndexs.size(); i++)
	{
		delete swapableIndexs[i];
	}

	SwapableLoops = swapedLoops;

	assert(SwapableLoops.size() <= inputLoopNum);
}

//Check the sanity of the two caluses:
//#pragma omp for prallel_depth(depth:seq1,seq2,...)
//and 
//#pragma omp for swap(seq1,seq2,...)
bool OpenCLLoopInterChange::CheckSanityofSwapIndexs(OMPParallelDepth& pd, vector<OMPSwapIndex>& swapIndexs)
{
	vector<string> seqs = pd.getSeq();

	//The user does specify any swap sequence in the parallel_depth clause
	if (seqs.empty() || swapIndexs.empty())
	{
		return true;
	}

	if (seqs.size() != swapIndexs.size())
	{
		return false;
	}

	for (unsigned i=0; i<seqs.size(); i++)
	{
		bool found = false;
		for (unsigned j=0; j<swapIndexs.size(); j++)
		{
			if (swapIndexs[j].getVariable() == seqs[i])
			{
				found = true;
				break;
			}
		}

		if (!found)
		{
			return false;
		}
	}

	return true;
}

void OpenCLLoopInterChange::InsertInnerLoops(vector<SwapLoopInfo>& newSwapLoops, vector<SwapLoopInfo>& SwapableLoops)
{
	for (unsigned k=0; k<SwapableLoops.size(); k++)
	{
		bool found =false;
		for (unsigned ki=0; ki<newSwapLoops.size(); ki++)
		{
			if (newSwapLoops[ki].getForStmt() == 
					SwapableLoops[k].getForStmt()
			   )
			{
				found = true;
				break;
			}
		}

		if (!found)
		{
			innerLoops.push_back(SwapableLoops[k]);
		}
	}
}

//
//This function performs user define loop interchange
//The user can define a loop interchange order through either the "parallel_depth" or "swap" clause
//
void OpenCLLoopInterChange::userDefInterChange()
{
	ForStmt* topLoop = getLoop()->getForStmt();
	OMPParallelDepth& pd = topLoop->getOMPFor().getParallelDepth();
	vector<string> seqs = pd.getSeq();

	int swapedLoops=1;
	Stmt* Kernel = topLoop->getBody();
	SwapableLoops.push_back(SwapLoopInfo(topLoop, 0));

	if (OCLCommon::isAPerfectNestedLoop(topLoop))
	{
		for (vector<ForStmt*>::iterator iter = getLoop()->subLoops.begin(); 
				iter != getLoop()->subLoops.end(); iter++)
		{
			SwapableLoops.push_back(SwapLoopInfo(*iter, swapedLoops));		
			Kernel = (*iter)->getBody();

			if (!OCLCommon::isAPerfectNestedLoop(*iter))
			{
				break;
			}
		}

		if (getLoop()->getOMPFor().getSwap() || seqs.size())
		{
			//this is the user-specific swapable indexs
			vector<OMPSwapIndex> swapIndexs = getLoop()->getOMPFor().getSwapIndexs();

			if (!CheckSanityofSwapIndexs(pd, swapIndexs))
			{
				SourceLocation loc = getLoop()->getForStmt()->getForLoc();
				cerr << "Warning: there are conflicts between the parallel_depth and swap clauses (for loop at line: " 
					<< OCLCommon::getLineNumber(getContext(), loc) << endl;
			}
			else if(swapIndexs.empty())
			{
				for (unsigned i=0; i<seqs.size(); i++)
				{
					swapIndexs.push_back(OMPSwapIndex(seqs[i]));
				}	
			}

			vector<SwapLoopInfo> newSwapLoops;
			vector<LoopIndex*> swapableIndexs;

			for (vector<SwapLoopInfo>::iterator iter = SwapableLoops.begin(); iter != SwapableLoops.end(); iter++)
			{
				ForStmt* for_stmt = iter->for_stmt;
				LoopIndex *l = OCLCommon::getLoopIndex(for_stmt);
				swapableIndexs.push_back(l);
			}

			//Do the actual user defined swaption
			for (unsigned ii=0; ii<swapIndexs.size(); ii++)
			{
				string index = swapIndexs[ii].getVariable();		
				unsigned j;

				for (j=0; j<swapableIndexs.size(); j++)
				{
					LoopIndex* l = swapableIndexs[j];				
					if (dyn_cast<DeclRefExpr>(l->variable))
					{
						DeclRefExpr *expr = (DeclRefExpr*) (l->variable);
						if (expr->getNameInfo().getAsString() == index)
						{
							break;
						}
					}
				}

				if (j >= swapableIndexs.size())
				{
					SourceLocation loc = topLoop->getForLoc();
					cerr << endl << "ERROR: The " << index << "th nested loop is not swaptable (loop at: line: " << OCLCommon::getLineNumber(getContext(), loc) << ")" << endl;
					exit(-1);
				}
				else
				{
					newSwapLoops.push_back(SwapableLoops[j]);
				}
			}

			swaped = true;
			innerLoops.clear();

			InsertInnerLoops(newSwapLoops, SwapableLoops);
			SwapableLoops = newSwapLoops;
		}
	}

	getLoop()->setKernel(Kernel);
}

void OpenCLLoopInterChange::VanillaInterChange()
{
	int max_loop_level;
	ForStmt* topLoop = getLoop()->getForStmt();
	OMPParallelDepth& pd = topLoop->getOMPFor().getParallelDepth();
	max_loop_level = pd.getDepth();
	bool lDepth = pd.isUserSetDepth();

	if (OCLCompilerOptions::UserDefParallelDepth)
	{
		lDepth = 1;
	}

	Stmt* Kernel = topLoop->getBody();

	int swapedLoops=1;
	SwapableLoops.push_back(SwapLoopInfo(topLoop, 0));

	//Only perfect nested loop can be swapped
	if (OCLCommon::isAPerfectNestedLoop(topLoop))
	{
		if (!lDepth || (lDepth && (swapedLoops < max_loop_level)))
		{
			for (vector<ForStmt*>::iterator iter = getLoop()->subLoops.begin(); iter != getLoop()->subLoops.end(); iter++)
			{
				SwapableLoops.push_back(SwapLoopInfo(*iter, swapedLoops));		
				Kernel = (*iter)->getBody();

				swapedLoops++;

				if (!OCLCommon::isAPerfectNestedLoop(*iter))
				{
					break;
				}
				//else
				//if (lDepth || !OCLCompilerOptions::EnableLoopInterchange)
				{
					if (swapedLoops >= max_loop_level)
					{
						break;
					}
				}
			}

			//This loop can be swapped	
			if (getLoop()->getOMPFor().getSwap())
			{
				//Loop interchange is turned on
				if (OCLCompilerOptions::EnableLoopInterchange)
				{
					InterChangeLoops(max_loop_level);
				}
			}
		}
	}

	assert(SwapableLoops.size() <= (unsigned)max_loop_level);

	getLoop()->setKernel(Kernel);
}

void OpenCLLoopInterChange::doIt()
{
	ForStmt* topLoop = getLoop()->getForStmt();
	OMPParallelDepth& pd = topLoop->getOMPFor().getParallelDepth();
	vector<string> seqs = pd.getSeq();

	if (getLoop()->getOMPFor().getSwap() && (seqs.size() || getLoop()->getOMPFor().getSwapIndexs().size()) )
	{
		userDefInterChange();
	}
	else
	{
		VanillaInterChange();
	}

	assert(SwapableLoops.size() <= (getLoop()->subLoops.size() + 1));
	
	getLoop()->setInnerLoops(innerLoops);
	getLoop()->setSwapLoops(SwapableLoops);
	getLoop()->swaped = swaped;
}
