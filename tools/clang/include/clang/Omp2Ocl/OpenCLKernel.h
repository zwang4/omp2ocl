#ifndef __OPENCLKERNEL_H__
#define __OPENCLKERNEL_H__
//#include "clang/AST/Expr.h"
//#include "clang/AST/Decl.h"
//#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include <vector>

using namespace std;
//using namespace clang;

namespace clang
{
	class CallExpr;
	class FunctionDecl;
	class ValueDecl;
	class DeclRefExpr;
	class ArraySubscriptExpr;
	class ForStmt;
	class BinaryOperator;

	class arraySubInfo
	{
		public:
			string name;
			vector<string> subExprs;
	};


	class FunctionLevelOCLBuffer
	{
		public:
			FunctionDecl* D;
			ValueDecl* E;
			FunctionLevelOCLBuffer(FunctionDecl* D, ValueDecl* E)
			{
				this->D = D;
				this->E = E;
			}
	};

	class CallArgInfo
	{
		bool isLocalBuf;

		public:
		DeclRefExpr* arg;
		unsigned index;
		bool isPointerAccess;
		string access_offset;
		unsigned lc;
		string name;
		string string_buf;
		bool isBuf;
		bool isGlobalMemThreadPrivate;

		CallArgInfo (bool isLocalBuf)
		{
			isBuf = false;
			isPointerAccess = false;
			isGlobalMemThreadPrivate = false;
			this->isLocalBuf = isLocalBuf;
			index = 0;
		}

		string getName()
		{
			return name;
		}

		void reset()
		{
			isBuf = false;
			isPointerAccess = false;
			isGlobalMemThreadPrivate = false;
			this->isLocalBuf = false;
			access_offset = "";
			lc = 0;
			index = 0;
		}

		void setAsLocalBuf()
		{
			isLocalBuf = true;
		}

		bool isALocalBuf()
		{
			return isLocalBuf;
		}
	};

	//This recoreds the callee functions whose arguments is passed as pointers where the pointers 
	//are pointed to a global memory buffer
	class CallArgInfoContainer {
		public:
			CallExpr *ce;
			vector<CallArgInfo> gCallArgs;
			unsigned curIndex;
			bool hasGlobalMemThreadPrivate;

			CallArgInfoContainer()
			{
				hasGlobalMemThreadPrivate = false;
				curIndex = 0;
			}
	};



	class globalVarIndex
	{
		bool isLocalBuf;
		string name;

		void init(unsigned ii, bool isP, bool isGTP, bool isLocalVar, string n);
		public:
		unsigned i;
		bool isPointerAccess;
		bool isGTP;

		globalVarIndex(unsigned ii, bool isP, bool isGTP, bool isLocalVar, string n, int a)
		{
			init(ii, isP, isGTP, isLocalVar, n);
		}

		globalVarIndex(CallArgInfo& ci)
		{
			init(ci.index, ci.isPointerAccess, 
					ci.isGlobalMemThreadPrivate, ci.isALocalBuf(), ci.getName() );
		}

		string getName()
		{
			return name;
		}

		void set2LocalBuf()
		{
			isLocalBuf = true;
		}

		bool isALocalBuf()
		{
			return isLocalBuf;
		}

		bool shouldTreadAsGlobalVar()
		{
			if (!isLocalBuf)
				return true;
			return false;
		}
	};

	class extendVarIndex
	{
		public:
			unsigned id;
			extendVarIndex(unsigned i)
			{
				id = i;
			}
	};

	class FuncProtoExt
	{
		public:
			DeclRefExpr* expr;
			unsigned extended_index;
			bool revised;
			bool isGlobalThreadPrivate;
			bool isLocalVar;

			FuncProtoExt(DeclRefExpr* e, unsigned ei, bool hasRevised, bool isGT, bool isLocalVar);

			bool isGTP() { return isGlobalThreadPrivate; }

			void setRevised()
			{
				revised = true;
			}

			bool hasRevised() { return revised; }

			bool isALocalVar() { return isLocalVar; }

			ValueDecl* getDecl();
	};



	class OCLGlobalMemVar
	{
		public:
			ValueDecl* v;
			bool isGlobalThreadPrivate;
			bool isFLevel;
			string declaredName;
			bool isThreadPrivate;
			bool isArray;

			OCLGlobalMemVar(ValueDecl* d, bool isG, bool isFLevel, bool isThreadPrivate);
			bool isArrayType()
			{
				return isArray;
			}

			void setDeclaredName(string strName)
			{
				declaredName = strName;
			}

			bool isFuncLevel() { return isFLevel; }

			OCLGlobalMemVar(FuncProtoExt& P);

			ValueDecl* getDecl() { return v; }
			QualType getType();
			string getNameAsString();

			string OCLKernelAccess(string str)
			{
				return str;
			}

			bool isAThreadPrivateVar()
			{
				return isThreadPrivate;
			}

			bool canbeDeclareAsGlobal();
			bool isDefinedOusideFunc();

			bool isGlobalTPBuf() { return isGlobalThreadPrivate; }
	};



	class RenamedFuncInfo
	{
		public:
			string origFuncName;
			vector<globalVarIndex> globalArugIds;
			vector<extendVarIndex> extendIndex;
			vector<OCLGlobalMemVar> oclMemVars;
			bool hasGlobalMemThreadPrivate;
			bool hasGenerated;
			string newName;
			bool enable_spec;

			string getOrigFuncName()
			{
				return origFuncName;
			}

			RenamedFuncInfo(string& oriName, vector<globalVarIndex>& gA, vector<extendVarIndex>& eid, string& newName, bool hasGMT);
			bool isInExtendId(unsigned id);
			bool isInExtendId(extendVarIndex& ei);
	};



	class PLoopParam {
		public:
			bool isWrite;
			PLoopParam(DeclRefExpr* ref, bool isWrite) {
				this->declRef = ref; 
				this->isWrite = isWrite;
			}
			DeclRefExpr* declRef;
			bool isWritten() { return isWrite; }

			string getName();
			ValueDecl* getDecl();
	};

	class LoopIndex {
		public:
			Stmt* variable;
			Stmt* init;
			ForStmt* for_stmt;

			LoopIndex(Stmt* index_variable, Stmt* index_init, ForStmt* for_stmt)
			{
				this->variable = index_variable;
				this->init = index_init;
				this->for_stmt = for_stmt;
			}
	};

	class ArraySubVariable
	{
		DeclRefExpr* base;
		public:
		ArraySubVariable() { base = NULL; }
		vector<DeclRefExpr*> v;
		void setBase(DeclRefExpr* base) { this->base = base; }
		DeclRefExpr* getBase() { return base; }
		void addElement(DeclRefExpr* e) { v.push_back(e); }
	};

	/****************************************************************************
	 *
	 * USED FOR VLOAD (BEGIN)
	 *
	 ****************************************************************************/
	class arrayBaseInfo
	{
		public:
			arraySubInfo ai;
			DeclRefExpr* t;
			bool isGlobalBuffer;
			bool isTGBuffer;
			string base_string;

			arrayBaseInfo()
			{
				isGlobalBuffer = false;
				t = NULL;
				isTGBuffer = false;
			}
	};

	class IndexStr
	{
		string str;
		Expr* expr;

		public:
		IndexStr(string s, Expr* e);
		/*
		IndexStr()
		{
			expr = NULL;
		}*/

		string getAsString()
		{
			return str;
		}

		const string getAsString() const
		{
			return str;
		}

		Expr* getExpr()
		{
			return expr;
		}

		const Expr* getExpr() const
		{
			return expr;
		}
	};

	class ArrayIndex
	{
		list<IndexStr> accessIndexs;
		ASTContext* pContext;
		
		//This will be set once the offset of
		//the local vtype array is know.
		//For example, the generated code declare the an array
		//double2 vI[2];
		//vIndex can be 0, which corresponds to vI[0]
		int vIndex;
	    //sIndex correspond to the offset of a vector structure
		//e.g. vI[0].x
		int sIndex;

		//Whether vIndex and sIndex are set;
		bool setVI;
		string orgLStr;
		string vDeclareName;
		unsigned width;
		string offsetStr;

		unsigned occurance;
		public:
		DeclRefExpr* base;
		ArraySubscriptExpr* Node;
		vector<ArrayIndex> indexs;
		bool hasIndirectAccess;
		vector<ArraySubscriptExpr*> asExprs;

		ArrayIndex(DeclRefExpr* e, ArraySubscriptExpr* Node, ASTContext* pCtx)
		{
			assert(e && "DeclRefExpr shouldn't be null");
			base = e;
			this->Node = Node;
			hasIndirectAccess = false;
			pContext = pCtx;
			addASE(Node);
			setVI = false;
			width=0;
			occurance=0;
		}

		void incOccurance()
		{
			occurance++;
		}

		unsigned getOccurance()
		{
			return occurance;
		}

		void setOffsetStr(string str)
		{
			this->offsetStr = str;
		}

		string getOffsetStr()
		{
			return offsetStr;
		}

		void setWidth(unsigned w)
		{
			width=w;
		}

		unsigned getWidth()
		{
			return width;
		}

		void setVDeclareName(string str)
		{
			vDeclareName = str;
		}

		string getVDeclareName()
		{
			return vDeclareName;
		}

		void setOrgLStr(string str)
		{
			orgLStr = str;
		}

		string getOrgLStr()
		{
			return orgLStr;
		}

		void setLIndexs(int v, int s)
		{
			vIndex = v;
			sIndex = s;
			setVI = true;
		}

		int getVIndex()
		{
			assert(setVI && "vIndex hasn't been set!");
			return vIndex;
		}

		int getSIndex()
		{
			assert(setVI && "sIndex hasn't been set!");
			return sIndex;
		}

		void addASE(ArraySubscriptExpr* expr)
		{
			for (unsigned i=0; i<asExprs.size(); i++)
			{
				if (asExprs[i] == expr)
				{
					return;
				}
			}

			asExprs.push_back(expr);
		}

		ArraySubscriptExpr* getASENode()
		{
			return Node;
		}

		ASTContext* getContext()
		{
			return pContext;
		}

		ASTContext* getContext() const
		{
			return pContext;
		}

		bool operator< (const ArrayIndex& rhs) const;

		void addStrIndex(string str, Expr* e)
		{
			accessIndexs.push_back(IndexStr(str, e));
		}

		const list<IndexStr>& getAccessIdx() const
		{
			return accessIndexs;
		}

		list<IndexStr>& getAccessIdx()
		{
			return accessIndexs;
		}

		string getName();
		bool hasIndirectAcc()
		{
			return hasIndirectAccess;
		}

		bool hasIndirectAcc() const
		{
			return hasIndirectAccess;
		}

		void setIndirectAcc()
		{
			hasIndirectAccess = true;
		}

		ArraySubscriptExpr* getNode() { return Node; }

		ValueDecl* getDecl();
	};

	class ArrayAccessInfo
	{
		public:
			ValueDecl* D;
			vector<ArrayIndex> VIs;
			ArrayAccessInfo (ValueDecl* d)
			{
				D = d;
			}

			ArrayAccessInfo(ValueDecl* d, ArrayIndex& I)
			{
				D = d;
				addArrayIndex(I);
			}

			void addArrayIndex(ArrayIndex I)
			{
				VIs.push_back(I);
			}

			string getName();

			vector<ArrayIndex>& getAIs()
			{
				return VIs;
			}

			const vector<ArrayIndex>& getAIs() const
			{
				return VIs;
			}

	};

	class OCLCompoundVLoadDeclareInfo
	{
		string dname;
		unsigned int vs; //vectorisation width
		vector< vector<ArrayIndex> > AIs;
		public:
		OCLCompoundVLoadDeclareInfo(string declareName, unsigned vs, vector<ArrayIndex>& AI)
		{
			this->dname = declareName;
			this->vs = vs;
			this->AIs.push_back(AI);
		}

		string getDeclareName()
		{
			return dname;
		}

		vector< vector<ArrayIndex> >& getAIs()
		{
			return AIs;
		}

		unsigned getVWidth()
		{
			return vs;
		}

		void addAI(vector<ArrayIndex>& v)
		{
			AIs.push_back(v);
		}

	};


	/****************************************************************************
	 *
	 * USED FOR VLOAD (END)
	 *
	 ****************************************************************************/
	class OpenCLNDRangeVar {
		public:
			std::string variable;
			std::string type;
			Stmt* Init;
			Stmt* Cond;
			Stmt* Inc;
			string cond_string;
			string cond_opcode_str;
			bool isCondInt;
			bool isIncInt;
			unsigned orig_loop_index;
			string increment;
			bool hasIncremental;

			OpenCLNDRangeVar(unsigned orig_loop_index)
			{
				Init = NULL;
				Cond = NULL;
				isCondInt = false;
				isIncInt = false;
				hasIncremental = false;
				this->orig_loop_index = orig_loop_index;
			}

			string getCondString(ASTContext& Context);
			string getName() { return variable; }

			OpenCLNDRangeVar(OpenCLNDRangeVar& g)
			{
				this->variable = g.variable;
				this->type = g.type;
				this->Init = g.Init;
				this->Cond = g.Cond;
				this->Inc = g.Inc;
				this->cond_string = g.cond_string;
				this->cond_opcode_str = g.cond_opcode_str;
				this->isCondInt = g.isCondInt;
				this->isIncInt = g.isIncInt;
				this->increment = g.increment;
				this->hasIncremental = g.hasIncremental;
				this->orig_loop_index = g.orig_loop_index;
			}

			OpenCLNDRangeVar(const OpenCLNDRangeVar& g)
			{
				this->variable = g.variable;
				this->type = g.type;
				this->Init = g.Init;
				this->Cond = g.Cond;
				this->Inc = g.Inc;
				this->cond_string = g.cond_string;
				this->cond_opcode_str = g.cond_opcode_str;
				this->isCondInt = g.isCondInt;
				this->isIncInt = g.isIncInt;
				this->increment = g.increment;
				this->hasIncremental = g.hasIncremental;
				this->orig_loop_index = g.orig_loop_index;
			}


	};


			class OpenCLBinarySquareOpt
			{
				BinaryOperator* opt;
				DeclRefExpr* LHS;	
				DeclRefExpr* RHS;
			public:
				OpenCLBinarySquareOpt(BinaryOperator *opt, DeclRefExpr* LHS, DeclRefExpr* RHS)
				{
					this->opt = opt;
					this->LHS = LHS;
					this->RHS = RHS;
				}

				BinaryOperator* getBOP()
				{
					return opt;
				}

				DeclRefExpr* getLHS()
				{
					return LHS;
				}

				DeclRefExpr* getRHS()
				{
					return RHS;
				}
			};


}


#endif
