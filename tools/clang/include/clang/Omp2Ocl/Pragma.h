#ifndef __OMP2OCL_PRAGMA_H__
#define __OMP2OCL_PRAGMA_H__
#include <vector>
#include <string>
#include <iostream>
#include "llvm/Support/raw_ostream.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Omp2Ocl/OpenCLCommon.h"

using namespace std;

namespace clang {
	/**
	 * Zheng: this is for openmp support
	 */
	enum ompSectionType {
		omp_parallel_for = 0,
		omp_parallel = 1,
		omp_parallel_reduction = 2,
		omp_private = 3,
		omp_firstprivate = 4,
		omp_copyin = 5,
		omp_tlsvar=6,
		omp_reduction=7,
		omp_swapindex = 8,
		omp_multiter = 9,
		ocl_localvar = 10,
		ocl_parallel_depth=11
	};

	class OMPObject {
		public:
			enum ompSectionType type;
			SourceLocation loc;
			std::string variable;

			void setVariable(std::string val) {variable = val;}
			std::string getVariable() { return variable; }
			std::string getVariable() const { return variable; }
			enum ompSectionType getType() { return type; }
			void print(llvm::raw_ostream& Out)
			{
				Out << this->variable;
			}

	};

	class OMPReductionObj : public OMPObject {
		public:
			string kind;

			OMPReductionObj () {
				type = omp_reduction;
			}

			string& getOperatorCode() { return kind; }

			OMPReductionObj(std::string val, string kind)
			{
				type = omp_reduction;
				setVariable(val);
				this->kind = kind;
			}
	};

	class OMPPrivate : public OMPObject{
		public:
			OMPPrivate() {type = omp_private;}
			OMPPrivate(std::string val) {
				type = omp_private;
				setVariable(val);
			}
	};


	class OMPFirstPrivate : public OMPObject{
		public:
			OMPFirstPrivate() {type = omp_firstprivate;}
			OMPFirstPrivate(std::string val) {
				type = omp_firstprivate;
				setVariable(val);
			}
	};

	class OMPMultIterIndex : public OMPObject {
		public:
			OMPMultIterIndex() { type = omp_multiter; }
			OMPMultIterIndex(std::string val)
			{
				type = omp_multiter;
				setVariable(val);
			}	
	};

	class OMPSwapIndex : public OMPObject {
		public:
			OMPSwapIndex() { type = omp_swapindex; }
			OMPSwapIndex(std::string val)
			{
				type = omp_swapindex;
				setVariable(val);
			}	
	};

	class OMPCopyIn : public OMPObject {
		public:
			OMPCopyIn() { type = omp_copyin; }

			OMPCopyIn(std::string val) {
				type = omp_copyin;
				setVariable(val);
			}
	};
	
	class OMPTLSVariable : public OMPObject {
		public:
			OMPTLSVariable() { type = omp_tlsvar; }

			OMPTLSVariable(std::string val) {
				type = omp_tlsvar;
				setVariable(val);
			}
	};

	class OMPThreadPrivateObject : public OMPObject
	{
		public:
			bool useGlobalMem;

			OMPThreadPrivateObject(string v, SourceLocation loc)
			{
				OMPThreadPrivateObject(v, loc, false);	
			}

			OMPThreadPrivateObject(string v, SourceLocation loc, bool ug)
			{
				setVariable(v);
				this->loc = loc;
				this->useGlobalMem = ug;
			}

			OMPThreadPrivateObject(OMPThreadPrivateObject& O)
			{
				setVariable(O.getVariable());
				this->loc = O.loc;
				this->useGlobalMem = O.isUseGlobalMem();
			}

			OMPThreadPrivateObject(const OMPThreadPrivateObject& O)
			{
				setVariable(O.getVariable());
				this->loc = O.loc;
				this->useGlobalMem = O.isUseGlobalMem();
			}

			void setUseGlobalMem(bool ib)
			{
				useGlobalMem = ib;
			}

			bool isUseGlobalMem() const
			{
				return useGlobalMem;
			}

			bool isUseGlobalMem()
			{
				return useGlobalMem;
			}
	};

	class OMPThreadPrivate
	{
		public:
			SourceLocation loc;
			vector<OMPThreadPrivateObject> objs;

			void addPrivateObj(OMPThreadPrivateObject& obj)
			{
				objs.push_back(obj);
			}

			bool isAThreadPrivateVariable(string& name);
			bool isAGlobalMemThreadPrivateVar(string& name);
			SourceLocation getLoc(string& name);
	};

	class OMPParallelDepth : public OMPObject {
		unsigned depth;
		vector<string> seqs;
		void init(Diagnostic &Diags, SourceLocation& loc, unsigned d);
		bool userSetDepth;
		void customDepth() { userSetDepth = true; }
		public:
		OMPParallelDepth(string depth, vector<string>& seq, Diagnostic &Diags, SourceLocation& Loc);

		OMPParallelDepth();
		OMPParallelDepth(unsigned d, Diagnostic &Diags, SourceLocation& loc);

		void setDepth(string d)
		{
			depth = atoi(d.c_str());
		}

		void setSeq(vector<string> seqs)
		{
			this->seqs = seqs;
		}

		bool isUserSetDepth() { return userSetDepth; }
		unsigned getDepth() { return depth; }
		vector<string> getSeq() { return seqs; }
	};

#define PRINT(__v__) {\
	for (unsigned i=0; i<__v__.size(); i++)\
	{\
		if (i > 0)\
		Out << ",";\
		__v__[i].print(Out);\
	}\
}



class OMPFor: public OMPObject {
	bool swap;
	bool isReduction;
	vector<OMPReductionObj> reducObjs;
	std::vector<OMPPrivate> privates;
	std::vector<OMPFirstPrivate> first_privates;
	std::vector<OMPCopyIn> copyins;
	std::vector<OMPSwapIndex> swapIndexs;
	std::vector<OMPMultIterIndex> multIterIndex;
	std::vector<OMPThreadPrivateObject> threadPrivates;
	std::vector<OMPTLSVariable> tlsVars;
	std::string schedule;
	OMPParallelDepth depth;
	bool enable_auto_tls_track;
	bool tls_check;

	public:
	OMPFor(bool red = false) { 
		type = omp_parallel_for; 
		swap = true;
		isReduction = red;
		enable_auto_tls_track = true;
		tls_check = true;
	}

	OMPFor operator=(OMPFor& rhs);


	void print(llvm::raw_ostream& Out);
	void set2Reduction();
	void enableTLSCheck() { tls_check = true; }
	void disableTLSCheck() { tls_check = false; }
	bool isTLSCheck() { return tls_check; }


	vector<OMPTLSVariable> getTLSVars() { return tlsVars; }
	bool isEnableAutoTLSTrack() { return enable_auto_tls_track; }
	vector<OMPMultIterIndex>& getMultIterIndex();
	void addMultIterIndex(OMPMultIterIndex& O);
	void addReductionVariable(OMPReductionObj& obj);
	vector<OMPReductionObj>& getReductionObjs();
	bool isACopyInVar(string& name);
	bool isReductionFor();
	bool getSwap();
	void setSwap(bool l);
	OMPParallelDepth& getParallelDepth();
	void addPrivateVariable(OMPPrivate &val);
	void addThreadPrivateVariable(OMPThreadPrivateObject& v);
	bool isAThreadPrivateVariable(string &v);
	void setParallelDepth(OMPParallelDepth depth);
	void addFirstPrivateVariable(OMPFirstPrivate &val);
	void addCopyInVariable(OMPCopyIn &val);
	void addTLSVariable(OMPTLSVariable &val);
	void addSwapIndex(OMPSwapIndex &val);
	void addSchedule(std::string sch);
	void print();
	std::vector<OMPSwapIndex> getSwapIndexs();
	std::vector<OMPPrivate> getPrivate();
	std::vector<OMPFirstPrivate> getFirstPrivate();
	std::vector<OMPCopyIn> getCopyIn();
	std::vector<OMPThreadPrivateObject> getThreadPrivate();
	std::string getSchedule();
	bool isVariablePrivate(std::string& v);
	bool isVariableFirstPrivate(std::string& v);
	bool isVariableCopyIns(std::string& v);
};


//OMP2OCL
enum Omp2OclType {
	omp2ocl_null = 0,
	omp2ocl_init = 1,
	omp2ocl_sync = 2,
	omp2ocl_flush = 3,
	omp2ocl_term = 4,
	omp2ocl_resetmlf = 5,
	omp2ocl_enablemlfrecord = 6,
	omp2ocl_disablemlfrecord = 7,
	omp2ocl_dumpmlf = 8,
	omp2ocl_hostflush = 9,
	omp2ocl_hostread = 10,
	omp2ocl_startprofile = 11,
	omp2ocl_stopprofile = 12,
	omp2ocl_dumpprofile = 13,
	omp2ocl_devread=14,
	omp2ocl_devwrite=15
};

class Omp2OclObj
{
	enum Omp2OclType type;
	SourceLocation Loc;
	public:
	Omp2OclObj(enum Omp2OclType t, SourceLocation Loc)
	{
		this->type = t;
	}

	int getType()
	{
		return type;
	}

	SourceLocation getLocation()
	{
		return Loc;
	}
};

class Omp2OclInit : public Omp2OclObj
{
	public:
		Omp2OclInit(SourceLocation Loc) : Omp2OclObj(omp2ocl_init, Loc) {}
};

class Omp2OclSync : public Omp2OclObj
{
	public:
		Omp2OclSync(SourceLocation Loc) : Omp2OclObj(omp2ocl_sync, Loc) {}
};

class Omp2OclFlush : public Omp2OclObj
{
	public:
		Omp2OclFlush(SourceLocation Loc) : Omp2OclObj(omp2ocl_flush, Loc) {}
};


class Omp2OclDevRead : public Omp2OclObj
{
	public:
		vector<string> vars;
		Omp2OclDevRead(SourceLocation Loc) : Omp2OclObj(omp2ocl_devread, Loc) {}
		void addVar(string name)
		{
			vars.push_back(name);
		}
		vector<string> getVars() { return vars; }
};

class Omp2OclDevWrite : public Omp2OclObj
{
	public:
		vector<string> vars;
		Omp2OclDevWrite(SourceLocation Loc) : Omp2OclObj(omp2ocl_devwrite, Loc) {}
		void addVar(string name)
		{
			vars.push_back(name);
		}
		vector<string> getVars() { return vars; }
};

class Omp2OclHostFlush : public Omp2OclObj
{
	public:
		vector<string> vars;
		Omp2OclHostFlush(SourceLocation Loc) : Omp2OclObj(omp2ocl_hostflush, Loc) {}
		void addVar(string name)
		{
			vars.push_back(name);
		}
		vector<string> getVars() { return vars; }
};

class Omp2OclHostRead : public Omp2OclObj
{
	public:
		vector<string> vars;
		Omp2OclHostRead(SourceLocation Loc) : Omp2OclObj(omp2ocl_hostread, Loc) {}
		void addVar(string name)
		{
			vars.push_back(name);
		}
		vector<string> getVars() { return vars; }
};

class Omp2OclTerm : public Omp2OclObj
{
	public:
		Omp2OclTerm(SourceLocation Loc) : Omp2OclObj(omp2ocl_term, Loc) {}
};

class Omp2OclStartProfile : public Omp2OclObj
{
	public:
		Omp2OclStartProfile(SourceLocation Loc) : Omp2OclObj(omp2ocl_startprofile, Loc) {}
};

class Omp2OclStopProfile : public Omp2OclObj
{
	public:
		Omp2OclStopProfile(SourceLocation Loc) : Omp2OclObj(omp2ocl_stopprofile, Loc) {}
};

class Omp2OclDumpProfile : public Omp2OclObj
{
	public:
		Omp2OclDumpProfile(SourceLocation Loc) : Omp2OclObj(omp2ocl_dumpprofile, Loc) {}
};

class Omp2OclResetMLF : public Omp2OclObj
{
	public:
		Omp2OclResetMLF(SourceLocation Loc) : Omp2OclObj(omp2ocl_resetmlf, Loc) {}
};

class Omp2OclDisableMLFRecord : public Omp2OclObj
{
	public:
		Omp2OclDisableMLFRecord(SourceLocation Loc) : Omp2OclObj(omp2ocl_disablemlfrecord, Loc) {}
};

class Omp2OclEnableMLFRecord : public Omp2OclObj
{
	public:
		Omp2OclEnableMLFRecord(SourceLocation Loc) : Omp2OclObj(omp2ocl_enablemlfrecord, Loc) {}
};

class Omp2OclDumpMLF : public Omp2OclObj
{
	public:
		Omp2OclDumpMLF(SourceLocation Loc) : Omp2OclObj(omp2ocl_dumpmlf, Loc) {}
};

class OCLLocalVar : public OMPObject
{
	string name;
	public:
	OCLLocalVar(string n)
	{
		type = ocl_localvar;
		name = n;
	}

	string getName()
	{
		return name;
	}
};


} /* end of namespace clang */



#endif
