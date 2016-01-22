#include "clang/Omp2Ocl/OpenCLKernel.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Stmt.h"
#include "clang/Omp2Ocl/OpenCLRoutine.h"

void globalVarIndex::init(unsigned ii, bool isP, bool isGTP, bool isL, string n)
{
	this->i = ii;
	this->isPointerAccess = isP;
	this->isGTP = isGTP;
	this->isLocalBuf = isL;
	this->name = n;
}

bool RenamedFuncInfo::isInExtendId(unsigned id)
{
	for (unsigned i=0; i<extendIndex.size(); i++)
	{
		//ZHENG FIXME: I SHOULD IMPLEMENT operator == for class extendVarIndex
		if (extendIndex[i].id == id)
			return true;
	}

	return false;
}

bool RenamedFuncInfo::isInExtendId(extendVarIndex& ei)
{
	for (unsigned i=0; i<extendIndex.size(); i++)
	{
		//ZHENG FIXME: I SHOULD IMPLEMENT operator == for class extendVarIndex
		if (extendIndex[i].id == ei.id)
			return true;
	}

	return false;
}

RenamedFuncInfo::RenamedFuncInfo(string& oriName, vector<globalVarIndex>& gA, vector<extendVarIndex>& eid, string& newName, bool hasGMT)
{
	this->origFuncName = oriName;
	this->globalArugIds = gA;
	this->newName = newName;
	this->extendIndex = eid;
	this->hasGlobalMemThreadPrivate = hasGMT;
	enable_spec = true;
	hasGenerated = false;
}

IndexStr::IndexStr(string s, Expr* e)
{
	str = s;
	trim(str);
	expr = e;
}
string PLoopParam::getName()
{
	return declRef->getNameInfo().getAsString();
}

ValueDecl* PLoopParam::getDecl()
{
	return declRef->getDecl();
}

string ArrayIndex::getName()
{
	return base->getNameInfo().getAsString();
}

ValueDecl* ArrayIndex::getDecl()
{
	return base->getDecl();
}

QualType OCLGlobalMemVar::getType() { 
	return v->getType(); 
}

bool OCLGlobalMemVar::canbeDeclareAsGlobal()
{
	//if (isGlobalThreadPrivate)
	//	return false;
	if (isFLevel)
		return false;

	return v->isDefinedOutsideFunctionOrMethod();
}


bool OCLGlobalMemVar::isDefinedOusideFunc()
{
	if (isFLevel)
		return false;

	return v->isDefinedOutsideFunctionOrMethod();
}
			
string ArrayAccessInfo::getName()
{
	return D->getNameAsString();
}
