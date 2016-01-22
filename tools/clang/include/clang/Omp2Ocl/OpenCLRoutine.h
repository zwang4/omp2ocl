#ifndef __OPENCLROUTINE_H__
#define __OPENCLROUTINE_H__

//This provides some common routines for the OMP 2 OCL tool

#include "clang/AST/Stmt.h"
#include "clang/AST/ASTContext.h"
#include "clang/Omp2Ocl/OpenCLKernel.h"

#include <string>
#include <vector>

using namespace std;

//
// Decide the simension of the array
// This is done by checking the type
//
//
string getVStructureName(unsigned vectorSize, unsigned idx);
void trim(string& str);
unsigned getArrayDimension(string type);
string getGlobalType(string type);
arraySubInfo getSubScripts(string& str);
vector<unsigned> getArrayDef(string type);
string calcArrayIndex(vector<unsigned>& defs, unsigned index, string subs);
void trim(string& str);
bool isAZeroInteger(string str);
string uint2String(unsigned i);
string mergeNameandType(string& type, string& name);
string getStringStmt(ASTContext& Context, Stmt* S);
vector<DeclRefExpr*> getDeclRefExprs(ASTContext& Context, Stmt* S);
string initValue(string type);
string getOpCodeFromString(string op);
string getOclVectorType(string type, unsigned vectorSize, bool isOCLKernel=false);
string reductVectorType2Scalar(string var, string op, unsigned vectorSize);
bool isOCLPremitiveType(string type);
string getVectorCopyInCode(string type, string& passInName, string& localName, vector<unsigned>& arrayInfo, unsigned group_size=0, string castType="__private ", string type_prefix="");
bool isVLoadable( DeclContext* DC, vector<QualType>& structTypes);
string generateVLoadForStructure(string copyInType, string elem_type, unsigned elem_num, string& passInName, string& localName, vector<unsigned>& arrayInfo, unsigned groupd_size=0, string type_prefix="");
void createOCLBuffer(llvm::raw_ostream& Out, ValueDecl* d);
void releaseOCLBuffer(llvm::raw_ostream& Out, ValueDecl* d);
string getCononicalType(ValueDecl* d);
string generateAlignCode(string pointerName, string alignSize, string sizeofType, string sizeofMult);
void createOCLBuffer(llvm::raw_ostream& Out, string& name, string& bufferName, string& sizeofType, string& bufferSize);
void releaseOCLBuffer(llvm::raw_ostream& Out, string& bufferName);
void strReplace(std::string& str, const std::string& pattern, const std::string& newStr);
void createOCLBuffer(llvm::raw_ostream& Out, ValueDecl* d, string name);
void releaseOCLBuffer(llvm::raw_ostream& Out, ValueDecl* d, string name);
string generateAlignCode(string pointerName, string alignSize, string size);
void createOCLBuffer(llvm::raw_ostream& Out, string& name, string& bufferName, string& bufferSize);
void createOCLBuffer(llvm::raw_ostream& Out, ValueDecl* d, string& hostName, string bufferName);
unsigned getArrayDimension(ValueDecl* d);
unsigned getArrayDimension(VarDecl* d);
string getStringExpr(ASTContext& Context, Expr* S);
#endif
