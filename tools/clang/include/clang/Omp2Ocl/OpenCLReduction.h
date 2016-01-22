/*
 * This class provides opencl reduction support
 *
 */

#ifndef __OPENCLREDUCTION_H__
#define __OPENCLREDUCTION_H__

#include "clang/AST/StmtVisitor.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/PrettyPrinter.h"
#include "llvm/Support/Format.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/Omp2Ocl/OpenCLPrinter.h"
#include "clang/AST/GlobalVariablePicker.h"
#include "clang/AST/DeclPrinter.h"
#include "clang/AST/CallArgReviseAction.h"
#include "clang/AST/GlobalCallArgPicker.h"
#include "clang/Omp2Ocl/OpenCLKerenelSchedule.h"

#include <fstream>
#include<iostream>
#include<string>
#include<stdio.h>
#include <vector>


class OpenCLReduction
{
	OMPThreadPrivate threadPrivates;
	OpenCLKernelLoop* oclLoop;
	ASTContext& Context; 

	public:
		OpenCLReduction(ASTContext& C, OMPThreadsPrivate ot, OpenCLKernelLoop* loop);

};

#endif
