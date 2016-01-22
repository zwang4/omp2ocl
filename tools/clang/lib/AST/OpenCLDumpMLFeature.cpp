#include "clang/Omp2Ocl/OpenCLDumpMLFeature.h"
#include "clang/Omp2Ocl/OpenCLRoutine.h"

static void getUnresolveLoopVariables(StmtPicker& V, ASTContext& Context, ForStmt* Node)
{
	Stmt* S = Node->getInit();
	BinaryOperator* oc = dyn_cast<BinaryOperator>(S);
	V.Visit(oc->getRHS());

	S = Node->getCond();
	oc = dyn_cast<BinaryOperator>(S);
	V.Visit(oc->getRHS());

	S = Node->getInc();
	oc = dyn_cast<BinaryOperator>(S);

	if (oc)
		V.Visit(oc->getRHS());


}

void OpenCLDumpMLFeature::generateScriptForKernel(string kernel_name)
{
	llvm::raw_ostream& Out = (*OS);
#if 0
	Out << "cmd = \"echo \\\"#define ENABLE_" << kernel_name << "\\\" > parse.cl\"\n";
	Out << "os.system(cmd)\n";
	Out << "cmd = \"cat " << KF << " >> parse.cl\"\n";
	Out << "os.system(cmd)\n\n";
	Out << "cmd = feature_extractor + \" parse.cl \"\n";
	Out << "cmd = cmd + cmd_vars  + \" > features/" << kernel_name << ".\" + " << "dataset" << "\n";

	Out << "print cmd\n";
	Out << "os.system(cmd)\n\n";
#endif

	Out << "proc_feature(\"" << kernel_name << "\")\n";
}

//Collecting ML features for normal loops
void OpenCLDumpMLFeature::generateScriptForKernel(string kernel_name,ASTContext& Context, Stmt* Kernel)
{
	llvm::raw_ostream& Out = (*OS);
	Out << "proc_feature(\"" << kernel_name << "\")\n";

	#if 0
	Out << "cmd = \"echo \\\"#define ENABLE_" << kernel_name << "\\\" > parse.cl\"\n";
	Out << "os.system(cmd)\n";
	Out << "cmd = \"cat " << KF << " >> parse.cl\"\n";
	Out << "os.system(cmd)\n\n";
	#endif

	StmtPicker dr (llvm::nulls(), Context, NULL, Context.PrintingPolicy, 0);
	dr.Visit(Kernel);
	vector<ForStmt*> forStmts = dr.getForStmts();
	StmtPicker V (llvm::nulls(), Context, NULL, Context.PrintingPolicy, 0);

	for (unsigned i=0; i<forStmts.size(); i++)
	{
		getUnresolveLoopVariables(V, Context, forStmts[i]);
	}

	vector<DeclRefExpr*>& decls = V.getDecl();

	for (unsigned i=0; i<decls.size(); i++)
	{
		DeclRefExpr* DRE = decls[i];
		string name = DRE->getNameInfo().getAsString();
		bool found = false;
		
		for (unsigned j=0; j<unResolveVars.size(); j++)
		{
			if (name == unResolveVars[j]->getNameInfo().getAsString())
			{
				found = true;
				break;
			}	
		}

		if (!found)
		{
			unResolveVars.push_back(DRE);
		}
	}


#if 0
	Out << "cmd = feature_extractor + \" parse.cl \"\n";
	Out << "#Unsolved variables: \n";
	for (unsigned i=0; i<unResolveVars.size(); i++)
	{
		DeclRefExpr* DRE = unResolveVars[i];
		string name = DRE->getNameInfo().getAsString();
		Out << "cmd = cmd + \"" << name << " \"\n";
		Out << "cmd = cmd + str(" << name << + ") + \" \"\n";
	}
	
	Out << "cmd = cmd + cmd_vars  + \" > features/" << kernel_name << ".\" + " << "dataset" << "\n";

	Out << "print cmd\n";
	Out << "os.system(cmd)\n\n";
#endif

	for (unsigned i=0; i<unResolveVars.size(); i++)
	{
		addUnresolveVar(unResolveVars[i]);
	}
}

void OpenCLDumpMLFeature::addUnresolveVar(DeclRefExpr* e)
{
	string name = e->getNameInfo().getAsString();
	bool found = false;

	for (unsigned i=0; i<declareVars.size(); i++)
	{
		if (name == declareVars[i]->getNameInfo().getAsString())
		{
			found = true;
			break;	
		}
	}

	if (!found)
	{
		declareVars.push_back(e);
	}
}

void OpenCLDumpMLFeature::generateHeader(llvm::raw_ostream& Out)
{
	Out << "import os, sys\n\n";
	Out << "cl=\"" << KF << "\"\n";
	Out << "feature_extractor = \"feat-extract\"\n";
	Out << "dataset=\n";

	vector<string> classes;
	classes.push_back("S");
	classes.push_back("W");
	classes.push_back("A");
	classes.push_back("B");
	classes.push_back("C");

	Out << "#The following variables need to define\n";
	for (unsigned j=0; j<classes.size(); j++)
	{
		string c = classes[j];
		
		Out << "if (dataset == \"" << c << "\"):\n";
		for (unsigned i=0; i<declareVars.size(); i++)
		{
			string name = declareVars[i]->getNameInfo().getAsString();
			Out << "	" << name << " = \n";
		}
	}

	Out << "\n";

	Out << "os.system(\"mkdir -p features\")\n\n";

	Out << "cmd_vars = \" \"";

	for (unsigned i=0; i<declareVars.size(); i++)
	{
		string name = declareVars[i]->getNameInfo().getAsString();
		Out << " + \" " << name << " \" + str(" << name << ") ";
	}

	Out << "\n\n";

	Out << "def proc_feature(kernel):\n";
	Out << "   global cl, unresolve_vars\n";
	Out << "   f = open(\"parse.cl\", \"w\")\n";
	Out << "   fs = open(cl, \"r\")\n";
	Out << "   f.write(\"#define ENABLE_\" + kernel + \"\\n\")\n";
	Out << "   for line in fs:\n";
	Out << "      f.write(line)\n";
	Out << "   fs.close()\n";
	Out << "   f.close()\n";
	Out << "   cmd = feature_extractor + \" parse.cl \"\n";
	Out << "   cmd = cmd + cmd_vars + \" > features/\"  + kernel + \".\" + dataset\n";
	Out << "   print cmd\n";
	Out << "   os.system(cmd)\n";
	Out << "\n";
}

void OpenCLDumpMLFeature::flush()
{
	string errf;
	string file = KF;
	strReplace(file, ".cl", ".py");

	llvm::raw_fd_ostream PO(file.data(), errf);

	generateHeader(PO);
	OS->flush();
	PO << buf;
	
	PO.flush();
	PO.close();
}
