#include "clang/Omp2Ocl/OpenCLCopyInRoutine.h"
#include "clang/Omp2Ocl/OpenCLRoutine.h"
#include "clang/Omp2Ocl/OpenCLCommon.h"

void OpenCLCopyInRoutine::genLocalCopyInCode(llvm::raw_ostream& Out, ValueDecl* d, string passInName, string type_prefix)
{
	string name = d->getNameAsString();
	string type = getCononicalType(d);
	string gtype = getGlobalType(type);

	vector<unsigned> arrayDef = getArrayDef(type);

	if (isOCLPremitiveType(gtype))
	{
		string cast_type = "";
		if (type_prefix.length())
		{
			cast_type = type_prefix + " ";
		}

		Out << getVectorCopyInCode(gtype, passInName, name, arrayDef, 0, cast_type, type_prefix);
	}
	else
	{
		const QualType ctype = d->getType().getCanonicalType();
		RecordDecl* rd = OCLCommon::getRecordDecl(ctype);	
		if (!rd)
		{
			cerr << "I cannot handle copyin of " << name << "(Type=" << type << ")\n";
			exit(-1);
		}
		else
		{
			vector<QualType> structTypes;
			if (isVLoadable(rd, structTypes))
			{
				assert(structTypes.size() && "No element type found!");
				string elem_type = getGlobalType(structTypes[0].getAsString());
				if (type_prefix.length())
				{
					cerr << "Warning: I may not able to handle a type_prefix as: " << type_prefix << ", " << __FILE__ << ":" << __LINE__ << endl;
				}

				Out << generateVLoadForStructure(DEFAULT_COPYIN_TYPE, elem_type, structTypes.size(), passInName, name, arrayDef, 0, type_prefix);
			}
			else
			{
				cerr << "A scalar copyin implementation for " << name << "(Type=" << type << ") is needed.\n";
				exit(-1);
			}	
		}
	}
}

void OpenCLCopyInRoutine::generateLocalCopyInCode(CopyInBuffer& buf)
{
	ValueDecl* d = buf.d;
	string name = d->getNameAsString();
	string passInName = "__ocl_copyin_" + name;

	genLocalCopyInCode(Out, d, passInName);
}


//Generate copyin code for the copyin buffer that is declared as __global
void OpenCLCopyInRoutine::generateGlobalCopyInCode(CopyInBuffer& buf)
{
	ValueDecl* d = buf.d;
	string name = d->getNameAsString();
	string type = getCononicalType(d);
	string gtype = getGlobalType(type);

	vector<OpenCLNDRangeVar> GV = loop->getOclLoopIndexs();
	unsigned group_size = GV.size();
	vector<unsigned> arrayDef = getArrayDef(type);

	string passInName = "__ocl_copyin_" + name;
	if (isOCLPremitiveType(gtype))
	{
		Out << getVectorCopyInCode(gtype, passInName, name, arrayDef, group_size, "__global ");
	}
	else
	{
		const QualType ctype = d->getType().getCanonicalType();
		RecordDecl* rd = OCLCommon::getRecordDecl(ctype);	
		if (!rd)
		{
			cerr << "I cannot handle copyin of " << name << "(Type=" << type << "), because I cannot find its definition.\n";
			exit(-1);
		}
		else
		{
			vector<QualType> structTypes;
			if (isVLoadable(rd, structTypes))
			{
				assert(structTypes.size() && "No element type found!");
				string elem_type = getGlobalType(structTypes[0].getAsString());
				Out << generateVLoadForStructure(DEFAULT_COPYIN_TYPE, elem_type, structTypes.size(), passInName, name, arrayDef, group_size);
			}
			else
			{
				cerr << "A scalar copyin implementation for " << name << "(Type=" << type << ") is needed.\n";
				exit(-1);
			}	
		}
	}
}


void OpenCLCopyInRoutine::doIt()
{
	vector<CopyInBuffer>& copyInBuffers = loop->getCopyInBuffers();
	
	if (copyInBuffers.size())
	{
		Out << "	//-------------------------------------------\n";
		Out << "	// Copy in (START)\n";
		Out << "	//-------------------------------------------\n";
		for (unsigned i=0; i<copyInBuffers.size(); i++)
		{
			CopyInBuffer& buf = copyInBuffers[i];
			if (!buf.isTPGlobalBuf)
			{
				generateLocalCopyInCode(buf);	
			}
			else
			{
				generateGlobalCopyInCode(buf);		
			}

		}
	
		Out << "	//-------------------------------------------\n";
		Out << "	// Copy in (END)\n";
		Out << "	//-------------------------------------------\n";
	}
}

void OpenCLCopyInRoutine::declareMultiFactor()
{
	Out << " unsigned " << COPYIN_MULTI_FACTOR_NAME << " = (";
	vector<OpenCLNDRangeVar> GV = loop->getOclLoopIndexs();
	for (unsigned i=0; i<GV.size(); i++)
	{
		if (i > 0)
			Out << " * ";
		Out << "gsize_" << i;
	}

	Out << ");\n";
}

void OpenCLCopyInRoutine::declareAddOffset()
{
	Out << " unsigned " << COPYIN_ADD_OFFSET_NAME << " = (";
	vector<OpenCLNDRangeVar> GV = loop->getOclLoopIndexs();

	for (int i=GV.size()-1; i>=0; i--)
	{
		if ( i < (GV.size()-1) )
			Out << " + ";

		Out << "(gid_" << i;

		for (int j=i-1; j>=0; j--)
		{
			Out << " * gsize_" << j;

		}

		Out << ")";
	}

	Out << ");\n";
}

//Generate copyIn code
void OpenCLCopyInRoutine::declareCopyInBuffers()
{
	vector<OpenCLNDRangeVar> GV = loop->getOclLoopIndexs();

	unsigned ii = 0;

	//This kernel has a copyin buffer on the global memory
	//so that I need to print out the global sizes
	if (loop->hasGlobalCopyInBuf)
	{
		for (vector<OpenCLNDRangeVar>::iterator iter = GV.begin(); iter != GV.end(); iter++)
		{
			Out << "	unsigned gsize_" << ii << " = get_global_size(" << ii << ");\n";
			Out << "	unsigned gid_" << ii << " = get_global_id(" << ii << ");\n";
			ii++;
		}
	
		Out << "\n";	
		declareMultiFactor();
		declareAddOffset();
		Out << "\n";	
	}

#if 0	
	vector<CopyInBuffer>& copyInBuffers = loop->getCopyInBuffers();
	for (unsigned i=0; i<copyInBuffers.size(); i++)
	{
		ValueDecl* d = copyInBuffers[i].d;
		string name = d->getNameAsString();

		if (loop->isAGlobalMemThreadPrivateVar(name))
			continue;
		
		string type = getCononicalType(d);
		string gtype = getGlobalType(type);

		Out << gtype << " " << name;

		vector<unsigned> arrayDef = getArrayDef(type);
		for (unsigned j=0; j<arrayDef.size(); j++)
		{
			Out << "[" << arrayDef[j] << "]";
		}

		Out << ";\n";
	}
#endif
}
