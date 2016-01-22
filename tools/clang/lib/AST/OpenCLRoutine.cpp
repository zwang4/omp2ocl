#include "clang/Omp2Ocl/OpenCLRoutine.h"
#include "clang/AST/StmtPrinter.h"
#include "clang/AST/DeclBase.h"
#include "clang/Omp2Ocl/OpenCLCommon.h"

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <stack>
#include <iostream>
#include <sstream>


using namespace std;
//trim a string
void trim(string& str)
{
	  string::size_type pos = str.find_last_not_of(' ');
	    if(pos != string::npos) {
			    str.erase(pos + 1);
				    pos = str.find_first_not_of(' ');
					    if(pos != string::npos) str.erase(0, pos);
						  }
		  else str.erase(str.begin(), str.end());
}

static string vectorGroupLoadPrefix(unsigned array_size, unsigned group_size) 
{
	string str;
	
	return str;
}

string generateAlignCode(string pointerName, string alignSize, string sizeofType, string sizeofMult)
{
	string str;
	llvm::raw_string_ostream OS(str);

	OS << "posix_memalign((void**)&" << pointerName << ", " << alignSize << ", sizeof(" << sizeofType << ") * " << sizeofMult << ");\n";

	OS << DYN_BUFFER_CHECK << "(" << pointerName << ", -1);\n";
	OS.flush();

	return str;
}

string generateAlignCode(string pointerName, string alignSize, string size)
{
	string str;
	llvm::raw_string_ostream OS(str);

	OS << "posix_memalign((void**)&" << pointerName << ", " << alignSize << ", " << size << ");\n";

	OS << DYN_BUFFER_CHECK << "(" << pointerName << ", -1);\n";
	OS.flush();

	return str;
}

static string vectorGroupLoadPostfix(unsigned total_size, string index, unsigned group_size) 
{
	string str = index;
	
	if (group_size)
	{
		str = "(" + str + " * ";
		str = str + COPYIN_MULTI_FACTOR_NAME;
		str = str + ") + ";

		str = str + COPYIN_ADD_OFFSET_NAME;
	}

	str = "(" + str + ")";

	return str;
}


bool isOCLPremitiveType(string type)
{
	if ( type == "unsigned" || type == "unsigned int" || type == "unsigned char" || 
			type == "float" || type == "double" || type == "long" || 
			type == "int" || type == "uchar" || type == "uint" || type == "ulong"){
		return true;
	}


	return false;
}

static void getArrayIndex(unsigned* va, vector<unsigned>& arrayInfo)
{
	unsigned arraySize = arrayInfo.size();
	int vi = arraySize-1;

	//I need to increment the upper dimension array index;
	if (va[vi] + 1 >= arrayInfo[vi])
	{
		while (vi >= 0 )
		{
			va[vi] = 0;
			vi--;
			if ((va[vi] + 1) < arrayInfo[vi])
			{
				va[vi]++;
				break;
			}
		}
	}
	else
	{
		va[vi]++;
	}
}

static void __genLoadStmt(llvm::raw_ostream& Out, string& passInName, string& localName, unsigned* va, unsigned times, 
		unsigned n, char** loadToken, vector<unsigned>& arrayInfo, unsigned& offset, string& indent, unsigned group_size, unsigned total_size, string type, string castType)
{

		string indexName = "_i_" + localName;
		string index = localName + "_offset";
		string vIndexArrayName = "_v_l_" + localName;

		Out << indent << "_vl" << n << " = vload" << n << "(" << 0 << ", " << passInName << ");\n";
		
#if LOCAL_SCALE_STORE
		unsigned m = 0;
		while (m < n)
		{
			Out << (indent) << localName;

			Out << "[" << vectorGroupLoadPrefix(1, group_size) << vectorGroupLoadPostfix(total_size, index, group_size) << "]";
			Out << " = " << "_vl" << n << "." << loadToken[m] << ";\n";
			Out << index << "++;\n";
			m++;
		}
#else
			Out << (indent) << "_vl_array_" << localName;

			Out << "[" << vIndexArrayName <<"]";
			Out << " = " << "_vl" << n << ";\n";
		
#endif

}

static void __generateLoadStmt(llvm::raw_ostream& Out, string& passInName, string& localName, unsigned* va, unsigned times, 
		unsigned n, char** loadToken, vector<unsigned>& arrayInfo, unsigned& offset, string& indent, unsigned group_size, unsigned total_size, string type, string castType)
{
	string indexName = "_i_" + localName;
	string index = localName + "_offset";
	string vIndexArrayName = "_v_l_" + localName;

	Out << "int _v_l_" + localName << " = __ocl_local_id;\n";//% " << times << ";\n";
	Out << "if (__ocl_local_id < " << times << "){\n";
	Out << "l_coeff = l_coeff + __ocl_local_id * " << n << ";\n";	
	__genLoadStmt(Out, passInName, localName, va, times, n, loadToken, arrayInfo, offset, indent, group_size, total_size, type, castType);
	Out << "}\n";

	Out << "#if 0\n";
	Out << "if (__ocl_dim_size < " << times << " && __ocl_local_id == 0) {\n";	
	Out << passInName << " = " << passInName << " + __ocl_dim_size * " << n << ";\n";
	Out << "for (unsigned " << indexName << "=__ocl_local_id; " << indexName << " < " << times << "; " << indexName << "++) {\n";
	__genLoadStmt(Out, passInName, localName, va, times, n, loadToken, arrayInfo, offset, indent, group_size, total_size, type, castType);
	Out << passInName << " = " << passInName << " + " << n << ";\n";
	Out << vIndexArrayName << "++;\n";
	Out << "}\n";
	Out << "}\n";
	Out << "#endif\n";
}


static QualType getDeclType(Decl* D) {
	if (TypedefDecl* TDD = dyn_cast<TypedefDecl>(D))
		return TDD->getUnderlyingType();
	if (ValueDecl* VD = dyn_cast<ValueDecl>(D))
		return VD->getType();
	return QualType();
}

//whether a data structure is vloadable
bool isVLoadable( DeclContext* DC, vector<QualType>& structTypes)
{
	vector<QualType> types;

	for (DeclContext::decl_iterator D = DC->decls_begin(), DEnd = DC->decls_end();
			D != DEnd; ++D) {

		if (isa<ObjCIvarDecl>(*D))
		{
			return false;
		}

		QualType CurDeclType = getDeclType(*D);
		types.push_back(CurDeclType);
	}

	string first;
	if (types.size())
	{
		first = types[0].getAsString();
		if (!isOCLPremitiveType(first))
		{
			return false;
		}
	}

	for (unsigned i=1; i<types.size(); i++)
	{
		if (first == types[i].getAsString())
			continue;
		else
			return false;
	}

	structTypes= types;

	return true;
}

string generateVLoadForStructure(string copyInType, string elem_type, unsigned elem_num, string& passInName, 
		string& localName, vector<unsigned>& arrayInfo, unsigned group_size, string type_prefix)
{
	string result;
	llvm::raw_string_ostream Out(result);
	string storeType = "__global ";


	string indent = "	";
	string LPointerName = "__ocl_p_" + localName;
	string PPointerName = "__ocl_copyin_p_" + localName;


	Out << copyInType << " " << elem_type << "* " << PPointerName << " =  " << passInName << ";\n";
	if (group_size)
	{
		Out << storeType;
	}

	Out << elem_type << "* " <<  LPointerName << " = " << localName;

	Out << ";\n\n";

	unsigned int total_size = 1;
	for (unsigned i=0; i<arrayInfo.size(); i++)
	{
		total_size = total_size * arrayInfo[i];
	}

	total_size = total_size * elem_num;

	vector<unsigned> newArrayInfo;
	newArrayInfo.push_back(total_size);

	Out.flush();

	result = result + getVectorCopyInCode(elem_type, PPointerName, LPointerName, newArrayInfo, group_size, storeType, type_prefix);

	return result;
}

string getVectorCopyInCode(string type, string& passInName, string& localName, vector<unsigned>& arrayInfo, unsigned group_size, string castType, string type_prefix)
{
	string result;
	llvm::raw_string_ostream Out(result);

	if (!isOCLPremitiveType(type))
	{	
		cerr << "Vector Load for " << type 
			<< " is not supported. A Scalar load implementation is needed!" << endl;
		exit(-1);
	}

	string indent = "	";

	unsigned int total_size = 1;
	for (unsigned i=0; i<arrayInfo.size(); i++)
	{
		total_size = total_size * arrayInfo[i];
	}
	
	unsigned u16 = 0;
	unsigned u8 = 0;
	unsigned u4 = 0;
	unsigned u2 = 0;
	unsigned tmp = 0;

	if (DEFAULT_LOAD_VECTOR == 16)
	{
		u16 = total_size / 16;
		tmp = total_size % 16;

		u8 = tmp / 8;
		tmp = tmp % 8;

		u4 = tmp / 4;
		tmp = tmp % 4;

		u2 = tmp / 2;
		tmp = tmp % 2;
	}
	else
		if (DEFAULT_LOAD_VECTOR == 8)
		{
			u8 = total_size / 8;
			tmp = total_size % 8;

			u4 = tmp / 4;
			tmp = tmp % 4;

			u2 = tmp / 2;
			tmp = tmp % 2;
		}
		else
			if (DEFAULT_LOAD_VECTOR == 4)
			{
				u4 = total_size / 4;
				tmp = total_size % 4;

				u2 = tmp / 2;
				tmp = tmp % 2;
			}
			else
				if (DEFAULT_LOAD_VECTOR == 2)
				{
					u2 = total_size / 2;
					tmp = total_size % 2;
				}

	Out << "{\n";

	//Declare vector values
	//FIXED ME: I SHOULD USE MACRO HERE
	int ts = 0;
	if (u16)
	{
		Out << indent << type << "16 _vl16;\n";
		ts = 16;
	}

	if (u8)
	{
		Out << (indent) << type << "8 _vl8;\n";
		ts = 8;
	}

	if (u4)
	{
		Out << (indent) << type << "4 _vl4;\n";
		ts = 4;
	}

	if (u2)
	{
		Out << (indent) << type << "2 _vl2;\n";
		ts = 2;
	}

	if (ts > 0)
	{
		Out << (indent) << "__local " << type << ts << "* _vl_array_" << localName << "=";
		Out << "(__local " << type << ts << "*)" << localName << ";\n";
	}

	Out << "\n";
	
	//FULLY UNLOAD
	unsigned offset=0;
	unsigned arraySize = arrayInfo.size();
	unsigned *va = new unsigned[arraySize];

	for (unsigned i=0; i<arraySize; i++)
	{
		va[i] = 0;
	}
	va[arraySize-1] = -1;

	char *v16[] = {"s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "sa", "sb", "sc", "sd", "se", "sf"};
	char *v8[] = {"s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7"};
	char *v4[] = {"x", "y", "z", "w"};
	char *v2[] = {"x", "y"};

	if (arrayInfo.size() > 1)
	{
		Out << type_prefix + " " + type << "* ocl_" << localName << "_ci = &" << localName;

		for (unsigned i=0; i<arrayInfo.size(); i++)
		{
			Out << "[0]";
		}
		
		Out << ";\n";
		//I have convert it to a 1 dimension array
		//so I chang its name
		localName = "ocl_" + localName + "_ci";
	}

	vector<unsigned> newAi;
	newAi.push_back(total_size);	
	
	Out << "unsigned " << localName << "_offset = 0;\n";

	if (u16)
	{
		//16-bit load
		__generateLoadStmt(Out, passInName, localName, va, u16, 16, v16, newAi, offset, indent, group_size, total_size, type, castType);
	}
	if (u8)
	{
		__generateLoadStmt(Out, passInName, localName, va, u8, 8, v8, newAi, offset, indent, group_size, total_size, type, castType);
	}

	if (u4)
	{
		__generateLoadStmt(Out, passInName, localName, va, u4, 4, v4, newAi, offset, indent, group_size, total_size, type, castType);
	}

	if (u2)
	{
		__generateLoadStmt(Out, passInName, localName, va, u2, 2, v2, newAi, offset, indent, group_size, total_size, type, castType);
	}

	if (tmp)
	{
		getArrayIndex(va, arrayInfo);
		
		char buf[64];
		snprintf(buf, 64, "%u", total_size-1);
		string str = buf;
		Out << "if(__ocl_local_id == 0){\n";
		Out << indent << localName << "[" << vectorGroupLoadPrefix(1, group_size) << vectorGroupLoadPostfix(total_size, str, group_size) << "]";
		Out << " = " << passInName << "[0];\n\n";
		Out << "}\n";
	}	

	Out << "}\n";

	delete []va;
	Out.flush();

	return result;
}

//get vectorize type, e.g.g double4
string getOclVectorType(string type, unsigned vectorSize, bool isOCLKernel)
{
	char buf[64];
	snprintf(buf, 64, "%u", vectorSize);

	string r;
	if (!isOCLKernel)
	{
		r = "cl_";
	}

	r = r + type + buf;

	return r;
}

string getVStructureName(unsigned vectorSize, unsigned idx)
{
		vector<string> subV;
		string result;

		if (vectorSize == 2)
		{
			subV.push_back("x");
			subV.push_back("y");
		}
		else
			if (vectorSize == 4)
			{
				subV.push_back("x");
				subV.push_back("y");
				subV.push_back("z");
				subV.push_back("w");
			}
			else
				if (vectorSize == 8)
				{
					subV.push_back("s0");
					subV.push_back("s1");
					subV.push_back("s2");
					subV.push_back("s3");
					subV.push_back("s4");
					subV.push_back("s5");
					subV.push_back("s6");
					subV.push_back("s7");

				}
				else
					if (vectorSize == 16)
					{
						subV.push_back("s0");
						subV.push_back("s1");
						subV.push_back("s2");
						subV.push_back("s3");
						subV.push_back("s4");
						subV.push_back("s5");
						subV.push_back("s6");
						subV.push_back("s7");
						subV.push_back("s8");
						subV.push_back("s9");
						subV.push_back("sa");
						subV.push_back("sb");
						subV.push_back("sc");
						subV.push_back("sd");
						subV.push_back("se");
						subV.push_back("sf");
					}

		assert((subV.size() > idx) && "INVALID IDX");
		result = subV[idx];

		return result;
}

//Reduce a vector type to a scalar
string reductVectorType2Scalar(string var, string op, unsigned vectorSize)
{
	string red;
	vector<string> subV;

	if (vectorSize == 1)
	{
		return var;
	}
	else
		if (vectorSize == 2)
		{
			subV.push_back("x");
			subV.push_back("y");
		}
		else
			if (vectorSize == 4)
			{
				subV.push_back("x");
				subV.push_back("y");
				subV.push_back("z");
				subV.push_back("w");
			}
			else
				if (vectorSize == 8)
				{
					subV.push_back("s0");
					subV.push_back("s1");
					subV.push_back("s2");
					subV.push_back("s3");
					subV.push_back("s4");
					subV.push_back("s5");
					subV.push_back("s6");
					subV.push_back("s7");

				}
				else
					if (vectorSize == 16)
					{
						subV.push_back("s0");
						subV.push_back("s1");
						subV.push_back("s2");
						subV.push_back("s3");
						subV.push_back("s4");
						subV.push_back("s5");
						subV.push_back("s6");
						subV.push_back("s7");
						subV.push_back("s8");
						subV.push_back("s9");
						subV.push_back("sa");
						subV.push_back("sb");
						subV.push_back("sc");
						subV.push_back("sd");
						subV.push_back("se");
						subV.push_back("sf");
					}

	for (unsigned i=0; i<subV.size(); i++)
	{
		if (i > 0)
			red = red + op;
		red = red + var + "." + subV[i];
	}

	red = "(" + red + ")";

	return red;
}

string initValue(string type)
{
	string result;

	if (type == "double")
	{
		result = "0.0";
	}
	else if (type == "float")
	{
		result = "0.0";
	}
	else
	{
		result = "0";
	}

	return result;
}

string getOpCodeFromString(string op)
{
	string r;
	if (op == "plus")
	{
		r = "+";
	}
	else
	{
		cerr << "Unknow op=" << op << endl;
		exit(-1);
	}

	return r;
}

string getStringExpr(ASTContext& Context, Expr* S)
{
	string buf;
	llvm::raw_string_ostream O(buf);
	StmtPrinter op(O, Context, NULL, Context.PrintingPolicy, -4);
	op.Visit(S);
	O.flush();

	return buf;
}

string getStringStmt(ASTContext& Context, Stmt* S)
{
	string buf;
	llvm::raw_string_ostream O(buf);
	StmtPrinter op(O, Context, NULL, Context.PrintingPolicy, -4);
	op.Visit(S);
	O.flush();

	return buf;
}

vector<DeclRefExpr*> getDeclRefExprs(ASTContext& Context, Stmt* S)
{
	string buf;
	llvm::raw_string_ostream O(buf);
	StmtPicker op(O, Context, NULL, Context.PrintingPolicy, -4);
	op.Visit(S);
	O.flush();

	return op.getDecl();
}

vector<unsigned> getArrayDef(string type)
{
	vector<unsigned> subs;
	string str="";
	unsigned i;

	for (i=0; i<type.length(); i++)
	{
		if (type[i] == '[')
			break;
	}

	for (; i<type.length(); i++)
	{
		if (type[i] == '[')
		{
			continue;
		}
		else
			if (type[i] == ']')
			{
				unsigned n = atoi(str.c_str());
				str="";
				subs.push_back(n);
				continue;
			}

		str = str + type[i];
	}

	return subs;
}

string mergeNameandType(string& type, string& name)
{
	string result = getGlobalType(type) + " " + name;

	if (type.find('[') != string::npos)
	{
		vector<unsigned> subs = getArrayDef(type);
		for (unsigned i=0; i<subs.size(); i++)
		{
			char buf[64];
			snprintf(buf, 64, "%u", subs[i]);
			result = result + "[";
			result = result + buf;
			result = result + "]";
		}
	}

	return result;

}

string getCononicalType(ValueDecl* d)
{
	string type = d->getType().getAsString();

	if (ParmVarDecl *Parm = dyn_cast<ParmVarDecl>(d))
	{
		type = Parm->getOriginalType().getAsString();
	}

	return type;
}

string calcArrayIndex(vector<unsigned>& defs, unsigned index, string subs)
{
	string results;
	string ss="";

	assert(defs.size() > 0 && "zero array");
	assert(defs.size() > index && "over subscript of an array");


	for (unsigned i=(index+1); i<defs.size(); i++)
	{
		char buf[64];

		snprintf(buf, 64, "%u", defs[i]);

		if (i > (index + 1))
			ss = ss + "*";

		ss = ss +  buf;
	}

	results = ss;
	results = "(" + results + ")";
	results = "( " + results + "*(" + subs + ") )";

	return results;
}

unsigned getArrayDimension(VarDecl* d)
{
	string type = getCononicalType(d);
	return getArrayDimension(type);
}

unsigned getArrayDimension(ValueDecl* d)
{
	string type = getCononicalType(d);
	return getArrayDimension(type);
}

unsigned getArrayDimension(string type)
{
	if ((type.find('[') == string::npos &&
	      type.find('*') == string::npos
	   ))
	{
		return 0;
	}

	unsigned dim = 0;
	unsigned starN = 0;
	for (unsigned i=0; i<type.length(); i++)
	{
		if (type[i] == '[')
			dim++;
		else
		if (type[i] == '*')
		{
			dim++;
			starN++;
		}
	}

	return dim;
}

string getGlobalType(string type)
{
	string argu="";

	for (size_t i=0; i<type.length(); i++)
	{
		if (type[i] == ' ' && argu != "struct")
		{
			break;
		}

		argu = argu + type[i];
	}

	return argu;
}


static string PopFromStack(stack<char>& tokenStack)
{
	stack<char> results;
	char c;

	while(tokenStack.size())	
	{
		c = tokenStack.top();
		tokenStack.pop();

		results.push(c);

		if (c == '[')
		{
			break;
		}

	}

	//reverse the results
	string ss="";
	while(results.size())
	{
		ss = ss + results.top();
		results.pop();
	}

	return ss;
}

/*!
 * The cunciton collect the subscriptions of an array
 *
 */
arraySubInfo getSubScripts(string& str)
{
	string name = "";
	stack<char> tokenStack;
	unsigned int i;
	arraySubInfo info;

	for (i=0; i<str.size(); i++)
	{
		if (str[i] == '[')
			break;

		name = name + str[i];
	}

	info.name = name;

	unsigned start = i;
	unsigned end = start;

	for ( ; i<str.size(); i++)
	{
		char c = str[i];
		if (c == ']')
		{
			string pc = PopFromStack(tokenStack);
			//add ']'
			end = end + 1;

			//add the push out stuff
			end = end + pc.length();

			if (!tokenStack.size())
			{
				//skip the first '['
				start++;
				//skip the last ']'
				info.subExprs.push_back(str.substr(start, end-start-1));
				start = end;
			}			

		}
		else
		{
			tokenStack.push(c);
		}
	}
#if 0
	if (tokenStack.size()  != 0)
	{
		cerr << "I am afraid something is wrong with the array: " << str << endl;
		cerr << "The extracted info is:" << endl;
		for (unsigned i=0; i<info.subExprs.size(); i++)
		{
			cerr << info.subExprs[i] << " ";
		}
		cerr << endl;

		cerr << "left:" << endl;

		while(tokenStack.size())
		{
			cerr << tokenStack.top() << " ";
			tokenStack.pop();
		}

		cerr << endl;
	}
#endif
	return info;
}


bool isAZeroInteger(string str)
{
	trim(str);
	if (str == "0" || str == "0;" || str == "0;\n")
		return true;
	else
		return false;
}

string uint2String(unsigned i)
{
	char buf[64];
	snprintf(buf, 64, "%u", i);

	string s = buf;
	return s;
}

void releaseOCLBuffer(llvm::raw_ostream& Out, ValueDecl* d)
{
	string indent = "	";
	string name = d->getNameAsString();
	Out << indent << " oclReleaseBuffer(__ocl_buffer_" << name << ");\n";
}


void releaseOCLBuffer(llvm::raw_ostream& Out, ValueDecl* d, string name)
{
	string indent = "	";
	Out << indent << " oclReleaseBuffer(__ocl_buffer_" << name << ");\n";
}


void releaseOCLBuffer(llvm::raw_ostream& Out, string& bufferName)
{
	Out << "	oclReleaseBuffer(" << bufferName << ");\n";
}

void createOCLBuffer(llvm::raw_ostream& Out, string& name, string& bufferName, string& sizeofType, string& bufferSize)
{
	string indent = "	";

	Out << indent << bufferName << " = oclCreateBuffer( " << name << ", sizeof(" << sizeofType << ") * " << bufferSize << ");\n";

	Out << DYN_BUFFER_CHECK << "(" << bufferName << ", -1);\n";
}


void createOCLBuffer(llvm::raw_ostream& Out, string& name, string& bufferName, string& bufferSize)
{
	string indent = "	";

	Out << indent << bufferName << " = oclCreateBuffer( " << name << ", " << bufferSize << ");\n";

	Out << DYN_BUFFER_CHECK << "(" << bufferName << ", -1);\n";
}

void createOCLBuffer(llvm::raw_ostream& Out, ValueDecl* d, string& hostName, string bufferName)
{
	QualType T = d->getType();

	if (ParmVarDecl *Parm = dyn_cast<ParmVarDecl>(d))
		T = Parm->getOriginalType();

	string type = T.getAsString();
	string indent = "	";

	string gtype = getGlobalType(type);
	vector<unsigned> dims = getArrayDef(type); 	

	Out << indent << bufferName << " = oclCreateBuffer( " << hostName << ", (";
	for (unsigned i=0; i<dims.size(); i++)
	{
		if (i) 
			Out << "*";

		Out << dims[i];
	}

	Out << ") * sizeof(" << gtype << ") );\n";

	Out << indent << DYN_BUFFER_CHECK << "(" << bufferName << ", -1);\n";
}

void createOCLBuffer(llvm::raw_ostream& Out, ValueDecl* d, string name)
{
	QualType T = d->getType();

	if (ParmVarDecl *Parm = dyn_cast<ParmVarDecl>(d))
		T = Parm->getOriginalType();

	string type = T.getAsString();
	string indent = "	";

	string gtype = getGlobalType(type);
	vector<unsigned> dims = getArrayDef(type); 	

	Out << indent << "__ocl_buffer_" << name << " = oclCreateBuffer( " << name << ", (";
	for (unsigned i=0; i<dims.size(); i++)
	{
		if (i) 
			Out << "*";

		Out << dims[i];
	}

	Out << ") * sizeof(" << gtype << ") );\n";

	Out << indent << DYN_BUFFER_CHECK << "(__ocl_buffer_" << name << ", -1);\n";
}

void createOCLBuffer(llvm::raw_ostream& Out, ValueDecl* d)
{
	string name = d->getNameAsString();
	QualType T = d->getType();

	if (ParmVarDecl *Parm = dyn_cast<ParmVarDecl>(d))
		T = Parm->getOriginalType();

	string type = T.getAsString();
	string indent = "	";

	string gtype = getGlobalType(type);
	vector<unsigned> dims = getArrayDef(type); 	

	Out << indent << "__ocl_buffer_" << name << " = oclCreateBuffer( " << name << ", (";
	for (unsigned i=0; i<dims.size(); i++)
	{
		if (i) 
			Out << "*";

		Out << dims[i];
	}

	Out << ") * sizeof(" << gtype << ") );\n";

	Out << indent << DYN_BUFFER_CHECK << "(__ocl_buffer_" << name << ", -1);\n";
}

void strReplace(std::string& str, const std::string& pattern, const std::string& newStr)
{
	size_t pos = 0;
	while((pos = str.find(pattern, pos)) != std::string::npos)
	{
		str.replace(pos, pattern.length(), newStr);
		pos += newStr.length();
	}
}
