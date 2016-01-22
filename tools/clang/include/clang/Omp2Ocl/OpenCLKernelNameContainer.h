#ifndef __OPENCLKERNELNAMECONTAINER_H__
#define __OPENCLKERNELNAMECONTAINER_H__

#include <vector>
#include <string>

using namespace std;

namespace clang
{
	class OpenCLKernelNameContainer
	{
		static vector<string> kernelNames;
		public:
			static void addKernelName(string name);
			static vector<string>& getKernelNames()
			{
				return kernelNames;
			}	
	};
}

#endif
