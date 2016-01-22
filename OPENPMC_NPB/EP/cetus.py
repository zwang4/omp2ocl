import os, sys

cfiles=""
cmd = "rm -f cetus_out/ep.cu"
os.system(cmd)

classpath="/home/zwang4/workspace/omp2ocl/cetus-1.3/lib/cetus.jar;/usr/share/antlr-2.7.7/antlr.jar"
#options="-I../common -cudaMemTrOptLevel=4 -useParallelLoopSwap -useUnrollingOnReduction -useLoopCollapse -useGlobalGMalloc -useMallocPitch -useMatrixTranspose -assumeNonZeroTripLoops -shrdSclrCachingOnReg -shrdArryElmtCachingOnReg"
options="-I../common -cudaMemTrOptLevel=4 -useParallelLoopSwap -useUnrollingOnReduction -useLoopCollapse -useGlobalGMalloc -useMatrixTranspose -assumeNonZeroTripLoops -shrdArryElmtCachingOnReg"
cuda_options="-cudaThreadBlockSize=1024"
#cuda_options="-cudaThreadBlockSize=1024 -cudaMaxGridDimSize=65535 -cudaGridDimSize=1024"

for i in range(1,len(sys.argv)):
	cfiles = cfiles + " " + sys.argv[i]


cmd ="omp2gpu " + options  + " " + cuda_options + " " +cfiles
print cmd
os.system(cmd)

cmd = "cp npbparam.h cetus_out/."
os.system(cmd)
