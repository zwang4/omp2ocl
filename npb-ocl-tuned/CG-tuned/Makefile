SHELL=/bin/sh
BENCHMARK=cg
BENCHMARKU=CG

include ../config/make.def
include ../../common/ocl.def

OBJS = cg.o ocl_runtime.o  \
       ${COMMON}/c_print_results.o ${COMMON}/c_${RAND}.o ${COMMON}/c_timers.o ${COMMON}/c_wtime.o

include ../sys/make.common

${PROGRAM}: config ${OBJS}
	${CLINK} ${CLINKFLAGS} -o ${PROGRAM} ${OBJS} ${C_LIB}

ocl_runtime.o: ../../common/ocl_runtime.c ../../common/ocl_runtime.h
	${CCOMPILE} ${CFLAGS} ../../common/ocl_runtime.c

.c.o:
	${CCOMPILE} ${CFLAGS} $<

clean:
	- rm -f *.o *~ 
	- rm -f npbparams.h core



