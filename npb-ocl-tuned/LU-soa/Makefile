SHELL=/bin/sh
BENCHMARK=lu
BENCHMARKU=LU

include ../config/make.def
include ../../common/ocl.def


OBJS =	lu.o \
	${COMMON}/c_print_results.o ${COMMON}/c_timers.o ${COMMON}/c_wtime.o ocl_runtime.o

include ../sys/make.common

${PROGRAM}: config ${OBJS}
	${CLINK} ${CLINKFLAGS} -o ${PROGRAM} ${OBJS} ${C_LIB}

ocl_runtime.o: ../../common/ocl_runtime.c ../../common/ocl_runtime.h
	${CCOMPILE} ${CFLAGS} ../../common/ocl_runtime.c

.c.o:
	${CCOMPILE} ${CFLAGS} $<


lu.o:             lu.c  npbparams.h

clean:
	- rm -f *.o *~ mputil*
	- rm -f  npbparams.h core
