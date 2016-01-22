SHELL=/bin/sh
BENCHMARK=ep
BENCHMARKU=EP

include ../config/make.def
include ../../common/ocl.def


OBJS =	ep.o ocl_runtime.o \
	${COMMON}/c_print_results.o ${COMMON}/c_timers.o ${COMMON}/c_${RAND}.o ${COMMON}/c_wtime.o

include ../sys/make.common

# npbparams.h is included by header.h
# The following rule should do the trick but many make programs (not gmake)
# will do the wrong thing and rebuild the world every time (because the
# mod time on header.h is not changed. One solution would be to 
# touch header.h but this might cause confusion if someone has
# accidentally deleted it. Instead, make the dependency on npbparams.h
# explicit in all the lines below (even though dependence is indirect). 

# header.h: npbparams.h

${PROGRAM}: config ${OBJS}
	${CLINK} ${CLINKFLAGS} -o ${PROGRAM} ${OBJS} ${C_LIB}

ocl_runtime.o: ../../common/ocl_runtime.c ../../common/ocl_runtime.h
	${CCOMPILE} ${CFLAGS} ../../common/ocl_runtime.c

.c.o:
	${CCOMPILE} ${CFLAGS} $<

ep.o:             ep.c  npbparams.h

clean:
	- rm -f *.o *~ mputil*
	- rm -f npbparams.h core
