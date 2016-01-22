#include "ocl_runtime.h"
#ifndef __OCL_DEF_H__
#define __OCL_DEF_H__

#ifndef likely
#define likely(x)       	__builtin_expect((x),1)
#endif
#ifndef unlikely
#define unlikely(x)       	__builtin_expect((x),0)
#endif

#define OCL_NEAREST_MULTD(a, n) { do { if (a < n) {a = 1;} else {if (a % n) a = (a / n) + 1; else a = a / n;}  } while(0); }
#ifdef DEBUG
#define DYN_BUFFER_CHECK(__name__,__line__) {\
if (unlikely (!__name__))\
{\
	fprintf (stderr,\
			"Failed to create the ocl buffer for %s at line: %d\n", #__name__, __line__);\
	exit (-1);\
}\
}
#else
#define DYN_BUFFER_CHECK(__name__,__line__) {}
#endif

#ifdef DEBUG
#define DYN_PROGRAM_CHECK(__name__) {\
if (unlikely (!__name__))\
{\
	fprintf (stderr,\
			"Failed to create the ocl kernel handle for %s \n", #__name__);\
	exit (-1);\
}\
}
#else
#define DYN_PROGRAM_CHECK(__name__) {}
#endif

#define CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_name_,__p_buf_name__,__buffer_name__,__buffer_size__,__type__) {\
	if (!__ocl_buffer_##__buffer_name__) {\
	if ((__type__*)__p_buf_name__ != (__type__*)__buffer_name__)\
	{\
		__ocl_buffer_name_ =\
			oclCreateBuffer (__buffer_name__, __buffer_size__ * sizeof (__type__));\
			DYN_BUFFER_CHECK (__ocl_buffer_name_, -1);\
			__p_buf_name__ = (__type__ *)__buffer_name__;\
		__oclLVarBufferList_t* p = malloc(sizeof(__oclLVarBufferList_t));\
			DYN_BUFFER_CHECK(p,__LINE__);\
			p->buf = __ocl_buffer_name_;\
			p->next = __ocl_lvar_buf_header;\
			__ocl_lvar_buf_header = p;\
	}\
	__ocl_buffer_##__buffer_name__ = __ocl_buffer_name_;\
	}\
}

#define CREATE_REDUCTION_STEP1_BUFFER(__buffer_size_keeper__,__buffer_size__,__ocl_buffer_handle__,__type__) {\
	        if (__buffer_size_keeper__ < __buffer_size__)\
	        {\
			                if (__buffer_size_keeper__ > 0)\
			                {\
						                        oclSync ();\
						                        oclReleaseBuffer (__ocl_buffer_handle__);\
						                }\
			                __ocl_buffer_handle__ =\
			                        oclCreateBuffer (NULL, sizeof (__type__) * __buffer_size__);\
			                DYN_BUFFER_CHECK (__ocl_buffer_handle__, -1);\
			                __buffer_size_keeper__ = __buffer_size__;\
			        }\
}

#define CREATE_REDUCTION_STEP2_BUFFER(__buffer_size_keeper__,__buffer_size__,__aligned_size__,__ocl_buffer_handle__,__buffer__,__type__) {\
if (__buffer_size_keeper__ < sizeof (__type__) * __buffer_size__)\
{\
	if (__buffer_size_keeper__ > 0)\
	{\
		oclSync ();\
			oclReleaseBuffer (__ocl_buffer_handle__);\
			free (__buffer__);\
	}\
	posix_memalign ((void **) &__buffer__, __aligned_size__,\
			sizeof (__type__) * __buffer_size__);\
		DYN_BUFFER_CHECK (__buffer__, -1);\
		__ocl_buffer_handle__ =\
		oclCreateBuffer (__buffer__,\
				sizeof (__type__) * __buffer_size__);\
		DYN_BUFFER_CHECK (__ocl_buffer_handle__, -1);\
		__buffer_size_keeper__ = sizeof (__type__) * __buffer_size__;\
}\
}

#define DECLARE_LOCALVAR_OCL_BUFFER(__variable_name__,__type__,__size__) \
	ocl_buffer * __ocl_buffer_##__variable_name__;\
	__ocl_buffer_##__variable_name__ = oclCreateBuffer(__variable_name__, sizeof(__type__)*__size__);\
	DYN_BUFFER_CHECK(__ocl_buffer_##__variable_name__,-1);\
	{\
		__oclLVarBufferList_t* p = malloc(sizeof(__oclLVarBufferList_t));\
			DYN_BUFFER_CHECK(p,__LINE__);\
			p->buf = __ocl_buffer_##__variable_name__;\
			p->next = __ocl_lvar_buf_header;\
			__ocl_lvar_buf_header = p;\
	}

#define RELEASE_LOCALVAR_OCL_BUFFERS() {\
	      	__oclLVarBufferList_t* header = __ocl_lvar_buf_header;\
	        while (header)\
	        {\
		                __oclLVarBufferList_t* p = header;\
			        header = header->next;\
			        oclReleaseBuffer(p->buf);\
			        free(p);\
		}\
}

#ifdef PROFILING
#define PROFILE_LOCALVAR_OCL_BUFFERS(__buffer__,__prof__) {\
	      	__oclLVarBufferList_t* header = __ocl_lvar_buf_header;\
	        while (header)\
	        {\
		                __oclLVarBufferList_t* p = header;\
			        header = header->next;\
				__buffer__ += oclDumpBufferProfiling (p->buf, __prof__);\
		}\
}
#endif

#define SYNC_LOCALVAR_OCL_BUFFERS() {\
	      	__oclLVarBufferList_t* header = __ocl_lvar_buf_header;\
	        while (header)\
	        {\
		                __oclLVarBufferList_t* p = header;\
			        header = header->next;\
				oclHostWrites(p->buf);\
		}\
}
#define REDUCTION_STEP1_MULT_NDRANGE() \
      	size_t __ocl_buf_size = __ocl_act_buf_size;\
            /*make sure the buffer length is multipled by (GROUP_SIZE * VECTOR_SIZE)*/\
            unsigned mulFactor = (GROUP_SIZE * 4);\
            __ocl_buf_size =\
                (__ocl_buf_size <\
				                  mulFactor) ? mulFactor : __ocl_buf_size;\
            __ocl_buf_size =\
                ((__ocl_buf_size / mulFactor) * mulFactor);\
            if (__ocl_buf_size < __ocl_act_buf_size) {\
				                __ocl_buf_size += mulFactor;\
		    }

#define CREATE_THREAD_PRIVATE_BUF(__buf__,__ocl_buf__,__type__,__size__,__align_size__) {\
        size_t buf_size = sizeof(__type__) * (__size__);\
        if (__buf__##_length < buf_size) {\
	                if (__buf__) {\
	                        oclSync();\
	                        oclReleaseBuffer(__ocl_buf__);\
	                        free(__buf__);\
	                }\
	                posix_memalign((void **)\
                                &__buf__, __align_size__, __size__);\
			                DYN_BUFFER_CHECK(__buf__, __LINE__);\
			                __ocl_buf__ = oclCreateBuffer(__buf__, buf_size);\
				                DYN_BUFFER_CHECK(__buf__, __LINE__);\
			                __buf__##_length = buf_size;\
			        }\
;}
#define GROUP_SIZE	128
#define DEFAULT_ALIGN_SIZE 16
//#define OCL_RELEASE_GTP_BUFFERS_IMMEDIATE /* if defined, __global threadprivate buffers will be released immediately after each use. This may somehow alleviate the memory pressure */

typedef struct __oclLVarBufferList {
	ocl_buffer *buf;
	struct __oclLVarBufferList *next;
} __oclLVarBufferList_t;

static __oclLVarBufferList_t *__ocl_lvar_buf_header = NULL;

static ocl_program *__ocl_program;

/** global data structures in : sp.c (BEGIN)*/
/** global data structures in : sp.c (END)*/
//OCL KERNELS (BEGIN)
static ocl_kernel *__ocl_add_0;
static ocl_kernel *__ocl_lhsinit_0;
static ocl_kernel *__ocl_lhsinit_1;
static ocl_kernel *__ocl_lhsx_0;
static ocl_kernel *__ocl_lhsx_1;
static ocl_kernel *__ocl_lhsx_2;
static ocl_kernel *__ocl_lhsx_3;
static ocl_kernel *__ocl_lhsx_4;
static ocl_kernel *__ocl_lhsx_5;
static ocl_kernel *__ocl_lhsy_0;
static ocl_kernel *__ocl_lhsy_1;
static ocl_kernel *__ocl_lhsy_2;
static ocl_kernel *__ocl_lhsy_3;
static ocl_kernel *__ocl_lhsy_4;
static ocl_kernel *__ocl_lhsy_5;
static ocl_kernel *__ocl_lhsz_0;
static ocl_kernel *__ocl_lhsz_1;
static ocl_kernel *__ocl_lhsz_2;
static ocl_kernel *__ocl_lhsz_3;
static ocl_kernel *__ocl_lhsz_4;
static ocl_kernel *__ocl_lhsz_5;
static ocl_kernel *__ocl_ninvr_0;
static ocl_kernel *__ocl_pinvr_0;
static ocl_kernel *__ocl_compute_rhs_0;
static ocl_kernel *__ocl_compute_rhs_1;
static ocl_kernel *__ocl_compute_rhs_2;
static ocl_kernel *__ocl_compute_rhs_3;
static ocl_kernel *__ocl_compute_rhs_4;
static ocl_kernel *__ocl_compute_rhs_5;
static ocl_kernel *__ocl_compute_rhs_6;
static ocl_kernel *__ocl_compute_rhs_7;
static ocl_kernel *__ocl_compute_rhs_8;
static ocl_kernel *__ocl_compute_rhs_9;
static ocl_kernel *__ocl_compute_rhs_10;
static ocl_kernel *__ocl_compute_rhs_11;
static ocl_kernel *__ocl_compute_rhs_12;
static ocl_kernel *__ocl_compute_rhs_13;
static ocl_kernel *__ocl_compute_rhs_14;
static ocl_kernel *__ocl_compute_rhs_15;
static ocl_kernel *__ocl_compute_rhs_16;
static ocl_kernel *__ocl_compute_rhs_17;
static ocl_kernel *__ocl_compute_rhs_18;
static ocl_kernel *__ocl_compute_rhs_19;
static ocl_kernel *__ocl_compute_rhs_20;
static ocl_kernel *__ocl_txinvr_0;
static ocl_kernel *__ocl_tzetar_0;
static ocl_kernel *__ocl_x_solve_0;
static ocl_kernel *__ocl_x_solve_1;
static ocl_kernel *__ocl_x_solve_2;
static ocl_kernel *__ocl_x_solve_3;
static ocl_kernel *__ocl_x_solve_4;
static ocl_kernel *__ocl_x_solve_5;
static ocl_kernel *__ocl_x_solve_6;
static ocl_kernel *__ocl_x_solve_7;
static ocl_kernel *__ocl_y_solve_0;
static ocl_kernel *__ocl_y_solve_1;
static ocl_kernel *__ocl_y_solve_2;
static ocl_kernel *__ocl_y_solve_3;
static ocl_kernel *__ocl_y_solve_4;
static ocl_kernel *__ocl_y_solve_5;
static ocl_kernel *__ocl_y_solve_6;
static ocl_kernel *__ocl_y_solve_7;
static ocl_kernel *__ocl_z_solve_0;
static ocl_kernel *__ocl_z_solve_1;
static ocl_kernel *__ocl_z_solve_2;
static ocl_kernel *__ocl_z_solve_3;
static ocl_kernel *__ocl_z_solve_4;
static ocl_kernel *__ocl_z_solve_5;
static ocl_kernel *__ocl_z_solve_6;
//OCL KERNELS (END)

//OCL BUFFERS (BEGIN)
static ocl_buffer *__ocl_buffer_u;
static ocl_buffer *__ocl_buffer_rhs;
static ocl_buffer *__ocl_buffer_lhs;
static ocl_buffer *__ocl_buffer_rho_i;
static ocl_buffer *__ocl_buffer_cv;
static ocl_buffer *__ocl_buffer_us;
static ocl_buffer *__ocl_buffer_rhon;
static ocl_buffer *__ocl_buffer_speed;
static ocl_buffer *__ocl_buffer_vs;
static ocl_buffer *__ocl_buffer_rhoq;
static ocl_buffer *__ocl_buffer_ws;
static ocl_buffer *__ocl_buffer_rhos;
static ocl_buffer *__ocl_buffer_square;
static ocl_buffer *__ocl_buffer_qs;
static ocl_buffer *__ocl_buffer_ainv;
static ocl_buffer *__ocl_buffer_forcing;
static ocl_buffer *__ocl_buffer_grid_points;
//OCL BUFFERS (END)

static void init_ocl_runtime();
static void create_ocl_buffers();
static void release_ocl_buffers();
static void sync_ocl_buffers();
static void flush_ocl_buffers();

#ifdef PROFILE_SWAP
static float t_swap = 0;
static float t_z = 0;
static float t_rhs = 0;
#endif

#endif
