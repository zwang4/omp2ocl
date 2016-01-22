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
#define OCL_RELEASE_GTP_BUFFERS_IMMEDIATE /* if defined, __global threadprivate buffers will be released immediately after each use. This may somehow alleviate the memory pressure */

typedef struct __oclLVarBufferList {
	ocl_buffer *buf;
	struct __oclLVarBufferList *next;
} __oclLVarBufferList_t;

static __oclLVarBufferList_t *__ocl_lvar_buf_header = NULL;

static ocl_program *__ocl_program;

/** global data structures in : ft.c (BEGIN)*/
/** global data structures in : ft.c (END)*/
//OCL KERNELS (BEGIN)
static ocl_kernel *__ocl_evolve_0;
static ocl_kernel *__ocl_compute_indexmap_0;
static ocl_kernel *__ocl_cffts1_0;
static ocl_kernel *__ocl_cffts2_0;
static ocl_kernel *__ocl_cffts3_0;
static ocl_kernel *__ocl_checksum_0_reduction_step0;
static ocl_kernel *__ocl_checksum_0_reduction_step1;
static ocl_kernel *__ocl_checksum_0_reduction_step2;
//OCL KERNELS (END)

//OCL BUFFERS (BEGIN)
static ocl_buffer *__ocl_buffer_ex;
static ocl_buffer *__ocl_buffer_u1_real_evolve = NULL;
static double *__ocl_p_u1_real_evolve = NULL;
static ocl_buffer *__ocl_buffer_u0_real_evolve = NULL;
static double *__ocl_p_u0_real_evolve = NULL;
static ocl_buffer *__ocl_buffer_indexmap_evolve = NULL;
static int *__ocl_p_indexmap_evolve = NULL;
static ocl_buffer *__ocl_buffer_u1_imag_evolve = NULL;
static double *__ocl_p_u1_imag_evolve = NULL;
static ocl_buffer *__ocl_buffer_u0_imag_evolve = NULL;
static double *__ocl_p_u0_imag_evolve = NULL;
static ocl_buffer *__ocl_buffer_indexmap_compute_indexmap = NULL;
static int *__ocl_p_indexmap_compute_indexmap = NULL;
static ocl_buffer *__ocl_buffer_yy0_real;
static ocl_buffer *__ocl_buffer_yy0_imag;
static ocl_buffer *__ocl_buffer_yy1_real;
static ocl_buffer *__ocl_buffer_yy1_imag;
static ocl_buffer *__ocl_buffer_u_real;
static ocl_buffer *__ocl_buffer_u_imag;
static double *__ocl_th_yy0_real_cffts1_0 = NULL;
static ocl_buffer *__ocl_buffer_yy0_real_cffts1_0 = NULL;
static unsigned __ocl_th_yy0_real_cffts1_0_length = 0;

static double *__ocl_th_yy0_imag_cffts1_0 = NULL;
static ocl_buffer *__ocl_buffer_yy0_imag_cffts1_0 = NULL;
static unsigned __ocl_th_yy0_imag_cffts1_0_length = 0;

static double *__ocl_th_yy1_real_cffts1_0 = NULL;
static ocl_buffer *__ocl_buffer_yy1_real_cffts1_0 = NULL;
static unsigned __ocl_th_yy1_real_cffts1_0_length = 0;

static double *__ocl_th_yy1_imag_cffts1_0 = NULL;
static ocl_buffer *__ocl_buffer_yy1_imag_cffts1_0 = NULL;
static unsigned __ocl_th_yy1_imag_cffts1_0_length = 0;

static ocl_buffer *__ocl_buffer_d_cffts1 = NULL;
static int *__ocl_p_d_cffts1 = NULL;
static ocl_buffer *__ocl_buffer_x_real_cffts1 = NULL;
static double *__ocl_p_x_real_cffts1 = NULL;
static ocl_buffer *__ocl_buffer_x_imag_cffts1 = NULL;
static double *__ocl_p_x_imag_cffts1 = NULL;
static ocl_buffer *__ocl_buffer_xout_real_cffts1 = NULL;
static double *__ocl_p_xout_real_cffts1 = NULL;
static ocl_buffer *__ocl_buffer_xout_imag_cffts1 = NULL;
static double *__ocl_p_xout_imag_cffts1 = NULL;
static double *__ocl_th_yy0_real_cffts2_0 = NULL;
static ocl_buffer *__ocl_buffer_yy0_real_cffts2_0 = NULL;
static unsigned __ocl_th_yy0_real_cffts2_0_length = 0;

static double *__ocl_th_yy0_imag_cffts2_0 = NULL;
static ocl_buffer *__ocl_buffer_yy0_imag_cffts2_0 = NULL;
static unsigned __ocl_th_yy0_imag_cffts2_0_length = 0;

static double *__ocl_th_yy1_real_cffts2_0 = NULL;
static ocl_buffer *__ocl_buffer_yy1_real_cffts2_0 = NULL;
static unsigned __ocl_th_yy1_real_cffts2_0_length = 0;

static double *__ocl_th_yy1_imag_cffts2_0 = NULL;
static ocl_buffer *__ocl_buffer_yy1_imag_cffts2_0 = NULL;
static unsigned __ocl_th_yy1_imag_cffts2_0_length = 0;

static ocl_buffer *__ocl_buffer_d_cffts2 = NULL;
static int *__ocl_p_d_cffts2 = NULL;
static ocl_buffer *__ocl_buffer_x_real_cffts2 = NULL;
static double *__ocl_p_x_real_cffts2 = NULL;
static ocl_buffer *__ocl_buffer_x_imag_cffts2 = NULL;
static double *__ocl_p_x_imag_cffts2 = NULL;
static ocl_buffer *__ocl_buffer_xout_real_cffts2 = NULL;
static double *__ocl_p_xout_real_cffts2 = NULL;
static ocl_buffer *__ocl_buffer_xout_imag_cffts2 = NULL;
static double *__ocl_p_xout_imag_cffts2 = NULL;
static double *__ocl_th_yy0_real_cffts3_0 = NULL;
static ocl_buffer *__ocl_buffer_yy0_real_cffts3_0 = NULL;
static unsigned __ocl_th_yy0_real_cffts3_0_length = 0;

static double *__ocl_th_yy0_imag_cffts3_0 = NULL;
static ocl_buffer *__ocl_buffer_yy0_imag_cffts3_0 = NULL;
static unsigned __ocl_th_yy0_imag_cffts3_0_length = 0;

static double *__ocl_th_yy1_real_cffts3_0 = NULL;
static ocl_buffer *__ocl_buffer_yy1_real_cffts3_0 = NULL;
static unsigned __ocl_th_yy1_real_cffts3_0_length = 0;

static double *__ocl_th_yy1_imag_cffts3_0 = NULL;
static ocl_buffer *__ocl_buffer_yy1_imag_cffts3_0 = NULL;
static unsigned __ocl_th_yy1_imag_cffts3_0_length = 0;

static ocl_buffer *__ocl_buffer_d_cffts3 = NULL;
static int *__ocl_p_d_cffts3 = NULL;
static ocl_buffer *__ocl_buffer_x_real_cffts3 = NULL;
static double *__ocl_p_x_real_cffts3 = NULL;
static ocl_buffer *__ocl_buffer_x_imag_cffts3 = NULL;
static double *__ocl_p_x_imag_cffts3 = NULL;
static ocl_buffer *__ocl_buffer_xout_real_cffts3 = NULL;
static double *__ocl_p_xout_real_cffts3 = NULL;
static ocl_buffer *__ocl_buffer_xout_imag_cffts3 = NULL;
static double *__ocl_p_xout_imag_cffts3 = NULL;
static ocl_buffer *__ocl_buffer_xstart;
static ocl_buffer *__ocl_buffer_xend;
static ocl_buffer *__ocl_buffer_ystart;
static ocl_buffer *__ocl_buffer_yend;
static ocl_buffer *__ocl_buffer_zstart;
static ocl_buffer *__ocl_buffer_zend;
static ocl_buffer *__ocl_buffer_chk_real_checksum_0 = NULL;
static unsigned __ocl_buffer_chk_real_checksum_0_size = 0;
static double *__ocl_output_chk_real_checksum_0 = NULL;
static ocl_buffer *__ocl_output_buffer_chk_real_checksum_0 = NULL;
static unsigned __ocl_output_chk_real_checksum_0_size = 0;
static ocl_buffer *__ocl_buffer_chk_imag_checksum_0 = NULL;
static unsigned __ocl_buffer_chk_imag_checksum_0_size = 0;
static double *__ocl_output_chk_imag_checksum_0 = NULL;
static ocl_buffer *__ocl_output_buffer_chk_imag_checksum_0 = NULL;
static unsigned __ocl_output_chk_imag_checksum_0_size = 0;
static ocl_buffer *__ocl_buffer_u1_real_checksum = NULL;
static double *__ocl_p_u1_real_checksum = NULL;
static ocl_buffer *__ocl_buffer_u1_imag_checksum = NULL;
static double *__ocl_p_u1_imag_checksum = NULL;
//OCL BUFFERS (END)

static void init_ocl_runtime();
static void create_ocl_buffers();
static void release_ocl_buffers();
static void sync_ocl_buffers();
static void flush_ocl_buffers();
#endif
