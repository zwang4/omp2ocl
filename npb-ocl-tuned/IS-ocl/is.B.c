//====== HOSTSIDE CODE START
#include "npbparams.h"
#include "stdlib.h"
#include "stdio.h"
#include "ocl_runtime.h"
#include <string.h>
#include <sys/time.h>

#if CLASS == 'S'
#define  TOTAL_KEYS_LOG_2    16
#define  MAX_KEY_LOG_2       11
#define  NUM_BUCKETS_LOG_2   9
#define  MULT_ITER_SIZE      512
#endif


/*************/
/*  CLASS W  */
/*************/
#if CLASS == 'W'
#define  TOTAL_KEYS_LOG_2    20
#define  MAX_KEY_LOG_2       16
#define  NUM_BUCKETS_LOG_2   10
#define  MULT_ITER_SIZE      1024
#endif

/*************/
/*  CLASS A  */
/*************/
#if CLASS == 'A'
#define  TOTAL_KEYS_LOG_2    23
#define  MAX_KEY_LOG_2       19
#define  NUM_BUCKETS_LOG_2   10
#define  MULT_ITER_SIZE      (8192*50)
#endif


/*************/
/*  CLASS B  */
/*************/
#if CLASS == 'B'
#define  TOTAL_KEYS_LOG_2    25
#define  MAX_KEY_LOG_2       21
#define  NUM_BUCKETS_LOG_2   10
#define  MULT_ITER_SIZE      (8192*500)
#endif


/*************/
/*  CLASS C  */
/*************/
#if CLASS == 'C'
#define  TOTAL_KEYS_LOG_2    27
#define  MAX_KEY_LOG_2       23
#define  NUM_BUCKETS_LOG_2   10
#define  MULT_ITER_SIZE	     (8192 * 20 * 50)
#endif


#define  TOTAL_KEYS          (1 << TOTAL_KEYS_LOG_2)
#define  MAX_KEY             (1 << MAX_KEY_LOG_2)
#define  NUM_BUCKETS         (1 << NUM_BUCKETS_LOG_2)
#define  NUM_KEYS            TOTAL_KEYS
#define  SIZE_OF_BUFFERS     NUM_KEYS  


#define  MAX_ITERATIONS      10
#define  TEST_ARRAY_SIZE     5

/**************************************** OCL ROUTINES DECL [BEGIN]******************************************/

#define GROUP_SIZE	128
//#define OCL_RELEASE_GTP_BUFFERS_IMMEDIATE /* if defined, __global threadprivate buffers will be released immediately after each use. This may somehow alleviate the memory pressure */

#define ENABLE_OCL_KERNEL_rank_0

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
				"Failed to create the ocl buffer for %s at line: %d\n", #__name__, #__line__);\
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
typedef struct __oclLVarBufferList
{
	ocl_buffer *buf;
	struct __oclLVarBufferList *next;
} __oclLVarBufferList_t;

static __oclLVarBufferList_t *__ocl_lvar_buf_header = NULL;

//OCL KERNELS (BEGIN)
static ocl_kernel *__ocl_rank_0;
static ocl_kernel *__ocl_rank_0_pre;
static ocl_kernel *__ocl_rank_0_reduction;
//OCL KERNELS (END)

static ocl_program *__ocl_program;

//OCL BUFFERS (BEGIN)
static ocl_buffer *__ocl_buffer_key_buff2;
static ocl_buffer *__ocl_buffer_key_array;
static ocl_buffer *__ocl_buffer_prv_buff1;
static int *__ocl_th_prv_buff1_rank_0 = NULL;
static ocl_buffer *__ocl_buffer_prv_buff1_rank_0 = NULL;
static unsigned __ocl_th_prv_buff1_rank_0_length = 0;

//OCL BUFFERS (END)

static void init_ocl_runtime ();
static void create_ocl_buffers ();
static void release_ocl_buffers ();
static void sync_ocl_buffers ();
static void flush_ocl_buffers ();
#ifdef PROFILING
static void dump_profiling ();
#endif
/**************************************** OCL ROUTINES DECL [END]*******************************************/

int *key_buff_ptr_global;         /* used by full_verify to get */
                                       /* copies of rank info        */

int      passed_verification;
                                 

/************************************/
/* These are the three main arrays. */
/* See SIZE_OF_BUFFERS def above    */
/************************************/
int key_array[SIZE_OF_BUFFERS],    
         key_buff1[SIZE_OF_BUFFERS],    
         key_buff2[SIZE_OF_BUFFERS],
         partial_verify_vals[TEST_ARRAY_SIZE];

#ifdef USE_BUCKETS
int bucket_size[NUM_BUCKETS],                    
         bucket_ptrs[NUM_BUCKETS];
#endif


/**********************/
/* Partial verif info */
/**********************/
int test_index_array[TEST_ARRAY_SIZE],
         test_rank_array[TEST_ARRAY_SIZE],

         S_test_index_array[TEST_ARRAY_SIZE] = 
                             {48427,17148,23627,62548,4431},
         S_test_rank_array[TEST_ARRAY_SIZE] = 
                             {0,18,346,64917,65463},

         W_test_index_array[TEST_ARRAY_SIZE] = 
                             {357773,934767,875723,898999,404505},
         W_test_rank_array[TEST_ARRAY_SIZE] = 
                             {1249,11698,1039987,1043896,1048018},

         A_test_index_array[TEST_ARRAY_SIZE] = 
                             {2112377,662041,5336171,3642833,4250760},
         A_test_rank_array[TEST_ARRAY_SIZE] = 
                             {104,17523,123928,8288932,8388264},

         B_test_index_array[TEST_ARRAY_SIZE] = 
                             {41869,812306,5102857,18232239,26860214},
         B_test_rank_array[TEST_ARRAY_SIZE] = 
                             {33422937,10244,59149,33135281,99}, 

         C_test_index_array[TEST_ARRAY_SIZE] = 
                             {44172927,72999161,74326391,129606274,21736814},
         C_test_rank_array[TEST_ARRAY_SIZE] = 
                             {61147,882988,266290,133997595,133525895};



double randlc (double *X, ocl_buffer * __ocl_buffer_X, double *A,
		ocl_buffer * __ocl_buffer_A);
void full_verify ();


	double
randlc (double *X, ocl_buffer * __ocl_buffer_X, double *A,
		ocl_buffer * __ocl_buffer_A)
{
	static int KS = 0;
	static double R23, R46, T23, T46;
	double T1, T2, T3, T4;
	double A1;
	double A2;
	double X1;
	double X2;
	double Z;
	int i, j;
	if (KS == 0)
	{
		R23 = 1.0;
		R46 = 1.0;
		T23 = 1.0;
		T46 = 1.0;
		for (i = 1; i <= 23; i++)
		{
			R23 = 0.50 * R23;
			T23 = 2.0 * T23;
		}
		for (i = 1; i <= 46; i++)
		{
			R46 = 0.50 * R46;
			T46 = 2.0 * T46;
		}
		KS = 1;
	}
	T1 = R23 * *A;
	j = T1;
	A1 = j;
	A2 = *A - T23 * A1;
	T1 = R23 * *X;
	j = T1;
	X1 = j;
	X2 = *X - T23 * X1;
	T1 = A1 * X2 + A2 * X1;
	j = R23 * T1;
	T2 = j;
	Z = T1 - T23 * T2;
	T3 = T23 * Z + A2 * X2;
	j = R46 * T3;
	T4 = j;
	*X = T3 - T46 * T4;
	return (R46 * *X);
}


void	create_seq( double seed, double a )
{
	double x;
	int    i, j, k;

	k = MAX_KEY/4;

	for (i=0; i<NUM_KEYS; i++)
	{
		x = randlc(&seed, NULL, &a, NULL);
		x += randlc(&seed, NULL, &a, NULL);
		x += randlc(&seed, NULL, &a, NULL);
		x += randlc(&seed, NULL, &a, NULL);  

		key_array[i] = k*x;
	}
}

void full_verify()
{
	int    i, j;
	int    k;
	int    m, unique_keys;



	/*  Now, finally, sort the keys:  */
	for( i=0; i<NUM_KEYS; i++ )
		key_array[--key_buff_ptr_global[key_buff2[i]]] = key_buff2[i];


	/*  Confirm keys correctly sorted: count incorrectly sorted keys, if any */

	j = 0;
	for( i=1; i<NUM_KEYS; i++ )
		if( key_array[i-1] > key_array[i] )
			j++;


	if( j != 0 )
	{
		printf( "Full_verify: number of keys out of sort: %d\n",
				j );
	}
	else
		passed_verification++;


}

static int prv_buff1[MAX_KEY];

void rank (int iteration)
{

	int    i, j, k;
	int    l, m;

	int    shift = MAX_KEY_LOG_2 - NUM_BUCKETS_LOG_2;
	int    key;
	int    min_key_val, max_key_val;
	
	key_array[iteration] = iteration;
	key_array[iteration+MAX_ITERATIONS] = MAX_KEY - iteration;

	/*  Determine where the partial verify test keys are, load into  */
	/*  top of array bucket_size                                     */
	for( i=0; i<TEST_ARRAY_SIZE; i++ )
		partial_verify_vals[i] = key_array[test_index_array[i]];

	/*  Clear the work array */
	for( i=0; i<MAX_KEY; i++ )
		key_buff1[i] = 0;

	for (i=0; i<MAX_KEY; i++)
		prv_buff1[i] = 0;

	oclHostWrites(__ocl_buffer_key_array);
	
	
	size_t __ocl_global_work_size[1];
	__ocl_global_work_size[0] = NUM_KEYS - (0);
	OCL_NEAREST_MULTD(__ocl_global_work_size[0], MULT_ITER_SIZE);
	oclGetWorkSize (1, __ocl_global_work_size, NULL);

	unsigned int __ocl_copyin_mult_factor = __ocl_global_work_size[0];

	size_t __ocl_bs_prv_buff1_rank_0 =
		sizeof (int) * (MAX_KEY * __ocl_copyin_mult_factor);

	if (!__ocl_buffer_prv_buff1_rank_0)
	{
		__ocl_buffer_prv_buff1_rank_0 =
			oclCreateBuffer (NULL,
					__ocl_bs_prv_buff1_rank_0);
	}

	//Fill up buffer with 0
	oclSetKernelArgBuffer(__ocl_rank_0_pre, 0, __ocl_buffer_prv_buff1_rank_0);
	{
		size_t gs[1]={__ocl_copyin_mult_factor * MAX_KEY};
		oclRunKernel (__ocl_rank_0_pre, 1, gs);
	}

	int __ocl_i_bound = NUM_KEYS;
	oclSetKernelArgBuffer (__ocl_rank_0, 0, __ocl_buffer_key_buff2);
	oclSetKernelArgBuffer (__ocl_rank_0, 1, __ocl_buffer_key_array);
	oclSetKernelArgBuffer (__ocl_rank_0, 2, __ocl_buffer_prv_buff1_rank_0);
	oclSetKernelArg (__ocl_rank_0, 3, sizeof (int), &__ocl_i_bound);

	oclDevReads  (__ocl_buffer_key_array);
	oclDevWrites (__ocl_buffer_key_buff2);

	oclRunKernel (__ocl_rank_0, 1, __ocl_global_work_size);

	oclSetKernelArgBuffer (__ocl_rank_0_reduction, 0, __ocl_buffer_prv_buff1);
	oclSetKernelArgBuffer (__ocl_rank_0_reduction, 1, __ocl_buffer_prv_buff1_rank_0);
	oclSetKernelArg (__ocl_rank_0_reduction, 2, sizeof (int), &__ocl_copyin_mult_factor);

	//Reduction
	oclDevWrites(__ocl_buffer_prv_buff1);

	{	
		size_t gs[1] = { MAX_KEY };
		oclRunKernel (__ocl_rank_0_reduction, 1, gs);
	}

	oclHostReads(__ocl_buffer_key_buff2);
	oclHostWrites(__ocl_buffer_prv_buff1);
	oclSync();
	
	
	for( i=0; i<MAX_KEY-1; i++ )   
		prv_buff1[i+1] += prv_buff1[i];  

	for( i=0; i<MAX_KEY; i++ )
	{
		key_buff1[i] += prv_buff1[i];
	}

	/*  To obtain ranks of each key, successively add the individual key
	    population, not forgetting to add m, the total of lesser keys,
	    to the first key population                                          */

	{

		/* This is the partial verify test section */
		/* Observe that test_rank_array vals are   */
		/* shifted differently for different cases */
		for( i=0; i<TEST_ARRAY_SIZE; i++ )
		{                                             
			k = partial_verify_vals[i];          /* test vals were put here */
			if( 0 <= k  &&  k <= NUM_KEYS-1 )
				switch( CLASS )
				{
					case 'S':
						if( i <= 2 )
						{
							if( key_buff1[k-1] != test_rank_array[i]+iteration )
							{
								printf( "Failed partial verification: "
										"iteration %d, test key %d, key_buff1:%d\n", 
										iteration, i, key_buff1[k-1]);
							}
							else
							{
								passed_verification++;
							}
						}
						else
						{
							if( key_buff1[k-1] != test_rank_array[i]-iteration )
							{
								printf( "Failed partial verification: "
										"iteration %d, test key %d, key_buff1:%d\n", 
										iteration, i, key_buff1[k-1]);
							}
							else
							{
								passed_verification++;
							}
						}
						break;
					case 'W':
						if( i < 2 )
						{
							if( key_buff1[k-1] != 
									test_rank_array[i]+(iteration-2) )
							{
								printf( "Failed partial verification: "
										"iteration %d, test key %d, key_buff1:%d\n", 
										iteration, i, key_buff1[k-1]);
							}
							else
								passed_verification++;
						}
						else
						{
							if( key_buff1[k-1] != test_rank_array[i]-iteration )
							{
								printf( "Failed partial verification: "
										"iteration %d, test key %d, key_buff1:%d\n", 
										iteration, i, key_buff1[k-1]);
							}
							else
								passed_verification++;
						}
						break;
					case 'A':
						if( i <= 2 )
						{
							if( key_buff1[k-1] != 
									test_rank_array[i]+(iteration-1) )
							{
								printf( "Failed partial verification: "
										"iteration %d, test key %d, key_buff1:%d\n", 
										iteration, i, key_buff1[k-1]);
							}
							else
								passed_verification++;
						}
						else
						{
							if( key_buff1[k-1] != 
									test_rank_array[i]-(iteration-1) )
							{
								printf( "Failed partial verification: "
										"iteration %d, test key %d, key_buff1:%d\n", 
										iteration, i, key_buff1[k-1]);
							}
							else
								passed_verification++;
						}
						break;
					case 'B':
						if( i == 1 || i == 2 || i == 4 )
						{
							if( key_buff1[k-1] != test_rank_array[i]+iteration )
							{
								printf( "Failed partial verification: "
										"iteration %d, test key %d\n", 
										iteration, i );
							}
							else
								passed_verification++;
						}
						else
						{
							if( key_buff1[k-1] != test_rank_array[i]-iteration )
							{
								printf( "Failed partial verification: "
										"iteration %d, test key %d\n", 
										iteration, i );
							}
							else
								passed_verification++;
						}
						break;
					case 'C':
						if( i <= 2 )
						{
							if( key_buff1[k-1] != test_rank_array[i]+iteration )
							{
								printf( "Failed partial verification: "
										"iteration %d, test key %d\n", 
										iteration, i );
							}
							else
								passed_verification++;
						}
						else
						{
							if( key_buff1[k-1] != test_rank_array[i]-iteration )
							{
								printf( "Failed partial verification: "
										"iteration %d, test key %d\n", 
										iteration, i );
							}
							else
								passed_verification++;
						}
						break;
				}        
		}




		/*  Make copies of rank info for use by full_verify: these variables
		    in rank are local; making them global slows down the code, probably
		    since they cannot be made register by compiler                        */

		if( iteration == MAX_ITERATIONS ) 
			key_buff_ptr_global = key_buff1;

	} /* end master */

}


int main( argc, argv )
    int argc;
    char **argv;
{

    int             i, iteration, itemp;
    int		    nthreads = 1;
    double          timecounter, maxtime;

    init_ocl_runtime();

/*  Initialize the verification arrays if a valid class */
    for( i=0; i<TEST_ARRAY_SIZE; i++ )
        switch( CLASS )
        {
            case 'S':
                test_index_array[i] = S_test_index_array[i];
                test_rank_array[i]  = S_test_rank_array[i];
                break;
            case 'A':
                test_index_array[i] = A_test_index_array[i];
                test_rank_array[i]  = A_test_rank_array[i];
                break;
            case 'W':
                test_index_array[i] = W_test_index_array[i];
                test_rank_array[i]  = W_test_rank_array[i];
                break;
            case 'B':
                test_index_array[i] = B_test_index_array[i];
                test_rank_array[i]  = B_test_rank_array[i];
                break;
            case 'C':
                test_index_array[i] = C_test_index_array[i];
                test_rank_array[i]  = C_test_rank_array[i];
                break;
        };

        

/*  Printout initial NPB info */
    printf( "\n\n NAS Parallel Benchmarks 2.3 OpenMP C version"
	    " - IS Benchmark\n\n" );
    printf( " Size:  %d  (class %c)\n", TOTAL_KEYS, CLASS );
    printf( " Iterations:   %d\n", MAX_ITERATIONS );

/*  Initialize timer  */             

/*  Generate random number sequence and subsequent keys on all procs */
    create_seq( 314159265.00,                    /* Random number gen seed */
                1220703125.00 );                 /* Random number gen mult */


/*  Do one interation for free (i.e., untimed) to guarantee initialization of  
    all data and code pages and respective tables */
////#pragma omp parallel    
    rank( 1 );  

/*  Start verification counter */
    passed_verification = 0;

    if( CLASS != 'S' ) printf( "\n   iteration\n" );

    timer_clear( 0 );
/*  Start timer  */             
    timer_start( 0 );

    struct timeval tv1, tv2;
    gettimeofday(&tv1, NULL);

/*  This is the main iteration */
    
////#pragma omp parallel private(iteration)    
    for( iteration=1; iteration<=MAX_ITERATIONS; iteration++ )
    {
////#pragma omp master	
        if( CLASS != 'S' ) printf( "        %d\n", iteration );
	
        rank( iteration );
	
#if defined(_OPENMP)	
////#pragma omp master
	nthreads = omp_get_num_threads();
#endif /* _OPENMP */	
    }

    gettimeofday(&tv2, NULL);
/*  End of timing, obtain maximum time of all processors */
    timer_stop( 0 );
    timecounter = (tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec) * 1E-6;


/*  This tests that keys are in sequence: sorting of last ranked key seq
    occurs here, but is an untimed operation                             */
    full_verify();



/*  The final printout  */
    if( passed_verification != 5*MAX_ITERATIONS + 1 )
        passed_verification = 0;
  
    c_print_results( "IS",
                     CLASS,
                     TOTAL_KEYS,
                     0,
                     0,
                     MAX_ITERATIONS,
		     nthreads,
                     timecounter,
                     ((double) (MAX_ITERATIONS*TOTAL_KEYS))
                                                  /timecounter/1000000.,
                     "keys ranked", 
                     passed_verification,
                     NPBVERSION,
                     COMPILETIME,
                     CC,
                     CLINK,
                     C_LIB,
                     C_INC,
                     CFLAGS,
                     CLINKFLAGS,
		     "randlc");

    oclDumpBytesTransferred (stdout);

  //  printf("TIME:%lf sec\n", (tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec) * 1E-6);
    return 0;

         /**************************/
}        /*  E N D  P R O G R A M  */

/**************************************** OCL ROUTINES DEFINITION [BEGIN]***********************************/

	static void
init_ocl_runtime ()
{
	int err;

	if (unlikely (err = oclInit ("NVIDIA", 1)))
	{
		fprintf (stderr, "Failed to init ocl runtime:%d.\n", err);
		exit (err);
	}

	__ocl_program = oclBuildProgram ("is.cl");
	if (unlikely (!__ocl_program))
	{
		fprintf (stderr, "Failed to build the program:%d.\n", err);
		exit (err);
	}

	__ocl_rank_0 = oclCreateKernel (__ocl_program, "rank_0");
	__ocl_rank_0_pre = oclCreateKernel (__ocl_program, "rank_0_pre");
	__ocl_rank_0_reduction = oclCreateKernel (__ocl_program, "rank_0_reduction");
	DYN_PROGRAM_CHECK (__ocl_rank_0);
	create_ocl_buffers ();
}

	static void
create_ocl_buffers ()
{
	__ocl_buffer_key_buff2 =
		oclCreateBuffer (key_buff2, (SIZE_OF_BUFFERS) * sizeof (int));
	DYN_BUFFER_CHECK (__ocl_buffer_key_buff2, -1);
	__ocl_buffer_key_array =
		oclCreateBuffer (key_array, (SIZE_OF_BUFFERS) * sizeof (int));
	DYN_BUFFER_CHECK (__ocl_buffer_key_array, -1);
	__ocl_buffer_prv_buff1 =
		oclCreateBuffer (prv_buff1, (MAX_KEY) * sizeof (int));
	DYN_BUFFER_CHECK (__ocl_buffer_prv_buff1, -1);
}

	static void
sync_ocl_buffers ()
{
	oclHostWrites (__ocl_buffer_key_buff2);
	oclHostWrites (__ocl_buffer_key_array);
	oclHostWrites (__ocl_buffer_prv_buff1);
	//SYNC_LOCALVAR_OCL_BUFFERS();
	oclSync ();
}

	static void
release_ocl_buffers ()
{
	oclReleaseBuffer (__ocl_buffer_key_buff2);
	oclReleaseBuffer (__ocl_buffer_key_array);
	oclReleaseBuffer (__ocl_buffer_prv_buff1);
	if (__ocl_th_prv_buff1_rank_0)
	{
		oclReleaseBuffer (__ocl_buffer_prv_buff1_rank_0);
		free (__ocl_th_prv_buff1_rank_0);
		__ocl_th_prv_buff1_rank_0_length = 0;
	}
	RELEASE_LOCALVAR_OCL_BUFFERS ();
}

#ifdef PROFILING
	static void
dump_profiling ()
{
	FILE *prof = fopen ("profiling-is", "w");
	float kernel = 0.0f, buffer = 0.0f;

	kernel += oclDumpKernelProfiling (__ocl_rank_0, prof);

	buffer += oclDumpBufferProfiling (__ocl_buffer_key_buff2, prof);
	buffer += oclDumpBufferProfiling (__ocl_buffer_key_array, prof);
	buffer += oclDumpBufferProfiling (__ocl_buffer_prv_buff1, prof);
	PROFILE_LOCALVAR_OCL_BUFFERS (buffer, prof);

	fprintf (stderr, "-- kernel: %.3fms\n", kernel);
	fprintf (stderr, "-- buffer: %.3fms\n", buffer);
	fclose (prof);
}
#endif

	static void
flush_ocl_buffers ()
{
	oclHostWrites (__ocl_buffer_key_buff2);
	oclHostWrites (__ocl_buffer_key_array);
	oclHostWrites (__ocl_buffer_prv_buff1);
	//SYNC_LOCALVAR_OCL_BUFFERS();
	oclSync ();
}

/**************************************** OCL ROUTINES DEFINITION [END]*************************************/
