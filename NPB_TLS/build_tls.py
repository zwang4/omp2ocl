import os, sys, string

classes=['S', 'W', 'A', 'B']
rd=os.getcwd()

scott_cmd="clang -cc1 -omp2ocl -gpu_tls=on -strict_tls_check=on -arch=AMD"
ocl_cmd="clang -cc1 -omp2ocl -ocl_tls=on -strict_tls_check=off -arch=AMD"

def gen_header(c, app):
	os.chdir(rd + "/NPB2.3-serial/" + app)
	cmd = "make %s CLASS=%s" % (app, c)
	os.system(cmd)

def copy_result(dst):
	cmd = "mkdir -p " + dst 
	os.system(cmd)
	cmd = "cp *.cl *.h *.c " + dst + "/."
	os.system(cmd)

def compile_app(c, app, bigapp, scheme):
	os.chdir(rd + "/" + bigapp + "/" + scheme + "/" + c )
	cmd = "cp ../../Makefile ."
	os.system(cmd)
	
	cmd = "cp ../../transpose.cl ."
	os.system(cmd)

	cmd = "make clean"
	os.system(cmd)

	cmd = "make"
	os.system(cmd)

def build_app(c, app, bigapp):
	cmd="mkdir -p scott"
	os.system(cmd)
	cmd = "mkdir -p ocl"
	os.system(cmd)

	gen_header(c, bigapp)
	os.chdir(rd + "/" + bigapp)
	cmd = "cp ../NPB2.3-serial/" + bigapp + "/npbparams.h ."
	os.system(cmd)

	cmd = scott_cmd + " " + app + ".c"
	os.system(cmd)
	copy_result("scott/" + c)
	
	cmd = ocl_cmd + " " + app + ".c"
	os.system(cmd)
	copy_result("ocl/" + c)

	compile_app(c, app, bigapp, "scott")
	compile_app(c, app, bigapp, "ocl")


#apps = ["lu"]
apps = ["sp", "lu", "cg"]

for app in apps:
	bigapp = app.swapcase()
	for c in classes:
		build_app(c, app, bigapp)
