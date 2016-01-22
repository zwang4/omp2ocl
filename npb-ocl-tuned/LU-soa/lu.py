import os, sys

cl="lu.cl"
feature_extractor = "feat-extract"
dataset=
#The following variables need to define
if (dataset == "S"):
	ny = 
	nz = 
	ist = 
	iend = 
	L2 = 
	ist1 = 
	iend1 = 
	jst = 
	jend = 
	jst1 = 
	jend1 = 
	nx = 
if (dataset == "W"):
	ny = 
	nz = 
	ist = 
	iend = 
	L2 = 
	ist1 = 
	iend1 = 
	jst = 
	jend = 
	jst1 = 
	jend1 = 
	nx = 
if (dataset == "A"):
	ny = 
	nz = 
	ist = 
	iend = 
	L2 = 
	ist1 = 
	iend1 = 
	jst = 
	jend = 
	jst1 = 
	jend1 = 
	nx = 
if (dataset == "B"):
	ny = 
	nz = 
	ist = 
	iend = 
	L2 = 
	ist1 = 
	iend1 = 
	jst = 
	jend = 
	jst1 = 
	jend1 = 
	nx = 
if (dataset == "C"):
	ny = 
	nz = 
	ist = 
	iend = 
	L2 = 
	ist1 = 
	iend1 = 
	jst = 
	jend = 
	jst1 = 
	jend1 = 
	nx = 

os.system("mkdir -p features")

cmd_vars = " " + " ny " + str(ny)  + " nz " + str(nz)  + " ist " + str(ist)  + " iend " + str(iend)  + " L2 " + str(L2)  + " ist1 " + str(ist1)  + " iend1 " + str(iend1)  + " jst " + str(jst)  + " jend " + str(jend)  + " jst1 " + str(jst1)  + " jend1 " + str(jend1)  + " nx " + str(nx) 

def proc_feature(kernel):
   global cl, unresolve_vars
   f = open("parse.cl", "w")
   fs = open(cl, "r")
   f.write("#define ENABLE_" + kernel + "\n")
   for line in fs:
      f.write(line)
   fs.close()
   f.close()
   cmd = feature_extractor + " parse.cl "
   cmd = cmd + cmd_vars + " > features/"  + kernel + "." + dataset
   print cmd
   os.system(cmd)

proc_feature("blts_0")
proc_feature("buts_0")
proc_feature("erhs_0")
proc_feature("erhs_1")
proc_feature("erhs_2")
proc_feature("erhs_3")
proc_feature("erhs_4")
proc_feature("erhs_5")
proc_feature("erhs_6")
proc_feature("jacld_0")
proc_feature("jacu_0")
proc_feature("l2norm_0_reduction_step0")
proc_feature("l2norm_0_reduction_step1")
proc_feature("l2norm_0_reduction_step2")
proc_feature("rhs_0")
proc_feature("rhs_1")
proc_feature("rhs_2")
proc_feature("rhs_3")
proc_feature("rhs_4")
proc_feature("rhs_5")
proc_feature("setbv_0")
proc_feature("setbv_1")
proc_feature("setbv_2")
proc_feature("setbv_3")
proc_feature("setbv_4")
proc_feature("setiv_0")
proc_feature("ssor_0")
proc_feature("ssor_1")
proc_feature("ssor_2")
