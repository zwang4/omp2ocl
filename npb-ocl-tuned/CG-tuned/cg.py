import os, sys

cl="cg.cl"
feature_extractor = "feat-extract"
dataset=
#The following variables need to define
if (dataset == "S"):
	rowstr = 
	j = 
if (dataset == "W"):
	rowstr = 
	j = 
if (dataset == "A"):
	rowstr = 
	j = 
if (dataset == "B"):
	rowstr = 
	j = 
if (dataset == "C"):
	rowstr = 
	j = 

os.system("mkdir -p features")

cmd_vars = " " + " rowstr " + str(rowstr)  + " j " + str(j) 

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

proc_feature("main_0")
proc_feature("main_1")
proc_feature("main_2_reduction_step0")
proc_feature("main_2_reduction_step1")
proc_feature("main_2_reduction_step2")
proc_feature("main_3")
proc_feature("main_4")
proc_feature("main_5_reduction_step0")
proc_feature("main_5_reduction_step1")
proc_feature("main_5_reduction_step2")
proc_feature("main_6")
proc_feature("conj_grad_0")
proc_feature("conj_grad_1_reduction_step0")
proc_feature("conj_grad_1_reduction_step1")
proc_feature("conj_grad_1_reduction_step2")
proc_feature("conj_grad_2")
proc_feature("conj_grad_3")
proc_feature("conj_grad_4")
proc_feature("conj_grad_5_reduction_step0")
proc_feature("conj_grad_5_reduction_step1")
proc_feature("conj_grad_5_reduction_step2")
proc_feature("conj_grad_6")
proc_feature("conj_grad_7_reduction_step0")
proc_feature("conj_grad_7_reduction_step1")
proc_feature("conj_grad_7_reduction_step2")
proc_feature("conj_grad_8")
proc_feature("conj_grad_9")
proc_feature("conj_grad_10")
proc_feature("conj_grad_11_reduction_step0")
proc_feature("conj_grad_11_reduction_step1")
proc_feature("conj_grad_11_reduction_step2")
