import os, sys

cl="mg.cl"
feature_extractor = "feat-extract"
dataset=
#The following variables need to define
if (dataset == "S"):
	n1 = 
	m2j = 
	m1j = 
	mm1 = 
	d2 = 
	mm2 = 
	d1 = 
if (dataset == "W"):
	n1 = 
	m2j = 
	m1j = 
	mm1 = 
	d2 = 
	mm2 = 
	d1 = 
if (dataset == "A"):
	n1 = 
	m2j = 
	m1j = 
	mm1 = 
	d2 = 
	mm2 = 
	d1 = 
if (dataset == "B"):
	n1 = 
	m2j = 
	m1j = 
	mm1 = 
	d2 = 
	mm2 = 
	d1 = 
if (dataset == "C"):
	n1 = 
	m2j = 
	m1j = 
	mm1 = 
	d2 = 
	mm2 = 
	d1 = 

os.system("mkdir -p features")

cmd_vars = " " + " n1 " + str(n1)  + " m2j " + str(m2j)  + " m1j " + str(m1j)  + " mm1 " + str(mm1)  + " d2 " + str(d2)  + " mm2 " + str(mm2)  + " d1 " + str(d1) 

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

proc_feature("psinv_0")
proc_feature("resid_0")
proc_feature("rprj3_0")
proc_feature("interp_0")
proc_feature("interp_1")
proc_feature("interp_2")
proc_feature("comm3_0")
proc_feature("comm3_1")
proc_feature("comm3_2")
proc_feature("zran3_0")
proc_feature("zran3_1")
proc_feature("zran3_2")
proc_feature("zero3_0")
