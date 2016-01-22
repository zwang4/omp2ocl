import os, sys

cl="ft.cl"
feature_extractor = "feat-extract"
dataset=
#The following variables need to define
if (dataset == "S"):
	fftblock = 
	d = 
if (dataset == "W"):
	fftblock = 
	d = 
if (dataset == "A"):
	fftblock = 
	d = 
if (dataset == "B"):
	fftblock = 
	d = 
if (dataset == "C"):
	fftblock = 
	d = 

os.system("mkdir -p features")

cmd_vars = " " + " fftblock " + str(fftblock)  + " d " + str(d) 

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

proc_feature("evolve_0")
proc_feature("compute_indexmap_0")
proc_feature("cffts1_0")
proc_feature("cffts2_0")
proc_feature("cffts3_0")
proc_feature("checksum_0_reduction_step0")
proc_feature("checksum_0_reduction_step1")
proc_feature("checksum_0_reduction_step2")
