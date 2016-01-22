import os, sys

cl="ft.host.cl"
feature_extractor = "feat-extract"
dataset=
#The following variables need to define
if (dataset == "S"):
if (dataset == "W"):
if (dataset == "A"):
if (dataset == "B"):
if (dataset == "C"):

os.system("mkdir -p features")

cmd_vars = " "

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

