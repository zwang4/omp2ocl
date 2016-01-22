import os, sys

cl="bt.cl"
feature_extractor = "feat-extract"
dataset=
#The following variables need to define
if (dataset == "S"):
	grid_points = 
if (dataset == "W"):
	grid_points = 
if (dataset == "A"):
	grid_points = 
if (dataset == "B"):
	grid_points = 
if (dataset == "C"):
	grid_points = 

os.system("mkdir -p features")

cmd_vars = " " + " grid_points " + str(grid_points) 

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

proc_feature("add_0")
proc_feature("exact_rhs_0")
proc_feature("exact_rhs_1")
proc_feature("exact_rhs_2")
proc_feature("exact_rhs_3")
proc_feature("exact_rhs_4")
proc_feature("initialize_0")
proc_feature("initialize_1")
proc_feature("initialize_2")
proc_feature("initialize_3")
proc_feature("initialize_4")
proc_feature("initialize_5")
proc_feature("initialize_6")
proc_feature("initialize_7")
proc_feature("lhsinit_0")
proc_feature("lhsinit_1")
proc_feature("lhsx_0")
proc_feature("lhsy_0")
proc_feature("lhsy_1")
proc_feature("lhsz_0")
proc_feature("lhsz_1")
proc_feature("compute_rhs_0")
proc_feature("compute_rhs_1")
proc_feature("compute_rhs_2")
proc_feature("compute_rhs_3")
proc_feature("compute_rhs_4")
proc_feature("compute_rhs_5")
proc_feature("compute_rhs_6")
proc_feature("compute_rhs_7")
proc_feature("compute_rhs_8")
proc_feature("compute_rhs_9")
proc_feature("compute_rhs_10")
proc_feature("compute_rhs_11")
proc_feature("compute_rhs_12")
proc_feature("compute_rhs_13")
proc_feature("compute_rhs_14")
proc_feature("compute_rhs_15")
proc_feature("compute_rhs_16")
proc_feature("compute_rhs_17")
proc_feature("compute_rhs_18")
proc_feature("compute_rhs_19")
proc_feature("compute_rhs_20")
proc_feature("x_backsubstitute_0")
proc_feature("x_solve_cell_0")
proc_feature("x_solve_cell_1")
proc_feature("x_solve_cell_2")
proc_feature("y_backsubstitute_0")
proc_feature("y_solve_cell_0")
proc_feature("y_solve_cell_1")
proc_feature("y_solve_cell_2")
proc_feature("z_backsubstitute_0")
proc_feature("z_solve_cell_0")
proc_feature("z_solve_cell_1")
proc_feature("z_solve_cell_2")
