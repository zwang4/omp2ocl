import os, sys

apps=["BT", "CG", "EP", "SP", "FT"]
classes=["S", "W", "A", "B", "C"]


rwd=os.getcwd()

def build(app):
	print app
	work_dir=rwd + "/" + app + "/cetus_output"
	os.chdir(work_dir)
	cmd = "mkdir -p binaries"
	os.system(cmd)
	for c in classes:
		cmd = "rm -f a.out"
		os.system(cmd)
		cmd = "cp ready/" + app.lower() + "." + c + " " + app.lower() + ".cu"
		os.system(cmd)
		cmd = "cp ready/npbparams." + c + " npbparams.h"
		os.system(cmd)	
		os.system("make > /dev/null 2>&1")
		cmd = "cp a.out ../../bin/" + app.lower() + "." + c
		os.system(cmd)

	print "[done]"

for a in apps:
	build(a)
