import os, sys

#apps=["BT", "CG", "EP", "SP", "FT"]
apps=["EP"]
classes=["S", "W", "A", "B"]
#classes=["S", "W", "A", "B", "C"]

runtimes=10
rwd=os.getcwd()

def run(app):
	print app
	work_dir=rwd + "/" + app + "/cetus_output"
	os.chdir(work_dir)
	log_dir=rwd + "/logs"
	cmd = "mkdir -p " + log_dir
	os.system(cmd)
	for c in classes:
		log_file= app.lower() + "." + c + ".log"
		cmd = "rm -f " + log_dir + "/" + log_file
		os.system(cmd)
		cmd = "./binaries/" + app.lower() + "." + c + " >> " + log_dir + "/" + log_file
		
		for i in range(0, runtimes):
			os.system(cmd)

	print "[done]"

for a in apps:
	run(a)
