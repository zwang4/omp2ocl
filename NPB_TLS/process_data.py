import os, sys, io_common, sqlite3

from io_common import *

conn = sqlite3.connect('spec.db')

token="Time in seconds ="

def get_time(line):
	line = line.replace('\n', '')
	l = len(token)
	ss=""
	for i in range(l+1, len(line)):
		ss = ss + line[i]
	return float(ss)

def process_file(fn):
	f = open(fn, "r")
	vd=[]
	for line in f:
		if line.find("Time in seconds =") >= 0:
			tt = get_time(line)
			vd.append(tt)
	return vd

def process_file_info(fname):
	fn = fname.split('.')
	app = fn[0]
	apps = app.split('/')
	app = apps[len(apps)-1]
	scheme=fn[1]
	dataset=fn[2]

	td = process_file(fname)
	if len(td) == 0:
		print "ERROR", fname
		sys.exit(-1)

	c = conn.cursor()
	i=0
	for t in td:
		sql = "insert into runtime values('%s', '%s', '%s', %d, %f)" % (app, dataset, scheme, i, t)
		i = i + 1
		c.execute(sql)
	conn.commit()


def list_apps(d):
	ds = listdir_fullpath(d)
	for di in ds:
		fs=ls_files(di, "*.log")
		for fn in fs:
			process_file_info(fn)


list_apps(sys.argv[1])
conn.close()
