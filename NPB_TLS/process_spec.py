import os, sys, io_common, sqlite3

from io_common import *

schemes=["ocl", "scott"]
conn = sqlite3.connect('spec.db')
c = conn.cursor()

def get_rows_float(rows):
	rs = []
	for r in rows:
		rs.append(float(r[0]))
	return rs

def get_exp_time(app, inp, table):
	conn_seq = sqlite3.connect('../omp2ocl-data/corei7.db')
	c = conn_seq.cursor()
	sql = "select time from %s where INPUT='%s'" % (table, inp)
	rows = c.execute(sql)

	rs = get_rows_float(rows)

	if (len(rs) == 0):
		return None

	return sum(rs) / len(rs)

def get_exp_ocl_time(app, inp, table):
	conn_seq = sqlite3.connect('../omp2ocl-data/radeon.db')
	c = conn_seq.cursor()
	sql = "select time from %s where INPUT='%s'" % (table, inp)
	rows = c.execute(sql)

	rs = get_rows_float(rows)

	if (len(rs) == 0):
		return None

	return sum(rs) / len(rs)

def get_omp_time(app, inp):
	table = app + "_omp"
	return get_exp_time(app, inp, table)

def get_seq_time(app, inp):
	table = app + "_seq"
	return get_exp_time(app, inp, table)

ocl_table={"bt" : "bt_tuned", "cg" : "cg_tuned", "ep" : "ep_private", "ft" : "ft_soa", "is" : "is_ocl", "lu" : "lu_soa",
		"mg" : "mg_swap", "sp" : "sp_tuned"};

def get_ocl_time(app, inp):
	table = ocl_table[app]
	return get_exp_ocl_time (app, inp, table)


def get_avg_time(app, dataset, scheme):
	global c
	sql="select runtime from runtime where app='%s' and class='%s' and scheme='%s'" % (app, dataset, scheme)
	rows = c.execute(sql)
	rs=[]
	for r in rows:
		rs.append(float(r[0]))
	
	return sum(rs) / len(rs)

def get_dist_apps():
	global c
	sql = "select distinct app from runtime"
	rows = c.execute(sql)
	apps = []

	for r in rows:
		apps.append(r[0])

	return apps


def get_data_sets(app):
	global c
	sql = "select distinct class from runtime where app='%s'" % app
	rows = c.execute(sql)
	datasets = []

	for r in rows:
		datasets.append(r[0])
	
	return datasets

def process():
	apps = get_dist_apps()
	res=dict()
	for a in apps:
		datasets = get_data_sets(a)
		for d in datasets:
			token = a + "." + d
			ts=[]
			for s in schemes:
				t = get_avg_time(a, d, s)
				ts.append(t)
			res[token] = ts
	return res


apps=["bt", "cg", "sp", "lu", "is", "ft", "mg", "ep"]
datasets=["S", "W", "A", "B"]
spec_res = process()
spec_apps = get_dist_apps()
f = open("result.csv", "w")
f.write("app,seq,omp,ocl,ocl_spec,scott\n")

for a in apps:
	for d in datasets:
		token = a + "." + d
		ocl_time = get_ocl_time(a, d)
		omp_time = get_omp_time(a, d)
		seq_time = get_seq_time(a, d)
		if ocl_time == None:
			continue
		if a in spec_apps:
			if token in spec_res:
				ocl_spec=spec_res[token][0]
				scott_spec=spec_res[token][1]
			else:
				continue
		else:
			ocl_spec = ocl_time
			scott_spec = ocl_time

		ss="%s,%f,%f,%f,%f,%f" % (token, seq_time, omp_time, ocl_time, ocl_spec, scott_spec)
		f.write(ss + "\n")
			
f.close()
conn.close()
