from __future__ import division

import os, sys
import subprocess
import numpy as np

from common import collect_train, collect_test

def print_data (feats, winners, ext, gpuname):
  dirname = "crossval-naive-%s" % gpuname
  if not os.path.exists (dirname): os.makedirs (dirname)
  outfile = open (os.path.join (dirname, "%s.csv" % ext), 'w')
  outfiled = open (os.path.join (dirname, "%s.dec.csv" % ext), 'w')

  # print values
  for i in range (winners.size):
    outfile.write (','.join ([ str(f) for f in feats[i,:] ]))
    outfile.write('\n');
    if winners[i] == 1:
       outfiled.write('GPU\n')
    else:
	   outfiled.write('CPU\n')

  outfile.close ()
  outfiled.close()

def print_data_time (feats, cpu_times, gpu_times, winners, ext):
  outfile = open ("%s.csv" % ext, 'w')

  print ("writing %s.csv" % ext)

  # print header
  outfile.write (','.join ([ "feat%d" % i for i in range(feats.shape[1]) ]))
  outfile.write (",cpu_time,gpu_time,decision\n")

  # print values
  for i in range (winners.size):
    outfile.write (','.join ([ str(f) for f in feats[i,:] ]))
    outfile.write (",%f" % (cpu_times[i]))
    outfile.write (",%f" % (gpu_times[i]))
    outfile.write (",%d\n" % (winners[i]))

  outfile.close ()


def cross_validation (gpuname):

  (train_feats, train_cpu, train_gpu) = collect_train ("corei7i",gpuname)
  (test_benchs,test_feats, test_cpu, test_gpu) = collect_test ("corei7",gpuname)

  tt1 = np.concatenate ((np.array(train_cpu,ndmin=2),np.array(train_gpu,ndmin=2)), axis=0).transpose()
  train_winners = np.argmin (tt1,axis=1) # winners[i]==0 iff CPU faster than GPU for benchmark i
  train_weights = abs (tt1[:,0] / tt1[:,1]) / tt1[:,0]

  tt2 = np.concatenate ((np.array(test_cpu,ndmin=2),np.array(test_gpu,ndmin=2)), axis=0).transpose()
  test_winners = np.argmin (tt2,axis=1) # winners[i]==0 iff CPU faster than GPU for benchmark i
  test_weights = abs (tt2[:,0] / tt2[:,1]) / tt2[:,0]

  # count benchmarks and assign IDs
  nbenchs = 0
  bench_names = []
  last_benchname = ""
  last_versionname = ""
  bench_ids = []
  version_ids = []
  for bench,version,inp in test_benchs:
    if bench != last_benchname:
      nbenchs += 1   # next benchmark ID
      version_id = 0 # reset version ID
      last_benchname = bench
      last_versionname = version
      bench_names.append (bench)
    elif version != last_versionname:
      version_id += 1
      last_versionname = version
    bench_ids.append (nbenchs-1)
    version_ids.append (version_id)

  #print (zip(bench_ids,version_ids))

  for bench_id in range(nbenchs):
    train_idxs = [ i for (i,bid,vid) in zip(range(len(bench_ids)),bench_ids,version_ids) if bid != bench_id ]
    test_idxs = [ i for (i,bid,vid) in zip(range(len(bench_ids)),bench_ids,version_ids) if bid == bench_id and vid == 1 ]
    if len (test_idxs) == 0:
      test_idxs = [ i for (i,bid,vid) in zip(range(len(bench_ids)),bench_ids,version_ids) if bid == bench_id and vid == 0 ]

    if len(sys.argv)>1 and sys.argv[1]=="-train-only":
      cross_train_feats = train_feats
      cross_train_winners = train_winners
    elif len(sys.argv)>1 and sys.argv[1]=="-test-only":
      cross_train_feats = test_feats[train_idxs,:]
      cross_train_winners = test_winners[train_idxs]
    else:
      cross_train_feats = np.concatenate ( (test_feats[train_idxs,:], train_feats), axis=0)
      cross_train_winners = np.concatenate ( (test_winners[train_idxs], train_winners) )
    #print (cross_train_feats.shape)
    #print (cross_train_winners.shape)

    cross_test_feats = test_feats[test_idxs,:]
    cross_test_winners = test_winners[test_idxs]
    #print (cross_test_feats.shape)
    #print (cross_test_winners.shape)

    # print data
    print_data (cross_train_feats, cross_train_winners, "%s_train" % bench_names[bench_id], gpuname)
    print_data (cross_test_feats, cross_test_winners, "%s_test" % bench_names[bench_id], gpuname)

    #yield bench_id



if __name__ == "__main__":

  cross_validation("tesla")
  cross_validation("radeon")

