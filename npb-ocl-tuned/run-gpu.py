import os
import sys
import subprocess
import re
import time
import datetime
import sqlite3

inputs = [ "B", "C" ]
#inputs = [ "S", "B" ]
#inputs = [ "S", "W", "A", "B", "C" ]
#inputs = [ "S", "W", "A", "B" ]

benchmarks = [
               #("bt", ["tuned"]),
               #("cg", ["ocl", "tuned"]),
               #("ep", ["global", "private"]),
               #("is", ["ocl"]),
               #("lu", ["soa"]),
               #("mg", ["swap", "global"]),
               #("sp", ["tuned"]),
               #("ft", ["soa"]),

               ("ep", ["global"]),
             ]

num_runs = 5
max_fails = 0
max_time_steps = {"S":5, "W":7, "A":9, "B":10, "C": 11}
max_time = {"S":60, "W":300, "A":1200, "B":2400, "C": 3600}


def create_table (cursor, bench, version):
  query = "CREATE TABLE IF NOT EXISTS %s_%s (input TEXT, time REAL, transfer INT)" % (bench, version.replace('-','_'))
  #print (query)
  cursor.execute (query)

def clear_runs (cursor, bench, version, inp):
  query = "DELETE FROM %s_%s WHERE input='%s'" % (bench,version.replace('-','_'),inp)
  #print (query)
  cursor.execute (query)

def record_run (cursor, bench, version, inp, time, transfer):
  query = "INSERT INTO %s_%s (input,time,transfer) VALUES ('%s',%f,%d)" % (bench,version.replace('-','_'),inp,time,transfer)
  #print (query)
  cursor.execute (query)


def parse_output (output):
  time = None
  verified = None
  bytes_transferred = 0

  for line in output:
    #print (line)
    m = re.match (" Verification\s+=\s+(\w+)", line)
    if m:
      if m.group(1) == "SUCCESSFUL":
        verified = True
      else:
        verified = False
      continue

    m = re.match (" Time in seconds\s+=\s+(\d+\.\d+)", line)
    if m:
      time = float (m.group(1))
      continue

    m = re.match ("bytes transferred: (\d+)", line)
    if m:
      bytes_transferred += int(m.group(1))
      continue

  if verified: return (time,bytes_transferred)
  else: return None



if __name__ == "__main__":

  top_dir = os.getcwd ()

  logfile = open ("gpu.log", 'a')
  logfile.write("%s\n\n" % (datetime.datetime.now().strftime ("%Y-%m-%d %H:%M:%S")))

  # database setup
  db = sqlite3.connect ("tesla.db")
  cursor = db.cursor ()

  for (bench,versions) in benchmarks:
    benchU = bench.upper();


    for version in versions:
      print ("")
      print ("")
      print ("%s-%s" % (bench,version))
      print ("")

      create_table (cursor, bench, version)

      os.chdir ("%s-%s" % (benchU,version))

      for inp in inputs:

        clear_runs (cursor, bench, version, inp)

        # compile benchmark for specific input
        cmd = "cp %s.%s.c %s.c && make clean && make CLASS=%s" % (bench, inp, bench, inp)
        #print (cmd)
        sys.stdout.flush()
        compiler_proc = subprocess.Popen (cmd, shell=True)
        rc = compiler_proc.wait ()
        if rc != 0:
          print ("compilation failed (rc: %d)" % rc)
          continue

        times = []
        transfers = []
        num_fails = 0

        # run benchmark number of times
        run = 0 # number of successful runs
        while run < num_runs and num_fails <= max_fails:
          # wait a bit between program runs
          #time.sleep (1)

          print (datetime.datetime.now().strftime ("%Y-%m-%d %H:%M:%S"))
          print ("%s.%s: run %d" % (bench, inp, run+1))

          #cmd = "timeout %d ./%s.%s" % (max_time[inp], bench, inp)
          cmd = "./%s.%s" % (bench, inp)
          sys.stdout.flush()
          bench_proc = subprocess.Popen (cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

          #try:
          #  rc = bench_proc.wait (timeout = max_time[inp])
          #except TimeoutExpired:
          #  print ("TIMEOUT")
          #  bench_proc.kill ()
          #  print ("kill send")
          #  bench_proc.wait ()
          #  print ("process terminated")
          #  num_fails = num_fails + 1
          #  continue
          #  
          #if rc != 0:
          #  print ("run failed (rc: %d)" % rc)
          #  num_fails = num_fails + 1
          #  continue


          # wait a minute, if compilation not finished by then program probably crashed
          #time_steps = 0
          #rc = bench_proc.poll ()
          #while rc == None:
          #  print ("sleep")
          #  time.sleep (10)
          #  time_steps += 1
          #  if time_steps >= 6:
          #    print ("TIMEOUT")
          #    break
          #  rc = bench_proc.poll ()
          #if rc == None:
          #  # check if compilation has finished
          #  comp_finished = False
          #  print ("flush stderr")
          #  bench_proc.stderr.flush();
          #  print ("done")
          #  for line in bench_proc.stderr:
          #    print (line)
          #    if line == "compilation finished\n":
          #      comp_finished = True
          #      break
          #  if not comp_finished:
          #    bench_proc.kill ()
          #    print ("kill send")
          #    bench_proc.wait ()
          #    print ("process terminated")
          #    num_fails = num_fails + 1
          #    continue

          #time_steps = 0
          #interval = 1
          #rc = bench_proc.poll ()
          #while rc == None:
          #  print ("sleep %d" % interval)
          #  time.sleep (interval)
          #  interval *= 2
          #  time_steps += 1
          #  if time_steps > max_time_steps[inp]:
          #    print ("TIMEOUT")
          #    break
          #  rc = bench_proc.poll ()
          #if rc == None:
          #  bench_proc.kill ()
          #  print ("kill send")
          #  bench_proc.wait ()
          #  print ("process terminated")
          #  num_fails = num_fails + 1
          #  continue
            
          rc = bench_proc.wait ()
          if rc != 0:
            print ("run failed (rc: %d)" % rc)
            num_fails = num_fails + 1
            continue

          result = parse_output (bench_proc.stdout)
          if result:
            times.append (result[0])
            transfers.append (result[1])
            record_run (cursor, bench, version, inp, result[0], result[1])
            run += 1
          else:
            print ("verification failed")
            num_fails = num_fails + 1
            continue

        # print results to file
        if num_fails > max_fails:
          logfile.write ("%s-%s.%s: %d failures in %d runs\n" % (bench,version,inp,num_fails,num_runs))
          print ("%d failures in %d runs" % (num_fails, num_runs))
          clear_runs (cursor, bench, version, inp) # delete data if too many runs failed
        print (times)
        print (transfers)
        if len(times) > 0: print ("AVG: %f" % (sum(times) / len(times)))

        db.commit ()

      # go back to top directory
      os.chdir (top_dir)

  logfile.write ("\n----------------------------------------\n");
  logfile.close()

  db.close()

