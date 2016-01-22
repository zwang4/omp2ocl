
import re
import sys
import subprocess
import time
import datetime

def parse_output (output):
  time = None
  t_swap = None
  t_z = None
  t_rhs = None
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

    m = re.match ("t_(\w+): (\d+\.\d+)", line)
    if m:
      if m.group(1) == "swap": t_swap = float (m.group(2))
      elif m.group(1) == "z": t_z = float (m.group(2))
      elif m.group(1) == "rhs": t_rhs = float (m.group(2))
      continue

  if verified: return (time,t_swap,t_z,t_rhs)
  else: return None


# assumes that current directory is the benchmark directory
def run_benchmark (bench, inp, num_runs, max_fails=2, swap=False):

  # compile benchmark for specific input
  cmd = "cp %s.%s.c %s.c && make clean && make CLASS=%s" % (bench, inp, bench, inp)
  #print (cmd)
  sys.stdout.flush()
  compiler_proc = subprocess.Popen (cmd, shell=True)
  rc = compiler_proc.wait ()
  if rc != 0:
    print ("compilation failed (rc: %d)" % rc)
    return None

  times = []
  if swap:
    t_swap = []
    t_z = []
    t_rhs = []
  num_fails = 0


  # run benchmark number of times
  run = 0 # number of successful runs
  while run < num_runs and num_fails <= max_fails:
    # wait a bit between program runs
    #time.sleep (1)

    print (datetime.datetime.now().strftime ("%Y-%m-%d %H:%M:%S"))
    print ("%s.%s: run %d" % (bench, inp, run+1))

    cmd = "./%s.%s" % (bench, inp)
    sys.stdout.flush()
    bench_proc = subprocess.Popen (cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    rc = bench_proc.wait ()
    if rc != 0:
      print ("run failed (rc: %d)" % rc)
      num_fails = num_fails + 1
      continue

    result = parse_output (bench_proc.stdout)
    if result:
      times.append (result[0])
      if swap:
        t_swap.append (result[1])
        t_z.append (result[2])
        t_rhs.append (result[3])
      run += 1
    else:
      print ("verification failed")
      num_fails = num_fails + 1
      continue

  if swap:
    return (num_fails, times, t_swap, t_z, t_rhs)
  else:
    return (num_fails, times)

