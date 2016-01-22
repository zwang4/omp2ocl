
import sys
import os

from util import run_benchmark

inputs = [ "S", "W", "A", "B", "C" ]
#inputs = [ "S", "W", "A", "B" ]
swap_configs = [ "NO", "RHS", "Z" ]
num_runs = 5


def main():
  program = sys.argv[1].lower()
  programU = program.upper()

  os.chdir ("%s-dimswap" % programU)

  # open output files
  outfiles = [ (open("%s-%s" % (program,config.lower()), "w"),
                open("%s-%s_swap" % (program,config.lower()), "w"), 
                open("%s-%s_z" % (program,config.lower()), "w"), 
                open("%s-%s_rhs" % (program,config.lower()), "w"))
                 for config in swap_configs ]

  for inp in inputs:
    print (inp)
    for config,outf in zip (swap_configs, outfiles):
      print (config)

      # copy files to where 'run_benchmark' expects them
      os.system ("cp %s.%s-%s.c %s.%s.c" % (program, inp, config, program, inp))
      os.system ("cp %s.%s-%s.cl %s.%s.cl" % (program, inp, config, program, inp))

      result = run_benchmark (program, inp, num_runs, swap=True)
      if result == None: continue

      num_fails,times,t_swap,t_z,t_rhs = result
      print (times)

      outf[0].write ( " ".join ( str(t) for t in times ) + "\n")
      outf[1].write ( " ".join ( str(t) for t in t_swap ) + "\n")
      outf[2].write ( " ".join ( str(t) for t in t_z ) + "\n")
      outf[3].write ( " ".join ( str(t) for t in t_rhs ) + "\n")

  # close files
  map (lambda f: [ f[i].close() for i in range(4) ], outfiles)


if __name__ == "__main__":
  main()

