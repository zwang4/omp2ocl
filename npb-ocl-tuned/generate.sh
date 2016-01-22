#!/bin/sh

cleanup () {
  rm -f $program.c
  rm -f $program.cl~ $program.host.c~
  rm -f $program.$class
  rm -f *.o
  rm -f npbparams.h
}

CLANG_DIR=../../build_tool
CLANG_BIN=../$CLANG_DIR/bin/clang

if [ $# -lt 2 ]
then
  echo "Please specify a program & class (optional: suffix)"
  exit 1
fi

program=`echo $1 | awk '{print tolower($0)}'`
programU=`echo $1 | awk '{print toupper($0)}'`
class=$2
if [ $# -ge 3 ] && [ "$3" != "" ]
then
  suffix=-$3
else
  suffix=
fi
flags=$4

echo "compiling for $program$suffix.$class"
echo "flags: '$flags'"

cd $programU$suffix
if [ $? -ne 0 ]
then
  echo "unknown benchmark"
  exit 2
fi

# generate npbparams.h
../sys/setparams $program $class

# find template file
if [ -f $program.$class.c-template ]
then
  temp_file=$program.$class.c-template
else
  temp_file=$program.c-template
fi

if ! [ -f $temp_file ]
then
  echo "template file missing"
  exit 2
fi

# generate OpenCL code
cp $temp_file $program.c
$CLANG_BIN -cc1 -omp2ocl -arch=nvidia $flags -I../common/ $program.c
if [ $? -ne 0 ]
then
  echo "code generation failed"
  cleanup
  exit 2
fi

# rename files
mv $program.cl $program.$class.cl
mv $program.host.c $program.$class.c

# correct OpenCL kernel file
sed -i s/$program.cl/$program.$class.cl/ $program.$class.c
sed -i s/"\"NVIDIA\", 0"/"\"NVIDIA\", 1"/ $program.$class.c
sed -i s/"\(oclSwapDimensions\)\s*(\(\w*\)\s*,"/"\1 (__ocl_buffer_\2,"/ $program.$class.c

echo ""

cleanup

