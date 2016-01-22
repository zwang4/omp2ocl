set(MODULE TRUE)

set( LLVM_USED_LIBS
  clangFrontend
  clangAST
  )

set( LLVM_LINK_COMPONENTS support mc)

add_clang_library(Omp2OpenCL Omp2OpenCL.cpp)

set_target_properties(Omp2OpenCL
  PROPERTIES
  LINKER_LANGUAGE CXX
  PREFIX "")
