; RUN: llc < %s -march=ptx -mattr=ptx20 | grep ".version 2.0"
; RUN: llc < %s -march=ptx -mattr=ptx21 | grep ".version 2.1"
; RUN: llc < %s -march=ptx -mattr=ptx22 | grep ".version 2.2"
; RUN: llc < %s -march=ptx -mattr=sm10 | grep ".target sm_10"
; RUN: llc < %s -march=ptx -mattr=sm13 | grep ".target sm_13"
; RUN: llc < %s -march=ptx -mattr=sm20 | grep ".target sm_20"

define ptx_device void @t1() {
	ret void
}
