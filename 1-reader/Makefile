CC = gcc
NVCC = nvcc

CCFLAGS = -Wall
LDFLAGS = -lm

DEBUG ?= 0

ifeq ($(DEBUG),1)
	CCFLAGS += -Og -g -pg -DSHOW_CYCLES
else
	CCFLAGS += -O3
endif

DIS_SRC = /home/inf5063-g03/exam3/DIS/src

INCLUDE_PATHS = $(DIS_SRC)/include/dis
INCLUDE_PATHS += $(DIS_SRC)/SISCI/api
INCLUDE_PATHS += $(DIS_SRC)/SISCI/src

CCFLAGS += $(addprefix -idirafter,$(INCLUDE_PATHS))
LDFLAGS += -L/opt/DIS/lib64 -lsisci

ALL_NVCCFLAGS = $(NVCCFLAGS)

ALL_LDFLAGS = $(addprefix -Xlinker ,$(LDFLAGS))

VIDEO ?= 0
FRAMES ?= 0

NODE_ID_ENCODER ?= 28
LOCAL_ADAPTER_NO ?= 0

ifeq ($(VIDEO),0)
	WIDTH = 352
	HEIGHT = 288
	INPUT_VIDEO = /opt/cipr/foreman.yuv
	OUTPUT_VIDEO = ~/yuv/test/foreman.yuv
	REFERENCE_VIDEO = ~/yuv/reference/foreman.yuv
else ifeq ($(VIDEO),1)
	WIDTH = 3840
	HEIGHT = 2160
	INPUT_VIDEO = /opt/cipr/foreman_4k.yuv
	OUTPUT_VIDEO = ~/yuv/test/foreman_4k.yuv
	REFERENCE_VIDEO = ~/yuv/reference/foreman_4k.yuv
else ifeq ($(VIDEO),2)
	WIDTH = 1920
	HEIGHT = 1080
	INPUT_VIDEO = /opt/cipr/tractor.yuv
	OUTPUT_VIDEO = ~/yuv/test/tractor.yuv
	REFERENCE_VIDEO = ~/yuv/reference/tractor.yuv
endif

all: c63enc

%.o: %.c
	$(CC) $(CCFLAGS) -c $< -o $@

c63enc: c63enc.o
	$(NVCC) $(ALL_NVCCFLAGS) $(ALL_LDFLAGS) -o $@ $^

clean:
	rm -f *.o c63enc temp/* $(OUTPUT_VIDEO)

encode: c63enc
	./c63enc -w $(WIDTH) -h $(HEIGHT) -f $(FRAMES) -a $(LOCAL_ADAPTER_NO) -r $(NODE_ID_ENCODER) $(INPUT_VIDEO)

gprof:
	gprof c63enc gmon.out -b
gprof-file:
	gprof c63enc gmon.out > temp/gprof-result.txt

cachegrind:
	valgrind --tool=cachegrind --branch-sim=yes --cachegrind-out-file=temp/cachegrind.out ./c63enc -w $(WIDTH) -h $(HEIGHT) -f 30 -o temp/test.c63 $(INPUT_VIDEO)
