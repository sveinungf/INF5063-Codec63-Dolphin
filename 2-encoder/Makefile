CC = gcc
NVCC = nvcc

DEBUG ?= 0

NODE_ID_READER ?= 4
NODE_ID_WRITER ?= 8
LOCAL_ADAPTER_NO ?= 0

CCFLAGS = -Wall
NVCCFLAGS = -std=c++11 -fmad=false -arch sm_50 -lineinfo
PTXFLAGS = -warn-double-usage -warn-lmem-usage -warn-spills
LDFLAGS = -lm

ifeq ($(DEBUG),1)
	CCFLAGS += -Og -g -pg
	NVCCFLAGS += -G -lineinfo
else
	CCFLAGS += -O3
endif

#SISCI
DIS_SRC = /home/inf5063-g03/exam3/DIS/src

INCLUDE_PATHS = $(DIS_SRC)/include/dis
INCLUDE_PATHS += $(DIS_SRC)/SISCI/api
INCLUDE_PATHS += $(DIS_SRC)/SISCI/src

CCFLAGS += $(addprefix -idirafter,$(INCLUDE_PATHS))
LDFLAGS += -L/opt/DIS/lib64 -lsisci

ALL_NVCCFLAGS = $(NVCCFLAGS)
ALL_NVCCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_NVCCFLAGS += $(addprefix -Xptxas ,$(PTXFLAGS))

ALL_LDFLAGS = $(addprefix -Xlinker ,$(LDFLAGS))

ifeq ($(VIDEO),0)
	REFERENCE_C63 = ~/c63/reference/foreman.c63
else ifeq ($(VIDEO),1)
	REFERENCE_C63 = ~/c63/reference/foreman_4k.yuv
else ifeq ($(VIDEO),2)
	REFERENCE_C63 = ~/c63/reference/tractor.c63
endif

C63_FILE = ../temp/test.c63
OUTPUT_VIDEO = ../temp/output.yuv

all: c63enc

%.o: %.c
	$(CC) $(CCFLAGS) -c $< -o $@
%.o: %.cu
	$(NVCC) $(ALL_NVCCFLAGS) -c $< -o $@

c63enc: c63enc.o dsp.o tables.o common.o me.o sisci.o
	$(NVCC) $(ALL_NVCCFLAGS) $(ALL_LDFLAGS) -o $@ $^

clean:
	rm -f *.o c63enc $(OUTPUT_VIDEO)

encode: c63enc
	./c63enc -a $(LOCAL_ADAPTER_NO) -r $(NODE_ID_READER) -w $(NODE_ID_WRITER)
decode:
	./c63dec $(C63_FILE) $(OUTPUT_VIDEO)

vlc:
	vlc --rawvid-width $(WIDTH) --rawvid-height $(HEIGHT) --rawvid-chroma I420 $(OUTPUT_VIDEO)

gprof:
	gprof c63enc gmon.out -b
nvprof:
	nvprof ./c63enc -a $(LOCAL_ADAPTER_NO) -r $(NODE_ID_READER) -w $(NODE_ID_WRITER)

cmp:
	@cmp --silent -n "$$(wc -c < $(C63_FILE))" $(C63_FILE) $(REFERENCE_C63) && echo "test == reference :D" || echo "test != reference :("
		