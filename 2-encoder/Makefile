CC = gcc
NVCC = nvcc

DEBUG ?= 0

VIDEO ?= 0
FRAMES ?= 0

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
	OUTPUT_VIDEO = ~/yuv/test/foreman.yuv
	REFERENCE_VIDEO = ~/yuv/reference/foreman.yuv
else ifeq ($(VIDEO),1)
	OUTPUT_VIDEO = ~/yuv/test/foreman_4k.yuv
	REFERENCE_VIDEO = ~/yuv/reference/foreman_4k.yuv
else ifeq ($(VIDEO),2)
	OUTPUT_VIDEO = ~/yuv/test/tractor.yuv
	REFERENCE_VIDEO = ~/yuv/reference/tractor.yuv
endif

all: c63enc

%.o: %.c
	$(CC) $(CCFLAGS) -c $< -o $@
%.o: %.cu
	$(NVCC) $(ALL_NVCCFLAGS) -c $< -o $@

c63enc: c63enc.o dsp.o tables.o io.o c63_write.o common.o me.o
	$(NVCC) $(ALL_NVCCFLAGS) $(ALL_LDFLAGS) -o $@ $^
c63dec: c63dec.c dsp.o tables.o io.o common.o me.o
	$(NVCC) $(ALL_NVCCFLAGS) $(ALL_LDFLAGS) -o $@ $^
c63pred: c63dec.c dsp.o tables.o io.o common.o me.o
	$(NVCC) $(ALL_NVCCFLAGS) $(ALL_LDFLAGS) -DC63_PRED -o $@ $^

clean:
	rm -f *.o c63enc temp/* $(OUTPUT_VIDEO)

encode: c63enc
	./c63enc -a $(LOCAL_ADAPTER_NO) -r $(NODE_ID_READER) -o temp/test.c63
decode:
	./c63dec temp/test.c63 $(OUTPUT_VIDEO)

vlc:
	vlc --rawvid-width $(WIDTH) --rawvid-height $(HEIGHT) --rawvid-chroma I420 $(OUTPUT_VIDEO)

gprof:
	gprof c63enc gmon.out -b
nvprof:
	nvprof ./c63enc -a $(LOCAL_ADAPTER_NO) -r $(NODE_ID_READER) -o temp/test.c63

cmp:
	@cmp --silent -n "$$(wc -c < $(OUTPUT_VIDEO))" $(OUTPUT_VIDEO) $(REFERENCE_VIDEO) && echo "test == reference :D" || echo "test != reference :("
