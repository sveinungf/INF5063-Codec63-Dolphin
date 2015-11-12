CC = gcc
NVCC = nvcc

DEBUG ?= 0

VIDEO ?= 0

NODE_ID_READER ?= 4
NODE_ID_WRITER ?= 8
LOCAL_ADAPTER_NO ?= 0

CCFLAGS = -Wall -march=native
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

CCFLAGS += -D_REENTRANT
CCFLAGS += $(addprefix -idirafter,$(INCLUDE_PATHS))
LDFLAGS += -L/opt/DIS/lib64 -lsisci

ALL_NVCCFLAGS = $(NVCCFLAGS)
ALL_NVCCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_NVCCFLAGS += $(addprefix -Xptxas ,$(PTXFLAGS))

ALL_LDFLAGS = $(addprefix -Xlinker ,$(LDFLAGS))

ifeq ($(VIDEO),0)
	WIDTH = 352
	HEIGHT = 288
	INPUT_VIDEO = /opt/cipr/foreman.yuv
	REFERENCE_VIDEO = ~/yuv/reference/foreman.yuv
	REFERENCE_C63 = ~/c63/reference/foreman.c63
else ifeq ($(VIDEO),1)
	WIDTH = 3840
	HEIGHT = 2160
	INPUT_VIDEO = /opt/cipr/foreman_4k.yuv
	REFERENCE_VIDEO = ~/yuv/reference/foreman_4k.yuv
	REFERENCE_C63 = ~/c63/reference/foreman_4k.yuv
else ifeq ($(VIDEO),2)
	WIDTH = 1920
	HEIGHT = 1080
	INPUT_VIDEO = /opt/cipr/tractor.yuv
	REFERENCE_VIDEO = ~/yuv/reference/tractor.yuv
	REFERENCE_C63 = ~/c63/reference/tractor.c63
endif

C63_FILE = ../temp/test.c63
OUTPUT_VIDEO = ../temp/output.yuv

all: c63enc

%.o: %.c
	$(CC) $(CCFLAGS) -c $< -o $@
%.o: %.cu
	$(NVCC) $(ALL_NVCCFLAGS) -c $< -o $@
	
OBJECTS_ROOT = allocation.o c63enc.o init.o sisci.o tables.o
OBJECTS_CUDA = dsp.o init_cuda.o me.o
OBJECTS_SIMD = dsp.o me.o

ALL_OBJECTS = $(OBJECTS_ROOT)
ALL_OBJECTS += $(addprefix cuda/,$(OBJECTS_CUDA))
ALL_OBJECTS += $(addprefix simd/,$(OBJECTS_SIMD))   

c63enc: $(ALL_OBJECTS)
	$(NVCC) $(ALL_NVCCFLAGS) $(ALL_LDFLAGS) -o $@ $^

clean:
	find . -type f -name '*.o' -delete
	rm -f c63enc $(OUTPUT_VIDEO)

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
	
PSNR_EXEC = ./tools/libyuv-tools-r634-linux-x86_64/bin/psnr

psnr:
	$(PSNR_EXEC) -s $(WIDTH) $(HEIGHT) -v $(INPUT_VIDEO) $(OUTPUT_VIDEO)
psnr-reference:
	$(PSNR_EXEC) -s $(WIDTH) $(HEIGHT) -v $(INPUT_VIDEO) $(REFERENCE_VIDEO)
psnr-diff:
	$(PSNR_EXEC) -s $(WIDTH) $(HEIGHT) -v $(REFERENCE_VIDEO) $(OUTPUT_VIDEO)

cmp:
	@cmp --silent -n "$$(wc -c < $(C63_FILE))" $(C63_FILE) $(REFERENCE_C63) && echo "test == reference :D" || echo "test != reference :("
		