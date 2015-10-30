CC = gcc

DEBUG ?= 0

VIDEO ?= 0
FRAMES ?= 0

NODE_ID_ENCODER ?= 28
LOCAL_ADAPTER_NO ?= 0

CCFLAGS = -Wall
LDFLAGS = -lm

ifeq ($(DEBUG),1)
	CCFLAGS += -Og -g -pg -DSHOW_CYCLES
else
	CCFLAGS += -O3
endif

# SISCI
DIS_SRC = /home/inf5063-g03/exam3/DIS/src

INCLUDE_PATHS = $(DIS_SRC)/include/dis
INCLUDE_PATHS += $(DIS_SRC)/SISCI/api
INCLUDE_PATHS += $(DIS_SRC)/SISCI/src

CCFLAGS += $(addprefix -idirafter,$(INCLUDE_PATHS))
LDFLAGS += -L/opt/DIS/lib64 -lsisci

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

all: c63write

%.o: %.c
	$(CC) $(CCFLAGS) -c $< -o $@
	
c63write: c63write.o tables.o io.o c63_write.o
	$(CC) $^ $(LDFLAGS) -o $@
	
clean:
	rm -f *.o c63write temp/* $(OUTPUT_VIDEO)

write: c63write
	./c63write -a $(LOCAL_ADAPTER_NO) -r $(NODE_ID_ENCODER) -o temp/test.c63
decode:
	./c63dec temp/test.c63 $(OUTPUT_VIDEO)

vlc:
	vlc --rawvid-width $(WIDTH) --rawvid-height $(HEIGHT) --rawvid-chroma I420 $(OUTPUT_VIDEO)
vlc-original:
	vlc --rawvid-width $(WIDTH) --rawvid-height $(HEIGHT) --rawvid-chroma I420 $(INPUT_VIDEO)
vlc-reference:
	vlc --rawvid-width $(WIDTH) --rawvid-height $(HEIGHT) --rawvid-chroma I420 $(REFERENCE_VIDEO)

gprof:
	gprof c63enc gmon.out -b
gprof-file:
	gprof c63enc gmon.out > temp/gprof-result.txt
nvprof:
	nvprof ./c63enc -w $(WIDTH) -h $(HEIGHT) -f $(FRAMES) -o temp/test.c63 $(INPUT_VIDEO)

psnr:
	./tools/yuv-tools/ycbcr.py psnr $(INPUT_VIDEO) $(WIDTH) $(HEIGHT) IYUV $(OUTPUT_VIDEO)
psnr-reference:
	./tools/yuv-tools/ycbcr.py psnr $(INPUT_VIDEO) $(WIDTH) $(HEIGHT) IYUV $(REFERENCE_VIDEO)
psnr-diff:
	./tools/yuv-tools/ycbcr.py psnr $(REFERENCE_VIDEO) $(WIDTH) $(HEIGHT) IYUV $(OUTPUT_VIDEO)
cmp:
	@cmp --silent -n "$$(wc -c < $(OUTPUT_VIDEO))" $(OUTPUT_VIDEO) $(REFERENCE_VIDEO) && echo "test == reference :D" || echo "test != reference :("
	
cachegrind:
	valgrind --tool=cachegrind --branch-sim=yes --cachegrind-out-file=temp/cachegrind.out ./c63enc -w $(WIDTH) -h $(HEIGHT) -f 30 -o temp/test.c63 $(INPUT_VIDEO)
