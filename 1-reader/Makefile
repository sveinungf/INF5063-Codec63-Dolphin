CC = gcc

DEBUG ?= 0

VIDEO ?= 0
FRAMES ?= 0

NODE_ID_ENCODER ?= 28
LOCAL_ADAPTER_NO ?= 0

CCFLAGS = -Wall
LDFLAGS = -lm

ifeq ($(DEBUG),1)
	CCFLAGS += -Og -g -pg
else
	CCFLAGS += -O3
endif

### SISCI ###
DIS_SRC ?= /home/inf5063-g03/exam3/DIS/src

INCLUDE_PATHS = $(DIS_SRC)/include/dis
INCLUDE_PATHS += $(DIS_SRC)/SISCI/api
INCLUDE_PATHS += $(DIS_SRC)/SISCI/src

CCFLAGS += -D_REENTRANT
CCFLAGS += $(addprefix -idirafter,$(INCLUDE_PATHS))
LDFLAGS += -L/opt/DIS/lib64 -lsisci
#############

DIR_COMMON = ../common

CCFLAGS += -idirafter$(DIR_COMMON)

ifeq ($(VIDEO),0)
	WIDTH = 352
	HEIGHT = 288
	INPUT_VIDEO = /opt/cipr/foreman.yuv
else ifeq ($(VIDEO),1)
	WIDTH = 3840
	HEIGHT = 2160
	INPUT_VIDEO = /opt/cipr/foreman_4k.yuv
else ifeq ($(VIDEO),2)
	WIDTH = 1920
	HEIGHT = 1080
	INPUT_VIDEO = /opt/cipr/tractor.yuv
else ifeq ($(VIDEO),3)
	WIDTH = 4096
	HEIGHT = 1680
	INPUT_VIDEO = /opt/cipr/bagadus.yuv
endif

all: c63read

%.o: %.c
	$(CC) $(CCFLAGS) -c $< -o $@

ALL_OBJECTS = c63read.o sisci.o

c63read: $(ALL_OBJECTS)
	$(CC) $^ $(LDFLAGS) $(CCFLAGS) -o $@

clean:
	rm -f c63read $(ALL_OBJECTS)

read: c63read
	./c63read -w $(WIDTH) -h $(HEIGHT) -f $(FRAMES) -a $(LOCAL_ADAPTER_NO) -r $(NODE_ID_ENCODER) $(INPUT_VIDEO)

gprof:
	gprof c63read gmon.out -b
