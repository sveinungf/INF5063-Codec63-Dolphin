CC = gcc

DEBUG ?= 0

NODE_ID_ENCODER ?= 28
LOCAL_ADAPTER_NO ?= 0

CCFLAGS = -Wall
LDFLAGS = -lm

ifeq ($(DEBUG),1)
	CCFLAGS += -Og -g -pg
else
	CCFLAGS += -O3
endif

# SISCI
DIS_SRC = /home/inf5063-g03/exam3/DIS/src

INCLUDE_PATHS = $(DIS_SRC)/include/dis
INCLUDE_PATHS += $(DIS_SRC)/SISCI/api
INCLUDE_PATHS += $(DIS_SRC)/SISCI/src

CCFLAGS += $(addprefix -idirafter,$(INCLUDE_PATHS))
LDFLAGS += -L/opt/DIS/lib64 -lsisci -lpthread

C63_FILE = ../temp/test.c63

all: c63write

%.o: %.c
	$(CC) $(CCFLAGS) -c $< -o $@
	
c63write: c63write.o tables.o c63_write.o sisci.o
	$(CC) $^ $(LDFLAGS) $(CCFLAGS) -o $@
	
clean:
	rm -f *.o c63write $(C63_FILE)

write: c63write
	./c63write -a $(LOCAL_ADAPTER_NO) -r $(NODE_ID_ENCODER) -o $(C63_FILE)

gprof:
	gprof c63write gmon.out -b
