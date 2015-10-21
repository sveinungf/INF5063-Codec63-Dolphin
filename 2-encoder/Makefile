# Acknowledgement: Functionality for creating make rules of dependencies is
# based on code presented here <http://codereview.stackexchange.com/q/11109>

# CC = gcc  # To specify compiler use: '$ CC=clang make clean all'
CFLAGS = -O3 -Wall -g
LDFLAGS = -lm
# Use the compiler to generate make rules. See gcc manual for details.
MFLAGS = -MMD -MP -MF

SOURCES = $(wildcard *.c)
OBJECTS = $(SOURCES:.c=.o)
DEPENDENCIES = $(addprefix .,$(SOURCES:.c=.d))  # Add dot prefix to hide files.

.PHONY: clean  all

all: c63enc c63dec c63pred

c63enc: c63enc.o dsp.o tables.o io.o c63_write.o common.o me.o
	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@
c63dec: c63dec.c dsp.o tables.o io.o common.o me.o
	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@
c63pred: c63dec.c dsp.o tables.o io.o common.o me.o
	$(CC) $^ -DC63_PRED $(CFLAGS) $(LDFLAGS) -o $@

%.o: %.c
	$(CC) $(CFLAGS) $(MFLAGS) $(addprefix .,$(patsubst %.o,%.d,$@)) -c $< -o $@

clean:
	$(RM) c63enc c63dec c63pred $(OBJECTS) $(DEPENDENCIES)

-include $(DEPENDENCIES)
