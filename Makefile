CC = gcc
SRC = src/
PROF_DIR = prof/
CFLAGS = -O3 -g

.DEFAULT_GOAL = MD.exe

MD.exe: $(SRC)/MD.c
	$(CC) $(CFLAGS) $(SRC)MD.c -lm -o MD.exe

MD_prof: $(SRC)/MD.c
	$(CC) $(CFLAGS) $(SRC)MD.c -pg -lm -o $(PROF_DIR)MD.exe

gprof: MD_prof
	$(PROF_DIR)/MD.exe < inputdata.txt
	gprof $(PROF_DIR)MD.exe > $(PROF_DIR)gprof.out

clean:
	rm ./MD.exe

run:
	./MD.exe < inputdata.txt
