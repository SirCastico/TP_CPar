CC = gcc
SRC = src/
CFLAGS = -O3 -g -Wall -mavx

.DEFAULT_GOAL = all

all: MDseq.exe MDpar.exe MDcuda.exe

MDpar.exe: $(SRC)/MDpar.c
	$(CC) $(CFLAGS) $(SRC)MDpar.c -lm -fopenmp -o MDpar.exe

MDseq.exe: $(SRC)/MDseq.c
	$(CC) $(CFLAGS) $(SRC)MDseq.c -lm -o MDseq.exe

MDcuda.exe: $(SRC)/MD.cu
	nvcc -O3 -g -std=c++11 -arch=sm_35 $(SRC)MD.cu -lm -Wno-deprecated-gpu-targets -o MDcuda.exe

clean:
	rm ./MDpar.exe ./MDseq.exe

runseq:
	./MDseq.exe < inputdata.txt
runpar:
	./MDpar.exe < inputdata.txt

cmp:
	python compare.py cp_average.txt seq_average.txt
	python compare.py cp_output.txt seq_output.txt
	python compare.py cp_traj.xyz seq_traj.xyz

runseq_perf:
	perf stat -d ./MDseq.exe < inputdata.txt

runpar_perf:
	perf stat -d ./MDpar.exe < inputdata.txt

runseq_annotate:
	perf record ./MDseq.exe < inputdata.txt
	perf annotate -n

runpar_annotate:
	perf record ./MDpar.exe < inputdata.txt
	perf annotate -n
