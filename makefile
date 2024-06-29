CC=clang //clang usage because of XCODE MacOS
CCC=clang_openmp //alias created for omp.h library
CCCC=clang_mpi  //alias created for mpi.h library
CFLAGS=-Wall -O3 -openmp //change -fopenmp with -openmp because of mac with M2, different comp way 
CFLAGS+=-DDEBUG
LDLIBS=-lm

#TODO: add the following cases: multistart_mds_omp multistart_mds_omp_tasks multistart_mds_mpi

all: multistart_mds_seq

multistart_mds_seq: multistart_mds_seq.c torczon.c Makefile
	$(CC) $(CFLAGS) -o multistart_mds_seq multistart_mds_seq.c torczon.c $(LDLIBS)

	$(CCC) $(CFLAGS) -o multistart_mds_omp multistart_mds_omp.c torczon.c $(LDLIBS)

	$(CCC) $(CFLAGS) -o multistart_mds_omp_tasks multistart_mds_omp_tasks.c torczon.c $(LDLIBS)

	$(CCCC) $(CFLAGS) -o multistart_mds_omp_tasks multistart_mds_omp_tasks.c torczon.c $(LDLIBS)


clean:
	rm -f multistart_mds_seq
