FC = ifort
FFLAGS = -g -O3 -m64 -fPIC -fp-model strict -xSSE4.2 -fomit-frame-pointer \
		 -axCORE-AVX2,COMMON-AVX512 -i8 -mkl=parallel
FSRC = lbfgsb.f linpack.f timer.f

all: $(FSRC:.f=.o)

%.o: %.f
	$(FC) $(FFLAGS) -c $< -o $@

