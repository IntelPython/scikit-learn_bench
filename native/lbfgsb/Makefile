FC = ifort
FFLAGS = -g -O
FSRC = lbfgsb.f linpack.f timer.f

all: $(FSRC:.f=.o)

%.o: %.f
	$(FC) $(FFLAGS) -c $< -o $@

