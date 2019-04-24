include ./arch.gnu
# OPTIMIZATION = -fast
# OPTIMIZATION = -O3
# DEBUG += -g


app:		cardiacsim

OBJECTS = cardiacsim.o splot.o cmdLine.o 

cardiacsim:	        $(OBJECTS) 
		$(C++LINK) $(LDFLAGS) -o $@ $(OBJECTS)  $(LDLIBS)
mpi: 				$(OBJECTS) 
		$(C++LINK) $(LDFLAGS) -o cardiacsim-mpi $(OBJECTS) $(LDLIBS)
serial: $(OBJECTS) 
		$(C++LINK) $(LDFLAGS) -o cardiacsim-serial $(OBJECTS) $(LDLIBS)
openmp: $(OBJECTS) 
		$(C++LINK) $(LDFLAGS) $(OPENMP) -o cardiacsim-openmp $(OBJECTS) $(LDLIBS)

test: test.o
	$(C++LINK) $(LDFLAGS) -o $@ test.o  $(LDLIBS)
clean:	
	$(RM) *.o cardiacsim *~;
	$(RM) cardiacsim-mpi *~;
	$(RM) cardiacsim-serial *~;
	$(RM) cardiacsim-openmp *~;
	$(RM) core;
