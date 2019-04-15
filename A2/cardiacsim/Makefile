include ./arch.gnu
# OPTIMIZATION = -fast
# OPTIMIZATION = -O3
# DEBUG += -g


app:		cardiacsim

OBJECTS = cardiacsim.o splot.o cmdLine.o

cardiacsim:	        $(OBJECTS) 
		$(C++LINK) $(LDFLAGS) -o $@ $(OBJECTS)  $(LDLIBS)

clean:	
	$(RM) *.o cardiacsim *~;
	$(RM) core;
