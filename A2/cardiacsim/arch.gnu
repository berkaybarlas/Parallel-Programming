#########################################################################
#									#
# Sample makefile header for running with Gnu compilers  		#
#  The makefile targets are appended to  the end of this file		#
#	 Don't change anything that comes before the targets 		#
#									#
#									#
#########################################################################


#For MPI uncomment this line
MPI_flag    = 1

RM		= rm -f
LN		= ln -s
ECHO		= echo


C++ 		= g++
CC		= gcc

#MPI compilers
ifdef MPI_flag
 C++             = mpic++
 CC              = mpicc
endif

C++LINK		= $(C++)
CLINK		= $(CC)



ARCH_FLAGS      = -DLINUX 
WARNINGS        = 
# OPTIMIZATION    =  -O3 -ftree-vectorize 
OPTIMIZATION    =  -O3 
#DEBUG          = -g

C++FLAGS        += $(INCLUDES) $(ARCH_FLAGS) $(WARNINGS) $(OPTIMIZATION) \
                  $(XTRAFLAGS) $(DEBUG)

CFLAGS		+= $(INCLUDES) $(ARCH_FLAGS) $(WARNINGS) $(OPTIMIZATION) \
                  $(XTRAFLAGS) $(DEBUG)

FFLAGS		= $(ARCH_FLAGS) -O2 -fno-second-underscore -ff90 -fugly-complex

LDFLAGS		= $(WARNINGS) $(OPTIMIZATION) $(DEBUG)
# *** LDFLAGS		= $(WARNINGS) $(OPTIMIZATION) $(DEBUG) -L/afs/nada.kth.se/home/5/u9034805/Public/lib/PPF 
# *** LDLIBS		= -lptools_ppf


ARCH_HAS_X	= arch_has_X



#########################################################################
# End of the System dependent prefix
#########################################################################


#########################################################################
#									#
# Suffixes for compiling most normal C++ and  C files		        #
#									#
#########################################################################

.SUFFIXES:
.SUFFIXES: .C .c .o

.C.o:
		@$(ECHO)
		@$(ECHO) "Compiling Source File --" $<
		@$(ECHO) "---------------------"
		$(C++) $(C++FLAGS) -c $<
		@$(ECHO)

.c.o:
		@$(ECHO)
		@$(ECHO) "Compiling Source File --" $<
		@$(ECHO) "---------------------"
		$(CC) $(CFLAGS) -c $<
		@$(ECHO)

