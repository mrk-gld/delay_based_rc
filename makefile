COMP = g++
OPT = -std=gnu++17 -Ofast -DARMA_DONT_USE_WRAPPER
LIB = -I ./include -lm -pthread -llapack -lopenblas -larmadillo -lboost_system -lboost_filesystem -lstdc++
BUILDDIR = bin/

delay_based_RC: delay_based_RC.cpp
		$(COMP) $(OPT) delay_based_RC.cpp $(LIB) -o $(BUILDDIR)/delay_based_RC

deep_delay_based_RC: deep_delay_based_RC.cpp
		$(COMP) $(OPT) deep_delay_based_RC.cpp $(LIB) -o $(BUILDDIR)/deep_delay_based_RC

delay_based_RC_ipc: delay_based_RC_ipc.cpp
		$(COMP) $(OPT) delay_based_RC_ipc.cpp $(LIB) -o $(BUILDDIR)/delay_based_RC_ipc