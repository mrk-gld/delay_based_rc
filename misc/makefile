COMP = g++
OPT = -std=gnu++17 -Ofast -DARMA_DONT_USE_WRAPPER
LIB = -I ../include -lm -pthread -llapack -lopenblas -larmadillo -lboost_system -lboost_filesystem -lstdc++

bifurcation_scan: bifurcation_scan.cpp ../include/reservoirs.h
		$(COMP) $(OPT) bifurcation_scan.cpp $(LIB) -o bifurcation_scan

generate_time_series: generate_time_series.cpp ../include/reservoirs.h
		$(COMP) $(OPT) generate_time_series.cpp $(LIB) -o generate_time_series