/*

written by: Mirko Goldmann
date: 2023-05-15

*/

#include <armadillo>
#include <cmath>
#include "reservoirs.h"
#include <ostream>

using namespace arma;
using namespace std;

#define MIN(a,b) (((a)<(b))?(a):(b))

//#define LOGFILE // comment out to disable logging
// be aware logging slows down the simulation crucially

//generate log file
#ifdef LOGFILE
    ofstream logfile("delay_rc_slo.log");
    ofstream state_file("delay_rc_slo_states.csv");
#endif

template<typename T>
vec integrate_dde(dde_reservoir<T> *RC, int length){

    vec states = vec(length,fill::zeros);
    for (int k = 0; k<length; k++){
        
        RC->runge_kutta_4th_order(0);

        #ifdef LOGFILE
            state_file << RC->readout() << "," << endl;
        #endif
        
        states(k) = RC->readout();
        
    }

    return states;
}

map<string, float> get_parameter_map_from_arg(int argc, char** argv){
    map<string,float> params;
    if( argc > 1 ) {
        // set parameters from command line using pattern -parameter=value
    
        for (int i=1; i < argc; i++){
            string arg = argv[i];
            cout << arg << endl;
            size_t pos = arg.find("=");
            string key = arg.substr(1,pos-1);
            string val = arg.substr(pos+1);
            float f_val = stof(val);
            params[key] = f_val;
        }
    }
    return params;
};

// Function to find local minima and maxima in a time series
pair<vec, vec> findLocalExtrema(const vec& timeSeries, uword n)
{
    const float minDistance = 0.01;
    vec minimaValues, maximaValues;

    if (timeSeries.n_elem < 2 * n + 1) {
        // Time series should have enough elements to have local extrema
        return make_pair(minimaValues, maximaValues);
    }

    for (uword i = n; i < timeSeries.n_elem - n; ++i) {
        bool isMinimum = true;
        bool isMaximum = true;

        for (uword j = i - n; j <= i + n; ++j) {
            if (j != i && (timeSeries(j) <= timeSeries(i))) {
                isMinimum = false;
            }
            if (j != i && (timeSeries(j) >= timeSeries(i))) {
                isMaximum = false;
            }
        }

        if (isMinimum) {
            // Found a local minimum
            bool isFarEnough = true;
            for (uword j = 0; j < minimaValues.n_elem; ++j) {
                if (abs(timeSeries(i) - minimaValues(j)) < minDistance) {
                    isFarEnough = false;
                    break;
                }
            }

            if (!isFarEnough) {
                continue;  // Skip this minimum
            }
            minimaValues.resize(minimaValues.n_elem + 1);
            minimaValues(minimaValues.n_elem - 1) = timeSeries(i);
        } else if (isMaximum) {

            bool isFarEnough = true;
            for (uword j = 0; j < maximaValues.n_elem; ++j) {
                if (abs(timeSeries(i) - maximaValues(j)) < minDistance) {
                    isFarEnough = false;
                    break;
                }
            }

            if (!isFarEnough) {
                continue;  // Skip this maximum
            }
            // Found a local maximum
            maximaValues.resize(maximaValues.n_elem + 1);
            maximaValues(maximaValues.n_elem - 1) = timeSeries(i);
        }
    }

    return make_pair(minimaValues, maximaValues);
}
/*
TODO:

*/

int main(int argc, char** argv){

    /* you can choose between different model for the delay-based reservoir:
        - class SLO: Stuart-Landau oscillator
        - class MGO: Mackey-Glass oscillator
        - class Ikeda: Ikeda oscillator
        - class LangKobayashi: Lang-Kobayashi model
    */

    // defining the stuart landau delay based reservoir
    //dde_reservoir<complex<float>> * rc = new SLO();

    // defining the mackey glass delay based reservoir
    //dde_reservoir<float> * rc = new MGO();

    // defining the ikeda delay based reservoir
    dde_reservoir<float> * rc = new Ikeda();

    // defining the lang kobayashi delay based reservoir
    //dde_reservoir<complex<float>> * rc = new LangKobayashi();

    //print selected reservoir
    cout << "Selected reservoir: " << rc->name << endl;

    // set reservoir parameters
    map<string,float> params = get_parameter_map_from_arg(argc, argv);

    // Open a file for writing the extremas
    ofstream outFile("extrema.csv");
    // Write the header
    outFile << "Type,Value,Parameter" << endl;

    //perform scan over certain parameter
    string param_name = "phi";
    const int num_step = 700;
    //generate vec that contains scan parameter values forward and backward
    vec scan_param_values_fwd = linspace<vec>(-0.2,3.3,num_step);    
    vec scan_param_values_bwd = linspace<vec>(3.29,-0.19,num_step);
    vec scan_param_values = join_cols(scan_param_values_fwd,scan_param_values_bwd);

    // several initial conditions in a list
    const int num_init_cond = 2;

    rc->set_parameters(params);
    rc->print_parameters();
    rc->init_delay();

    const float init_time = 50000;
    const float compute_time = 2000;
    for (int k= 0; k < scan_param_values.n_elem; k++){

        params[param_name] = scan_param_values(k);
        printf("Parameter %s = %f\n",param_name.c_str(),scan_param_values(k));

        rc->set_parameters(params);
        rc->print_parameters();
        
        // run reservoir without inputs
        // initial transient
        const int init_length = int(init_time/rc->integ_step);
        integrate_dde(rc,init_length);

        // run reservoir 
        const int length = int(compute_time/rc->integ_step);
        vec states = integrate_dde(rc,length);

        //compute the local minima and maxima of the states
        
        

        // check if extrema were found
        if (stddev(states) < 1e-3) {
            outFile << "Fixed_Point," << mean(states) << "," << scan_param_values(k) << endl;
        }
        else{
            pair<vec, vec> extrema = findLocalExtrema(states,1);
            // Write the local minima
            for (uword i = 0; i < MIN(extrema.first.n_elem, 20); ++i) {
                outFile << "Minimum," << extrema.first(i) << "," << scan_param_values(k) << endl;
            }

            // Write the local maxima
            for (uword i = 0; i < MIN(extrema.second.n_elem, 20); ++i) {
                outFile << "Maximum," << extrema.second(i) << "," << scan_param_values(k) << endl;
            }
            if (extrema.first.n_elem == 0 && extrema.second.n_elem == 0) {
                outFile << "Plateau," << max(states) <<","<< scan_param_values(k) << endl;
                outFile << "Plateau," << min(states) <<","<< scan_param_values(k) << endl;
            }
        }
    }

    // Close the file
    outFile.close();

    cout << "Extrema values have been written to extrema.csv" << endl;
        
    return 0;
};
