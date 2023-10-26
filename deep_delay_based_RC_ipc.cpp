/*

written by: Mirko Goldmann
date: 2023-05-15

*/

#include <armadillo>
#include <cmath>
#include "reservoirs.h"
#include "information_processing_capacity.h"
#include <ostream>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace arma;
using namespace std;

//#define LOGFILE // comment out to disable logging
// be aware logging slows down the simulation crucially

//generate log file
#ifdef LOGFILE
    ofstream logfile("delay_rc_slo.log");
    ofstream state_file("delay_rc_slo_states.csv");
#endif

template<typename T>
mat integrate_dde_reservoir(vector<dde_reservoir<T>*> RC, vec u_t, vec mask){

    int len_time = u_t.n_elem;
    const int num_nodes = RC[0]->num_nodes * RC.size();
    mat states(len_time, num_nodes, fill::ones);

    double* m_ptr = mask.memptr();
    double* u_ptr = u_t.memptr();

    int steps_per_node = int(RC[0]->theta / RC[0]->integ_step);


    for (int k = 0; k < len_time;k++){
        for (int n=0; n < RC[0]->num_nodes; n++){
            for (int i = 0; i < steps_per_node; i++){
                /*
                you can decide between different integration schemes:
                - RK4 is faster as it allows for larger integration steps however does not include noise
                - Euler-Maruyama is slower due to smaller integration steps needed but includes noise
                 */

                //RC->runge_kutta_4th_order(m_ptr[n] * u_ptr[k]);
                for(int l = 0; l < RC.size(); l++){
                    if (l == 0){
                        RC[l]->euler_maruyama(m_ptr[n] * u_ptr[k]);
                    }
                    else{
                        RC[l]->euler_maruyama(RC[l-1]->readout());
                    }
                        
                }

                #ifdef LOGFILE
                    state_file << RC->readout() << "," << endl;
                #endif
            }
            for(int l = 0; l < RC.size(); l++){
                const int start = l * RC[0]->num_nodes;
                states(k, start+n) = RC[l]->readout();
            }
            
        }
    }

    return states;
}

struct csvLogger{

    string filename;
    ofstream file;

    void init(string filename){
        this->filename = filename;
        file.open(filename);
    }

    void log(string line){
        file << line << endl;
    }

    void close(){
        file.close();
    }

    template<typename T>
    void writeHeader(dde_reservoir<T> *RC){
        file << "reservoir,delay,num_nodes,theta,integ_step,noise_amp, , ," << endl;
        file << RC->name << ","<< RC->delay << "," << RC->num_nodes << "," << RC->theta << "," << RC->integ_step << "," << RC->noise_amp << " , , ," << endl;
        file << RC->csv_header();
    }

    template<typename T>
    csvLogger& operator<<(const T& line){
        file << line;
        return *this;
    }

    // allows for chaining of << operator
    csvLogger& operator<<(std::ostream& (*pManip)(std::ostream&)) {
        file << pManip;
        return *this;
    }

};

std::vector<map<string, float>> get_parameter_map_from_arg(int argc, char** argv){
    std::vector<map<string, float>> layer_params;
    for (int i=1; i < argc; i++){
        string arg = argv[i];
        size_t pos = arg.find("=");
        string key = arg.substr(1, pos-1);
        string val = arg.substr(pos+1);
        float f_val = stof(val);

        // Extract layer index, if present.
        size_t layer_pos = key.find("_layer");
        if (layer_pos != string::npos) {
            int layer_idx = stoi(key.substr(layer_pos + 6));
            key = key.substr(0, layer_pos);
            if (layer_idx >= layer_params.size()) {
                layer_params.resize(layer_idx + 1);
            }
            layer_params[layer_idx][key] = f_val;
        } else {
            // Global parameter, apply to all layers.
            for (auto &params : layer_params) {
                params[key] = f_val;
            }
        }
    }
    return layer_params;
}


void printMemIdx(const double result, vector<pair<int, int>> idx)
{
    cout << result << ",";
    for (auto it = idx.begin(); it != idx.end(); it++) {
        cout << it->first << "," << it->second << ",";
    }
    cout << endl;
}

#define MEMORY_THRESHOLD 0.1

int main(int argc, char** argv){

    std::vector<map<string, float>> layer_params = get_parameter_map_from_arg(argc, argv);
    if (layer_params[0].find("seed") == layer_params[0].end()){
        cout << "No seed parameter found, using default seed" << endl;
        layer_params[0]["seed"] = 0;
    }
    // loading input data

    const int init_length = 1000;
    const int train_length = 5000;
    const int input_length = init_length + train_length;

    vec inputs = generateRandomInputSequenceUni(input_length);

    vec u_init = inputs.subvec(0,init_length-1);
    vec u_train = inputs.subvec(init_length,init_length+train_length-1);

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
    vector<dde_reservoir<float>*> deep_reservoir;
    for(int i = 0; i < layer_params.size(); i++){
        dde_reservoir<float> * rc = new Ikeda();
        cout << "Reservoir layer: " << i+1 << endl;
        cout << "Selected reservoir: " << rc->name << endl;

        rc->set_parameters(layer_params[i]);
        rc->init_delay();
        rc->print_parameters();
        deep_reservoir.push_back(rc);
    }
    
    cout << "Depth of reservoir = " << deep_reservoir.size() << endl;

    // defining the lang kobayashi delay based reservoir
    //dde_reservoir<complex<float>> * rc = new LangKobayashi();    

    // logging to file
    csvLogger logger;
    // check if logs folder exists
    struct stat info;
    if (stat("logs", &info) != 0){
        // create logs folder
        mkdir("logs", 0777);
    }
    
    logger.init("logs/delay_rc_output.csv");
    logger.writeHeader(deep_reservoir[0]);

    // generate random mask wit fixed seed for reproducibility
    // check if there is a seed parameter in the command line

    arma_rng::set_seed(int(layer_params[0]["seed"]));
    vec mask = vec(deep_reservoir[0]->num_nodes, arma::fill::randu)-0.5;

    // run reservoir with inputs
    cout << "running initial phase" << endl;
    integrate_dde_reservoir(deep_reservoir, u_init, mask);

    cout << "running training phase" << endl;
    mat states_train = integrate_dde_reservoir(deep_reservoir, u_train, mask);

    cout << "normalize states" << endl;
    states_train.resize(states_train.n_rows, states_train.n_cols - 1);
    for (int i = 0; i < states_train.n_cols; i++) {
        states_train.col(i) -= mean(states_train.col(i));
        states_train.col(i) /= stddev(states_train.col(i));
    }

    cout << "precompute sst" << endl;
    arma::Mat<double> sst = states_train.t() * states_train / states_train.n_rows;
    arma::Mat<double> ssti = arma::pinv(sst);

    double totalMemCap = 0.f;
    const int maxDegree = 5;
    for (int d=1; d <= maxDegree; d++){
        logger << "MC_d=" << d << ",";
    }
    logger << "total MC," << endl;

    const int past_max = 1000;
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    int nc = 0;
    cout << "start loops" << endl;
    for (int deg = 1; deg <= maxDegree; deg++) {
        double memCapDeg = 0.f;
        for (int numVar = 1; numVar <= deg; numVar++) {
            vector<vector<int>> powerlist = getPowerListsByTotalDegree(deg, numVar);

            for (int i = 0; i < powerlist.size(); i++) {
                const int maxDelay = past_max - powerlist[i].size() + 1;

                vector<int> NUMBER_LOW_MEM_ACC_LAST_PEAKS(powerlist[i].size(), 0);

                for (int delay = 0; delay < maxDelay; delay++) {

                    vector<pair<int, int>> startSteps = getInitIdx(powerlist[i], delay);
                    vector<pair<int, int>> steps = startSteps;

                    const int pos = steps.size() - 1;
                    bool force = false;
                    do {
                        force = false;

                        vec targets = setNonlinearMemoryTask(inputs, steps, init_length);

                        double resTesting = calculateMemoryCapacityFast(states_train, ssti, targets);
                        nc++;

                        if (resTesting > MEMORY_THRESHOLD) {
                            memCapDeg += resTesting;
                            printMemIdx(resTesting, steps);

                            for (int j = 0; j < NUMBER_LOW_MEM_ACC_LAST_PEAKS.size(); j++) {
                                NUMBER_LOW_MEM_ACC_LAST_PEAKS[j] = steps[j].first - startSteps[j].first;
                            }

                        } else {
                            const int lastIdx = powerlist[i].size() - 1;

                            if (steps[lastIdx].first > startSteps[lastIdx].first + NUMBER_LOW_MEM_ACC_LAST_PEAKS[lastIdx] + MAX_RANGE_AFTER_PEAK) {
                                force = true;
                            }
                        }
                    } while (getNext(steps, NUMBER_LOW_MEM_ACC_LAST_PEAKS, startSteps, past_max, pos, force));

                    if (steps[0].first > startSteps[0].first + NUMBER_LOW_MEM_ACC_LAST_PEAKS[0] + MAX_RANGE_AFTER_PEAK) {
                        delay = maxDelay - 1;
                    }
                }
            }
        }
        cout << "Memory Capacity (Degree = " << deg << "):" << memCapDeg << endl;
        logger << memCapDeg << ",";
        totalMemCap += memCapDeg;
    }
    logger << totalMemCap << ",";
    return 0;
};
