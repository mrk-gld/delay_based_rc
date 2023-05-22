#include <armadillo>
#include <cmath>
#include "reservoirs.h"
#include <ostream>

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
mat integrate_dde_reservoir(dde_reservoir<T> *RC, vec u_t, vec mask){

    int len_time = u_t.n_elem;
    mat states(len_time, RC->num_nodes+1,fill::ones);

    double* m_ptr = mask.memptr();
    double* u_ptr = u_t.memptr(); 

    int steps_per_node = int(RC->theta / RC->integ_step);
    

    for (int k = 0; k < len_time;k++){
        for (int n=0; n < RC->num_nodes; n++){
            for (int i = 0; i < steps_per_node; i++){
                /* 
                you can decide between different integration schemes:
                - RK4 is faster as it allows for larger integration steps however does not include noise
                - Euler-Maruyama is slower due to smaller integration steps needed but includes noise
                 */

                RC->runge_kutta_4th_order(m_ptr[n] * u_ptr[k]);
                //RC->euler_marayuma(m_ptr[n] * u_ptr[k]);

                #ifdef LOGFILE
                    state_file << RC->readout() << "," << endl;
                #endif
            }
            states(k, n) = RC->readout();
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
/*
TODO: 
- generate README.md
- include Lang-Kobayashi model and adapt the integration methods for handling 2 dimension state variables
*/

int main(int argc, char** argv){

    // loading input data
    vec mg_t;
    mg_t.load("mackey_glass_tau17.csv", csv_ascii);
    
    // normalize input data
    mg_t -= mean(mg_t);
    mg_t /= stddev(mg_t);

    // split into target and input data
    const int pred_steps = 1;
    const int input_length = 2500;
    vec u_t = mg_t.subvec(0,input_length);
    vec y_t = mg_t.subvec(pred_steps,input_length+pred_steps);

    // split into training and testing data
    
    const int init_length = 1000;
    const int train_length = 1000;
    if (init_length + train_length > input_length){
        cout << "Error: init_length + train_length > input_length" << endl;
        return 1;
    }

    cout << "Splitting data into training and testing sets" << endl;
    const int test_length = input_length - train_length - init_length;
    vec u_init = u_t.subvec(0,init_length);

    vec u_train = u_t.subvec(init_length,init_length+train_length-1);
    vec y_train = y_t.subvec(init_length,init_length+train_length-1);

    vec u_test = u_t.subvec(init_length+train_length,init_length+train_length+test_length-1);
    vec y_test = y_t.subvec(init_length+train_length,init_length+train_length+test_length-1);

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
    //dde_reservoir<float> * rc = new Ikeda();

    // defining the lang kobayashi delay based reservoir
    dde_reservoir<complex<float>> * rc = new LangKobayashi();
    

    //print selected reservoir
    cout << "Selected reservoir: " << rc->name << endl;

    if( argc > 1 ) {
        // set parameters from command line using pattern -parameter=value
        map<string,float> params;
        for (int i=1; i < argc; i++){
            string arg = argv[i];
            size_t pos = arg.find("=");
            string key = arg.substr(1,pos-1);
            string val = arg.substr(pos+1);
            float f_val = stof(val);
            params[key] = f_val;
        }
        rc->set_parameters(params);
    }

    rc->init_delay();
    rc->print_parameters();

    // logging to file
    csvLogger logger;

    logger.init("delay_rc_output.csv");
    logger.writeHeader(rc);

    // generate random mask wit fixed seed for reproducibility
    arma_rng::set_seed(42);
    vec mask = vec(rc->num_nodes, arma::fill::randu)-0.5;

    // run reservoir with inputs
    cout << "running initial phase" << endl;
    integrate_dde_reservoir(rc, u_init, mask);

    cout << "running training phase" << endl;
    mat states_train = integrate_dde_reservoir(rc, u_train, mask);

    cout << "running testing phase" << endl;
    mat states_test = integrate_dde_reservoir(rc, u_test, mask);

    cout << "training output layer" << endl;
    vec w_out = solve(states_train, y_train);

    cout << "computing training predictions" << endl;
    vec y_pred_train(train_length);
    for (int i=0; i < train_length;i++){
        vec state_col = states_train.row(i).t();
        y_pred_train(i) = dot(w_out, state_col);
    }

    //compute training nrmse
    double nrmse_train = sqrt(mean(pow(y_pred_train - y_train,2)));
    cout << "Training NRMSE = " << nrmse_train << endl;
    
    cout << "computing testing predictions" << endl;
    vec y_pred(test_length);
    for (int i=0; i < test_length;i++){
        vec state_col = states_test.row(i).t();
        y_pred(i) = dot(w_out, state_col);
    }

    //compute test nrmse
    double nrmse = sqrt(mean(pow(y_pred - y_test,2)));
    cout << "Testing NRMSE = " << nrmse << endl;

    logger << "Training NRMSE, Testing NRMSE" << std::endl;
    logger << nrmse_train << "," << nrmse << std::endl;

    logger.close();

    // save results
    y_pred.save("y_pred.csv", csv_ascii);
    y_test.save("y_test.csv", csv_ascii);

    return 0;
};