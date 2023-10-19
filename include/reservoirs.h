/*

written by: Mirko Goldmann
date: 2023-05-15

*/

#include <armadillo>
#include <cmath>

using namespace arma;
using namespace std;

#define LONG_LINE "______________________"

// noise generator
mt19937 gen;
normal_distribution<float> noise;

// delay-based RC Base struct, all other delay-based RCs inherit from this and can override its functionality
template <typename T>
struct dde_reservoir
{

    float delay = 80.f;
    int num_nodes = 50;
    float theta = 1.4f; // time per virtual node
    float integ_step = 0.01;
    float noise_amp = 1e-3f;
    string name = "dde_reservoir";

    T z_t;
    vector<T> z_tau;

    void init_delay(T x_0)
    {
        int steps_per_delay = int(delay / integ_step);
        z_t = x_0;
        z_tau = vector<float>(steps_per_delay);
        for (int i = 0; i < steps_per_delay; i++)
        {
            z_tau[i] =  x_0 + noise_amp * noise(gen);
        }
    };

    virtual void init_delay()
    {
        
    };

    virtual float readout() { return 0.0; };

    virtual T dde_equation(T z_t, T z_tau, float u_t) { return -z_t; };

    virtual void set_parameters(map<string, float> params){};

    virtual void print_parameters()
    {
        cout << endl
             << endl
             << LONG_LINE << endl;
        cout << "General RC parameters:" << endl;
        cout << "delay = " << delay << endl;
        cout << "num_nodes = " << num_nodes << endl;
        cout << "theta = " << theta << endl;
        cout << "input time = " << theta * num_nodes << endl;
        cout << "integ_step = " << integ_step << endl;
        cout << "noise_amp = " << noise_amp << endl;
        cout << LONG_LINE << endl;
    };

    void set_rc_parameters(map<string, float> params)
    {
        // check if parameter is in map
        if (params.find("delay") != params.end())
        {
            delay = params["delay"];
        }
        if (params.find("num_nodes") != params.end())
        {
            num_nodes = int(params["num_nodes"]);
        }
        if (params.find("theta") != params.end())
        {
            theta = params["theta"];
        }
        if (params.find("integ_step") != params.end())
        {
            integ_step = params["integ_step"];
        }
        if (params.find("noise_amp") != params.end())
        {
            noise_amp = params["noise_amp"];
        }
    };

    virtual string csv_header()
    {
        stringstream ss;
        ss << endl;
        return ss.str();
    }

    void euler_maruyama(float u_t)
    {

        T dzdt = dde_equation(z_t, z_tau[0], u_t);

        // generate float noise term
        T real_noise = noise(gen);
        T noise_term = noise_amp * real_noise;

        // in case state variable is complex, generate complex noise term
        if constexpr (std::is_same_v<T, std::complex<float>>)
        {
            std::complex<float> complex_noise(noise(gen), noise(gen));
            noise_term = noise_amp * complex_noise;
        }

        T new_zk1 = z_t + integ_step * dzdt + noise_term * sqrt(integ_step);

        z_t = new_zk1;

        // update delay line
        z_tau.push_back(new_zk1);
        z_tau.erase(z_tau.begin());
    }

    virtual void runge_kutta_4th_order(float u_t)
    {

        T k1 = dde_equation(z_t, z_tau[0], u_t);
        T zk1 = z_t + k1 * integ_step / 2.f;

        T k2 = dde_equation(zk1, z_tau[0], u_t);
        T zk2 = z_t + k2 * integ_step / 2.f;

        T k3 = dde_equation(zk2, z_tau[0], u_t);
        T zk3 = z_t + k3 * integ_step;

        T k4 = dde_equation(zk3, z_tau[1], u_t);

        T z_kp1 = z_t + (1.f / 6.f) * (k1 + 2.f * k2 + 2.f * k3 + k4) * integ_step;

#ifdef LOGFILE
        logfile << k1 << endl;
        logfile << k2 << endl;
        logfile << k3 << endl;
        logfile << k4 << endl;
#endif

        z_t = z_kp1;

        // update delay line
        z_tau.push_back(z_kp1);
        z_tau.erase(z_tau.begin());
    }
};

complex<float> i = complex<float>(0, 1); // imaginary unit

// stuart landau model inhereting from dde_reservoir
struct SLO : dde_reservoir<complex<float>>
{

    float lambda = -0.02;
    float eta = 0.01;
    float omega = 0;
    float gamma = -0.1;
    float kappa = 0.1;

    SLO()
    {
        name = "stuart landau";
    }

    void set_parameters(map<string, float> params) override
    {
        // check if parameter is in map
        if (params.find("lambda") != params.end())
        {
            lambda = params["lambda"];
        }
        if (params.find("eta") != params.end())
        {
            eta = params["eta"];
        }
        if (params.find("omega") != params.end())
        {
            omega = params["omega"];
        }
        if (params.find("gamma") != params.end())
        {
            gamma = params["gamma"];
        }
        if (params.find("kappa") != params.end())
        {
            kappa = params["kappa"];
        }
        set_rc_parameters(params);
    }

    void init_delay()
    {
        int steps_per_delay = int(delay / integ_step);
        z_t = complex<float>(0.1, 0.1);
        z_tau = vector<complex<float>>(steps_per_delay, complex<float>(0.1, 0.1));
    }

    float readout()
    {
        return pow(abs(z_t), 2);
    }

    // Runge Kutte 4th order method for stuart landau oscillator
    complex<float> dde_equation(complex<float> z_t, complex<float> z_tau, float u_t)
    {
        return (lambda + eta * u_t + i * omega + gamma * abs(z_t) * abs(z_t)) * z_t + kappa * z_tau;
    }

    void print_parameters()
    {
        cout << endl
             << endl
             << LONG_LINE << endl;
        cout << "Stuart Landau RC parameters:" << endl;
        cout << "lambda = " << lambda << endl;
        cout << "eta = " << eta << endl;
        cout << "omega = " << omega << endl;
        cout << "gamma = " << gamma << endl;
        cout << "kappa = " << kappa << endl;
        cout << LONG_LINE << endl;
        dde_reservoir::print_parameters();
    }

    string csv_header()
    {
        stringstream ss;
        ss << "lambda,eta,omega,gamma,kappa" << endl;
        ss << lambda << "," << eta << "," << omega << "," << gamma << "," << kappa << endl;
        return ss.str();
    }
};

// mackey glass model inhereting from dde_reservoir
struct MGO : dde_reservoir<float>
{

    // parameters as in Appeltant et al. 2011
    float eta = 0.2f;
    float epsilon = 1.f;
    float p = 10.f;
    float gamma = 0.1f;

    MGO()
    {
        name = "mackey glass";
    }

    void set_parameters(map<string, float> params)
    {
        // check if parameter is in map
        if (params.find("eta") != params.end())
        {
            eta = params["eta"];
        }
        if (params.find("epsilon") != params.end())
        {
            epsilon = params["epsilon"];
        }
        if (params.find("p") != params.end())
        {
            p = params["p"];
        }
        if (params.find("gamma") != params.end())
        {
            gamma = params["gamma"];
        }
        set_rc_parameters(params);
    }

    void init_delay()
    {
        int steps_per_delay = int(delay / integ_step);
        z_t = 0.1f;
        z_tau = vector<float>(steps_per_delay, 0.1f);
    }

    float readout()
    {
        return z_t;
    }

    // Runge Kutte 4th order method for mackey glass oscillator
    float dde_equation(float z_t, float z_tau, float u_t)
    {
        return eta * (z_tau + gamma * u_t) / (1.f + pow(z_tau + gamma * u_t, p)) - epsilon * z_t;
    }

    void print_parameters()
    {
        cout << endl
             << endl
             << LONG_LINE << endl;
        cout << "Mackey Glass RC parameters:" << endl;
        cout << "eta = " << eta << endl;
        cout << "epsilon = " << epsilon << endl;
        cout << "p = " << p << endl;
        cout << "gamma = " << gamma << endl;
        cout << LONG_LINE << endl;
        dde_reservoir::print_parameters();
    }

    string csv_header()
    {
        stringstream ss;
        ss << "eta,epsilon,p,gamma" << endl;
        ss << eta << "," << epsilon << "," << p << "," << gamma << endl;
        return ss.str();
    }
};

// Ikeda model inhereting from dde_reservoir
struct Ikeda : dde_reservoir<float>
{
    float beta = 1.6;
    float gamma = 0.9;
    float epsilon = 1.f;
    float phi = 0.2;

    Ikeda()
    {
        name = "ikeda";
    }

    void set_parameters(map<string, float> params)
    {
        // check if parameter is in map
        if (params.find("beta") != params.end())
        {
            beta = params["beta"];
        }
        if (params.find("gamma") != params.end())
        {
            gamma = params["gamma"];
        }
        if (params.find("epsilon") != params.end())
        {
            epsilon = params["epsilon"];
        }
        if (params.find("phi") != params.end())
        {
            phi = params["phi"];
        }
        set_rc_parameters(params);
    }

    void print_parameters()
    {
        cout << endl
             << endl
             << LONG_LINE << endl;
        cout << "Ikeda RC parameters:" << endl;
        cout << "beta = " << beta << endl;
        cout << "gamma = " << gamma << endl;
        cout << "epsilon = " << epsilon << endl;
        cout << "phi = " << phi << endl;
        cout << LONG_LINE << endl;
        dde_reservoir::print_parameters();
    }

    void init_delay() override
    {
        int steps_per_delay = int(delay / integ_step);
        z_t = 0.1f;
        z_tau = vector<float>(steps_per_delay, 0.1f);
    }

    float readout()
    {
        return z_t;
    }

    // Runge Kutte 4th order method for mackey glass oscillator
    float dde_equation(float z_t, float z_tau, float u_t)
    {
        float sin_term = sin(z_tau + gamma * u_t + phi);
        return -epsilon * z_t + beta * sin_term * sin_term;
    }

    string csv_header()
    {
        stringstream ss;
        ss << "beta,gamma,epsilon,phi" << endl;
        ss << beta << "," << gamma << "," << epsilon << "," << phi << endl;
        return ss.str();
    }
};

const float pi = 3.14159265358979323846f;
const float c_const = 299792458.f;
const float q_const = 1.60217662e-19f;

// Lang-Kobayashi model inheriting from dde_reservoir
struct LangKobayashi : dde_reservoir<complex<float>>
{
    // implementation of the model used in Estebanez et al. 2021

    const float alpha = 3.f;      // linewidth enhancement factor
    const float lambda = 1550e-9; // wavelength in m
    const float omega_0 = (2.0 * pi * c_const) / lambda;
    const float N_0 = 1.5e8; // carriers at transparency
    const float g_n = 1.2e-5 / 1e-9;
    const float s = 5e-7;        // photon decay in Hz
    const float t_s = 2e-9;      // carrier life time
    const float t_ph = 2e-12;    // photon lifetime
    const float J_th = 15.37e-3; // threshold gain
    const float t_in = 0.01e-9;  // SL round trip
    const float transwatt =
        (6.626e-13 * (omega_0 / (2.0 * pi)) * (0.3 / 0.8) * s); // For mWatt conversion

    float phi_0 = 0.0;            // phase offset
    float kappa_fb = 0.05 / t_in; // feedback attenuation
    float kappa_inj = 2 / t_in;   // injection gain
    float detuning = 20 * 1e9;    // frequency detuning
    float delta_omega = 2.0 * pi * detuning;
    float bias_rel = 0.995;

    /*MZM parameters*/
    float mzm_bias = 0;
    float mzm_range = pi * 0.5;

    float t = 0.f; // time variable
    float N_t;     // carrier number is handled internally
    float G_t;     // gain is handled internally

    LangKobayashi()
    {
        name = "Lang-Kobayashi";
        delay = 1e-9;         // length of the cavity in ns
        integ_step = 0.1e-12; // integration step in ns
        theta = 11e-12;       // time per virtual node in ns
    }

    // set parameters by name from a map, first it is checked if the parameter is in the map
    void set_parameters(map<string, float> params)
    {
        // check if parameter is in map
        if (params.find("phi_0") != params.end())
        {
            phi_0 = params["phi_0"];
        }
        if (params.find("mzm_bias") != params.end())
        {
            mzm_bias = params["mzm_bias"];
        }
        if (params.find("mzm_range") != params.end())
        {
            mzm_range = params["mzm_range"];
        }
        if (params.find("kappa_fb") != params.end())
        {
            kappa_fb = params["kappa_fb"];
        }
        if (params.find("kappa_inj") != params.end())
        {
            kappa_inj = params["kappa_inj"];
        }
        if (params.find("detuning") != params.end())
        {
            detuning = params["detuning"];
        }
        if (params.find("bias_rel") != params.end())
        {
            bias_rel = params["bias_rel"];
        }
        set_rc_parameters(params);
    }

    void init_delay()
    {
        int steps_per_delay = int(delay / integ_step);
        z_t = complex<float>(1.0, 0.0);
        z_tau = vector<complex<float>>(steps_per_delay, complex<float>(1.0, 0.0));
    }

    float readout()
    {
        return pow(abs(z_t), 2) * transwatt;
    }

    // dde of the electric field -> Lang-Kobayashi model
    complex<float> dde_equation(complex<float> E_t, complex<float> E_tau, float N, float G_t, float u_t)
    {

        const complex<float> E_0inj(100, 0);
        complex<float> E_inj = E_0inj * sin(mzm_bias + mzm_range * u_t);

        const complex<float> injection =
            kappa_inj * E_inj * std::exp(-i * delta_omega * t);

        const complex<float> feedback =
            kappa_fb * E_tau * std::exp(i * (omega_0 * delay + phi_0));

        return 0.5f * (1.0f + complex<float>(0.0f, alpha)) * (G_t - 1.0f / t_ph) * E_t +
               feedback + injection;
    }

    // Lang-Kobayashi model for carrier number
    float carriers_LK(complex<float> E_t, float N_t, float G_t)
    {
        return (bias_rel * J_th) / q_const - N_t / t_s - G_t * pow(abs(E_t), 2.0);
    }

    // Lang-Kobayashi model for gain
    float gain_LK(complex<float> E_t, float N_t)
    {
        return (g_n * (N_t - N_0)) / (1.0 + s * pow(abs(E_t), 2.0));
    }

    // Runge-Kutta 4th order solver for Lang-Kobayashi model
    void runge_kutta_4th_order(float u_t) override
    {

        // Obtain the current values of E(t) and N(t)
        complex<float> E_t = z_t;
        complex<float> E_tau = z_tau[0];

        // Compute k1
        complex<float> k1_E = dde_equation(E_t, E_tau, N_t, G_t, u_t);
        float k1_N = carriers_LK(E_t, N_t, G_t);

        complex<float> E_t_k1 = E_t + 0.5f * integ_step * k1_E;
        float N_t_k1 = N_t + 0.5f * integ_step * k1_N;
        float G_th1 = gain_LK(E_t_k1, N_t_k1);

        // Compute k2
        complex<float> k2_E = dde_equation(E_t_k1, E_tau, N_t_k1, G_th1, u_t);
        float k2_N = carriers_LK(E_t_k1, N_t_k1, G_th1);

        complex<float> E_t_k2 = E_t + 0.5f * integ_step * k2_E;
        float N_t_k2 = N_t + 0.5f * integ_step * k2_N;
        float G_th2 = gain_LK(E_t_k2, N_t_k2);

        // Compute k3
        complex<float> k3_E = dde_equation(E_t_k2, E_tau, N_t_k2, G_th2, u_t);
        float k3_N = carriers_LK(E_t_k2, N_t_k2, G_th2);

        complex<float> E_t_k3 = E_t + integ_step * k3_E;
        float N_t_k3 = N_t + integ_step * k3_N;
        float G_th3 = gain_LK(E_t_k3, N_t_k3);

        // Compute k4
        complex<float> k4_E = dde_equation(E_t_k3, E_tau, N_t_k3, G_th3, u_t);
        float k4_N = carriers_LK(E_t_k3, N_t_k3, G_th3);

        // Compute the next values of E(t) and N(t)
        E_t = E_t + (integ_step / 6.0f) * (k1_E + 2.0f * k2_E + 2.0f * k3_E + k4_E);
        N_t = N_t + (integ_step / 6.0f) * (k1_N + 2.0f * k2_N + 2.0f * k3_N + k4_N);
        G_t = gain_LK(E_t, N_t);

        t += integ_step;

        z_t = E_t;
        z_tau.push_back(E_tau);
        z_tau.erase(z_tau.begin());
    }

    void print_parameters()
    {
        cout << endl
             << endl
             << LONG_LINE << endl;
        cout << "Lang-Kobayashi RC parameters:" << endl;
        cout << "phi_0 = " << phi_0 << endl;
        cout << "mzm_bias = " << mzm_bias << endl;
        cout << "mzm_range = " << mzm_range << endl;
        cout << "kappa_fb = " << kappa_fb << endl;
        cout << "kappa_inj = " << kappa_inj << endl;
        cout << "detuning = " << detuning << endl;
        cout << "bias_rel = " << bias_rel << endl;

        cout << LONG_LINE << endl;
        dde_reservoir::print_parameters();
    }

    string csv_header() override
    {
        stringstream ss;
        ss << "phi_0,mzm_bias,mzm_range,kappa_fb,kappa_inj,detuning,bias_rel" << endl;
        ss << phi_0 << "," << mzm_bias << "," << mzm_range << "," << kappa_fb << "," << kappa_inj << "," << detuning << "," << bias_rel << endl;
        return ss.str();
    }
};
