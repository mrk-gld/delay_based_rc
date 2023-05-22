# Delay-Based Reservoir Computing

This repository contains code for implementing a delay-based reservoir computing model. The code is mainly written in C++ and uses the Armadillo library for linear algebra operations.

## Code Files

The repository contains the following code files:

- `delay_based_RC.cpp`: The main code file that implements the delay-based reservoir computing model.
- `reservoirs.h`: Header file containing the definitions of different types of reservoirs that can be used in the model.
    - with the reservoirs.h there is a Base model called dde_reservoir where all other reservoir model are derived from 
    - within the code one can decide between different models, the available models are:
        - real value models
            - Mackey Glass Oscillator
            - Ikeda Oscillator
        - complex valued models
            - Stuart Landau Oscillator
            - Lang Kobayashi Oscillator    
- `plot_reservoir_behavior.ipynb` a jupyter notebook to plot the results of the simulations
- ` bayes_opt_wrapper.py` a python file that performs bayesian optimization of hyperparameters while running the simulations using the delay_based_RC.cpp implementation 

## Usage

To use the code, follow these steps:

1. Clone the repository to your local machine.
2. Install the Armadillo library. (https://arma.sourceforge.net/)
3. Compile the code using the corresponding make file
4. Run the compiled executable with the following command: `./delay_based_RC`
5. Hyperparameter can be set using the command line. To set the delay parameter to 5 and the number of nodes to 100, run the executable with the following command: `./delay_based_RC -delay=5 -num_nodes=100`

## Parameters
# General reservoir parameters
You can set the parameters of the model by passing command line arguments to the executable. The following parameters are available:

- `-delay`: The delay parameter of the reservoir.
- `-num_nodes`: The number of nodes in the reservoir.
- `-theta`: The time constant of the reservoir.
- `-integ_step`: The integration step size.
- `-noise_amp`: The amplitude of the noise added to the reservoir.

# Model specific parameters
Every model has specific parameter that initially are set to default values usually taken from literature examples. Using the command line arguments one can change many of the parameters directly. For the parameter names and their corresponding property look at the models in reservoirs.h. 

## What the code does

- Loads input data from a CSV file named `mackey_glass_tau17.csv` and normalizes it. (Replace this file with your own input data file to change task.)
- Splits the data into training and testing sets.
- Selects a delay-based reservoir model (Lang-Kobayashi).
- Sets reservoir parameters based on command-line arguments.
- Initializes and runs the reservoir with the input data.
- Trains the output layer using the training data.
- Computes and prints the normalized root mean square error (NRMSE) for training and testing predictions.
- Logs the NRMSE values to a file.
- Saves the predicted and actual values of the testing data to separate CSV files.

## Output
The code runs the simulation and save a file called "delay_rc_output.csv" that contains the model and reservoir parameter such as the training and testing performance. Additionally, the code generates two output files: `y_pred.csv` and `y_test.csv`. These files contain the predicted and actual output values, respectively.

## Debugging
In the "delay_based_RC.cpp" one can uncomment the LOGFILE parameter and compile the code again. Now the reservoir states will be written to a log *.csv-file. Be aware this generates a lot of data and slows down the simulation crucially. The outputted data can be plotted within the jupyter notebook. 
