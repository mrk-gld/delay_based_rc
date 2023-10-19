
# building a wrapper for the bayesian optimization that calles a cpp function and returns the result
# the cpp function is called with the parameters as a string and returns the result as a string

import subprocess
import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer
from subprocess import Popen, PIPE

test_error_phrase = "Testing NRMSE = "
train_error_phrase = "Training NRMSE = "

def eval_performance(params):

    print("evaluating parameters: ", params)
    eta = params[0]
    gamma = params[1]
    kappa = params[2]

    p = Popen(['./bin/delay_based_RC', f"-eta={eta}",f"-gamma={gamma}", f"-kappa={kappa}", "-num_nodes=10" ,"-theta=2", f"-delay={10*2.05}"], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    output, err = p.communicate(b"input data that is passed to subprocess' stdin")

    output = str(output)
    print(output)
    if train_error_phrase in output:
        nrmse_str = output.split(train_error_phrase)[1]
        nrmse_str = nrmse_str.split("\\n")[0]
        nrmse = float(nrmse_str)
        return nrmse
    else:
        print("error: ", err)
        return 1000.0

search_space = [
        Real(1e-4, 3, name='eta',prior='log-uniform'),
        Real(-0.4, 0.0, name='gamma'),
        Real(0.01, 1, name="kappa")
    ]

n_epoch_bayes_opt = 75
n_random_bayes_opt = 20

subprocess.call(["make"])
result = gp_minimize(func=eval_performance, dimensions=search_space, n_calls=n_epoch_bayes_opt, n_random_starts=n_random_bayes_opt, verbose=True)

print("best result: ", result.fun)
print("best parameters: ", result.x)
