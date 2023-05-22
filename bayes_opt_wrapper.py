
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

    #p = Popen(['./delay_rc_slo', str(params[0]), str(params[1]), str(params[2])], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    p = Popen(['./delay_rc_slo', str(params[0])], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    output, err = p.communicate(b"input data that is passed to subprocess' stdin")
    
    output = str(output)
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
        #Real(-0.03, 0.0, name='lambda'),
        #Real(-0.4, 0.0, name='gamma'),
    ]

n_epoch_bayes_opt = 75
n_random_bayes_opt = 20

subprocess.call(["make"])
result = gp_minimize(func=eval_performance, dimensions=search_space, n_calls=n_epoch_bayes_opt, n_random_starts=n_random_bayes_opt, verbose=True)

print("best result: ", result.fun)
print("best parameters: ", result.x)