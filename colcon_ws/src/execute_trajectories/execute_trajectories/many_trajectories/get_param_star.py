import numpy as np
from skopt.utils import create_result
from skopt import expected_minimum
from skopt import load
# make sure your optimizer stored a model:
# Optimizer(..., model_queue_size=1, n_initial_points=0) and you called opt.tell(...)
opt = load("opt_state.pkl")  # Load your optimizer state
Xi = [list(x) for x in opt.Xi]
yi = [float(y) for y in opt.yi]
res = create_result(Xi, yi, space=opt.space, models=opt.models, specs=None)

# Optionally control restarts via n_random_starts (defaults to 20)
x_star, y_star = expected_minimum(res, random_state=0)  # returns (x_pred, y_pred)
print("Model-predicted minimum:", y_star)
print('end=', x_star)