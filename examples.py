import numpy as np

def objective_qp(x, t):
    obj_value = x[0]**2 + x[1]**2 + (x[2]+1)**2 - (1/t) * np.sum(np.log(x))
    grad = np.array([2*x[0] - (1/t)/x[0], 2*x[1] - (1/t)/x[1], 2*(x[2] + 1) - (1/t)/x[2]])
    hessian = np.diag([2 + (1/t)/x[0]**2, 2 + (1/t)/x[1]**2, 2 + (1/t)/x[2]**2])
    return obj_value, grad, hessian

def objective_lp(x, t):
    if t == 0:
        t = 1e-10  # Small value to avoid division by zero, adjust as needed
    obj_value = -(x[0] + x[1]) - (1/t) * np.sum(np.log([1 - x[0], 1- x[1], x[0] - x[1] + 1, 2 - x[0], x[1]]))
    grad = np.array([-1 +(1/t)/(1-x[0]) - (1/t)/(x[0] - x[1] + 1)+ (1/t)/(2 - x[0]),
                     -1 +(1/t)/(1-x[1])+ (1/t)/(x[0]- x[1] + 1) +(1/t)/x[1]])
    hessian = np.array([
        [(1/t)/(1 - x[0])**2 +(1/t)/(x[0] -x[1] + 1)**2 + (1/t)/(2- x[0])**2, -(1/t)/(x[0] - x[1] + 1)**2],
        [-(1/t)/(x[0] - x[1] +1)**2,(1/t)/(1 -x[1])**2 +(1/t)/(x[0] -x[1] +1)**2 +(1/t)/x[1]**2]
    ])
    return obj_value, grad, hessian


def setup_qp():
    ineq_constraints = [
        lambda x: x[0],
        lambda x: x[1],
        lambda x: x[2]
    ]
    eq_constraints_mat = np.array([[1, 1, 1]])
    eq_constraints_rhs = np.array([1])
    x0 = np.array([0.1, 0.2, 0.7])
    return objective_qp, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0

def setup_lp():
    ineq_constraints = [
        lambda x: 1 - x[0],
        lambda x: 1 - x[1],
        lambda x: x[0] - x[1] + 1,
        lambda x: 2 - x[0],
        lambda x: x[1]
    ]
    x0 = np.array([0.5, 0.75])
    return objective_lp, ineq_constraints, None, None, x0
