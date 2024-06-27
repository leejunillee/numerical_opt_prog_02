import numpy as np
import math

def calculate_stop_criterion(direction, hessian_matrix):
    return math.sqrt(np.dot(direction.T, np.dot(hessian_matrix, direction)))

def interior_pt(func, ineq_constraints, eq_constraints_mat=None, eq_constraints_rhs=None, x0=None):
    t = 1.0
    mu = 10.0
    num_constraints = len(ineq_constraints)

    barrier_value = num_constraints / t
    tol_outer = 1e-3
    tol_inner = 1e-3
    obj_inner = float('inf')
    pointer = x0
    a = 0.01
    iter_outer = 0
    iter_inner = 0
    trajectory = []

    while barrier_value > tol_outer:
        while obj_inner > tol_inner:
            obj_value, grad, hessian_matrix = func(pointer, t)
            if eq_constraints_mat is not None:
                lhs_upper = np.hstack((hessian_matrix, eq_constraints_mat.T))
                lhs_lower = np.hstack((eq_constraints_mat, np.zeros((eq_constraints_mat.shape[0], eq_constraints_mat.shape[0]))))
                lhs = np.vstack((lhs_upper, lhs_lower))
                rhs = np.hstack((-grad, np.zeros(eq_constraints_rhs.shape)))
                solution = np.linalg.solve(lhs, rhs)
                newton_step = solution[:x0.shape[0]]
            else:
                newton_step = np.linalg.solve(hessian_matrix, -grad)
            trajectory.append({'position': pointer.copy(), 'objective_value': obj_value})
            pointer += a * newton_step
            obj_inner = 0.5 * calculate_stop_criterion(newton_step, hessian_matrix)**2
            iter_inner += 1
        t *= mu
        barrier_value = num_constraints / t
        iter_outer += 1

    return trajectory
