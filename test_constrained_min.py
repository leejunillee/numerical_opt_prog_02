import unittest
import numpy as np
from src.constrained_min import interior_pt
from examples import setup_qp, setup_lp
import matplotlib.pyplot as plt

class TestConstrainedMin(unittest.TestCase):
    def test_qp(self):
        func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0 = setup_qp()
        result_path = interior_pt(func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0)
        plot_results_qp(result_path)
        final_position = result_path[-1]['position']
        final_value = result_path[-1]['objective_value']
        print("Final position (QP):", final_position)
        print("Final objective value (QP):", final_value)
        self.assertTrue(np.allclose(final_position.sum(), 1, atol=1e-3))
        self.assertTrue(np.all(final_position >= 0))
        
    def test_lp(self):
        func, ineq_constraints, _, _, x0 = setup_lp()
        result_path = interior_pt(func, ineq_constraints, x0=x0)
        plot_results_lp(result_path)
        final_position = result_path[-1]['position']
        final_value = result_path[-1]['objective_value']
        print("Final position (LP):", final_position)
        print("Final objective value (LP):", final_value)
        self.assertTrue(final_position[0] <= 2)
        self.assertTrue(final_position[1] <= 1)
        self.assertTrue(final_position[1] >= -final_position[0] + 1)


def plot_results_qp(result_path):
    final_position = result_path[-1]['position']
    final_value = result_path[-1]['objective_value']
    positions = np.array([entry['position'] for entry in result_path])
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], marker='x', label='Path')
    ax.scatter(final_position[0], final_position[1], final_position[2], marker='o', color='r', label='Final Candidate')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'QP Path and Final Position (Value: {final_value:.4f})')
    ax.legend()
    plt.show()

def plot_results_lp(result_path):
    final_position = result_path[-1]['position']
    final_value = result_path[-1]['objective_value']
    positions = np.array([entry['position'] for entry in result_path])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(positions[:, 0], positions[:, 1], marker='x', label='Path')
    ax.scatter(final_position[0], final_position[1], marker='o', color='r', label='Final Candidate')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'LP Path and Final Position (Value: {final_value:.4f})')
    
    x = np.linspace(0, 2, 1500)
    ax.fill_between(x, np.maximum(0, -x + 1), 1, color='gray', alpha=0.3, label='Feasible Region')
    
    ax.legend()
    plt.show()

if __name__ == '__main__':
    unittest.main()
