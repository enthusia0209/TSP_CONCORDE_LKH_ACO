from typing import List
from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphProblem
import asyncio
import time
from graphite.solvers.libs.concorde.tsp import TSPSolver
import numpy as np

class CONSolver(BaseSolver):
    def __init__(self, problem_types: List[GraphProblem] = [GraphProblem(n_nodes=2), GraphProblem(n_nodes=2, directed=True, problem_type='General TSP')]):
        super().__init__(problem_types=problem_types)

    def problem_transformations(self, problem: GraphProblem):
        return problem.nodes

    def solve_concorde(self, matrix):
        solver = TSPSolver.from_data(
            xs=[coord[0] for coord in matrix],  # Extract x values
            ys=[coord[1] for coord in matrix],  # Extract y values
            norm="EUC_2D"  # Specifies that the distances are Euclidean in 2D
        )
        solution = solver.solve()
        return solution.tour

    def symmetricize(self, m, high_int=None):
        if high_int is None:
            high_int = round(10*m.max())
            
        m_bar = m.copy()
        np.fill_diagonal(m_bar, 0)
        u = np.matrix(np.ones(m.shape) * high_int)
        np.fill_diagonal(u, 0)
        m_symm_top = np.concatenate((u, np.transpose(m_bar)), axis=1)
        m_symm_bottom = np.concatenate((m_bar, u), axis=1)
        m_symm = np.concatenate((m_symm_top, m_symm_bottom), axis=0)

        return m_symm.astype(int) # Concorde requires integer weights

    async def solve(self, distance_matrix, future_id: int, directed=False) -> List[int]:
        n_cities = len(distance_matrix)
        # durations = np.matrix(distance_matrix)
        # durations_symm = self.symmetricize(durations)
        solution = self.solve_concorde(distance_matrix)
        # solution = [city for city in solution if city < n_cities]
        solution = np.append(solution, 0)

        return solution

if __name__ == '__main__':
    n_nodes = 100  # Adjust as needed
    test_problem = GraphProblem(n_nodes=n_nodes)
    solver = CONSolver(problem_types=[test_problem.problem_type])
    start_time = time.time()
    route = asyncio.run(solver.solve_problem(test_problem))
    if route is None:
        print(f"{solver.__class__.__name__} No solution found.")
    else:
        print(f"{solver.__class__.__name__} Best Solution: {route}")
    print(f"{solver.__class__.__name__} Time Taken: {time.time() - start_time}")
