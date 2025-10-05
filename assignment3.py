import random
import time
from collections import deque
import heapq
from typing import List, Tuple, Dict

class KSATGenerator:
    def __init__(self, k: int, m: int, n: int):
        self.k = k
        self.m = m
        self.n = n
        self.variables = [f"x{i}" for i in range(1, n + 1)]
    
    def generate_clause(self) -> List[Tuple[str, bool]]:
        """Generate a single clause with k distinct literals"""
        selected_vars = random.sample(self.variables, self.k)
        clause = []
        for var in selected_vars:
            is_negated = random.choice([True, False])
            clause.append((var, is_negated))
        return clause
    
    def generate_formula(self) -> List[List[Tuple[str, bool]]]:
        """Generate a complete k-SAT formula"""
        formula = []
        for _ in range(self.m):
            clause = self.generate_clause()
            formula.append(clause)
        return formula

class SATSolver:
    def __init__(self, formula: List[List[Tuple[str, bool]]]):
        self.formula = formula
        self.variables = self._extract_variables()
    
    def _extract_variables(self):
        variables = set()
        for clause in self.formula:
            for var, _ in clause:
                variables.add(var)
        return list(variables)
    
    def evaluate_assignment(self, assignment: Dict) -> Tuple[int, int]:
        """Return (satisfied_clauses, unsatisfied_clauses)"""
        satisfied = 0
        for clause in self.formula:
            clause_satisfied = False
            for var, negated in clause:
                if var in assignment:
                    value = not assignment[var] if negated else assignment[var]
                    if value:
                        clause_satisfied = True
                        break
            if clause_satisfied:
                satisfied += 1
        return satisfied, len(self.formula) - satisfied
    
    def heuristic1(self, assignment: Dict) -> int:
        """Heuristic 1: Number of unsatisfied clauses"""
        _, unsatisfied = self.evaluate_assignment(assignment)
        return unsatisfied
    
    def heuristic2(self, assignment: Dict) -> float:
        """Heuristic 2: Weighted clause satisfaction"""
        score = 0
        for clause in self.formula:
            true_literals = 0
            for var, negated in clause:
                if var in assignment:
                    value = not assignment[var] if negated else assignment[var]
                    if value:
                        true_literals += 1
            if true_literals == 0:
                score += 10  # Heavy penalty for unsatisfied clauses
            else:
                score -= true_literals  # Reward clauses with more true literals
        return score
    
    def generate_random_assignment(self) -> Dict:
        """Generate random complete assignment"""
        return {var: random.choice([True, False]) for var in self.variables}
    
    def get_neighbors(self, assignment: Dict) -> List[Dict]:
        """Generate neighbors by flipping one variable"""
        neighbors = []
        for var in self.variables:
            neighbor = assignment.copy()
            neighbor[var] = not neighbor[var]
            neighbors.append(neighbor)
        return neighbors

class HillClimbingSolver(SATSolver):
    def solve(self, max_iterations: int = 1000, heuristic_func: str = 'h1') -> Tuple[Dict, int, int]:
        """Solve using Hill Climbing"""
        current = self.generate_random_assignment()
        current_score = self.heuristic1(current) if heuristic_func == 'h1' else self.heuristic2(current)
        iterations = 0
        nodes_generated = 1
        
        for _ in range(max_iterations):
            iterations += 1
            neighbors = self.get_neighbors(current)
            nodes_generated += len(neighbors)
            
            best_neighbor = None
            best_score = current_score
            
            for neighbor in neighbors:
                score = self.heuristic1(neighbor) if heuristic_func == 'h1' else self.heuristic2(neighbor)
                if score < best_score:
                    best_score = score
                    best_neighbor = neighbor
            
            if best_neighbor is None:
                break  # Local optimum
            
            current = best_neighbor
            current_score = best_score
            
            if current_score == 0:  # All clauses satisfied
                break
        
        return current, iterations, nodes_generated

class BeamSearchSolver(SATSolver):
    def solve(self, beam_width: int, max_iterations: int = 1000, heuristic_func: str = 'h1') -> Tuple[Dict, int, int]:
        """Solve using Beam Search"""
        # Initialize beam with random assignments
        beam = [self.generate_random_assignment() for _ in range(beam_width)]
        nodes_generated = beam_width
        iterations = 0
        
        best_assignment = beam[0]
        best_score = self.heuristic1(best_assignment) if heuristic_func == 'h1' else self.heuristic2(best_assignment)
        
        for _ in range(max_iterations):
            iterations += 1
            all_neighbors = []
            
            # Generate all neighbors from current beam
            for assignment in beam:
                neighbors = self.get_neighbors(assignment)
                all_neighbors.extend(neighbors)
            
            nodes_generated += len(all_neighbors)
            
            # Score all neighbors
            scored_neighbors = []
            for neighbor in all_neighbors:
                score = self.heuristic1(neighbor) if heuristic_func == 'h1' else self.heuristic2(neighbor)
                scored_neighbors.append((score, neighbor))
            
            # Sort by score and select top beam_width
            scored_neighbors.sort(key=lambda x: x[0])
            beam = [neighbor for _, neighbor in scored_neighbors[:beam_width]]
            
            # Update best solution
            current_best_score, current_best_assignment = scored_neighbors[0]
            if current_best_score < best_score:
                best_score = current_best_score
                best_assignment = current_best_assignment
            
            if best_score == 0:
                break
        
        return best_assignment, iterations, nodes_generated

class VariableNeighborhoodDescentSolver(SATSolver):
    def get_neighbors_large_step(self, assignment: Dict, step_size: int = 2) -> List[Dict]:
        """Generate neighbors by flipping multiple variables"""
        neighbors = []
        for _ in range(len(self.variables)):
            neighbor = assignment.copy()
            vars_to_flip = random.sample(self.variables, min(step_size, len(self.variables)))
            for var in vars_to_flip:
                neighbor[var] = not neighbor[var]
            neighbors.append(neighbor)
        return neighbors
    
    def get_neighbors_targeted(self, assignment: Dict) -> List[Dict]:
        """Generate neighbors targeting variables in unsatisfied clauses"""
        unsatisfied_vars = set()
        for clause in self.formula:
            clause_satisfied = False
            for var, negated in clause:
                if var in assignment:
                    value = not assignment[var] if negated else assignment[var]
                    if value:
                        clause_satisfied = True
                        break
            if not clause_satisfied:
                for var, _ in clause:
                    unsatisfied_vars.add(var)
        
        neighbors = []
        for var in unsatisfied_vars:
            neighbor = assignment.copy()
            neighbor[var] = not neighbor[var]
            neighbors.append(neighbor)
        
        return neighbors if neighbors else self.get_neighbors(assignment)
    
    def solve(self, max_iterations: int = 1000, heuristic_func: str = 'h1') -> Tuple[Dict, int, int]:
        """Solve using Variable Neighborhood Descent"""
        current = self.generate_random_assignment()
        current_score = self.heuristic1(current) if heuristic_func == 'h1' else self.heuristic2(current)
        iterations = 0
        nodes_generated = 1
        
        neighborhood_functions = [
            self.get_neighbors,
            self.get_neighbors_targeted,
            lambda x: self.get_neighbors_large_step(x, 2)
        ]
        
        for _ in range(max_iterations):
            iterations += 1
            improved = False
            
            for neighborhood_func in neighborhood_functions:
                neighbors = neighborhood_func(current)
                nodes_generated += len(neighbors)
                
                for neighbor in neighbors:
                    score = self.heuristic1(neighbor) if heuristic_func == 'h1' else self.heuristic2(neighbor)
                    if score < current_score:
                        current = neighbor
                        current_score = score
                        improved = True
                        break
                
                if improved:
                    break
            
            if not improved or current_score == 0:
                break
        
        return current, iterations, nodes_generated

def run_experiments():
    """Run comprehensive experiments comparing all algorithms"""
    
    # Test configurations (m clauses, n variables)
    test_configs = [
        (50, 20),   # Configuration 1
        (75, 25),   # Configuration 2  
        (100, 30),  # Configuration 3
    ]
    
    algorithms = {
        'Hill Climbing (H1)': (HillClimbingSolver, 'h1'),
        'Hill Climbing (H2)': (HillClimbingSolver, 'h2'),
        'Beam Search w=3 (H1)': (BeamSearchSolver, 'h1', 3),
        'Beam Search w=3 (H2)': (BeamSearchSolver, 'h2', 3),
        'Beam Search w=4 (H1)': (BeamSearchSolver, 'h1', 4),
        'Beam Search w=4 (H2)': (BeamSearchSolver, 'h2', 4),
        'VND (H1)': (VariableNeighborhoodDescentSolver, 'h1'),
        'VND (H2)': (VariableNeighborhoodDescentSolver, 'h2'),
    }
    
    results = {}
    
    for m, n in test_configs:
        print(f"\n{'='*60}")
        print(f"Testing configuration: m={m}, n={n}")
        print(f"{'='*60}")
        
        # Generate 5 different random instances
        instances = []
        for i in range(5):
            generator = KSATGenerator(3, m, n)
            formula = generator.generate_formula()
            instances.append(formula)
        
        config_results = {}
        
        for alg_name, alg_config in algorithms.items():
            print(f"\n{alg_name}:")
            total_satisfied = 0
            total_iterations = 0
            total_nodes = 0
            total_time = 0
            solved_count = 0
            
            for i, formula in enumerate(instances):
                solver_class = alg_config[0]
                heuristic = alg_config[1]
                
                if alg_name.startswith('Beam Search'):
                    beam_width = alg_config[2]
                    solver = solver_class(formula)
                    start_time = time.time()
                    assignment, iterations, nodes = solver.solve(beam_width=beam_width, heuristic_func=heuristic)
                    end_time = time.time()
                else:
                    solver = solver_class(formula)
                    start_time = time.time()
                    assignment, iterations, nodes = solver.solve(heuristic_func=heuristic)
                    end_time = time.time()
                
                satisfied, unsatisfied = solver.evaluate_assignment(assignment)
                runtime = end_time - start_time
                
                total_satisfied += satisfied
                total_iterations += iterations
                total_nodes += nodes
                total_time += runtime
                
                if unsatisfied == 0:
                    solved_count += 1
                
                print(f"  Instance {i+1}: {satisfied}/{m} satisfied, "
                      f"{iterations} iterations, {nodes} nodes, {runtime:.3f}s")
            
            avg_satisfied = total_satisfied / (5 * m)
            avg_iterations = total_iterations / 5
            avg_nodes = total_nodes / 5
            avg_time = total_time / 5
            success_rate = solved_count / 5
            
            # Calculate penetrance
            penetrance = success_rate / avg_nodes if avg_nodes > 0 else 0
            
            config_results[alg_name] = {
                'avg_satisfied': avg_satisfied,
                'success_rate': success_rate,
                'avg_iterations': avg_iterations,
                'avg_nodes': avg_nodes,
                'avg_time': avg_time,
                'penetrance': penetrance
            }
            
            print(f"  Average: {avg_satisfied:.3f} satisfaction rate, "
                  f"{success_rate:.3f} success rate, {avg_iterations:.1f} iterations, "
                  f"{avg_nodes:.1f} nodes, {avg_time:.3f}s, penetrance: {penetrance:.6f}")
        
        results[(m, n)] = config_results
    
    # Print comparative analysis
    print(f"\n{'='*80}")
    print("COMPARATIVE ANALYSIS")
    print(f"{'='*80}")
    
    for m, n in test_configs:
        print(f"\nConfiguration: m={m}, n={n}")
        print("-" * 80)
        print(f"{'Algorithm':<25} {'Success Rate':<12} {'Avg Nodes':<10} {'Penetrance':<12} {'Time (s)':<10}")
        print("-" * 80)
        
        config_data = results[(m, n)]
        for alg_name in algorithms.keys():
            data = config_data[alg_name]
            print(f"{alg_name:<25} {data['success_rate']:<12.3f} "
                  f"{data['avg_nodes']:<10.1f} {data['penetrance']:<12.6f} "
                  f"{data['avg_time']:<10.3f}")

def demonstrate_single_instance():
    """Demonstrate solving a single k-SAT instance"""
    print("DEMONSTRATION: Solving a single 3-SAT instance")
    print("=" * 50)
    
    # Generate a small instance for demonstration
    generator = KSATGenerator(3, 10, 5)
    formula = generator.generate_formula()
    
    print("Generated 3-SAT formula:")
    for i, clause in enumerate(formula):
        clause_str = " ∨ ".join([f"¬{var}" if negated else var for var, negated in clause])
        print(f"  Clause {i+1}: ({clause_str})")
    
    # Test all algorithms on this instance
    algorithms = [
        ("Hill Climbing H1", HillClimbingSolver(formula), 'h1'),
        ("Hill Climbing H2", HillClimbingSolver(formula), 'h2'),
        ("Beam Search w=3 H1", BeamSearchSolver(formula), 'h1', 3),
        ("Beam Search w=3 H2", BeamSearchSolver(formula), 'h2', 3),
        ("VND H1", VariableNeighborhoodDescentSolver(formula), 'h1'),
        ("VND H2", VariableNeighborhoodDescentSolver(formula), 'h2'),
    ]
    
    print("\nAlgorithm Performance:")
    print("-" * 70)
    print(f"{'Algorithm':<20} {'Satisfied':<10} {'Iterations':<10} {'Nodes':<10} {'Time (s)':<10}")
    print("-" * 70)
    
    for alg_name, solver, heuristic, *extra in algorithms:
        start_time = time.time()
        
        if alg_name.startswith('Beam Search'):
            beam_width = extra[0]
            assignment, iterations, nodes = solver.solve(beam_width=beam_width, heuristic_func=heuristic)
        else:
            assignment, iterations, nodes = solver.solve(heuristic_func=heuristic)
        
        end_time = time.time()
        satisfied, unsatisfied = solver.evaluate_assignment(assignment)
        runtime = end_time - start_time
        
        print(f"{alg_name:<20} {satisfied}/{len(formula):<9} {iterations:<10} {nodes:<10} {runtime:<10.4f}")
        
        if unsatisfied == 0:
            print(f"  ✓ SOLUTION FOUND: {assignment}")

if __name__ == "__main__":
    print("k-SAT Problem Solver")
    print("Testing Uniform Random 3-SAT Problems")
    print("=" * 60)
    
    # Run comprehensive experiments
    run_experiments()
    
    print("\n" + "=" * 60)
    print("SINGLE INSTANCE DEMONSTRATION")
    print("=" * 60)
    
    # Demonstrate on a single instance
    demonstrate_single_instance()
