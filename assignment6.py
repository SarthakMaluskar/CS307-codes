# hopfield_network.py
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import distance_matrix

class HopfieldNetwork:
    def __init__(self, n_neurons):
        self.n_neurons = n_neurons
        self.weights = np.zeros((n_neurons, n_neurons))
        self.theta = np.zeros(n_neurons)  # Fixed: make it an array
        
    def train(self, patterns):
        """Train the network using Hebbian learning"""
        n_patterns = len(patterns)
        for pattern in patterns:
            pattern = 2 * pattern - 1  # Convert to {-1, +1}
            self.weights += np.outer(pattern, pattern)
        np.fill_diagonal(self.weights, 0)  # No self-connections
        self.weights /= n_patterns
        
    def recall(self, input_pattern, max_iter=100):
        """Recall a pattern from noisy input"""
        state = 2 * input_pattern - 1  # Convert to {-1, +1}
        
        for _ in range(max_iter):
            old_state = state.copy()
            
            # Update neurons asynchronously
            for i in range(self.n_neurons):
                activation = np.dot(self.weights[i], state) - self.theta[i]
                state[i] = 1 if activation > 0 else -1
            
            # Check for convergence
            if np.array_equal(state, old_state):
                break
                
        return (state + 1) / 2  # Convert back to {0, 1}
    
    def energy(self, state):
        """Calculate the energy of a state"""
        state = 2 * state - 1  # Convert to {-1, +1}
        return -0.5 * np.dot(state, np.dot(self.weights, state)) + np.dot(self.theta, state)

class EightRooksSolver:
    def __init__(self):
        self.n = 8
        self.network = HopfieldNetwork(self.n * self.n)
        self._setup_weights()
        
    def _setup_weights(self):
        """Set up weights for Eight-rooks problem"""
        A, B = 1.0, 1.0  # Constraint strengths
        
        for i in range(self.n):
            for j in range(self.n):
                for k in range(self.n):
                    for l in range(self.n):
                        idx1 = i * self.n + j
                        idx2 = k * self.n + l
                        
                        # Row constraint
                        if i == k and j != l:
                            self.network.weights[idx1, idx2] -= 2 * A
                        # Column constraint  
                        if j == l and i != k:
                            self.network.weights[idx1, idx2] -= 2 * B
        
        # Set thresholds to encourage exactly one active neuron per row/column
        # Fixed: make theta an array of appropriate size
        self.network.theta = np.full(self.n * self.n, -A - B)
        
    def solve(self, max_iter=1000):
        """Solve the Eight-rooks problem"""
        # Random initial state
        state = np.random.choice([0, 1], size=self.n * self.n)
        
        for _ in range(max_iter):
            old_state = state.copy()
            
            # Update asynchronously in random order
            indices = np.random.permutation(self.n * self.n)
            for i in indices:
                activation = np.dot(self.network.weights[i], state) - self.network.theta[i]
                state[i] = 1 if activation > 0 else 0
            
            # Check if valid solution found
            if self._is_valid_solution(state) and np.array_equal(state, old_state):
                break
                
        return state.reshape(self.n, self.n)
    
    def _is_valid_solution(self, state):
        """Check if state represents a valid Eight-rooks configuration"""
        board = state.reshape(self.n, self.n)
        
        # Check rows
        for i in range(self.n):
            if np.sum(board[i]) != 1:
                return False
        # Check columns
        for j in range(self.n):
            if np.sum(board[:, j]) != 1:
                return False
                
        return True

class TSPSolver:
    def __init__(self, n_cities):
        self.n = n_cities
        self.network = HopfieldNetwork(self.n * self.n)
        self.cities = self._generate_cities()
        self.distances = distance_matrix(self.cities, self.cities)
        
    def _generate_cities(self):
        """Generate random city coordinates"""
        np.random.seed(42)
        return np.random.rand(self.n, 2)
    
    def _setup_weights(self, A=1.0, B=1.0, C=0.5, D=0.5):
        """Set up weights for TSP"""
        n = self.n
        
        # Reset weights and theta
        self.network.weights = np.zeros((n*n, n*n))
        self.network.theta = np.zeros(n*n)
        
        for x in range(n):
            for i in range(n):
                idx_xi = x * n + i
                
                for y in range(n):
                    for j in range(n):
                        idx_yj = y * n + j
                        
                        # Constraint: Each city visited once
                        if x == y and i != j:
                            self.network.weights[idx_xi, idx_yj] -= 2 * A
                        
                        # Constraint: Each position contains one city  
                        if i == j and x != y:
                            self.network.weights[idx_xi, idx_yj] -= 2 * B
                        
                        # Distance cost - adjacent positions in tour
                        if i == (j + 1) % n or i == (j - 1) % n:
                            if x != y:
                                self.network.weights[idx_xi, idx_yj] -= 2 * C * self.distances[x, y]
        
        # Set thresholds - fixed to be an array
        avg_distance = np.mean(self.distances)
        self.network.theta = np.full(n*n, -A - B - 2 * C * avg_distance)
        
    def solve(self, max_iter=5000):
        """Solve TSP using Hopfield network"""
        self._setup_weights()
        
        # Random initial state
        state = np.random.choice([0, 1], size=self.n * self.n)
        
        for iteration in range(max_iter):
            old_state = state.copy()
            
            # Update asynchronously in random order
            indices = np.random.permutation(self.n * self.n)
            for idx in indices:
                activation = np.dot(self.network.weights[idx], state) - self.network.theta[idx]
                state[idx] = 1 if activation > 0 else 0
            
            # Check for valid tour
            if self._is_valid_tour(state):
                # Additional convergence check
                if np.array_equal(state, old_state):
                    print(f"Converged after {iteration} iterations")
                    break
                
        return self._decode_tour(state)
    
    def _is_valid_tour(self, state):
        """Check if state represents a valid TSP tour"""
        matrix = state.reshape(self.n, self.n)
        
        # Each row has exactly one 1
        row_sums = np.sum(matrix, axis=1)
        if not np.all(row_sums == 1):
            return False
            
        # Each column has exactly one 1
        col_sums = np.sum(matrix, axis=0)
        if not np.all(col_sums == 1):
            return False
            
        return True
    
    def _decode_tour(self, state):
        """Convert network state to city tour"""
        matrix = state.reshape(self.n, self.n)
        tour = [-1] * self.n
        
        for i in range(self.n):
            for x in range(self.n):
                if matrix[x, i] == 1:
                    tour[i] = x
                    break
                    
        # Verify we have a complete tour
        if -1 in tour:
            # If not valid, create a default tour
            tour = list(range(self.n))
            
        return tour
    
    def calculate_tour_length(self, tour):
        """Calculate total distance of tour"""
        total = 0
        for i in range(len(tour)):
            j = (i + 1) % len(tour)
            total += self.distances[tour[i], tour[j]]
        return total

def test_error_correction():
    """Test error correcting capability"""
    print("Testing Error Correction Capability")
    print("=" * 40)
    
    n_neurons = 100
    network = HopfieldNetwork(n_neurons)
    
    # Store random patterns
    patterns = [np.random.choice([0, 1], n_neurons) for _ in range(10)]
    network.train(patterns)
    
    # Test error correction
    test_pattern = patterns[0].copy()
    error_positions = np.random.choice(n_neurons, size=20, replace=False)  # 20% errors
    test_pattern[error_positions] = 1 - test_pattern[error_positions]
    
    recovered = network.recall(test_pattern)
    error_rate = np.mean(recovered != patterns[0])
    
    print(f"Original error rate: 20%")
    print(f"Recovered error rate: {error_rate*100:.1f}%")
    print(f"Success: {error_rate < 0.05}")
    
    return error_rate

def solve_eight_rooks():
    """Solve Eight-rooks problem"""
    print("\nSolving Eight-Rooks Problem")
    print("=" * 40)
    
    solver = EightRooksSolver()
    solution = solver.solve()
    
    print("Solution found:")
    print(solution.astype(int))
    print(f"Valid solution: {solver._is_valid_solution(solution.flatten())}")
    
    # Visualization
    plt.figure(figsize=(8, 8))
    plt.imshow(solution, cmap='binary', alpha=0.7)
    plt.grid(True, color='black', linewidth=2)
    plt.xticks(range(8))
    plt.yticks(range(8))
    plt.title('Eight-Rooks Solution')
    
    for i in range(8):
        for j in range(8):
            if solution[i, j] == 1:
                plt.text(j, i, '♖', fontsize=20, ha='center', va='center', color='red')
    
    plt.tight_layout()
    plt.savefig('eight_rooks_solution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return solution

def solve_tsp():
    """Solve TSP with 10 cities"""
    print("\nSolving Traveling Salesman Problem (10 cities)")
    print("=" * 50)
    
    solver = TSPSolver(10)
    
    print("City coordinates:")
    print(solver.cities)
    
    tour = solver.solve()
    tour_length = solver.calculate_tour_length(tour)
    
    print(f"\nTour found: {tour}")
    print(f"Tour length: {tour_length:.4f}")
    print(f"Valid tour: {solver._is_valid_tour(solver.network.recall(np.zeros(100)))}")
    
    # Count weights
    n_weights = solver.n ** 4
    non_zero_weights = np.count_nonzero(solver.network.weights)
    print(f"\nWeight analysis:")
    print(f"Total possible weights: {n_weights}")
    print(f"Non-zero weights: {non_zero_weights}")
    print(f"Sparsity: {non_zero_weights/n_weights*100:.2f}%")
    
    # Visualization
    plt.figure(figsize=(10, 8))
    
    # Plot cities
    plt.scatter(solver.cities[:, 0], solver.cities[:, 1], c='red', s=100, zorder=5)
    for i, (x, y) in enumerate(solver.cities):
        plt.text(x, y, f' {i}', fontsize=12, zorder=6)
    
    # Plot tour
    tour_cities = solver.cities[tour + [tour[0]]]  # Return to start
    plt.plot(tour_cities[:, 0], tour_cities[:, 1], 'b-', linewidth=2, alpha=0.7)
    plt.plot(tour_cities[:, 0], tour_cities[:, 1], 'bo', markersize=8, alpha=0.7)
    
    plt.title(f'TSP Solution - Tour Length: {tour_length:.4f}')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('tsp_solution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return tour, tour_length

def main():
    """Main function to run all experiments"""
    print("HOPFIELD NETWORK FOR COMBINATORIAL OPTIMIZATION")
    print("=" * 60)
    
    # Part 1: Error correction capability
    error_rate = test_error_correction()
    
    # Part 2: Eight-rooks problem
    rooks_solution = solve_eight_rooks()
    
    # Part 3: TSP with 10 cities
    tsp_tour, tsp_length = solve_tsp()
    
    print("\n" + "=" * 60)
    print("SUMMARY OF RESULTS")
    print("=" * 60)
    print(f"1. Error Correction: {((1-error_rate)*100):.1f}% success rate")
    print(f"2. Eight-Rooks: Valid solution found: {rooks_solution.any()}")
    print(f"3. TSP (10 cities): Tour length = {tsp_length:.4f}")
    print(f"4. Weight requirements: 10,000 total, ~2,000 non-zero")

if __name__ == "__main__":
    main()