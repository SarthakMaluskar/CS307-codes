from collections import deque
import time

class RabbitLeap:
    def __init__(self):
        self.initial_state = "EEE_WWW"
        self.goal_state = "WWW_EEE"
    
    def is_valid_state(self, state):
        """Check if state has exactly 3 E, 3 W, and 1 _"""
        return (state.count('E') == 3 and 
                state.count('W') == 3 and 
                state.count('_') == 1 and
                len(state) == 7)
    
    def get_valid_moves(self, state):
        """Generate all valid moves from current state"""
        if not self.is_valid_state(state):
            return []
            
        moves = []
        empty_idx = state.index('_')
        
        # Check east-bound moves (E moves right)
        if empty_idx > 0 and state[empty_idx - 1] == 'E':  # Slide east
            new_state = self.swap_positions(state, empty_idx, empty_idx - 1)
            if self.is_valid_state(new_state):
                moves.append(new_state)
        
        if empty_idx > 1 and state[empty_idx - 2] == 'E' and state[empty_idx - 1] == 'W':  # Jump east
            new_state = self.swap_positions(state, empty_idx, empty_idx - 2)
            if self.is_valid_state(new_state):
                moves.append(new_state)
        
        # Check west-bound moves (W moves left)
        if empty_idx < len(state) - 1 and state[empty_idx + 1] == 'W':  # Slide west
            new_state = self.swap_positions(state, empty_idx, empty_idx + 1)
            if self.is_valid_state(new_state):
                moves.append(new_state)
        
        if empty_idx < len(state) - 2 and state[empty_idx + 2] == 'W' and state[empty_idx + 1] == 'E':  # Jump west
            new_state = self.swap_positions(state, empty_idx, empty_idx + 2)
            if self.is_valid_state(new_state):
                moves.append(new_state)
        
        return moves
    
    def swap_positions(self, state, i, j):
        """Swap two positions in the state string"""
        state_list = list(state)
        state_list[i], state_list[j] = state_list[j], state_list[i]
        return ''.join(state_list)
    
    def bfs_solve(self):
        """Solve using Breadth-First Search"""
        queue = deque()
        visited = set()
        parent = {}
        
        queue.append(self.initial_state)
        visited.add(self.initial_state)
        parent[self.initial_state] = None
        
        nodes_explored = 0
        
        while queue:
            current_state = queue.popleft()
            nodes_explored += 1
            
            if current_state == self.goal_state:
                # Reconstruct path
                path = []
                while current_state:
                    path.append(current_state)
                    current_state = parent[current_state]
                return path[::-1], nodes_explored
            
            for next_state in self.get_valid_moves(current_state):
                if next_state not in visited:
                    visited.add(next_state)
                    parent[next_state] = current_state
                    queue.append(next_state)
        
        return None, nodes_explored
    
    def dfs_solve(self):
        """Solve using Depth-First Search"""
        stack = []
        visited = set()
        parent = {}
        
        stack.append(self.initial_state)
        visited.add(self.initial_state)
        parent[self.initial_state] = None
        
        nodes_explored = 0
        
        while stack:
            current_state = stack.pop()
            nodes_explored += 1
            
            if current_state == self.goal_state:
                # Reconstruct path
                path = []
                while current_state:
                    path.append(current_state)
                    current_state = parent[current_state]
                return path[::-1], nodes_explored
            
            # Reverse the moves to explore in consistent order (optional)
            moves = self.get_valid_moves(current_state)
            moves.reverse()  # For consistent exploration
            
            for next_state in moves:
                if next_state not in visited:
                    visited.add(next_state)
                    parent[next_state] = current_state
                    stack.append(next_state)
        
        return None, nodes_explored
    
    def print_solution(self, path, algorithm_name):
        """Print the solution path"""
        if not path:
            print(f"No solution found using {algorithm_name}!")
            return
        
        print(f"\n{algorithm_name} Solution ({len(path)-1} steps):")
        print("-" * 50)
        for i, state in enumerate(path):
            move_type = self.get_move_type(path[i-1], state) if i > 0 else "Initial"
            print(f"Step {i:2d}: {state} {f'({move_type})' if i > 0 else ''}")
    
    def get_move_type(self, prev_state, current_state):
        """Determine the type of move made"""
        prev_empty = prev_state.index('_')
        curr_empty = current_state.index('_')
        
        diff = curr_empty - prev_empty
        
        if diff == 1: return "W slides left"
        elif diff == -1: return "E slides right"
        elif diff == 2: return "W jumps over E"
        elif diff == -2: return "E jumps over W"
        else: return "Unknown move"

def main():
    problem = RabbitLeap()
    
    print("Rabbit Leap Problem Solver")
    print("=" * 50)
    print(f"Initial State: {problem.initial_state}")
    print(f"Goal State:    {problem.goal_state}")
    print("=" * 50)
    
    # BFS Solution
    print("\nSolving with BFS...")
    start_time = time.time()
    bfs_path, bfs_nodes = problem.bfs_solve()
    bfs_time = time.time() - start_time
    
    if bfs_path:
        problem.print_solution(bfs_path, "BFS")
        print(f"\nBFS Statistics:")
        print(f"  Steps: {len(bfs_path)-1}")
        print(f"  Nodes explored: {bfs_nodes}")
        print(f"  Time: {bfs_time:.4f} seconds")
    else:
        print("No BFS solution found!")
    
    print("\n" + "=" * 50)
    
    # DFS Solution
    print("\nSolving with DFS...")
    start_time = time.time()
    dfs_path, dfs_nodes = problem.dfs_solve()
    dfs_time = time.time() - start_time
    
    if dfs_path:
        problem.print_solution(dfs_path, "DFS")
        print(f"\nDFS Statistics:")
        print(f"  Steps: {len(dfs_path)-1}")
        print(f"  Nodes explored: {dfs_nodes}")
        print(f"  Time: {dfs_time:.4f} seconds")
    else:
        print("No DFS solution found!")
    
    # Comparison
    print("\n" + "=" * 50)
    print("COMPARISON SUMMARY:")
    print("=" * 50)
    if bfs_path and dfs_path:
        print(f"{'Algorithm':<10} {'Steps':<6} {'Nodes':<8} {'Time (s)':<10} {'Optimal'}")
        print(f"{'BFS':<10} {len(bfs_path)-1:<6} {bfs_nodes:<8} {bfs_time:.4f}    {'Yes'}")
        print(f"{'DFS':<10} {len(dfs_path)-1:<6} {dfs_nodes:<8} {dfs_time:.4f}    {'No'}")

def test_moves():
    """Test function to verify move generation"""
    problem = RabbitLeap()
    
    print("Testing move generation:")
    test_states = [
        "EEE_WWW",  # Initial
        "EE_EWWW",  # After E slide
        "EEWE_WW",  # After E jump
    ]
    
    for state in test_states:
        moves = problem.get_valid_moves(state)
        print(f"\nState: {state}")
        print(f"Valid moves: {moves}")

if __name__ == "__main__":
    main()
    
    # Uncomment to test move generation
    # print("\n" + "="*50)
    # test_moves()
