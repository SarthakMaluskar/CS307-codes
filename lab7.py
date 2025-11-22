import numpy as np
import matplotlib.pyplot as plt
import random

# ==========================================
# PART 1: MENACE (Tic-Tac-Toe RL)
# ==========================================

class MENACE:
    def __init__(self):
        # Crucial Part 1: The 'Matchboxes'
        # Dictionary mapping board_state (str) -> {move_index: bead_count}
        self.matchboxes = {}
        self.moves_played = [] # History of (state, action) for the current game
        
        # Hyperparameters
        self.initial_beads = 3  # Start with equal probability
        self.win_reward = 3
        self.draw_reward = 1
        self.lose_penalty = 1

    def get_board_string(self, board):
        return "".join(board)

    def get_move(self, board):
        state = self.get_board_string(board)
        available_moves = [i for i, x in enumerate(board) if x == ' ']
        
        # If we haven't seen this state, initialize a new 'matchbox'
        if state not in self.matchboxes:
            self.matchboxes[state] = {m: self.initial_beads for m in available_moves}
        
        # Crucial Part 2: Probabilistic Action Selection (Drawing a bead)
        # Probability of action 'a' is count(a) / sum(counts)
        bead_counts = self.matchboxes[state]
        moves = list(bead_counts.keys())
        weights = list(bead_counts.values())
        
        # Handle case where a box becomes empty (resign or random reset)
        if sum(weights) == 0:
            chosen_move = random.choice(available_moves)
        else:
            chosen_move = random.choices(moves, weights=weights, k=1)[0]
            
        self.moves_played.append((state, chosen_move))
        return chosen_move

    def update(self, result):
        # Crucial Part 3: Reinforcement (Adding/Removing beads)
        # result: 1 (Win), 0 (Draw), -1 (Loss)
        
        if result == 1:
            delta = self.win_reward
        elif result == 0:
            delta = self.draw_reward
        else:
            delta = -self.lose_penalty

        for state, move in self.moves_played:
            if state in self.matchboxes:
                current_beads = self.matchboxes[state].get(move, 0)
                # Ensure beads don't drop below 0 (optional variation: 0 removes move permanently)
                new_count = max(0, current_beads + delta)
                self.matchboxes[state][move] = new_count
        
        # Clear history for next game
        self.moves_played = []

# (Helper function to run a quick MENACE training loop is at the bottom of main)

# ==========================================
# PART 2: BINARY BANDIT (Stationary)
# ==========================================

class BinaryBandit:
    def __init__(self, p_success):
        """
        p_success: list of probabilities for getting 1 for each arm.
        """
        self.probs = p_success

    def pull(self, action):
        # Returns 1 with probability p_success[action], else 0
        return 1 if np.random.random() < self.probs[action] else 0

def solve_binary_bandit(bandit_instance, steps=1000, epsilon=0.1):
    n_arms = len(bandit_instance.probs)
    Q = np.zeros(n_arms)      # Estimated value
    N = np.zeros(n_arms)      # Count of pulls
    
    history_Q = np.zeros((steps, n_arms)) # To plot convergence

    for t in range(steps):
        # Epsilon-Greedy Logic
        if np.random.random() < epsilon:
            action = np.random.randint(n_arms)
        else:
            # Random tie-breaking for argmax
            best_actions = np.flatnonzero(Q == Q.max())
            action = np.random.choice(best_actions)
        
        reward = bandit_instance.pull(action)
        
        # Update Estimates (Sample Average)
        N[action] += 1
        Q[action] = Q[action] + (1.0 / N[action]) * (reward - Q[action])
        
        history_Q[t] = Q
        
    return Q, history_Q

# ==========================================
# PART 3: NON-STATIONARY 10-ARMED BANDIT
# ==========================================

class NonStationaryBandit:
    def __init__(self, k_arms=10):
        self.k = k_arms
        # "All ten mean-rewards start out equal" (Initialized to 0 here)
        self.q_true = np.zeros(self.k) 
        
    def step(self):
        # "Add normally distributed increment with mean 0, std 0.01 to all mean-rewards"
        self.q_true += np.random.normal(0, 0.01, self.k)
        
    def pull(self, action):
        # Reward is the true mean + unit variance noise (Standard Sutton/Barto setup)
        return np.random.normal(self.q_true[action], 1.0)

def run_nonstat_experiment(steps=10000, runs=200, epsilon=0.1, alpha=None):
    """
    If alpha is None: Uses Sample Average (Standard Epsilon Greedy)
    If alpha is set: Uses Constant Step Size (Modified Epsilon Greedy)
    """
    rewards = np.zeros((runs, steps))
    optimal_action_counts = np.zeros((runs, steps))
    
    for r in range(runs):
        bandit = NonStationaryBandit()
        Q = np.zeros(bandit.k)
        N = np.zeros(bandit.k)
        
        for t in range(steps):
            bandit.step() # Evolution of the environment
            
            # Action Selection
            if np.random.random() < epsilon:
                action = np.random.randint(bandit.k)
            else:
                best_actions = np.flatnonzero(Q == Q.max())
                action = np.random.choice(best_actions)
                
            # Check if optimal for stats
            if action == np.argmax(bandit.q_true):
                optimal_action_counts[r, t] = 1
            
            # Get Reward
            reward = bandit.pull(action)
            
            # Update Rule
            if alpha is None:
                # Standard Sample Average method
                N[action] += 1
                Q[action] += (1.0 / N[action]) * (reward - Q[action])
            else:
                # Modified Constant Step-size method
                Q[action] += alpha * (reward - Q[action])
                
            rewards[r, t] = reward
            
    # Average over runs
    avg_rewards = rewards.mean(axis=0)
    return avg_rewards

# ==========================================
# MAIN EXECUTION & PLOTTING
# ==========================================

def main():
    print("Running Simulations...")

    # --- 1. Binary Bandits ---
    # Define Bandit A and Bandit B
    # Bandit A: Arm 0 is bad (0.2), Arm 1 is good (0.8)
    bandit_a = BinaryBandit([0.2, 0.8])
    # Bandit B: Arm 0 is okay (0.6), Arm 1 is bad (0.4)
    bandit_b = BinaryBandit([0.6, 0.4])

    print("Solving Binary Bandit A...")
    FinalQ_A, HistQ_A = solve_binary_bandit(bandit_a)
    print("Solving Binary Bandit B...")
    FinalQ_B, HistQ_B = solve_binary_bandit(bandit_b)

    # Plotting Fig 1
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(HistQ_A[:, 0], label='Action 0 (True: 0.2)')
    plt.plot(HistQ_A[:, 1], label='Action 1 (True: 0.8)')
    plt.title('Bandit A')
    plt.xlabel('Steps')
    plt.ylabel('Estimated Q Value')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(HistQ_B[:, 0], label='Action 0 (True: 0.6)')
    plt.plot(HistQ_B[:, 1], label='Action 1 (True: 0.4)')
    plt.title('Bandit B')
    plt.xlabel('Steps')
    plt.legend()
    plt.grid(True)

    plt.suptitle('Fig. 1. Expected Rewards (Q-Values) for Bandit A and Bandit B')
    plt.tight_layout()
    plt.savefig('Fig1_Binary_Bandits.png')
    print("Saved Fig1_Binary_Bandits.png")

    # --- 2. Non-Stationary Bandit ---
    print("\nRunning Non-Stationary Experiment (This takes a moment)...")
    # Simulation parameters
    n_steps = 10000 
    n_runs = 200 # Averaging over runs to get a smooth curve
    
    # Run 1: Standard Sample Average
    print(f"Agent 1: Sample Average (1/n) - {n_runs} runs...")
    avg_rewards_sample = run_nonstat_experiment(steps=n_steps, runs=n_runs, epsilon=0.1, alpha=None)
    
    # Run 2: Constant Alpha
    print(f"Agent 2: Constant Step-size (alpha=0.1) - {n_runs} runs...")
    avg_rewards_const = run_nonstat_experiment(steps=n_steps, runs=n_runs, epsilon=0.1, alpha=0.1)

    # Plotting Fig 2
    plt.figure(figsize=(10, 6))
    plt.plot(avg_rewards_sample, label='Sample Average (Standard)', alpha=0.8)
    plt.plot(avg_rewards_const, label='Constant Alpha=0.1 (Modified)', alpha=0.8)
    
    plt.title('Fig. 2. Average Reward over Steps (Non-Stationary 10-Armed Bandit)')
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Fig2_NonStationary_Rewards.png')
    print("Saved Fig2_NonStationary_Rewards.png")
    
    

if __name__ == "__main__":
    main()