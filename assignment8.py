import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson
import time

MAX_BIKES = 20
MAX_MOVE = 5
GAMMA = 0.9
RENTAL_REWARD = 10
MOVE_COST = 2
PARKING_COST = 4
PARKING_LIMIT = 10

LAMBDA_REQ_A = 3
LAMBDA_RET_A = 3
LAMBDA_REQ_B = 4
LAMBDA_RET_B = 2

THETA = 1e-4

class GbikeMDP:
    def __init__(self):
        self.poisson_cache = {}
        self.max_poisson_n = 29 
        
        print("Precomputing transition dynamics...")
        self.trans_prob_A, self.exp_reward_A = self._precompute_dynamics(LAMBDA_REQ_A, LAMBDA_RET_A)
        self.trans_prob_B, self.exp_reward_B = self._precompute_dynamics(LAMBDA_REQ_B, LAMBDA_RET_B)
        print("Precomputation complete.")

        self.V = np.zeros((MAX_BIKES + 1, MAX_BIKES + 1))
        self.policy = np.zeros((MAX_BIKES + 1, MAX_BIKES + 1), dtype=int)
        self.actions = np.arange(-MAX_MOVE, MAX_MOVE + 1)

    def _get_poisson_prob(self, n, lam):
        key = (n, lam)
        if key not in self.poisson_cache:
            self.poisson_cache[key] = poisson.pmf(n, lam)
        return self.poisson_cache[key]

    def _precompute_dynamics(self, lambda_req, lambda_ret):
        transition_probs = np.zeros((MAX_BIKES + 1, MAX_BIKES + 1))
        expected_rewards = np.zeros(MAX_BIKES + 1)

        for s_morning in range(MAX_BIKES + 1):
            prob_mass = 0.0
            
            for req in range(self.max_poisson_n):
                prob_req = self._get_poisson_prob(req, lambda_req)
                rentals = min(s_morning, req)
                expected_rewards[s_morning] += prob_req * (rentals * RENTAL_REWARD)
                
                for ret in range(self.max_poisson_n):
                    prob_ret = self._get_poisson_prob(ret, lambda_ret)
                    joint_prob = prob_req * prob_ret
                    
                    next_s = s_morning - rentals + ret
                    next_s = min(next_s, MAX_BIKES)
                    next_s = max(next_s, 0)
                    
                    transition_probs[s_morning, next_s] += joint_prob
                    prob_mass += joint_prob

        return transition_probs, expected_rewards

    def calculate_moving_cost(self, action):
        cost = 0
        if action > 0:
            count = action - 1
            if count > 0:
                cost += count * MOVE_COST
        elif action < 0:
            cost += abs(action) * MOVE_COST
        return cost

    def calculate_parking_cost(self, num_bikes):
        return PARKING_COST if num_bikes > PARKING_LIMIT else 0

    def expected_return(self, state, action, V):
        n_A, n_B = state
        
        move = action
        if move > 0:
            move = min(move, n_A)
            move = min(move, MAX_BIKES - n_B)
        else:
            move = max(move, -n_B)
            move = max(move, -(MAX_BIKES - n_A))
            
        morning_A = n_A - move
        morning_B = n_B + move
        
        cost = self.calculate_moving_cost(move)
        cost += self.calculate_parking_cost(morning_A)
        cost += self.calculate_parking_cost(morning_B)
        
        exp_rental_A = self.exp_reward_A[morning_A]
        exp_rental_B = self.exp_reward_B[morning_B]
        total_immediate_reward = exp_rental_A + exp_rental_B - cost
        
        probs_A = self.trans_prob_A[morning_A]
        probs_B = self.trans_prob_B[morning_B]
        
        weighted_V_by_B = np.dot(V, probs_B)
        expected_V_next = np.dot(probs_A, weighted_V_by_B)
        
        return total_immediate_reward + GAMMA * expected_V_next

    def policy_evaluation(self):
        print("  Evaluated Policy...", end=" ")
        while True:
            delta = 0
            new_V = np.copy(self.V)
            
            for i in range(MAX_BIKES + 1):
                for j in range(MAX_BIKES + 1):
                    action = self.policy[i, j]
                    new_val = self.expected_return((i, j), action, self.V)
                    delta = max(delta, abs(new_val - self.V[i, j]))
                    new_V[i, j] = new_val
            
            self.V = new_V
            if delta < THETA:
                break
        print(f"Converged (Delta: {delta:.5f})")

    def policy_improvement(self):
        print("  Improving Policy...")
        policy_stable = True
        
        for i in range(MAX_BIKES + 1):
            for j in range(MAX_BIKES + 1):
                old_action = self.policy[i, j]
                action_returns = []
                
                possible_actions = np.arange(-MAX_MOVE, MAX_MOVE + 1)
                
                for action in possible_actions:
                    if (action > 0 and action > i) or (action < 0 and abs(action) > j):
                        action_returns.append(-np.inf)
                        continue
                    if (action > 0 and (j + action) > MAX_BIKES) or (action < 0 and (i + abs(action)) > MAX_BIKES):
                        action_returns.append(-np.inf)
                        continue
                        
                    val = self.expected_return((i, j), action, self.V)
                    action_returns.append(val)
                
                best_action_idx = np.argmax(action_returns)
                best_action = possible_actions[best_action_idx]
                
                self.policy[i, j] = best_action
                
                if old_action != best_action:
                    policy_stable = False
                    
        return policy_stable

    def solve(self):
        print("Starting Policy Iteration...")
        start_time = time.time()
        iterations = 0
        while True:
            iterations += 1
            print(f"Iteration {iterations}:")
            self.policy_evaluation()
            stable = self.policy_improvement()
            if stable:
                print(f"Policy stable after {iterations} iterations.")
                break
        print(f"Total time: {time.time() - start_time:.2f}s")

    def plot_results(self):
        fig, ax = plt.subplots(1, 2, figsize=(18, 7))
        
        sns.heatmap(self.policy, ax=ax[0], cmap="coolwarm", annot=True, fmt="d", 
                    cbar_kws={'label': 'Net Bikes Moved (A -> B)'})
        ax[0].invert_yaxis()
        ax[0].set_title("Optimal Policy\n(Positive = Move A->B, Negative = Move B->A)")
        ax[0].set_xlabel("Bikes at B")
        ax[0].set_ylabel("Bikes at A")

        sns.heatmap(self.V, ax=ax[1], cmap="viridis", 
                    cbar_kws={'label': 'Expected Value'})
        ax[1].invert_yaxis()
        ax[1].set_title("Optimal Value Function")
        ax[1].set_xlabel("Bikes at B")
        ax[1].set_ylabel("Bikes at A")
        
        plt.tight_layout()
        
        plt.savefig('policy_and_value_heatmap.png', dpi=300)
        print("Plots saved to 'policy_and_value_heatmap.png'")
        
        plt.show()

if __name__ == "__main__":
    solver = GbikeMDP()
    solver.solve()
    solver.plot_results()
