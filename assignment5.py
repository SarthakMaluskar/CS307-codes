# hmm_financial_analysis.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn import hmm
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class FinancialHMM:
    def __init__(self, n_states=3):
        self.n_states = n_states
        self.model = None
        self.hidden_states = None
        
    def create_synthetic_data(self, start_date='2014-01-01', end_date='2023-12-31'):
        """Create realistic synthetic stock price data with regime changes"""
        print("Creating synthetic financial data...")
        
        # Create business days only (exclude weekends)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        dates = dates[dates.dayofweek < 5]  # Monday=0, Friday=4
        n_days = len(dates)
        
        np.random.seed(42)  # For reproducibility
        
        # Start with initial price
        initial_price = 100
        prices = [initial_price]
        log_returns = []
        
        # Define market regimes with different characteristics
        regimes = [
            # State 0: Bull market (high positive returns, low volatility)
            {'mean': 0.0012, 'std': 0.014, 'color': 'green', 'label': 'Bull Market'},
            # State 1: Bear market (negative returns, high volatility)  
            {'mean': -0.0008, 'std': 0.025, 'color': 'red', 'label': 'Bear Market'},
            # State 2: Transitional (moderate returns and volatility)
            {'mean': 0.0004, 'std': 0.018, 'color': 'orange', 'label': 'Transitional'}
        ]
        
        # Generate regime sequence with persistence
        true_states = [0]
        current_state = 0
        state_duration = 1
        
        for i in range(1, n_days):
            # Change state with probability based on current duration
            change_prob = min(0.02 + state_duration * 0.001, 0.1)
            if np.random.random() < change_prob:
                # Prefer to move to adjacent states
                possible_states = [s for s in range(self.n_states) if s != current_state]
                current_state = np.random.choice(possible_states)
                state_duration = 1
            else:
                state_duration += 1
            true_states.append(current_state)
        
        # Generate returns based on true states
        for i in range(n_days - 1):  # Generate n_days-1 returns
            state = true_states[i]
            regime = regimes[state]
            
            # Generate return with some autocorrelation
            if i > 0:
                prev_return = log_returns[-1]
                # Add momentum effect
                momentum = 0.1 * prev_return
            else:
                momentum = 0
                
            return_val = np.random.normal(regime['mean'] + momentum, regime['std'])
            log_returns.append(return_val)
            
            # Calculate new price
            new_price = prices[-1] * np.exp(return_val)
            prices.append(new_price)
        
        # Create DataFrame - ensure all arrays have same length
        # We have n_days prices but n_days-1 returns
        self.stock_data = pd.DataFrame({
            'Open': prices[:-1],
            'High': [p * (1 + abs(r)/2) for p, r in zip(prices[:-1], log_returns)],
            'Low': [p * (1 - abs(r)/2) for p, r in zip(prices[:-1], log_returns)],
            'Close': prices[:-1],
            'Adj Close': prices[:-1],
            'Volume': np.random.lognormal(15, 1, len(prices[:-1])),
            'Log_Returns': log_returns,
            'True_State': true_states[:-1]  # Match the length of returns
        }, index=dates[:len(prices[:-1])])  # Match the length
        
        # Calculate simple returns as well
        self.stock_data['Returns'] = self.stock_data['Adj Close'].pct_change()
        # Remove first row with NaN
        self.stock_data = self.stock_data.iloc[1:]
        
        print(f"Created synthetic data with {len(self.stock_data)} trading days")
        print(f"Date range: {self.stock_data.index[0].date()} to {self.stock_data.index[-1].date()}")
        
        return self.stock_data
    
    def preprocess_data(self):
        """Preprocess data - mainly for consistency with real data workflow"""
        # Remove any remaining NaN values
        self.stock_data = self.stock_data.dropna()
        
        print(f"Data preprocessing complete. {len(self.stock_data)} trading days available.")
        print(f"Average daily return: {self.stock_data['Log_Returns'].mean():.6f}")
        print(f"Volatility: {self.stock_data['Log_Returns'].std():.6f}")
        
        return self.stock_data
    
    def fit_hmm(self, returns_column='Log_Returns'):
        """Fit Gaussian HMM to the returns data"""
        # Prepare data for HMM
        returns = self.stock_data[returns_column].values.reshape(-1, 1)
        
        print(f"Fitting HMM with {self.n_states} states on {len(returns)} data points...")
        
        # Create and fit HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=1000,
            random_state=42
        )
        
        # Fit the model first, then predict states
        self.model.fit(returns)
        self.hidden_states = self.model.predict(returns)
        self.stock_data['Hidden_State'] = self.hidden_states
        
        print("HMM fitting completed successfully.")
        return self.model, self.hidden_states
    
    def analyze_states(self):
        """Analyze the characteristics of each hidden state"""
        state_analysis = {}
        
        for state in range(self.n_states):
            state_returns = self.stock_data[self.stock_data['Hidden_State'] == state]['Log_Returns']
            if len(state_returns) > 0:  # Only analyze states that have data
                state_analysis[state] = {
                    'mean_return': state_returns.mean(),
                    'volatility': state_returns.std(),
                    'count': len(state_returns),
                    'proportion': len(state_returns) / len(self.stock_data)
                }
        
        return state_analysis
    
    def plot_results(self, symbol='SYNTHETIC'):
        """Create comprehensive visualizations"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Colors for different states
        colors = ['green', 'red', 'orange', 'blue', 'purple']
        
        # Plot 1: Stock price with hidden states
        for state in range(self.n_states):
            mask = self.stock_data['Hidden_State'] == state
            if mask.any():  # Only plot if state exists in data
                axes[0].plot(self.stock_data.index[mask], 
                            self.stock_data['Adj Close'][mask],
                            'o', color=colors[state], alpha=0.7, 
                            label=f'State {state}', markersize=2)
        
        # Add overall price trend
        axes[0].plot(self.stock_data.index, self.stock_data['Adj Close'], 
                    'k-', alpha=0.3, linewidth=0.5)
        axes[0].set_title(f'{symbol} Stock Price with Hidden Market Regimes')
        axes[0].set_ylabel('Price ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Returns with hidden states
        for state in range(self.n_states):
            mask = self.stock_data['Hidden_State'] == state
            if mask.any():  # Only plot if state exists in data
                axes[1].plot(self.stock_data.index[mask], 
                            self.stock_data['Log_Returns'][mask],
                            'o', color=colors[state], alpha=0.7,
                            label=f'State {state}', markersize=2)
        
        axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[1].set_title('Daily Log Returns with Hidden States')
        axes[1].set_ylabel('Log Returns')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: State transitions over time
        axes[2].plot(self.stock_data.index, self.stock_data['Hidden_State'], 
                    linewidth=1)
        axes[2].set_title('Hidden State Transitions Over Time')
        axes[2].set_ylabel('Hidden State')
        axes[2].set_xlabel('Date')
        axes[2].set_yticks(range(self.n_states))
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('hmm_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_state_statistics(self, state_analysis):
        """Plot statistics for each hidden state"""
        if not state_analysis:
            print("No state analysis data available.")
            return None
            
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        states = list(state_analysis.keys())
        means = [state_analysis[state]['mean_return'] * 100 for state in states]  # Convert to percentage
        volatilities = [state_analysis[state]['volatility'] * 100 for state in states]  # Convert to percentage
        proportions = [state_analysis[state]['proportion'] for state in states]
        
        # Mean returns and volatilities
        x_pos = np.arange(len(states))
        width = 0.35
        
        bars1 = axes[0].bar(x_pos - width/2, means, width, label='Mean Return (%)', alpha=0.7, color='blue')
        bars2 = axes[0].bar(x_pos + width/2, volatilities, width, label='Volatility (%)', alpha=0.7, color='red')
        axes[0].set_xlabel('Hidden State')
        axes[0].set_ylabel('Percentage (%)')
        axes[0].set_title('Mean Returns and Volatility by Hidden State')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels([f'State {s}' for s in states])
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}%', ha='center', va='bottom', fontsize=9)
        
        for bar in bars2:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}%', ha='center', va='bottom', fontsize=9)
        
        # State proportions
        colors = ['green', 'red', 'orange']
        wedges, texts, autotexts = axes[1].pie(proportions, 
                                              labels=[f'State {s}' for s in states], 
                                              autopct='%1.1f%%', 
                                              startangle=90, 
                                              colors=colors[:len(states)])
        axes[1].set_title('Proportion of Time in Each Hidden State')
        
        plt.tight_layout()
        plt.savefig('state_statistics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig

    def print_transition_matrix(self):
        """Print the transition matrix in a readable format"""
        if self.model is None:
            print("Model not fitted yet.")
            return
        
        transition_matrix = self.model.transmat_
        print("\nTransition Probability Matrix:")
        print("+" + "-"*50 + "+")
        print("| From \\ To", end="")
        for j in range(self.n_states):
            print(f"|   State {j}   ", end="")
        print("|")
        print("+" + "-"*50 + "+")
        
        for i in range(self.n_states):
            print(f"| State {i:4} ", end="")
            for j in range(self.n_states):
                print(f"|   {transition_matrix[i,j]:6.4f}   ", end="")
            print("|")
        print("+" + "-"*50 + "+")
        
        # Calculate state persistence
        print("\nState Persistence (Diagonal Elements):")
        for i in range(self.n_states):
            persistence = 1 / (1 - transition_matrix[i,i]) if transition_matrix[i,i] < 1 else float('inf')
            print(f"State {i}: {transition_matrix[i,i]:.4f} → ~{persistence:.1f} days expected duration")

def main():
    """Main function to run the complete analysis"""
    print("=" * 70)
    print("FINANCIAL TIME SERIES ANALYSIS USING GAUSSIAN HIDDEN MARKOV MODELS")
    print("=" * 70)
    
    # Initialize HMM analyzer
    hmm_analyzer = FinancialHMM(n_states=3)
    
    # Part 1: Data Creation and Preprocessing
    print("\n1. CREATING AND PREPROCESSING SYNTHETIC FINANCIAL DATA...")
    stock_data = hmm_analyzer.create_synthetic_data()
    processed_data = hmm_analyzer.preprocess_data()
    
    print(f"📊 Data Overview:")
    print(f"   • Period: {processed_data.index[0].date()} to {processed_data.index[-1].date()}")
    print(f"   • Trading Days: {len(processed_data):,}")
    print(f"   • Final Price: ${processed_data['Adj Close'].iloc[-1]:.2f}")
    print(f"   • Total Return: {(processed_data['Adj Close'].iloc[-1]/100-1)*100:.1f}%")
    
    # Part 2: Fit Gaussian HMM
    print("\n2. FITTING GAUSSIAN HIDDEN MARKOV MODEL...")
    model, hidden_states = hmm_analyzer.fit_hmm()
    
    # Part 3: Parameter Analysis
    print("\n3. HIDDEN STATE ANALYSIS...")
    state_analysis = hmm_analyzer.analyze_states()
    
    if not state_analysis:
        print("No states found in the analysis. Exiting.")
        return
    
    print("\n" + "=" * 60)
    print("HIDDEN STATE CHARACTERISTICS")
    print("=" * 60)
    for state, stats in state_analysis.items():
        mean_pct = stats['mean_return'] * 100 * 252  # Annualized
        vol_pct = stats['volatility'] * 100 * np.sqrt(252)  # Annualized
        print(f"\n📍 State {state}:")
        print(f"   📈 Mean Return: {stats['mean_return']:.6f} ({mean_pct:.1f}% annualized)")
        print(f"   📊 Volatility:  {stats['volatility']:.6f} ({vol_pct:.1f}% annualized)")
        print(f"   ⏱️  Proportion:  {stats['proportion']:.3f} ({stats['count']} days)")
    
    # Transition Matrix Analysis
    hmm_analyzer.print_transition_matrix()
    
    # Part 4: Visualization
    print("\n4. GENERATING VISUALIZATIONS...")
    hmm_analyzer.plot_results('SYNTHETIC')
    hmm_analyzer.plot_state_statistics(state_analysis)
    
    # Part 5: Interpretation and Insights
    print("\n5. MARKET REGIME INTERPRETATION...")
    print("=" * 50)
    
    # Identify state characteristics
    print("\n🔮 MARKET REGIME CLASSIFICATION:")
    regime_descriptions = []
    for state, stats in state_analysis.items():
        mean_return = stats['mean_return']
        volatility = stats['volatility']
        
        if mean_return > 0.0008 and volatility < 0.016:
            regime_type = "BULL MARKET"
            description = "High returns with low volatility - Ideal for growth investing"
            emoji = "🟢"
            strategy = "Aggressive growth strategy recommended"
        elif mean_return < -0.0003 and volatility > 0.022:
            regime_type = "BEAR MARKET" 
            description = "Negative returns with high volatility - Risk management crucial"
            emoji = "🔴"
            strategy = "Defensive positioning and risk reduction advised"
        elif mean_return > 0.0002 and volatility > 0.018:
            regime_type = "VOLATILE GROWTH"
            description = "Positive but risky returns - Requires careful position sizing"
            emoji = "🟡"
            strategy = "Balanced approach with moderate risk exposure"
        else:
            regime_type = "SIDEWAYS MARKET"
            description = "Moderate conditions with limited trends"
            emoji = "🔵"
            strategy = "Range-bound strategies may be effective"
        
        regime_descriptions.append((state, regime_type, description, strategy, emoji))
        print(f"{emoji} State {state}: {regime_type}")
        print(f"   📝 {description}")
        print(f"   💡 Strategy: {strategy}")
    
    # Future state prediction
    print("\n6. FUTURE STATE PREDICTION...")
    print("-" * 35)
    current_state = hidden_states[-1]
    transition_matrix = model.transmat_
    next_state_probs = transition_matrix[current_state]
    most_likely_next = np.argmax(next_state_probs)
    
    current_regime = next((r[1] for r in regime_descriptions if r[0] == current_state), "Unknown")
    next_regime = next((r[1] for r in regime_descriptions if r[0] == most_likely_next), "Unknown")
    
    print(f"🕒 Current market state: {current_state} ({current_regime})")
    print(f"🔮 Most likely next state: {most_likely_next} ({next_regime})")
    print(f"📊 Transition probability: {next_state_probs[most_likely_next]:.3f}")
    
    # Risk management insights
    print("\n7. RISK MANAGEMENT INSIGHTS...")
    print("-" * 35)
    high_vol_state = max(state_analysis.items(), key=lambda x: x[1]['volatility'])[0]
    low_vol_state = min(state_analysis.items(), key=lambda x: x[1]['volatility'])[0]
    
    high_vol_regime = next((r[1] for r in regime_descriptions if r[0] == high_vol_state), "Unknown")
    low_vol_regime = next((r[1] for r in regime_descriptions if r[0] == low_vol_state), "Unknown")
    
    print(f"⚡ Highest volatility: State {high_vol_state} ({high_vol_regime}) - {state_analysis[high_vol_state]['volatility']*100:.2f}% daily")
    print(f"🍃 Lowest volatility:  State {low_vol_state} ({low_vol_regime}) - {state_analysis[low_vol_state]['volatility']*100:.2f}% daily")
    print(f"🔄 Probability of entering high volatility: {transition_matrix[:, high_vol_state].mean():.3f}")
    
    # Model performance assessment
    print("\n8. MODEL PERFORMANCE ASSESSMENT...")
    print("-" * 35)
    
    # Calculate overall model log likelihood
    returns = processed_data['Log_Returns'].values.reshape(-1, 1)
    log_likelihood = model.score(returns)
    print(f"📈 Model Log-Likelihood: {log_likelihood:.2f}")
    
    # State persistence analysis
    state_changes = np.sum(np.diff(hidden_states) != 0)
    avg_persistence = len(hidden_states) / (state_changes + 1)
    print(f"⏱️  Average state persistence: {avg_persistence:.1f} days")
    print(f"🔄 Total state changes: {state_changes}")
    
    return hmm_analyzer, state_analysis, regime_descriptions

if __name__ == "__main__":
    try:
        # Run the analysis
        analyzer, analysis, regimes = main()
        
        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\n📁 Generated Files:")
        print("   • hmm_analysis_results.png - Price and returns with regimes")
        print("   • state_statistics.png - Statistical analysis of states")
        print("\n💡 Key Insights:")
        print("   • Market regimes successfully identified and characterized")
        print("   • Transition probabilities calculated for risk management")
        print("   • Visualizations created for pattern recognition")
        print("\n🎯 Next Steps:")
        print("   • Use identified regimes for tactical asset allocation")
        print("   • Monitor transition probabilities for early regime change detection")
        print("   • Apply similar analysis to portfolio of assets")
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        print("\n🔧 Troubleshooting Tips:")
        print("1. Ensure all dependencies are installed:")
        print("   pip install numpy pandas matplotlib seaborn scikit-learn hmmlearn")
        print("2. Check that you have write permissions in the current directory")
        print("3. Try running with different random seeds if results seem unstable")