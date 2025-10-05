import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import math
from collections import deque
import time
from tqdm import tqdm

def load_and_split_image(mat_path, grid_size, var_name='Iscrambled'):
    """Load image from .mat file and split into puzzle pieces with error handling."""
    try:
        with open(mat_path, 'r') as f:
            lines = f.readlines()
        
        # Skip the first line (creation comment)
        header_lines = lines[1:5]
        data_lines = lines[5:]
        
        # Parse header
        name_line = header_lines[0].strip()
        if not name_line.startswith('# name:'):
            raise ValueError("Invalid .mat file format")
        actual_var_name = name_line.split(': ')[1]
        if actual_var_name != var_name:
            print(f"Warning: Expected {var_name}, found {actual_var_name}")
        
        type_line = header_lines[1].strip()
        if not type_line.startswith('# type:'):
            raise ValueError("Invalid .mat file format")
        
        ndims_line = header_lines[2].strip()
        if not ndims_line.startswith('# ndims:'):
            raise ValueError("Invalid .mat file format")
        ndims = int(ndims_line.split(': ')[1])
        
        dims_line = header_lines[3].strip()
        dims = list(map(int, dims_line.split()))
        if ndims == 2:
            height, width = dims
        else:
            raise ValueError("Only 2D matrices supported")
        
        # Parse data
        data_flat = []
        for line in data_lines:
            if line.strip() and not line.startswith('#'):
                data_flat.extend(map(int, line.strip().split()))
        
        img_array = np.array(data_flat, dtype=np.uint8).reshape((height, width))
        # Convert grayscale to RGB
        img_array = np.stack([img_array] * 3, axis=-1)
        
        piece_height = height // grid_size
        piece_width = width // grid_size
        
        # Verify divisible dimensions
        if height % grid_size != 0 or width % grid_size != 0:
            print(f"Warning: Image dimensions {height}x{width} not perfectly divisible by {grid_size}")
            piece_height = height // grid_size
            piece_width = width // grid_size
        
        pieces = []
        for i in range(grid_size):
            for j in range(grid_size):
                top = i * piece_height
                left = j * piece_width
                bottom = min(top + piece_height, height)
                right = min(left + piece_width, width)
                piece = img_array[top:bottom, left:right, :]
                pieces.append(piece)
        
        print(f"✅ Successfully loaded {len(pieces)} pieces of size ~{piece_width}x{piece_height}")
        return pieces, piece_width, piece_height, grid_size
        
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        # Fallback: create a demo image
        return create_demo_image(grid_size)

def create_demo_image(grid_size=4):
    """Create a demo image for testing if file loading fails."""
    print("🔄 Creating demo image...")
    size = 512
    # Create a gradient test image
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    X, Y = np.meshgrid(x, y)
    demo_array = (np.sin(10*X) * np.cos(10*Y) * 127 + 128).astype(np.uint8)
    demo_array = np.stack([demo_array] * 3, axis=-1)
    
    piece_size = size // grid_size
    pieces = []
    for i in range(grid_size):
        for j in range(grid_size):
            top = i * piece_size
            left = j * piece_size
            piece = demo_array[top:top+piece_size, left:left+piece_size, :]
            pieces.append(piece)
    
    return pieces, piece_size, piece_size, grid_size

def get_oriented_piece(piece, orientation):
    """Apply orientation to a piece (0=normal, 1=flipped vertically)."""
    if orientation == 0:
        return piece
    else:
        return np.flipud(piece)

def compute_edge_dissimilarity(piece1, piece2, direction):
    """Compute dissimilarity between adjacent edges using L2 norm with normalization."""
    if direction == 'right':
        edge1 = piece1[:, -1, :].flatten()
        edge2 = piece2[:, 0, :].flatten()
    elif direction == 'bottom':
        edge1 = piece1[-1, :, :].flatten()
        edge2 = piece2[0, :, :].flatten()
    else:
        return 0
    
    # Normalize and use L2 norm
    edge1 = edge1.astype(np.float32) / 255.0
    edge2 = edge2.astype(np.float32) / 255.0
    return np.sqrt(np.sum((edge1 - edge2) ** 2))

def compute_corner_consistency(state, original_pieces, grid_size):
    """Compute corner consistency between 4 adjacent pieces."""
    consistency = 0
    for i in range(grid_size - 1):
        for j in range(grid_size - 1):
            # Get 4 pieces at corner
            tl_idx = i * grid_size + j
            tr_idx = i * grid_size + j + 1
            bl_idx = (i + 1) * grid_size + j
            br_idx = (i + 1) * grid_size + j + 1
            
            tl_piece = get_oriented_piece(original_pieces[state[tl_idx][0]], state[tl_idx][1])
            tr_piece = get_oriented_piece(original_pieces[state[tr_idx][0]], state[tr_idx][1])
            bl_piece = get_oriented_piece(original_pieces[state[bl_idx][0]], state[bl_idx][1])
            br_piece = get_oriented_piece(original_pieces[state[br_idx][0]], state[br_idx][1])
            
            # Check all 4 corner meeting points
            corners = [
                (tl_piece[-1, -1, :], tr_piece[-1, 0, :], bl_piece[0, -1, :], br_piece[0, 0, :]),  # Center point
            ]
            
            for corner_group in corners:
                corner_array = np.array(corner_group, dtype=np.float32) / 255.0
                variance = np.var(corner_array, axis=0).sum()
                consistency += variance
    
    return consistency

def cost_function(state, original_pieces, grid_size, use_corners=True):
    """Enhanced cost function with edge matching and corner consistency."""
    total_cost = 0
    idx = 0
    
    # Edge dissimilarity
    for i in range(grid_size):
        for j in range(grid_size):
            piece_idx, orient = state[idx]
            current_piece = get_oriented_piece(original_pieces[piece_idx], orient)
            
            # Right neighbor
            if j < grid_size - 1:
                right_idx = idx + 1
                right_piece_idx, right_orient = state[right_idx]
                right_piece = get_oriented_piece(original_pieces[right_piece_idx], right_orient)
                total_cost += compute_edge_dissimilarity(current_piece, right_piece, 'right')
            
            # Bottom neighbor
            if i < grid_size - 1:
                bottom_idx = idx + grid_size
                bottom_piece_idx, bottom_orient = state[bottom_idx]
                bottom_piece = get_oriented_piece(original_pieces[bottom_piece_idx], bottom_orient)
                total_cost += compute_edge_dissimilarity(current_piece, bottom_piece, 'bottom')
            
            idx += 1
    
    # Corner consistency (lower is better)
    if use_corners:
        corner_cost = compute_corner_consistency(state, original_pieces, grid_size)
        total_cost += corner_cost * 0.05  # Reduced weight for corners
    
    return total_cost

def generate_neighbor(current_state, temperature):
    """Generate neighbor states with temperature-adaptive strategies."""
    new_state = current_state.copy()
    num_pieces = len(current_state)
    grid_size = int(np.sqrt(num_pieces))
    
    # Temperature-adaptive move selection
    if temperature > 1000:  # High temp: more exploration
        if random.random() < 0.7:
            # Large moves
            idx1, idx2 = random.sample(range(num_pieces), 2)
            new_state[idx1], new_state[idx2] = new_state[idx2], new_state[idx1]
        else:
            # Orientation flips
            idx = random.randint(0, num_pieces - 1)
            piece_idx, orient = new_state[idx]
            new_state[idx] = (piece_idx, 1 - orient)
    
    else:  # Low temp: more exploitation
        rand = random.random()
        if rand < 0.3:  # Simple swap
            idx1, idx2 = random.sample(range(num_pieces), 2)
            new_state[idx1], new_state[idx2] = new_state[idx2], new_state[idx1]
        elif rand < 0.5:  # Chain swap (3 pieces)
            indices = random.sample(range(num_pieces), min(3, num_pieces))
            values = [new_state[i] for i in indices]
            rotated = values[1:] + [values[0]]
            for i, idx in enumerate(indices):
                new_state[idx] = rotated[i]
        elif rand < 0.7 and grid_size > 2:  # Block rotation
            i = random.randint(0, grid_size - 2)
            j = random.randint(0, grid_size - 2)
            tl = i * grid_size + j
            tr = i * grid_size + j + 1
            bl = (i + 1) * grid_size + j
            br = (i + 1) * grid_size + j + 1
            temp = new_state[tl]
            new_state[tl] = new_state[bl]
            new_state[bl] = new_state[br]
            new_state[br] = new_state[tr]
            new_state[tr] = temp
        elif rand < 0.9:  # Multiple swaps
            num_swaps = random.randint(2, 4)
            for _ in range(num_swaps):
                idx1, idx2 = random.sample(range(num_pieces), 2)
                new_state[idx1], new_state[idx2] = new_state[idx2], new_state[idx1]
        else:  # Flip orientation
            idx = random.randint(0, num_pieces - 1)
            piece_idx, orient = new_state[idx]
            new_state[idx] = (piece_idx, 1 - orient)
    
    return new_state

def adaptive_simulated_annealing(initial_state, original_pieces, grid_size, 
                                max_iter=20000, initial_temp=None):
    """Enhanced simulated annealing with better adaptive parameters."""
    current_state = initial_state.copy()
    current_cost = cost_function(current_state, original_pieces, grid_size)
    
    # Adaptive initial temperature
    if initial_temp is None:
        sample_costs = []
        for _ in range(100):
            sample_state = generate_neighbor(current_state, 1000)
            sample_cost = cost_function(sample_state, original_pieces, grid_size)
            sample_costs.append(abs(sample_cost - current_cost))
        initial_temp = np.percentile(sample_costs, 75)  # Use 75th percentile
    
    best_state = current_state.copy()
    best_cost = current_cost
    
    # Improved temperature schedule
    cooling_rate = 0.997  # Slower cooling
    min_temp = 0.01
    
    # Tracking
    accept_history = deque(maxlen=200)
    improvement_history = deque(maxlen=100)
    temp = initial_temp
    
    print(f"🚀 Starting SA: Initial cost = {current_cost:.2f}, Initial temp = {initial_temp:.2f}")
    
    start_time = time.time()
    
    for iteration in tqdm(range(max_iter), desc="Optimizing"):
        # Generate temperature-adaptive neighbor
        new_state = generate_neighbor(current_state, temp)
        new_cost = cost_function(new_state, original_pieces, grid_size)
        
        delta = new_cost - current_cost
        
        # Metropolis acceptance criterion
        if delta < 0 or random.random() < math.exp(-delta / max(temp, 1e-10)):
            current_state = new_state
            current_cost = new_cost
            accept_history.append(1)
            
            if current_cost < best_cost:
                best_state = current_state.copy()
                best_cost = current_cost
                improvement_history.append(best_cost)
        else:
            accept_history.append(0)
        
        # Adaptive cooling based on acceptance rate
        accept_rate = np.mean(accept_history) if accept_history else 0
        if accept_rate < 0.1 and temp > min_temp * 10:
            cooling_rate = 0.999  # Slow down if accepting too few
        else:
            cooling_rate = 0.997
        
        # Cool down
        temp = max(min_temp, temp * cooling_rate)
        
        # Progress logging
        if iteration % 1000 == 0 and iteration > 0:
            improvement = ((improvement_history[0] - best_cost) / improvement_history[0] * 100) if improvement_history else 0
            print(f"Iter {iteration}: Cost = {current_cost:.1f}, Best = {best_cost:.1f}, "
                  f"Temp = {temp:.3f}, Accept = {accept_rate:.1%}, Improv = {improvement:.1f}%")
        
        # Early stopping conditions
        if best_cost < 10.0:  # Very good solution
            break
        if len(improvement_history) > 50 and improvement_history[0] - improvement_history[-1] < 1.0:
            # Stagnation detection
            break
    
    total_time = time.time() - start_time
    print(f"✅ Optimization completed in {total_time:.1f}s")
    print(f"📊 Final cost: {best_cost:.2f} ({(current_cost - best_cost)/current_cost*100:.1f}% improvement)")
    
    return best_state

def multi_start_solve(original_pieces, grid_size, num_starts=3):
    """Multiple random starts with different strategies."""
    num_pieces = grid_size * grid_size
    best_overall_state = None
    best_overall_cost = float('inf')
    
    for start in range(num_starts):
        print(f"\n🎯 Attempt {start + 1}/{num_starts}")
        
        if start == 0:
            # Completely random
            initial_state = [(i, random.randint(0, 1)) for i in range(num_pieces)]
            random.shuffle(initial_state)
        elif start == 1:
            # Smart initialization with edge matching
            initial_state = smart_initialization(original_pieces, grid_size)
        else:
            # Mixed strategy
            initial_state = [(i, 0) for i in range(num_pieces)]
            random.shuffle(initial_state)
            # Flip some pieces randomly
            for i in random.sample(range(num_pieces), num_pieces // 4):
                initial_state[i] = (initial_state[i][0], 1)
        
        # Run simulated annealing
        solved_state = adaptive_simulated_annealing(
            initial_state, original_pieces, grid_size, max_iter=15000
        )
        
        final_cost = cost_function(solved_state, original_pieces, grid_size)
        print(f"Attempt {start + 1} final cost: {final_cost:.2f}")
        
        if final_cost < best_overall_cost:
            best_overall_state = solved_state
            best_overall_cost = final_cost
            print(f"🎉 New best solution found!")
    
    return best_overall_state

def smart_initialization(pieces, grid_size):
    """Create smarter initial state using greedy edge matching."""
    num_pieces = len(pieces)
    used = [False] * num_pieces
    state = []
    
    # Start with random piece
    first_idx = random.randint(0, num_pieces - 1)
    state.append((first_idx, 0))
    used[first_idx] = True
    
    # Greedy placement
    for pos in range(1, num_pieces):
        best_piece = None
        best_orientation = 0
        best_cost = float('inf')
        
        for piece_idx in range(num_pieces):
            if used[piece_idx]:
                continue
            
            # Try both orientations
            for orientation in [0, 1]:
                temp_state = state + [(piece_idx, orientation)]
                local_cost = 0
                row = pos // grid_size
                col = pos % grid_size
                
                # Check left neighbor
                if col > 0:
                    left_idx = state[pos - 1][0]
                    left_orient = state[pos - 1][1]
                    left_piece = get_oriented_piece(pieces[left_idx], left_orient)
                    current_piece = get_oriented_piece(pieces[piece_idx], orientation)
                    local_cost += compute_edge_dissimilarity(left_piece, current_piece, 'right')
                
                # Check top neighbor
                if row > 0:
                    top_idx = state[pos - grid_size][0]
                    top_orient = state[pos - grid_size][1]
                    top_piece = get_oriented_piece(pieces[top_idx], top_orient)
                    current_piece = get_oriented_piece(pieces[piece_idx], orientation)
                    local_cost += compute_edge_dissimilarity(top_piece, current_piece, 'bottom')
                
                if local_cost < best_cost:
                    best_cost = local_cost
                    best_piece = piece_idx
                    best_orientation = orientation
        
        state.append((best_piece, best_orientation))
        used[best_piece] = True
    
    return state

def reconstruct_image(state, original_pieces, grid_size, piece_width, piece_height):
    """Reconstruct the full image from puzzle state."""
    img_height = grid_size * piece_height
    img_width = grid_size * piece_width
    img_array = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    
    idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            piece_idx, orient = state[idx]
            piece = get_oriented_piece(original_pieces[piece_idx], orient)
            
            # Handle edge cases for non-divisible dimensions
            actual_height = min(piece_height, img_height - i * piece_height)
            actual_width = min(piece_width, img_width - j * piece_width)
            
            start_i = i * piece_height
            start_j = j * piece_width
            img_array[start_i:start_i+actual_height, start_j:start_j+actual_width] = piece[:actual_height, :actual_width]
            idx += 1
    
    return Image.fromarray(img_array)

def visualize_progress(original_pieces, states, costs, grid_size, pw, ph):
    """Visualize the progression of the solution."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flat
    
    # Show different stages
    stages = [0, len(states)//4, len(states)//2, 3*len(states)//4, -1]
    for i, stage_idx in enumerate(stages[:5]):
        if stage_idx < len(states):
            img = reconstruct_image(states[stage_idx], original_pieces, grid_size, pw, ph)
            axes[i].imshow(img)
            axes[i].set_title(f'Stage {i+1}\nCost: {costs[stage_idx]:.1f}')
            axes[i].axis('off')
    
    # Show cost progression
    axes[5].plot(costs)
    axes[5].set_xlabel('Iteration')
    axes[5].set_ylabel('Cost')
    axes[5].set_title('Cost Progression')
    axes[5].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main(mat_path, grid_size=4, var_name='Iscrambled', use_multi_start=True):
    """Main function to solve the jigsaw puzzle with enhanced features."""
    print("🎯 Enhanced Jigsaw Puzzle Solver")
    print("=" * 50)
    
    # Load and split image
    original_pieces, pw, ph, gs = load_and_split_image(mat_path, grid_size, var_name)
    
    print(f"📊 Puzzle Info: {gs}x{gs} grid, {len(original_pieces)} pieces, {pw}x{ph} pixels each")
    
    # Create scrambled state
    num_pieces = gs * gs
    scrambled_state = [(i, random.randint(0, 1)) for i in range(num_pieces)]
    random.shuffle(scrambled_state)
    
    initial_cost = cost_function(scrambled_state, original_pieces, gs)
    print(f"🔀 Initial scrambled cost: {initial_cost:.2f}")
    
    # Solve puzzle
    print("\n🔄 Starting optimization...")
    start_time = time.time()
    
    if use_multi_start and grid_size >= 4:
        solved_state = multi_start_solve(original_pieces, gs)
    else:
        solved_state = adaptive_simulated_annealing(scrambled_state, original_pieces, gs)
    
    total_time = time.time() - start_time
    
    # Calculate results
    final_cost = cost_function(solved_state, original_pieces, gs)
    improvement = (initial_cost - final_cost) / initial_cost * 100
    
    print(f"\n🎉 FINAL RESULTS:")
    print(f"   Initial cost: {initial_cost:.2f}")
    print(f"   Final cost: {final_cost:.2f}")
    print(f"   Improvement: {improvement:.1f}%")
    print(f"   Total time: {total_time:.1f} seconds")
    
    # Reconstruct and display images
    initial_img = reconstruct_image(scrambled_state, original_pieces, gs, pw, ph)
    final_img = reconstruct_image(solved_state, original_pieces, gs, pw, ph)
    
    # Display results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(initial_img)
    ax1.set_title(f'Scrambled\nCost: {initial_cost:.2f}')
    ax1.axis('off')
    
    ax2.imshow(final_img)
    ax2.set_title(f'Solved\nCost: {final_cost:.2f} ({improvement:.1f}% improvement)')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return solved_state, final_cost

if __name__ == "__main__":
    # Install required: pip install tqdm
    try:
        solved_state, cost = main('scrambled_lena.mat', grid_size=4, use_multi_start=True)
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure the file path is correct and required packages are installed:")
        print("   pip install numpy pillow matplotlib tqdm")