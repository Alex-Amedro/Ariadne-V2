# Paper Outline - Co-Evolution with Diversity

## 2. PROJECT JOURNEY (2-3 pages)

### 2.1 Initial Hypothesis
- Co-evolution = automatic curriculum learning
- Generator creates progressively harder levels
- Agent improves by facing challenging environments
- Expected: co-evolution >> random baseline
- Goal: 15-20 page technical report

### 2.2 Implementation - Three Systems

**System 1: Vanilla Co-evolution**
- Generator: MLP 8→64→64→4
- Agent: PPO, 50k timesteps/epoch
- Loss: MSE on level parameters
- Result: 20% → 80% (final), 59.3% (mean)
- Duration: 20 epochs, ~4.5 hours

**System 2: Random Baseline**
- No generator training
- Uniform sampling: grid_size [5,12], obstacles [0,5], doors/keys [0,2]
- Same agent training protocol
- Result: 53.3% → 100% (final), 73% (mean)
- Duration: 20 epochs, ~3.8 hours

**System 3: Diversity Co-evolution**
- Added: Novelty Search objective
- Archive: 100 last generated levels
- Loss: L_performance + 0.5 * L_diversity
- Result: 73.3% → 100% (final), 92.3% (mean)
- Duration: 25 epochs, ~4.2 hours

### 2.3 The Surprise
- Baseline (random) outperformed vanilla co-evolution
- 73% vs 59.3% mean success rate
- p = 0.0415 (statistically significant)
- Cohen's d = -0.68 (medium effect)
- Counter-intuitive: no learning > learned curriculum

**Investigation:**
- Difficulty hypothesis: tested, rejected (0.527 vs 0.504 similar)
- Variance hypothesis: baseline has more variance (0.142 vs 0.089)
- Generator behavior: converges to local optimum
- Diversity collapse identified

### 2.4 Solution - Diversity Mechanism

**Core Idea:**
- Generator needs to be REWARDED for creating diverse levels
- Not just tracking diversity, but OPTIMIZING for it
- Add diversity as explicit objective in loss function

**Archive System:**
- Keep memory of last 100 generated levels
- Like a "history book" of what generator already tried
- FIFO queue: new levels push out old ones
- Prevents generator from repeating same patterns

**How It Works Step-by-Step:**

1. **Generate new level:** z → Generator → (grid_size=7, obstacles=3, doors=1, keys=1)

2. **Check novelty:** How different is this from archive?
   - Convert to vector: [7/12, 3/5, 1/2, 1/2] = [0.58, 0.60, 0.50, 0.50]
   - Compare to each level in archive
   - Find 15 nearest neighbors (most similar levels)
   - Calculate distances: d₁, d₂, ..., d₁₅
   - Novelty score = average(d₁, ..., d₁₅)
   - High score = very different from archive = GOOD

3. **Compute batch diversity:**
   - Generate 20 levels in batch
   - Calculate ALL pairwise distances (20×19/2 = 190 pairs)
   - Average all distances = batch diversity score
   - High score = levels different from each other = GOOD

4. **Combined loss function:**
   ```
   L_performance = -SuccessRate(agent on levels)
     → Want HIGH SR = agent struggles = good curriculum
     → Negative sign = minimize means maximize SR
   
   L_diversity = -mean_distance(batch_levels)
     → Want HIGH distance = diverse levels = good exploration
     → Negative sign = minimize means maximize diversity
   
   L_total = L_performance + 0.5 × L_diversity
     → Balance both objectives
     → 0.5 = equal weight
   ```

5. **Gradient descent:**
   - Backpropagate through L_total
   - Generator learns to create levels that:
     a) Challenge the agent (high SR loss)
     b) Are different from each other (high diversity)
     c) Are different from archive (high novelty)

**Why 0.5 weight?**
- Too high (λ=1.0): generator ignores agent performance, just makes random stuff
- Too low (λ=0.1): generator ignores diversity, converges like vanilla
- 0.5 = empirically found sweet spot

**Example Evolution:**
```
Epoch 1:  Archive empty → all levels novel
          Batch: grid_size ∈ [5,8], diverse

Epoch 10: Archive has 100 levels
          Generator learns: "avoid grid_size=7-8, already explored"
          Batch: grid_size ∈ [5,6,9,10], different regions

Epoch 20: Archive covers most space
          Generator forced to explore edges
          Batch: mix of easy (5×5) and hard (11×11)
```

**Key Difference from Vanilla:**
- Vanilla: "I see diversity is 69.4%" (just observing)
- Diversity: "Increase diversity or get penalized" (actively optimizing)

### 2.5 Validation
- Statistical tests: all p < 0.0001 (except baseline vs vanilla: p=0.0415)
- Effect sizes: Cohen's d = 1.37-1.96 (large to very large)
- Robustness: 100% SR maintained for 20 consecutive epochs
- Stability: lowest std (9.4% vs 17.6% vs 22.1%)

**Key Finding:**
Diversity co-evolution: +33% vs vanilla, +19.3% vs baseline

---

## 3. TECHNICAL APPROACH (3 pages)

### 3.1 Environment Design

**Base: MiniGrid-Empty**
- Grid-based navigation
- Goal: reach green square
- Obstacles, doors, keys as challenges

**Parametric Extension:**
```python
grid_size: [5, 12]        # int
num_obstacles: [0, 5]      # int
num_doors: [0, 2]          # int
num_keys: [0, 2]           # int
```

**Reward Function:**
```python
r = +1.0 (goal reached)
  - 0.001 * steps
  + 0.01 * (new cells visited)
```

**Constraints:**
- num_keys ≥ num_doors (solvability)
- BFS verification: level must be solvable
- Regenerate if unsolvable

### 3.2 Generator Architecture

**Why Neural Network?**
- Want smooth mapping from random noise → level parameters
- Gradient descent can optimize this mapping
- Learned distribution adapts during co-evolution

**Network Structure:**
```
Input:  z ∈ R^8 (random Gaussian noise, sampled fresh each time)
        Example: z = [-0.5, 1.2, 0.3, -0.8, 0.1, -1.1, 0.9, 0.4]

Layer 1: z → FC1(8→64) → ReLU → h1
        Expands 8D noise to 64D hidden representation
        ReLU removes negatives: max(0, x)

Layer 2: h1 → FC2(64→64) → ReLU → h2
        Processes in 64D space (learning complexity)

Layer 3: h2 → FC3(64→4) → raw_output
        Compresses to 4 values (our 4 parameters)
        raw_output might be anything: [-5.2, 2.1, -0.3, 1.8]
```

**Parameter Mapping (Critical Part):**
```python
# Need to convert raw network output to valid ranges

raw = network_forward(z)  # Might be any real numbers

# Sigmoid squashes to [0,1]
grid_size = sigmoid(raw[0]) * 7 + 5
  → sigmoid(-5.2) = 0.005 → 0.005*7+5 = 5.04 ✓
  → sigmoid(2.1) = 0.891 → 0.891*7+5 = 11.24 ✓
  → Always in [5, 12] range

obstacles = sigmoid(raw[1]) * 5
  → sigmoid(2.1) = 0.891 → 0.891*5 = 4.45 ✓
  → Always in [0, 5] range

doors = sigmoid(raw[2]) * 2
  → Always in [0, 2] range

keys = sigmoid(raw[3]) * 2
  → Always in [0, 2] range

# Final step: round to integers
grid_size = int(grid_size)  # 11.24 → 11
obstacles = int(obstacles)  # 4.45 → 4
doors = int(doors)
keys = int(keys)
```

**Why This Design?**
- Sigmoid guarantees valid ranges (can't generate grid_size=50)
- Smooth function = gradient descent works
- Random z input = stochastic generator
- Different z → different levels

**Training the Generator:**
```python
# Each training iteration:

1. Sample random z ~ N(0,1)  # Fresh noise each time
2. Generate level: params = generator(z)
3. Evaluate: SR = test_agent_on_level(params)
4. Compute loss: L = -SR + 0.5*(-diversity)
5. Backpropagate: ∂L/∂weights
6. Update: weights -= lr * gradient

# After update, generator.forward() changes
# Same z will now produce DIFFERENT level
# Hopefully one that increases SR or diversity
```

**Optimizer Details:**
- Adam: adaptive learning rate (good for non-convex)
- lr=1e-3: not too fast (stable), not too slow
- Gradient clipping max_norm=1.0:
  - If ||gradient|| > 1.0, scale it down
  - Prevents exploding gradients
  - Stabilizes training
- 20 iterations/epoch: multiple small steps better than 1 big step

### 3.3 Co-evolution Algorithm

**Phase 0: Initial Training**
- Generate 20 random levels
- Train agent from scratch
- 100k timesteps
- Establishes baseline competence

**Per Epoch:**
1. Generate 20 levels (generator)
2. Train agent 50-60k timesteps (PPO)
3. Evaluate agent on 5 random levels
4. Update generator based on agent performance
5. Log metrics (SR, diversity, losses)

**Vanilla Loss:**
```python
for level in batch:
    SR = evaluate_agent(level)
    if SR < 0.3:
        target[i] += noise * 0.2
loss = MSE(generated, target)
```

### 3.4 Diversity Mechanism (Novelty Search)

**Problem Being Solved:**
- Vanilla generator converges to "sweet spot" (e.g., always grid_size=8)
- Need to FORCE exploration of entire parameter space

**Archive System (The "Memory"):**
```python
self.archive = []  # Starts empty
self.max_size = 100

def add_to_archive(level):
    archive.append(level)
    if len(archive) > 100:
        archive.pop(0)  # Remove oldest (FIFO = First In First Out)

# After 100 epochs, archive contains last 100 generated levels
# Represents "what we've already tried"
```

**Distance Function (How "Different" Are Two Levels?):**
```python
def distance(level1, level2):
    # Normalize to [0,1] so all dimensions equal weight
    v1 = [level1['grid_size']/12,
          level1['obstacles']/5,
          level1['doors']/2,
          level1['keys']/2]
    
    v2 = [level2['grid_size']/12,
          level2['obstacles']/5,
          level2['doors']/2,
          level2['keys']/2]
    
    # Euclidean distance
    return sqrt((v1[0]-v2[0])² + (v1[1]-v2[1])² + 
                (v1[2]-v2[2])² + (v1[3]-v2[3])²)

# Example:
# level1 = (5, 0, 0, 0) → v1 = [0.42, 0.00, 0.00, 0.00]
# level2 = (12, 5, 2, 2) → v2 = [1.00, 1.00, 1.00, 1.00]
# distance = sqrt(0.58² + 1² + 1² + 1²) = 1.75 (very different!)

# level1 = (7, 2, 1, 1) → v1 = [0.58, 0.40, 0.50, 0.50]
# level2 = (8, 3, 1, 1) → v2 = [0.67, 0.60, 0.50, 0.50]
# distance = sqrt(0.09² + 0.20² + 0² + 0²) = 0.22 (similar)
```

**Novelty Metric (Is This Level "New"?):**
```python
def compute_novelty(new_level, archive):
    if len(archive) == 0:
        return 1.0  # First level = maximally novel
    
    # Calculate distance to every archived level
    distances = []
    for archived_level in archive:
        d = distance(new_level, archived_level)
        distances.append(d)
    
    # Sort to find nearest neighbors
    distances.sort()
    
    # Take 15 nearest (or fewer if archive < 15)
    k = min(15, len(distances))
    k_nearest = distances[:k]
    
    # Average distance to nearest neighbors = novelty
    novelty_score = sum(k_nearest) / k
    
    return novelty_score

# High novelty = far from archive = UNEXPLORED region
# Low novelty = close to archive = ALREADY TRIED region
```

**Why k=15 Nearest Neighbors?**
- Don't want just 1 nearest (too noisy)
- Don't want all 100 (too averaged out)
- 15 = local neighborhood
- Standard in Novelty Search literature

**Batch Diversity (Are Levels in Batch Different From Each Other?):**
```python
def compute_batch_diversity(batch_of_20_levels):
    vectors = [normalize(level) for level in batch_of_20_levels]
    
    # Calculate ALL pairwise distances
    distances = []
    for i in range(20):
        for j in range(i+1, 20):
            d = distance(vectors[i], vectors[j])
            distances.append(d)
    
    # Total: 20*19/2 = 190 distances
    
    diversity = mean(distances)
    return diversity

# High diversity = levels spread out in parameter space
# Low diversity = levels clustered together
```

**Combined Loss (The Key Innovation):**
```python
# Per training iteration:

# 1. Generate 20 levels
batch = [generator(random_z()) for _ in range(20)]

# 2. Evaluate agent performance on 5 of them
performance_losses = []
for level in batch[:5]:
    SR = evaluate_agent(level)
    performance_losses.append(-SR)  # Want HIGH SR
L_performance = mean(performance_losses)

# 3. Calculate batch diversity
batch_diversity = compute_batch_diversity(batch)
L_diversity = -batch_diversity  # Want HIGH diversity

# 4. Combined loss
L_total = L_performance + 0.5 * L_diversity
         = -SR + 0.5 * (-diversity)

# Example numbers:
# SR = 0.8 → L_performance = -0.8
# diversity = 2.5 → L_diversity = -2.5
# L_total = -0.8 + 0.5*(-2.5) = -0.8 - 1.25 = -2.05

# 5. Backpropagate and update generator
# Gradient tells generator:
#   - Make levels harder (increase SR loss)
#   - Make levels more diverse (increase diversity)
```

**λ = 0.5 Weight Explained:**
```
λ = 0.0  → L_total = L_performance only
          → Vanilla co-evolution (mode collapse)

λ = 0.1  → L_total = L_performance + 0.1*L_diversity
          → Diversity helps a bit, but not enough

λ = 0.5  → L_total = L_performance + 0.5*L_diversity
          → BALANCED: both objectives matter equally
          → Our choice (empirically best)

λ = 1.0  → L_total = L_performance + 1.0*L_diversity
          → Diversity dominates
          → Generator might ignore agent performance
          → Creates diverse but useless levels

λ = 5.0  → L_total = L_performance + 5.0*L_diversity
          → ONLY diversity matters
          → Basically random generator
```

**Update Archive Each Epoch:**
```python
# After generating batch:
for level in batch:
    add_to_archive(level)

# Archive grows: 0 → 20 → 40 → ... → 100 → stays at 100
# Novelty calculation becomes more meaningful over time
```

---

## 4. EXPERIMENTAL RESULTS (3-4 pages)

### 4.1 Setup

**Hardware:**
- CPU: Intel i7
- RAM: 16GB
- No GPU used

**Hyperparameters:**
```
Agent (PPO):
  learning_rate: 3e-4
  n_steps: 512
  batch_size: 128
  gamma: 0.99
  gae_lambda: 0.95

Generator:
  learning_rate: 1e-3
  hidden_dim: 64
  latent_dim: 8

Training:
  batch_size: 20 levels
  agent_timesteps: 50-60k/epoch
  generator_updates: 20/epoch
  eval_episodes: 3/level
```

**Training Time:**
- Vanilla: 4h 30min (20 epochs)
- Baseline: 3h 45min (20 epochs)
- Diversity: 4h 10min (25 epochs)

### 4.2 Main Results

**Success Rate Evolution:**
```
            Initial  Final   Mean    Std     Best
Vanilla     20.0%   80.0%   59.3%   22.1%   86.7%
Baseline    53.3%   100.0%  73.0%   17.6%   100.0%
Diversity   73.3%   100.0%  92.3%   9.4%    100.0%
```

**Learning Curves:**
- Vanilla: slow start, plateaus at 80%
- Baseline: steady improvement, reaches 100% epoch 18
- Diversity: fast start (73%), 100% by epoch 6, maintains

**Diversity Metrics:**
- Vanilla: 69.4% unique configs (but similar parameters)
- Diversity: batch distance 2.5 ± 0.5 (maintained throughout)
- Diversity: novelty score 0.8 ± 0.3

### 4.3 Statistical Tests

**Purpose:**
- Verify if differences between methods are real or just random chance
- Quantify effect sizes (how big are the differences?)

**t-tests (two-tailed):**
```
Diversity vs Baseline:  t=4.603,  p<0.0001  ✓
Diversity vs Vanilla:   t=6.582,  p<0.0001  ✓
Baseline vs Vanilla:    t=-2.110, p=0.0415  ✓
```

**What This Means:**
- **p < 0.0001:** Less than 0.01% chance results are random
- **p = 0.0415:** 4.15% chance results are random (still significant at α=0.05)
- All three comparisons: statistically significant differences exist

**Effect Sizes (Cohen's d):**
```
Diversity vs Baseline:  d=1.37   (large effect)
Diversity vs Vanilla:   d=1.96   (very large effect)
Baseline vs Vanilla:    d=-0.68  (medium effect)
```

**What This Means:**
- **d > 1.2:** Very large practical difference (not just statistical)
- **d = 1.37:** Diversity beats baseline by ~1.37 standard deviations
- **d = 1.96:** Diversity beats vanilla by ~2 standard deviations (huge!)
- **d = -0.68:** Vanilla worse than baseline by ~0.7 std (moderate)

**Interpretation:**
1. **Diversity >> Baseline:** Highly significant (p<0.0001) + very large effect (d=1.37)
   - Not just statistically different, practically much better
   
2. **Diversity >> Vanilla:** Extremely significant (p<0.0001) + very large effect (d=1.96)
   - Biggest improvement, strongest evidence
   
3. **Baseline > Vanilla:** Significant (p=0.0415) + medium effect (d=-0.68)
   - Surprising: random sampling beats learned co-evolution
   - Medium effect = meaningful practical difference

**Visual Representation:**
- Figures show clear separation in box plots
- Learning curves diverge significantly
- Confidence intervals don't overlap
    queue = [(start, 0)]  # (position, keys_collected)
    visited = set()
    
    while queue:
        pos, keys = queue.pop(0)
        
        if pos == goal:
            return True  # Found path!
        
        if (pos, keys) in visited:
            continue
        visited.add((pos, keys))
        
        for neighbor in neighbors(pos):
            cell = grid[neighbor]
            
            if cell == 'wall' or cell == 'obstacle':
                continue  # Can't pass
            
            if cell == 'door':
                if keys > 0:
                    queue.append((neighbor, keys-1))  # Use key
            elif cell == 'key':
                queue.append((neighbor, keys+1))  # Collect key
            else:
                queue.append((neighbor, keys))  # Empty cell
    
    return False  # No path found
```

**Impact:**
- ~5-10% of generated levels are unsolvable
- Rejection sampling ensures only valid levels used
- Small computational overhead (~0.01s per check)
**Finding:**
- Strong specialization to parametric environment
- No generalization to different layouts
- Agent learns environment-specific strategies

### 4.5 Difficulty Analysis

**Difficulty Score:**
```python
d = 0.4 * grid_size/12 
  + 0.3 * obstacles/5
  + 0.15 * doors/2
  + 0.15 * keys/2
```

**Results (1000 samples each):**
```
           Mean    Std
Vanilla    0.527   0.089
Baseline   0.504   0.142
```

**Conclusion:**
- Similar mean difficulty (p>0.05)
- Baseline has higher variance
- Variance aids generalization

---

## 5. ANALYSIS & DISCUSSION (2 pages)

### 5.1 Why Vanilla Fails - Mode Collapse

**Observable Symptoms:**
- Success rate plateaus at 80-86%
- Parameters cluster around "sweet spot"
- Diversity tracking shows 69.4% unique configs BUT similar actual difficulty
- Variance decreases over epochs

**Example Convergence Pattern:**
```
Epoch 1-5:   grid_size ∈ [5, 11], std=2.1 (exploring)
Epoch 6-10:  grid_size ∈ [6, 10], std=1.5 (narrowing)
Epoch 11-15: grid_size ∈ [7, 9],  std=0.8 (converging)
Epoch 16-20: grid_size ∈ [7.8, 8.2], std=0.3 (collapsed!)
```

**Root Cause Analysis:**

1. **Loss Function Limitation:**
   - Vanilla uses: L = MSE(generated_params, target_params)
   - Target only changes if SR < 0.3 (add noise)
   - No explicit reward for exploration
   - Gradient naturally converges to local minimum

2. **Feedback Loop Problem:**
   ```
   Epoch N:   Generator finds grid_size=8 works well
              Agent trains, achieves 80% SR
   
   Epoch N+1: Generator gradient says "keep grid_size=8"
              Agent specializes further to grid_size=8
   
   Epoch N+2: Even harder to escape grid_size=8
              Agent over-specialized, can't handle grid_size=6
   
   Result: Stuck in local optimum
   ```

3. **Passive vs Active Diversity:**
   - Vanilla MEASURES diversity: "69.4% unique configs"
   - But doesn't OPTIMIZE for it
   - Like counting steps but not trying to walk more
   - No gradient signal to increase diversity

**Comparison to Baseline (Why Random Wins):**

| Aspect | Vanilla | Random Baseline |
|--------|---------|----------------|
| Exploration | Converges to sweet spot | Always explores full space |
| Variance | Decreases over time | Constant high variance |
| Agent adaptation | Over-specialized | Forced to generalize |
| Final SR | 59.3% mean | 73% mean |

**Key Insight:**
- Random's "stupidity" is actually an advantage
- No learning = no convergence = no collapse
- Natural variance > learned convergence (without diversity objective)

### 5.2 How Diversity Solves This

**Mechanism - Archive-Based Novelty:**

1. **Historical Memory:**
   - Archive stores last 100 generated levels
   - Represents "explored regions" of parameter space
   - Prevents generator from repeating same patterns

2. **Novelty Gradient:**
   ```
   L_total = L_performance + 0.5 * L_diversity
           = -SR(agent) + 0.5 * (-mean_distance)
   
   Generator gets TWO signals:
   - Make levels hard (maximize SR loss)
   - Make levels novel (maximize distances)
   ```

3. **Active Exploration Pressure:**
   - If generator tries grid_size=8 again
   - Archive already has many grid_size=8 levels
   - Novelty score LOW
   - Diversity loss HIGH
   - Gradient pushes AWAY from 8
   - Generator explores 6, 10, etc.

**Behavioral Evidence:**
```
Diversity System - Parameter Evolution:

Epoch 1-5:   grid_size ∈ [5, 11], std=2.3 (initial exploration)
Epoch 6-10:  grid_size ∈ [5, 12], std=2.5 (maintained diversity!)
Epoch 11-15: grid_size ∈ [5, 11], std=2.4 (still exploring)
Epoch 16-20: grid_size ∈ [6, 10], std=2.2 (continues exploration)
Epoch 21-25: grid_size ∈ [5, 12], std=2.6 (no collapse!)

Compare to Vanilla: std drops from 2.1 → 0.3
Diversity: std stays around 2.3-2.6 throughout
```

**Concrete Example - Batch Comparison:**

Vanilla Epoch 15 batch:
```
Level 1: (8, 3, 1, 1)
Level 2: (8, 2, 1, 1)  ← Very similar to 1
Level 3: (7, 3, 1, 1)  ← Very similar to 1
Level 4: (8, 3, 1, 1)  ← Duplicate!
...
Diversity = 0.25 (low)
```

Diversity Epoch 15 batch:
```
Level 1: (5, 0, 0, 0)   ← Easy
Level 2: (11, 5, 2, 2)  ← Very hard
Level 3: (7, 2, 0, 0)   ← Medium-easy
Level 4: (9, 4, 1, 1)   ← Medium-hard
...
Diversity = 2.8 (high!)
```

**Quantitative Benefits:**

1. **No Plateau:** 100% SR achieved and MAINTAINED (not just hit once)
2. **Lower Variance:** 9.4% std vs 22.1% vanilla (more stable learning)
3. **Broader Coverage:** Agent sees full parameter space
4. **Robust Generalization:** Handles any grid_size, not just 8

**Why λ=0.5 is Critical:**
- λ=0: Pure vanilla (collapse)
- λ=0.5: Balanced (our choice)
- λ=1.0: Too much diversity (ignores performance)

### 5.3 Transfer Learning - Specialization Trade-off

**Experimental Results:**
```
Test Environment          Success Rate
MiniGrid-Empty-5x5        100% ✓
MiniGrid-Empty-8x8        0%
MiniGrid-DoorKey-5x5      0%
All other variants        0%
```

**Why Zero Transfer?**

1. **Layout Differences:**
   - Parametric env: obstacles placed randomly
   - Standard envs: fixed layouts
   - Agent learns "find goal in random grid"
   - Can't apply to "navigate fixed maze"

2. **State Space Mismatch:**
   - Trained on varying grid sizes (5-12)
   - But each episode has ONE fixed size
   - Standard envs: fixed 5×5 or 8×8
   - Different observation distributions

3. **Strategy Specialization:**
   - Agent learns: "wall-following + random search"
   - Works for random obstacles
   - Fails for structured layouts (rooms, corridors)

**Why Empty-5x5 Works:**
- Only standard env similar to parametric
- No obstacles = like our grid_size=5, obstacles=0
- Closest match in distribution

**Broader Implications:**
- Co-evolution optimizes for TASK FAMILY
- Not general RL agent
- Trade-off: specialization vs generalization
- Our goal: master parametric family ✓
- Not goal: solve all MiniGrid variants

**Comparison to Literature:**
- PAIRED: also poor transfer (different env family)
- PLR: better transfer (trains on diverse fixed envs)
- Our approach: different use case

### 5.4 Limitations & Future Work

**Current Limitations:**

1. **Computational Cost:**
   - 4+ hours training (diversity)
   - Evaluation expensive (3 episodes × 20 levels/epoch)
   - Can't scale to very large batches

2. **Single Environment Family:**
   - Only tested on MiniGrid parametric
   - Unknown if approach generalizes to other PCG domains
   - Need validation on different games

3. **Hyperparameter Sensitivity:**
   - λ=0.5 found empirically
   - Archive size=100 not thoroughly explored
   - k=15 neighbors: standard but not tuned

4. **No Multi-Task Learning:**
   - Trains on one task family
   - Could improve with mixed objectives
   - Future: combine parametric + standard envs

**Future Directions:**

1. **MAP-Elites Integration (2-3 weeks):**
   - Replace archive with quality-diversity grid
   - 2D bins: (grid_size, num_obstacles)
   - Better coverage guarantee

2. **PAIRED Comparison (3-4 weeks):**
   - Implement regret-based antagonist
   - Direct benchmark comparison
   - Validate competitive performance

3. **Multi-Environment Training (2-3 weeks):**
   - Train on parametric + standard MiniGrid mix
   - Test generalization improvement
   - Balance specialization vs transfer

4. **Adaptive λ Schedule (1 week):**
   - Start high (explore), decrease (exploit)
   - Could combine benefits of both phases
   - Easy to test
weights drawn from N(0,1)
- Forward pass: values explode or vanish
- Example: raw output = [50.2, -30.1, 100.5, -15.3]
- After sigmoid: all ≈1.0 or ≈0.0 (saturated)
- Generator stuck in corners of parameter space

**Why Random Init Fails:**
```python
# Bad initialization:
W1 ~ N(0, 1)  # 8×64 weights from standard normal

# Forward pass:
h1 = ReLU(W1 @ z)  # h1 can be HUGE or tiny
h2 = ReLU(W2 @ h1)  # Compounds the problem
out = W3 @ h2  # Completely unpredictable

# Gradient:
grad = ∂L/∂out @ ∂out/∂W3 @ ∂W3/∂h2 @ ...
# If values too large → gradient explodes (NaN)
# If values too small → gradient vanishes (no learning)
```

**Solution: Xavier (Glorot) Initialization**
```python
def init_weights(m):
    if isinstance(m, nn.Linear):
        # Xavier uniform initialization
        nn.init.xavier_uniform_(m.weight)
        # Formula: W ~ U[-√(6/(n_in + n_out)), √(6/(n_in + n_out))]
        
        # Example for 8→64 layer:
        # limit = √(6/(8+64)) = √(6/72) = √0.083 = 0.29
        # W ~ U[-0.29, 0.29]
        
        # Bias always zero
        nn.init.zeros_(m.bias)

# Why this works:
# - Keeps variance of activations constant across layers
# - Prevenidea: generator should make levels where SR ≈ 0.5
- If SR = 0.5, agent succeeds half the time = "just right"
- But: where to set threshold?

**Why Threshold Matters:**
```
Threshold too HIGH (e.g., 0.8):
- SR < 0.8 → add noise to make harder
- But agent improving → SR increasing naturally
- Generator keeps adding difficulty
- Agent can't keep up → SR crashes

Threshold too LOW (e.g., 0.1):
- SR < 0.1 → add noise
- But SR already very low = levels TOO HARD
- Adding more difficulty = impossible levels
- Agent learns nothing

Threshold just right (0.3):
- SR < 0.3 → levels too hard, make easier
- SR ∈ [0.3, 0.7] → good curriculum range
- SR > 0.7 → levels too easy, keep as is
- Balances challenge and learning
```

**Solution Code:**
```python
# In vanilla co-evolution generator training:

for level in evaluated_batch:
    SR = level['success_rate']
    
    if SR < 0.3:
        # Level too hard, agent failing
        # Add random noise to parameters
        target[i] = current_params[i] + torch.randn_like(current_params[i]) * 0.2
        # This pushes generator to CHANGE (hopefully make easier)
    
    elif SR > 0.7:
        # Level too easy, agent succeeding
        # Keep as is (no change needed)
        target[i] = current_params[i]
    
    else:
        # SR ∈ [0.3, 0.7] = sweet spot
        # Keep as is
        target[i] = current_params[i]

# Then minimize MSE(generated, target)
loss = MSE(generated_params, target)
```

**Why 0.3 Specifically?**
- Empirically tested: 0.2, 0.3, 0.4, 0.5
- 0.3 gave best results:
  * 0.2: too harsh, generator changes too much
  * 0.3: good balance ✓
  * 0.4: too lenient, levels stay easy
  * 0.5: agent plateaus early

**Alternative Approach (Not Used):**
```python
# Could use smooth target:
target_SR = 0.5
difficulty_adjustment = (current_SR - target_SR)
# But: less interpretable, needs hyperparameter tuning
```

**Impact:**
- Without threshold: vanilla SR peaked at 73%, crashed to 60%
- With 0.3 threshold: vanilla SR stable at 80%
- Still not as good as diversity (92%), but more stable
Epoch 1: raw outputs = [45.2, -38.1, 92.3, -11.5]
         sigmoid = [1.0, 0.0, 1.0, 0.0]
         Always generates: (12, 0, 2, 0) ← stuck!

After Xavier:
Epoch 1: raw outputs = [0.5, -0.3, 1.2, 0.8]
         sigmoid = [0.62, 0.43, 0.77, 0.69]
         Generates: (9, 2, 2, 1) ← reasonable!

Epoch 10: raw outputs = [1.8, 0.2, -0.5, 1.1]
          sigmoid = [0.86, 0.55, 0.38, 0.75]
          Generates: (11, 3, 1, 2) ← diverse!
```

**Alternative: He Initialization**
- We tried: nn.init.kaiming_uniform_
- Designed for ReLU activations
- Similar results to Xavier
- Stuck with Xavier (more common in literature)
1. Parametric vs fixed layouts
2. Different action distributions
3. Reward function specific
4. No multi-task training

**Implications:**
- Co-evolution optimizes for task family
- Not general RL agent
- Trade-off: specialization vs generalization

### 5.4 Limitations

**Computational:**
- 4+ hours training time
- Single-task optimization
- No parallel evaluation

**Methodological:**
- One environment family tested
- Manual hyperparameter tuning
- Limited architecture search

**Generalization:**
- Transfer learning fails
- Environment-specific strategies
- No cross-task learning

---

## 6. CHALLENGES & SOLUTIONS (1-2 pages)

### 6.1 Technical Challenges

**Problem 1: Unsolvable Levels**
- Generator creates impossible levels
- Doors without keys
- Obstacles block all paths

**Solution:**
```python
def is_solvable(level):
    if level['num_keys'] < level['num_doors']:
        return False
    return BFS_check(level)
```
- Constraint: keys ≥ doors
- BFS verification before training
- Regenerate if fails

**Problem 2: Generator Initialization**
- Random init: unstable outputs
- Outputs outside valid range
- Training diverges early

**Solution:**
```python
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
```
- Xavier initialization
- Stabilizes early training
- Outputs in valid range from start

**Problem 3: Evaluation Variance**
- 1 episode per level: high variance
- Success rate fluctuates ±30%
- Unstable generator gradients

**Solution:**
- 3 episodes per level evaluation
- Average success rate
- More stable gradients
- Trade-off: 3x evaluation time

### 6.2 Experimental Challenges

**Problem 4: Training Instability**
- Vanilla co-evolution: SR drops epoch 16→17 (86.7% → 80%)
- Generator updates too aggressive
- Agent forgets previous strategies

**Solution:**
- Gradient clipping: max_norm=1.0
- Reduced generator learning rate: 1e-3 → 5e-4 (tested)
- 20 small updates vs 5 large updates

**Problem 5: Target SR Tuning**
- Initial target: SR > 0.5
- Too easy: generator stagnates
- Too hard: agent never succeeds

**Solution:**
```python
if SR < 0.3:
    target += noise * 0.2  # make harder
elif SR > 0.7:
    pass  # keep as is
```
- Adaptive target based on current SR
- Empirically found 0.3 threshold

**Problem 6: Diversity Measurement**
- Initial: count unique configs
- Not enough: (5,2,1,1) vs (5,3,1,1) counted as different
- But very similar in difficulty

**Solution:**
- Continuous distance metric
- Normalized Euclidean distance
- Captures similarity better than discrete counting

### 6.3 Debugging Process

**Generator Mode Collapse Detection:**
- Printed level parameters every epoch
- Noticed clustering around grid_size=8
- Std deviation decreased over time
- Led to diversity solution

**Baseline Comparison:**
- Initially expected vanilla >> baseline
- Result shocked: baseline better
- Systematic investigation:
  1. Check difficulty (rejected)
  2. Check variance (confirmed)
  3. Identify diversity collapse

---

## 7. LESSONS LEARNED & FUTURE WORK (1 page)

### 7.1 What Worked

**Diversity Mechanism:**
- Novelty Search solves mode collapse
- Simple to implement
- Large effect (d=1.96)
- Archive approach effective

**Parametric Environment:**
- Clean experimental setup
- Easy to control difficulty
- Fast iteration cycles
- Good for ablation studies

**Statistical Validation:**
- t-tests give confidence
- Effect sizes show practical significance
- Rigorous comparison methodology

### 7.2 What Didn't Work

**Passive Diversity Tracking:**
- Measuring ≠ optimizing
- Need explicit gradient
- Logging not enough

**Single Environment Family:**
- Limited generalization
- Transfer learning fails
- Need multi-task approach

**Generator Complexity:**
- MLP sufficient
- Tried deeper (8→128→128→64→4): no improvement
- Simpler = better here

### 7.3 If I Had More Time...

**MAP-Elites Implementation:**
- Quality-Diversity algorithm
- 2D archive: (grid_size, obstacles)
- Better coverage of parameter space
- Estimated: 2-3 weeks

**PAIRED Comparison:**
- State-of-art co-evolution method
- Protagonist-Antagonist setup
- Direct benchmark comparison
- Estimated: 3-4 weeks

**Multi-Task Training:**
- Train on multiple MiniGrid variants
- Improve generalization
- Test transfer learning again
- Estimated: 2-3 weeks

**Curriculum Pacing:**
- Adaptive difficulty adjustment
- Based on current agent SR
- Smoother learning curves
- Estimated: 1 week

### 7.4 Broader Implications

**For PCG:**
- Diversity objective is crucial
- Random sampling = strong baseline
- Need explicit exploration mechanisms

**For RL:**
- Curriculum learning not automatic
- Teacher needs diversity incentive
- Co-evolution can fail without care

**For Research:**
- Always compare to random baseline
- Statistical tests essential
- Negative results informative

**Key Takeaway:**
Co-evolution is powerful but requires careful design. Diversity mechanisms prevent mode collapse and enable robust learning.

---

## QUICK STATS FOR ABSTRACT/INTRO

- 3 systems compared
- 92.3% mean SR achieved
- +33% improvement over vanilla
- +19.3% improvement over baseline
- p < 0.0001 (highly significant)
- Cohen's d = 1.96 (very large effect)
- 100% SR maintained 20 epochs
- 4.2 hours training time
