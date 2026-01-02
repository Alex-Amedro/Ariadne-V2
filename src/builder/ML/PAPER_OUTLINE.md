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
- Novelty Search from evolutionary algorithms
- Archive-based approach
- Distance metric: Euclidean in normalized parameter space
- k-NN novelty: mean distance to 15 nearest neighbors
- λ = 0.5 balances performance and exploration

**Implementation:**
```
L_total = -SR(agent) + 0.5 * (-diversity)
diversity = mean(pairwise_distances(batch))
```

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

**Network:**
```
Input:  z ∈ R^8 (Gaussian noise)
FC1:    8 → 64, ReLU
FC2:    64 → 64, ReLU
FC3:    64 → 4, Sigmoid + scaling
Output: (grid_size, obstacles, doors, keys)
```

**Parameter Mapping:**
```python
grid_size = sigmoid(x[0]) * 7 + 5
obstacles = sigmoid(x[1]) * 5
doors = sigmoid(x[2]) * 2
keys = sigmoid(x[3]) * 2
```

**Optimizer:**
- Adam, lr=1e-3
- Gradient clipping: max_norm=1.0
- Updates: 20 iterations per epoch

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

**Archive:**
- FIFO queue, size 100
- Stores last generated levels
- Normalized parameter vectors

**Novelty Metric:**
```python
def novelty(level):
    distances = [dist(level, archived) for archived in archive]
    k_nearest = sorted(distances)[:15]
    return mean(k_nearest)
```

**Diversity Loss:**
```python
# Batch diversity
vectors = [normalize(level) for level in batch]
distances = pairwise_euclidean(vectors)
diversity = mean(distances)

# Total loss
L_total = L_performance + λ * (-diversity)
```

**Parameters:**
- λ = 0.5 (diversity weight)
- k = 15 (nearest neighbors)
- Archive size = 100

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

**t-tests (two-tailed):**
```
Diversity vs Baseline:  t=4.603,  p<0.0001  ✓
Diversity vs Vanilla:   t=6.582,  p<0.0001  ✓
Baseline vs Vanilla:    t=-2.110, p=0.0415  ✓
```

**Effect Sizes (Cohen's d):**
```
Diversity vs Baseline:  d=1.37   (large)
Diversity vs Vanilla:   d=1.96   (very large)
Baseline vs Vanilla:    d=-0.68  (medium)
```

**Interpretation:**
- All differences statistically significant
- Diversity shows very large improvements
- Random baseline beats vanilla (medium effect)

### 4.4 Transfer Learning

**Test Environments (MiniGrid standard):**
```
Empty-5x5:        100% ✓
Empty-8x8:        0%
DoorKey-5x5:      0%
DoorKey-8x8:      0%
MultiRoom-N2-S4:  0%
FourRooms:        0%
KeyCorridorS3R3:  0%
```

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

### 5.1 Why Vanilla Fails

**Mode Collapse:**
- Generator converges to local optimum
- Parameters cluster around "sweet spot"
- Example: grid_size≈8, obstacles≈2.5, doors≈1
- Agent over-specializes to narrow distribution

**Evidence:**
- Diversity tracking shows 69.4% unique configs
- But parameter variance decreases over epochs
- Success rate plateaus at 80-86%

**Root Cause:**
- Loss function: MSE(generated, target)
- No explicit exploration incentive
- Gradient pushes toward convergence
- Passive diversity tracking ≠ active diversity optimization

**Comparison to Baseline:**
- Random sampling: natural variance
- Each epoch = independent draws
- Covers broader parameter space
- Forces agent to generalize

### 5.2 How Diversity Helps

**Novelty Search Mechanism:**
- Archive stores history
- New levels must differ from archive
- Gradient toward unexplored regions
- Maintains exploration pressure

**Behavioral Differences:**
```
Vanilla epochs 15-20:  grid_size ∈ [7.8, 8.2]
Diversity epochs 15-20: grid_size ∈ [6.5, 10.5]
```

**Benefits:**
1. Prevents convergence to local optimum
2. Agent sees diverse challenges
3. Learns robust strategies
4. 100% SR maintained (not just achieved)

**Trade-off:**
- Performance vs exploration: λ = 0.5 balances
- Too high λ: ignore agent performance
- Too low λ: converge like vanilla
- 0.5 empirically optimal

### 5.3 Transfer Learning Analysis

**Limited Generalization:**
- 100% on Empty-5x5 only
- 0% on all other environments

**Reasons:**
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
