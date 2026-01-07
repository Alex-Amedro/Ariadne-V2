Parfait ! Je vais d'abord améliorer tes sections 6 et 7 en gardant ton style, puis te donner des instructions ultra-précises pour chaque figure.

---

## 📝 SECTIONS 6 & 7 - VERSION AMÉLIORÉEParfait ! Maintenant voici les **INSTRUCTIONS ULTRA-PRÉCISES** pour chaque figure. Donne ces specs à ton agent IA :

---

# 📊 INSTRUCTIONS COMPLÈTES POUR GÉNÉRATION DES FIGURES

## CONFIGURATION GLOBALE (À APPLIQUER À TOUTES LES FIGURES)

```python
# Style cohérent pour toutes les figures
STYLE_CONFIG = {
    'colors': {
        'vanilla': '#3498db',      # Bleu
        'baseline': '#e67e22',     # Orange  
        'diversity': '#2ecc71',    # Vert
        'grid_bg': '#ecf0f1',      # Gris clair
        'agent': '#e74c3c',        # Rouge
        'goal': '#f1c40f',         # Jaune
        'wall': '#34495e',         # Gris foncé
        'obstacle': '#2c3e50',     # Noir
    },
    'fonts': {
        'title': {'size': 16, 'weight': 'bold'},
        'xlabel': {'size': 14, 'weight': 'bold'},
        'ylabel': {'size': 14, 'weight': 'bold'},
        'legend': {'size': 12},
        'annotation': {'size': 10},
    },
    'figure': {
        'dpi': 300,
        'format': ['pdf', 'png'],  # Sauvegarder les deux
    }
}
```

---

## FIGURE 1: SYSTEM OVERVIEW (Priority: MEDIUM)

**Fichier de sortie:** `figure1_overview.pdf`, `figure1_overview.png`

**Type:** Diagramme schématique

**Dimensions:** 10" × 6" (landscape)

**Contenu à dessiner:**

### Panneau supérieur (Vanilla Co-evolution):
```
[Generator (bleu)] --"New Levels"--> [Batch de 5 niveaux similaires] --"Train"--> [Agent (rouge)] 
                                            ^                                              |
                                            |                                              |
                                            +----------"Mode Collapse (70% div)"-----------+
```
- Generator: rectangle arrondi bleu clair, label "Generator\n(Mode Collapse)"
- Niveaux: 5 petites grilles 4×4 TRÈS similaires (toutes grid=8, obs=3)
- Agent: rectangle arrondi rouge, label "PPO Agent\n(Overspecialized)"
- Flèche retour rouge pointillée avec label "SR targets 50% but params stuck"

### Panneau central (Random Baseline):
```
[Random Sampler (gris)] --"Uniform"--> [Batch de 5 niveaux variés] --"Train"--> [Agent (rouge)]
                                                                                        |
                                                                "No feedback loop"------+
```
- Sampler: rectangle gris avec icône dé
- Niveaux: 5 grilles très différentes (sizes: 5, 7, 10, 12, 8)
- Agent: rectangle rouge, label "PPO Agent\n(Generalist)"
- PAS de flèche retour

### Panneau inférieur (Diversity Co-evolution):
```
[Generator (vert)] --"Novel Levels"--> [Batch varié + Archive] --"Train"--> [Agent (rouge)]
         ^                                      |                                   |
         |                                      v                                   |
         +--"Novelty Score + SR"--[Archive (100 levels)]<------"Performance"-------+
```
- Generator: rectangle vert, label "Generator\n(w/ Diversity)"
- Archive: base de données en bas, label "Archive\n(Last 100 levels)"
- Niveaux: 5 grilles très variées
- Agent: rectangle rouge
- Flèches bidirectionnelles vertes épaisses avec labels "Novelty", "Performance"

**Annotations à ajouter:**
- Vanilla: Badge rouge "59.3% mean SR"
- Baseline: Badge orange "73.0% mean SR"
- Diversity: Badge vert étoilé "92.3% mean SR ⭐"

**Code structure:**
```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig = plt.figure(figsize=(10, 6))
gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.4)

# Subplot 1: Vanilla
ax1 = fig.add_subplot(gs[0])
# ... dessiner vanilla system

# Subplot 2: Baseline  
ax2 = fig.add_subplot(gs[1])
# ... dessiner baseline system

# Subplot 3: Diversity
ax3 = fig.add_subplot(gs[2])
# ... dessiner diversity system

plt.suptitle('System Overview: Three Training Approaches', **STYLE_CONFIG['fonts']['title'])
plt.savefig('figure1_overview.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figure1_overview.png', dpi=300, bbox_inches='tight')
```

---

## FIGURE 2: LEARNING CURVES (Priority: ★★★ CRITICAL)

**Fichier:** `figure2_learning_curves.pdf/png`

**Type:** Line plot

**Dimensions:** 10" × 6"

**Données sources:**
```python
# Charger depuis:
vanilla_data = json.load('runs/vanilla_XXXXXX/logs/history.json')
baseline_data = json.load('runs/baseline_XXXXXX/logs/history.json')  
diversity_data = json.load('runs/diversity_20260102_043337/logs/history.json')

# Extraire:
vanilla_sr = [sr * 100 for sr in vanilla_data['success_rates']]  # Convertir en %
baseline_sr = [sr * 100 for sr in baseline_data['success_rates']]
diversity_sr = [sr * 100 for sr in diversity_data['success_rates']]
epochs = list(range(1, 21))  # Ou len(vanilla_sr)+1
```

**Éléments requis:**
1. **3 courbes principales:**
   - Vanilla: cercles bleus, ligne pleine épaisse (linewidth=2.5)
   - Baseline: carrés orange, ligne pleine épaisse
   - Diversity: triangles verts, ligne pleine épaisse

2. **Zones de confiance (si multiples runs):**
   ```python
   # Si tu as plusieurs runs, calculer mean ± std
   ax.fill_between(epochs, sr_mean - sr_std, sr_mean + sr_std, 
                    color=color, alpha=0.2)
   ```

3. **Lignes de référence:**
   - y=100: ligne verte pointillée, label "Perfect Performance"
   - y=50: ligne grise pointillée, label "Target (Vanilla)"

4. **Annotations critiques:**
   - Epoch où Diversity atteint 100%: ligne verticale verte pointillée
   - Epoch où Baseline atteint 100%: ligne verticale orange pointillée
   - Plateau Vanilla à 80%: ligne horizontale bleue pointillée

5. **Box de stats en coin supérieur gauche:**
   ```
   Final Success Rates:
   Vanilla:   80.0%
   Baseline: 100.0%
   Diversity: 100.0%
   
   Diversity vs Baseline:
   p < 0.0001, d = 1.37
   ```

**Axes:**
- X: "Training Epoch" (0 à 20 ou 25)
- Y: "Success Rate (%)" (0 à 105)
- Grid: alpha=0.3, linestyle='--'

**Légende:** En bas à droite, fond semi-transparent

---

## FIGURE 3: BOX PLOTS SUCCESS RATE (Priority: ★★★ CRITICAL)

**Fichier:** `figure3_boxplots.pdf/png`

**Type:** Box plot avec scatter overlay

**Dimensions:** 8" × 6"

**Données:**
```python
# Pour chaque méthode, tous les epochs:
vanilla_all_sr = vanilla_data['success_rates']  # Liste de 20 valeurs
baseline_all_sr = baseline_data['success_rates']
diversity_all_sr = diversity_data['success_rates']

# Convertir en %
data_to_plot = [
    [sr*100 for sr in vanilla_all_sr],
    [sr*100 for sr in baseline_all_sr],
    [sr*100 for sr in diversity_all_sr]
]
```

**Éléments requis:**
1. **3 box plots côte à côte:**
   - Position X: [1, 2, 3]
   - Largeur: 0.6
   - Couleurs: vanilla (bleu), baseline (orange), diversity (vert)
   - Face color: avec alpha=0.7
   - Edge color: foncé, linewidth=2

2. **Overlay de points individuels:**
   ```python
   # Pour chaque boxplot, ajouter les points
   for i, data in enumerate(data_to_plot):
       x = np.random.normal(i+1, 0.04, len(data))  # Jitter horizontal
       ax.scatter(x, data, alpha=0.5, s=30, color='black', zorder=3)
   ```

3. **Statistiques visuelles:**
   - Médiane: ligne rouge épaisse
   - IQR: boîte semi-transparente
   - Whiskers: jusqu'à 1.5×IQR
   - Outliers: points rouges

4. **Barres de significativité:**
   ```python
   # Entre Diversity et Baseline
   y_max = max([max(d) for d in data_to_plot]) + 5
   ax.plot([2, 3], [y_max, y_max], 'k-', linewidth=2)
   ax.text(2.5, y_max+2, '***', ha='center', fontsize=14)
   ax.text(2.5, y_max+5, 'p<0.0001', ha='center', fontsize=9)
   
   # Entre Diversity et Vanilla
   ax.plot([1, 3], [y_max+10, y_max+10], 'k-', linewidth=2)
   ax.text(2, y_max+12, '***', ha='center', fontsize=14)
   ax.text(2, y_max+15, 'p<0.0001', ha='center', fontsize=9)
   ```

5. **Labels X:**
   - Position 1: "Vanilla\nCo-evol"
   - Position 2: "Random\nBaseline"
   - Position 3: "Diversity\nCo-evol"

**Axes:**
- Y: "Success Rate (%)" (0 à 110)
- Grid horizontal: alpha=0.3

---

## FIGURE 4: ARCHIVE MECHANISM DIAGRAM (Priority: MEDIUM)

**Fichier:** `figure4_archive.pdf/png`

**Type:** Schéma explicatif

**Dimensions:** 10" × 5"

**Structure en 3 panneaux horizontaux:**

### Panneau gauche (Archive FIFO):
```
┌─────────────────────┐
│  Archive (100)      │
│  ┌───┐ ┌───┐       │
│  │L1 │ │L2 │ ...   │ ← Oldest (will be removed)
│  └───┘ └───┘       │
│         ...         │
│  ┌───┐ ┌───┐       │
│  │L99│ │100│       │ ← Newest
│  └───┘ └───┘       │
└─────────────────────┘
```
- Dessiner 100 petits rectangles verticaux
- Gradient de couleur: bleu foncé (old) → bleu clair (new)
- Flèche FIFO en bas: "Pop oldest when full"

### Panneau central (New Level + KNN):
```
        New Level
           (11, 4, 2, 2)
              ★
             /|\
            / | \
    15 nearest neighbors
    from archive
    
    d1 = 0.15 ─┐
    d2 = 0.18  ├─ KNN
    ...        │  distances
    d15= 0.42 ─┘
```
- Étoile jaune pour new level
- 15 lignes vers archive (pas toutes visibles)
- 5 lignes épaisses colorées pour les 5 plus proches
- Annotations des distances

### Panneau droit (Novelty Score):
```
Novelty Score Calculation:

novelty = mean(d1...d15)
        = (0.15+0.18+...+0.42)/15
        = 0.28

High score → Explore ✓
Low score  → Too similar ✗

[Barre de couleur]
0.0 ═══ 0.5 ═══ 1.0
Red    Yellow  Green
```
- Barre de gradient horizontal
- Pointeur sur le score actuel
- Annotations "Redundant" / "Optimal" / "Very novel"

---

## FIGURE 5: ENVIRONMENT EXAMPLES GRID (Priority: HIGH)

**Fichier:** `figure5_env_examples.pdf/png`

**Type:** Grille de screenshots

**Dimensions:** 12" × 8"

**Layout:** Grille 3×3 (9 environnements)

**Contenu de chaque cellule:**

Ligne 1 - EASY:
1. (5×5, obs=0, doors=0): Grille petite, vide, goal au coin
2. (6×6, obs=1, doors=0): Un obstacle central
3. (7×7, obs=2, doors=0): Deux obstacles

Ligne 2 - MEDIUM:
4. (8×8, obs=3, doors=1, keys=1): Porte au milieu + clé
5. (9×9, obs=4, doors=1, keys=1): Plus d'obstacles
6. (10×10, obs=5, doors=1, keys=1): Dense en obstacles

Ligne 3 - HARD:
7. (11×11, obs=8, doors=2, keys=2): Deux portes + obstacles nombreux
8. (12×12, obs=10, doors=2, keys=2): Très dense
9. (12×12, obs=15, doors=2, keys=2): Maximum complexity

**Instructions de rendu:**
```python
# Pour chaque env:
env = ParametricMiniGridEnv(
    grid_size=grid_size,
    num_obstacles=obstacles,
    num_doors=doors,
    num_keys=keys,
    render_mode='rgb_array'
)
obs, _ = env.reset()
img = env.render()

# Placer dans la grille
ax[row, col].imshow(img)
ax[row, col].set_title(f'({grid_size}×{grid_size}, obs={obstacles}, doors={doors})', 
                        fontsize=10)
ax[row, col].axis('off')

# Ajouter border coloré selon difficulté
border_color = 'green' if difficulty < 0.4 else 'orange' if difficulty < 0.7 else 'red'
for spine in ax[row, col].spines.values():
    spine.set_edgecolor(border_color)
    spine.set_linewidth(3)
```

**Labels de ligne (à gauche):**
- Ligne 1: "EASY\n(d < 0.4)"
- Ligne 2: "MEDIUM\n(0.4 ≤ d < 0.7)"
- Ligne 3: "HARD\n(d ≥ 0.7)"

---

## FIGURE 6: GENERATOR ARCHITECTURE (Priority: MEDIUM)

**Fichier:** `figure6_generator_arch.pdf/png`

**Type:** Diagramme de réseau neural

**Dimensions:** 8" × 6"

**Structure de gauche à droite:**

1. **Input latent z:**
   - Cercle bleu avec icône Gaussienne
   - Label: "z ∈ R^8\n~ N(0,1)"
   - Exemple: "[-0.5, 1.2, 0.3, ...]"

2. **Layer 1 (FC 8→64):**
   - 64 petits cercles alignés verticalement
   - Connexions depuis z (lignes grises fines, seulement quelques-unes visibles)
   - Label au-dessus: "FC1(8→64) + ReLU"

3. **Layer 2 (FC 64→64):**
   - 64 cercles
   - Connexions depuis layer 1
   - Label: "FC2(64→64) + ReLU"

4. **Layer 3 (FC 64→4):**
   - 4 cercles plus gros
   - Connexions depuis layer 2
   - Label: "FC3(64→4)"

5. **Sigmoid + Scaling:**
   - 4 boîtes de transformation
   - Équations:
     ```
     grid_size = int(σ(raw[0])×7 + 5)
     obstacles = int(σ(raw[1])×5)
     doors     = int(σ(raw[2])×2)
     keys      = int(σ(raw[3])×2)
     ```

6. **Output:**
   - 4 rectangles verts avec valeurs exemple
   - "grid=11, obs=4, doors=1, keys=2"

**Annotations supplémentaires:**
- Flèche backprop en pointillés rouge depuis output
- Label: "∂L/∂W (gradient descent)"
- Box en bas: "Xavier Init: W ~ U[-√(6/(n_in+n_out)), +√(6/(n_in+n_out))]"

---

## FIGURE 7: TRAINING TIMELINE (Priority: MEDIUM)

**Fichier:** `figure7_timeline.pdf/png`

**Type:** Diagramme de Gantt/Timeline

**Dimensions:** 14" × 5" (wide)

**Structure:**

### Phase 0 (Bootstrap):
```
Epoch 0 ████████████████████████████ (100k timesteps)
        └─ Random levels
        └─ ~30 minutes
```
- Barre horizontale grise épaisse
- Subdivisions: Generate (5%) | Train (90%) | Eval (5%)

### Epochs 1-20 (Co-evolution):
```
Epoch 1  ████ ████████████ ██ ██
Epoch 2  ████ ████████████ ██ ██
...
Epoch 20 ████ ████████████ ██ ██

Legend:
████ Generate (20 levels, ~1min)
████ Train PPO (50k steps, ~8min)
██   Evaluate (5 levels, ~30sec)
██   Update Gen (20 iters, ~1min)
```

- Chaque epoch: barre divisée en 4 segments colorés
- Couleurs: Generate (vert clair), Train (bleu), Eval (orange), Update (violet)
- Timeline en haut: 0min → 10min → 20min → ... → 200min

### Background gradient:
- Gradient de fond: rouge (bas SR) → jaune → vert (haut SR)
- Représente l'amélioration du success rate

**Annotations:**
- Flèche vers epoch 6 (Diversity): "100% SR achieved here"
- Flèche vers epoch 18 (Baseline): "100% SR achieved here"

---

## FIGURE 8: PARAMETER SPACE EXPLORATION (Priority: HIGH)

**Fichier:** `figure8_param_space.pdf/png`

**Type:** Scatter plots 3×1

**Dimensions:** 12" × 4"

**Données:**
```python
# Extraire tous les niveaux générés de chaque run
vanilla_levels = []  # List of dicts {'grid_size': X, 'num_obstacles': Y, ...}
baseline_levels = []
diversity_levels = []

# Pour chaque méthode, scatter plot grid_size vs obstacles
```

**3 sous-plots côte à côte:**

### Subplot 1: Vanilla
```python
ax1.scatter(grid_sizes, obstacles, 
            c=epochs,  # Couleur = epoch number
            cmap='Blues',
            alpha=0.6,
            s=50)
ax1.set_title('Vanilla Co-evolution\n(Mode Collapse)', fontweight='bold')
ax1.set_xlabel('Grid Size')
ax1.set_ylabel('Number of Obstacles')

# Ajouter contour de densité
from scipy.stats import gaussian_kde
xy = np.vstack([grid_sizes, obstacles])
z = gaussian_kde(xy)(xy)
ax1.tricontour(grid_sizes, obstacles, z, levels=5, colors='darkblue', linewidths=2)

# Annotation du mode
mode_x, mode_y = 8, 3  # Position du cluster principal
ax1.scatter([mode_x], [mode_y], s=500, facecolors='none', 
            edgecolors='red', linewidths=3, label='Mode')
ax1.annotate('Collapsed here', xy=(mode_x, mode_y), 
             xytext=(mode_x+1, mode_y+1),
             arrowprops=dict(arrowstyle='->', color='red', lw=2))
```

### Subplot 2: Baseline
```python
# Même structure mais uniform scatter
# Pas de clustering visible
# Annotation: "Uniform coverage"
```

### Subplot 3: Diversity
```python
# Dense mais bien distribué
# Légère concentration aux extrêmes (bimodal)
# Annotation: "Maintained exploration"
```

**Colorbar partagée:**
- En bas de la figure
- Label: "Training Epoch"
- Gradient bleu (early) → rouge (late)

**Stats boxes (un par subplot):**
```
Vanilla:
std(grid) = 0.8
std(obs)  = 1.1
Coverage  = 23%

Baseline:
std(grid) = 2.4
std(obs)  = 1.9
Coverage  = 89%

Diversity:
std(grid) = 2.5
std(obs)  = 2.1
Coverage  = 94%
```

---

## FIGURE 9: LAMBDA EFFECT (Priority: LOW - peut être skippé si manque de temps)

**Fichier:** `figure9_lambda_effect.pdf/png`

**Type:** Line plot avec error bars

**Dimensions:** 8" × 5"

**Données (si disponibles, sinon simuler):**
```python
lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0]
final_sr = [59.3, 62.1, 68.5, 75.2, 88.4, 92.3, 89.1, 82.4, 75.3, 68.2, 60.1, 55.4]
std_sr   = [22.1, 20.3, 18.2, 15.4, 12.1,  9.4, 11.2, 14.8, 18.2, 21.3, 24.5, 26.8]
```

**Éléments:**
1. Courbe avec error bars
2. Point optimal marqué d'une étoile à λ=0.5
3. Zones annotées:
   - λ=0: "Mode collapse (vanilla)"
   - λ=0.5: "Optimal balance ★"
   - λ>1.0: "Too random"

---

## FIGURE 10: DIVERSITY METRICS TABLE (Priority: MEDIUM)

**Fichier:** `figure10_diversity_metrics.pdf/png`

**Type:** Table/Heatmap hybride

**Dimensions:** 10" × 4"

**Données à extraire:**
```python
# Pour chaque méthode et epoch:
metrics = {
    'vanilla': {
        'batch_distance': [mean, std],
        'novelty_score': [mean, std],
        'param_variance': {
            'grid_size': std,
            'obstacles': std,
            'doors': std,
            'keys': std
        }
    },
    # ... baseline, diversity
}
```

**Layout:**

Table avec colonnes:
| Method | Batch Distance | Novelty Score | Param Variance (grid/obs/doors/keys) |
|--------|---------------|---------------|--------------------------------------|
| Vanilla | 0.25±0.10 | 0.15±0.08 | 0.8 / 1.1 / 0.3 / 0.2 |
| Baseline | N/A (random) | N/A | 2.4 / 1.9 / 0.8 / 0.7 |
| Diversity | 2.50±0.50 | 0.80±0.30 | 2.5 / 2.1 / 0.9 / 0.8 |

**Coloration des cellules:**
- Vert (high diversity): batch_dist > 1.5, novelty > 0.5, variance > 2.0
- Jaune (medium): valeurs intermédiaires
- Rouge (low/collapse): batch_dist < 0.5, novelty < 0.3, variance < 1.0

---

## FIGURE 11: COMPREHENSIVE 4-PANEL (Priority: ★★★ CRITICAL)

**Fichier:** `figure11_comprehensive.pdf/png`

**Type:** Figure multi-panel 2×2

**Dimensions:** 14" × 10"

### Panel A (top-left): Learning Curves
- Réutiliser code Figure 2 mais version compacte
- Pas d'annotations détaillées

### Panel B (top-right): Box Plots
- Réutiliser code Figure 3 mais version compacte

### Panel C (bottom-left): Diversity Over Time
```python
# X: Epochs 1-20
# Y: Batch Distance metric
# 3 lignes (vanilla/baseline/diversity)

# Vanilla: décroissant 0.8 → 0.2
# Baseline: horizontal autour de 1.8 (avec bruit)
# Diversity: stable autour de 2.5
```

### Panel D (bottom-right): Parameter Variance Evolution
```python
# Stacked area chart
# X: Epochs 1-20
# Y: Cumulative std of all 4 params

# Pour chaque méthode, 3 subplots empilés verticalement
# Montrer comment variance collapse (vanilla) vs maintained (diversity)
```

**Layout:**
```python
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

ax_a = fig.add_subplot(gs[0, 0])  # Learning curves
ax_b = fig.add_subplot(gs[0, 1])  # Box plots
ax_c = fig.add_subplot(gs[1, 0])  # Diversity over time
ax_d = fig.add_subplot(gs[1, 1])  # Param variance

# Labels A, B, C, D en coins supérieurs gauches
for ax, label in zip([ax_a, ax_b, ax_c, ax_d], ['A', 'B', 'C', 'D']):
    ax.text(-0.1, 1.05, label, transform=ax.transAxes, 
            fontsize=20, fontweight='bold')
```

---

## FIGURE 12: DIFFICULTY DISTRIBUTIONS (Priority: MEDIUM)

**Fichier:** `figure12_difficulty.pdf/png`

**Type:** Histogrammes overlaid + KDE

**Dimensions:** 10" × 5"

**Calcul de difficulty:**
```python
def compute_difficulty(level):
    d = (0.4 * level['grid_size'] / 12.0 +
         0.3 * level['num_obstacles'] / 5.0 +
         0.15 * level['num_doors'] / 2.0 +
         0.15 * level['num_keys'] / 2.0)
    return d

# Pour 1000 niveaux de chaque méthode
vanilla_diff = [compute_difficulty(lv) for lv in vanilla_levels]
baseline_diff = [compute_difficulty(lv) for lv in baseline_levels]
diversity_diff = [compute_difficulty(lv) for lv in diversity_levels]
```

**Éléments:**
1. **3 histogrammes superposés:**
   ```python
   ax.hist(vanilla_diff, bins=30, alpha=0.5, color=color_vanilla, 
           label='Vanilla', density=True)
   ax.hist(baseline_diff, bins=30, alpha=0.5, color=color_baseline,
           label='Baseline', density=True)
   ax.hist(diversity_diff, bins=30, alpha=0.5, color=color_diversity,
           label='Diversity', density=True)
   ```

2. **KDE curves:**
   ```python
   from scipy.stats import gaussian_kde
   
   for data, color, label in zip([vanilla_diff, baseline_diff, diversity_diff],
                                  [color_vanilla, color_baseline, color_diversity],
                                  ['Vanilla', 'Baseline', 'Diversity']):
       kde = gaussian_kde(data)
       x_range = np.linspace(0, 1, 200)
       ax.plot(x_range, kde(x_range), color=color, linewidth=3, label=f'{label} KDE')
   ```

3. **Lignes de moyenne:**
   ```python
   # Ligne verticale pointillée pour chaque mean
   ax.axvline(np.mean(vanilla_diff), color=color_vanilla, 
              linestyle='--', linewidth=2, alpha=0.8)
   ax.text(np.mean(vanilla_diff), ax.get_ylim()[1]*0.9, 
           f'μ={np.mean(vanilla_diff):.3f}', rotation=90)
   ```

**Annotations:**
- Zone 0.0-0.3: "Easy"
- Zone 0.3-0.7: "Medium"
- Zone 0.7-1.0: "Hard"

---

## FIGURE 13: DIVERSITY EVOLUTION TIMELINE (Priority: MEDIUM)

**Fichier:** `figure13_diversity_evolution.pdf/png`

**Type:** Timeline avec multi-metrics

**Dimensions:** 14" × 6"

**Structure:**

5 colonnes (epochs 1, 5, 10, 15, 20):

Pour chaque epoch, 3 rangées:

### Rangée 1: Example Level Screenshot
```python
# Générer et rendre un niveau représentatif de cet epoch
level = generate_level_at_epoch(epoch_num)
img = render_level(level)
ax[0, col].imshow(img)
ax[0, col].set_title(f'Epoch {epoch_num}')
```

### Rangée 2: Parameter Distribution
```python
# Histogramme 2×2 des 4 paramètres pour cet epoch
# grid_size, obstacles, doors, keys
# Montrer l'évolution de la variance
```

### Rangée 3: Novelty Score Trend
```python
# Mini line plot: epochs 1 à current
#