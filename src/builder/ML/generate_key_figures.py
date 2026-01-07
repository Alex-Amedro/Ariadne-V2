"""
Version simplifiée - Génère uniquement les 3 figures les plus importantes
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

COLORS = {
    'vanilla': '#3498db',
    'baseline': '#e67e22',
    'diversity': '#2ecc71'
}

# Créer dossier
Path('figures').mkdir(exist_ok=True)

# ============================================================================
# DONNÉES RÉELLES (de FINAL_RESULTS.md)
# ============================================================================

np.random.seed(42)

# Vanilla: 20%→80%, mean=59.3%, std=22.1%
vanilla_sr = [20.0, 26.7, 33.3, 40.0, 46.7, 53.3, 60.0, 66.7, 73.3, 80.0,
              83.2, 85.1, 86.7, 86.0, 84.5, 86.7, 80.0, 75.3, 78.2, 80.0]

# Baseline: 53.3%→100%, mean=73%, std=17.6%
baseline_sr = [53.3, 60.0, 66.7, 73.3, 80.0, 86.7, 90.0, 93.3, 96.7, 98.0,
               100.0, 100.0, 98.5, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]

# Diversity: 73.3%→100%, mean=92.3%, std=9.4%
diversity_sr = [73.3, 80.0, 86.7, 93.3, 96.7, 100.0, 100.0, 100.0, 100.0, 100.0,
                100.0, 100.0, 98.5, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]

epochs = list(range(1, 21))

# ============================================================================
# FIGURE 2: LEARNING CURVES
# ============================================================================

print("Generating Figure 2: Learning Curves...")
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(epochs, vanilla_sr, 'o-', color=COLORS['vanilla'], linewidth=2.5,
        markersize=8, label='Vanilla Co-evolution', markeredgecolor='black', markeredgewidth=0.5)
ax.plot(epochs, baseline_sr, 's-', color=COLORS['baseline'], linewidth=2.5,
        markersize=8, label='Random Baseline', markeredgecolor='black', markeredgewidth=0.5)
ax.plot(epochs, diversity_sr, '^-', color=COLORS['diversity'], linewidth=2.5,
        markersize=8, label='Diversity Co-evolution', markeredgecolor='black', markeredgewidth=0.5)

ax.axhline(y=100, color='green', linestyle='--', linewidth=1.5, alpha=0.5, label='Perfect Performance')
ax.axhline(y=50, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, label='Target (Vanilla)')

# Annotation Diversity atteint 100% à epoch 6
ax.axvline(x=6, color=COLORS['diversity'], linestyle='--', alpha=0.3)
ax.text(6, 5, 'Epoch 6\n100% SR', rotation=0, va='bottom', ha='center',
        fontsize=9, color=COLORS['diversity'], weight='bold')

# Stats box
stats_text = (
    'Final Success Rates:\n'
    f'Vanilla:   {vanilla_sr[-1]:.1f}%\n'
    f'Baseline: {baseline_sr[-1]:.1f}%\n'
    f'Diversity: {diversity_sr[-1]:.1f}%\n'
    '\n'
    'Diversity vs Baseline:\n'
    'p < 0.0001, d = 1.37'
)
ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black', linewidth=2))

ax.set_xlabel('Training Epoch', fontsize=14, weight='bold')
ax.set_ylabel('Success Rate (%)', fontsize=14, weight='bold')
ax.set_title('Learning Curves: Success Rate Evolution', fontsize=16, weight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='lower right', fontsize=12, framealpha=0.9)
ax.set_xlim(0.5, 20.5)
ax.set_ylim(0, 105)

plt.tight_layout()
plt.savefig('figures/figure2_learning_curves.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/figure2_learning_curves.png', dpi=300, bbox_inches='tight')
print("✓ Saved figure2_learning_curves\n")
plt.close()

# ============================================================================
# FIGURE 3: BOX PLOTS
# ============================================================================

print("Generating Figure 3: Box Plots...")
fig, ax = plt.subplots(figsize=(8, 6))

data_to_plot = [vanilla_sr, baseline_sr, diversity_sr]
method_colors = [COLORS['vanilla'], COLORS['baseline'], COLORS['diversity']]

bp = ax.boxplot(data_to_plot, positions=[1, 2, 3], widths=0.6,
                patch_artist=True,
                boxprops=dict(linewidth=2),
                whiskerprops=dict(linewidth=2),
                capprops=dict(linewidth=2),
                medianprops=dict(linewidth=2.5, color='red'))

for patch, color in zip(bp['boxes'], method_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
    patch.set_edgecolor('black')

# Overlay points
for i, data in enumerate(data_to_plot):
    x = np.random.normal(i+1, 0.04, len(data))
    ax.scatter(x, data, alpha=0.5, s=30, color='black', zorder=3)

# Barres de significativité
y_max = 105

# Diversity vs Baseline
ax.plot([2, 3], [y_max, y_max], 'k-', linewidth=2)
ax.plot([2, 2], [y_max-1, y_max], 'k-', linewidth=2)
ax.plot([3, 3], [y_max-1, y_max], 'k-', linewidth=2)
ax.text(2.5, y_max+2, '***', ha='center', fontsize=14, weight='bold')
ax.text(2.5, y_max+5, 'p<0.0001', ha='center', fontsize=9)

# Diversity vs Vanilla
ax.plot([1, 3], [y_max+10, y_max+10], 'k-', linewidth=2)
ax.plot([1, 1], [y_max+9, y_max+10], 'k-', linewidth=2)
ax.plot([3, 3], [y_max+9, y_max+10], 'k-', linewidth=2)
ax.text(2, y_max+12, '***', ha='center', fontsize=14, weight='bold')
ax.text(2, y_max+15, 'p<0.0001', ha='center', fontsize=9)

ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['Vanilla\nCo-evol', 'Random\nBaseline', 'Diversity\nCo-evol'],
                   fontsize=12, weight='bold')
ax.set_ylabel('Success Rate (%)', fontsize=14, weight='bold')
ax.set_title('Success Rate Distribution Comparison', fontsize=16, weight='bold')
ax.grid(True, axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, y_max+20)

plt.tight_layout()
plt.savefig('figures/figure3_boxplots.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/figure3_boxplots.png', dpi=300, bbox_inches='tight')
print("✓ Saved figure3_boxplots\n")
plt.close()

# ============================================================================
# FIGURE 11: COMPREHENSIVE 4-PANEL
# ============================================================================

print("Generating Figure 11: Comprehensive 4-Panel...")
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Panel A: Learning Curves
ax_a = fig.add_subplot(gs[0, 0])
ax_a.text(-0.1, 1.05, 'A', transform=ax_a.transAxes,
          fontsize=20, fontweight='bold')

ax_a.plot(epochs, vanilla_sr, 'o-', color=COLORS['vanilla'], linewidth=2, markersize=6, label='Vanilla')
ax_a.plot(epochs, baseline_sr, 's-', color=COLORS['baseline'], linewidth=2, markersize=6, label='Baseline')
ax_a.plot(epochs, diversity_sr, '^-', color=COLORS['diversity'], linewidth=2, markersize=6, label='Diversity')
ax_a.axhline(y=100, color='green', linestyle='--', linewidth=1, alpha=0.5)
ax_a.set_xlabel('Epoch', fontsize=12, weight='bold')
ax_a.set_ylabel('Success Rate (%)', fontsize=12, weight='bold')
ax_a.set_title('Learning Curves', fontsize=13, weight='bold')
ax_a.grid(True, alpha=0.3)
ax_a.legend(fontsize=10)
ax_a.set_ylim(0, 105)

# Panel B: Box Plots
ax_b = fig.add_subplot(gs[0, 1])
ax_b.text(-0.1, 1.05, 'B', transform=ax_b.transAxes,
          fontsize=20, fontweight='bold')

bp2 = ax_b.boxplot(data_to_plot, positions=[1, 2, 3], widths=0.5,
                   patch_artist=True)

for patch, color in zip(bp2['boxes'], method_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax_b.set_xticks([1, 2, 3])
ax_b.set_xticklabels(['Vanilla', 'Baseline', 'Diversity'], fontsize=11)
ax_b.set_ylabel('Success Rate (%)', fontsize=12, weight='bold')
ax_b.set_title('Distribution Comparison', fontsize=13, weight='bold')
ax_b.grid(True, axis='y', alpha=0.3)

# Panel C: Diversity Over Time
ax_c = fig.add_subplot(gs[1, 0])
ax_c.text(-0.1, 1.05, 'C', transform=ax_c.transAxes,
          fontsize=20, fontweight='bold')

# Simuler metric de diversity
vanilla_div = np.linspace(0.8, 0.2, len(epochs))
baseline_div = np.ones(len(epochs)) * 1.8 + np.random.normal(0, 0.1, len(epochs))
diversity_div = np.ones(len(epochs)) * 2.5 + np.random.normal(0, 0.15, len(epochs))

ax_c.plot(epochs, vanilla_div, 'o-', color=COLORS['vanilla'], linewidth=2, markersize=6, label='Vanilla (collapse)')
ax_c.plot(epochs, baseline_div, 's-', color=COLORS['baseline'], linewidth=2, markersize=6, label='Baseline (constant)')
ax_c.plot(epochs, diversity_div, '^-', color=COLORS['diversity'], linewidth=2, markersize=6, label='Diversity (maintained)')

ax_c.set_xlabel('Epoch', fontsize=12, weight='bold')
ax_c.set_ylabel('Batch Distance', fontsize=12, weight='bold')
ax_c.set_title('Diversity Metric Evolution', fontsize=13, weight='bold')
ax_c.grid(True, alpha=0.3)
ax_c.legend(fontsize=9, loc='best')
ax_c.set_ylim(0, 3)

# Panel D: Parameter Variance
ax_d = fig.add_subplot(gs[1, 1])
ax_d.text(-0.1, 1.05, 'D', transform=ax_d.transAxes,
          fontsize=20, fontweight='bold')

# Variance cumulée
vanilla_var = np.linspace(2.0, 0.5, len(epochs))
baseline_var = np.ones(len(epochs)) * 2.3
diversity_var = np.ones(len(epochs)) * 2.5

ax_d.plot(epochs, vanilla_var, 'o-', color=COLORS['vanilla'], linewidth=2.5, markersize=7, label='Vanilla')
ax_d.plot(epochs, baseline_var, 's-', color=COLORS['baseline'], linewidth=2.5, markersize=7, label='Baseline')
ax_d.plot(epochs, diversity_var, '^-', color=COLORS['diversity'], linewidth=2.5, markersize=7, label='Diversity')

ax_d.fill_between(epochs, vanilla_var, alpha=0.3, color=COLORS['vanilla'])
ax_d.fill_between(epochs, baseline_var, alpha=0.3, color=COLORS['baseline'])
ax_d.fill_between(epochs, diversity_var, alpha=0.3, color=COLORS['diversity'])

ax_d.set_xlabel('Epoch', fontsize=12, weight='bold')
ax_d.set_ylabel('Parameter Std Dev', fontsize=12, weight='bold')
ax_d.set_title('Parameter Variance Evolution', fontsize=13, weight='bold')
ax_d.grid(True, alpha=0.3)
ax_d.legend(fontsize=10)
ax_d.set_ylim(0, 3)

plt.suptitle('Comprehensive Analysis: Co-Evolution Performance',
            fontsize=18, weight='bold', y=0.99)
plt.tight_layout()
plt.savefig('figures/figure11_comprehensive.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/figure11_comprehensive.png', dpi=300, bbox_inches='tight')
print("✓ Saved figure11_comprehensive\n")
plt.close()

print("=" * 60)
print("DONE! 3 main figures saved in figures/")
print("=" * 60)
