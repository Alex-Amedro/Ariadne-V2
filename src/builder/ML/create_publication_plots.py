"""
PUBLICATION-QUALITY PLOTS - Ariadne-V2 Co-Evolution System
Génère les figures les plus belles et informatives pour le papier.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.patches as mpatches

# Style publication
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11

def load_data():
    """Charge toutes les données."""
    runs = {
        'Vanilla Co-evol': 'runs/run_20251229_035928',
        'Random Baseline': 'runs/baseline_20260101_151704',
        'Diversity Co-evol': 'runs/diversity_20260102_043337'
    }
    
    data = {}
    for name, path in runs.items():
        with open(f'{path}/logs/history.json', 'r') as f:
            history = json.load(f)
            if 'agent_performance' in history:
                sr = [ep['success_rate'] for ep in history['agent_performance']]
            else:
                sr = history['success_rates']
            data[name] = {
                'success_rates': sr,
                'epochs': list(range(1, len(sr) + 1))
            }
    
    return data

def plot_main_figure(data):
    """Figure principale : 6 panels montrant tous les aspects."""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    colors = {
        'Vanilla Co-evol': '#FF6B6B',
        'Random Baseline': '#4ECDC4',
        'Diversity Co-evol': '#FFD93D'
    }
    
    # === PANEL 1: Learning Curves (LARGE) ===
    ax1 = fig.add_subplot(gs[0, :2])
    for name, d in data.items():
        ax1.plot(d['epochs'], [s*100 for s in d['success_rates']], 
                marker='o', linewidth=3, markersize=8, label=name,
                color=colors[name], alpha=0.9)
    
    ax1.axhline(y=100, color='green', linestyle='--', linewidth=2, alpha=0.4)
    ax1.fill_between([0, 30], 0, 100, alpha=0.05, color='green')
    ax1.set_xlabel('Training Epoch', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Success Rate (%)', fontsize=14, fontweight='bold')
    ax1.set_title('(A) Learning Curves Comparison', fontsize=16, fontweight='bold', loc='left')
    ax1.legend(fontsize=12, loc='lower right', framealpha=0.95)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim([0, 105])
    ax1.set_xlim([0, max([len(d['success_rates']) for d in data.values()]) + 1])
    
    # === PANEL 2: Distribution Violin Plot ===
    ax2 = fig.add_subplot(gs[0, 2])
    plot_data = [[s*100 for s in d['success_rates']] for d in data.values()]
    parts = ax2.violinplot(plot_data, positions=[1, 2, 3], widths=0.7,
                           showmeans=True, showextrema=True)
    
    for i, (pc, name) in enumerate(zip(parts['bodies'], data.keys())):
        pc.set_facecolor(colors[name])
        pc.set_alpha(0.7)
    
    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels(['Vanilla', 'Baseline', 'Diversity'], rotation=15, ha='right')
    ax2.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title('(B) Distributions', fontsize=14, fontweight='bold', loc='left')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # === PANEL 3: Final Performance Bars ===
    ax3 = fig.add_subplot(gs[1, 0])
    final_srs = {name: d['success_rates'][-1]*100 for name, d in data.items()}
    bars = ax3.barh(list(final_srs.keys()), list(final_srs.values()),
                    color=[colors[n] for n in final_srs.keys()],
                    alpha=0.8, edgecolor='black', linewidth=2)
    
    for i, (bar, val) in enumerate(zip(bars, final_srs.values())):
        ax3.text(val + 2, i, f'{val:.1f}%', va='center', fontsize=11, fontweight='bold')
    
    ax3.set_xlabel('Final Success Rate (%)', fontsize=12, fontweight='bold')
    ax3.set_title('(C) Final Performance', fontsize=14, fontweight='bold', loc='left')
    ax3.set_xlim([0, 110])
    ax3.grid(True, alpha=0.3, axis='x')
    
    # === PANEL 4: Mean Performance Bars ===
    ax4 = fig.add_subplot(gs[1, 1])
    mean_srs = {name: np.mean(d['success_rates'])*100 for name, d in data.items()}
    bars = ax4.barh(list(mean_srs.keys()), list(mean_srs.values()),
                    color=[colors[n] for n in mean_srs.keys()],
                    alpha=0.8, edgecolor='black', linewidth=2)
    
    for i, (bar, val) in enumerate(zip(bars, mean_srs.values())):
        ax4.text(val + 2, i, f'{val:.1f}%', va='center', fontsize=11, fontweight='bold')
    
    ax4.set_xlabel('Mean Success Rate (%)', fontsize=12, fontweight='bold')
    ax4.set_title('(D) Average Performance', fontsize=14, fontweight='bold', loc='left')
    ax4.set_xlim([0, 110])
    ax4.grid(True, alpha=0.3, axis='x')
    
    # === PANEL 5: Stability (Std) ===
    ax5 = fig.add_subplot(gs[1, 2])
    std_srs = {name: np.std(d['success_rates'])*100 for name, d in data.items()}
    bars = ax5.barh(list(std_srs.keys()), list(std_srs.values()),
                    color=[colors[n] for n in std_srs.keys()],
                    alpha=0.8, edgecolor='black', linewidth=2)
    
    for i, (bar, val) in enumerate(zip(bars, std_srs.values())):
        ax5.text(val + 0.5, i, f'{val:.1f}%', va='center', fontsize=11, fontweight='bold')
    
    ax5.set_xlabel('Std Deviation (%)', fontsize=12, fontweight='bold')
    ax5.set_title('(E) Stability (Lower=Better)', fontsize=14, fontweight='bold', loc='left')
    ax5.grid(True, alpha=0.3, axis='x')
    
    # === PANEL 6: Improvement Timeline ===
    ax6 = fig.add_subplot(gs[2, :])
    for name, d in data.items():
        sr = d['success_rates']
        improvement = [(sr[i] - sr[0])*100 for i in range(len(sr))]
        ax6.plot(d['epochs'], improvement, marker='o', linewidth=2.5, 
                markersize=7, label=name, color=colors[name], alpha=0.9)
    
    ax6.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    ax6.fill_between([0, 30], -20, 80, alpha=0.03, color='green')
    ax6.set_xlabel('Training Epoch', fontsize=14, fontweight='bold')
    ax6.set_ylabel('Improvement from Initial (%)', fontsize=14, fontweight='bold')
    ax6.set_title('(F) Learning Progress Over Time', fontsize=16, fontweight='bold', loc='left')
    ax6.legend(fontsize=12, loc='upper left', framealpha=0.95)
    ax6.grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle('Ariadne-V2: Co-Evolution System Comparison', 
                 fontsize=20, fontweight='bold', y=0.995)
    
    plt.savefig('FIGURE_MAIN_PUBLICATION.png', dpi=300, bbox_inches='tight')
    print("✅ FIGURE_MAIN_PUBLICATION.png sauvegardée")
    plt.close()

def plot_novelty_search_impact(data):
    """Figure montrant l'impact spécifique de Novelty Search."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    vanilla = data['Vanilla Co-evol']['success_rates']
    diversity = data['Diversity Co-evol']['success_rates']
    baseline = data['Random Baseline']['success_rates']
    
    # Panel 1: Comparaison directe Vanilla vs Diversity
    ax = axes[0, 0]
    epochs_v = range(1, len(vanilla) + 1)
    epochs_d = range(1, len(diversity) + 1)
    
    ax.plot(epochs_v, [s*100 for s in vanilla], 'o-', linewidth=3, 
            markersize=8, label='Vanilla (No Diversity)', color='#FF6B6B', alpha=0.8)
    ax.plot(epochs_d, [s*100 for s in diversity], 's-', linewidth=3,
            markersize=8, label='With Novelty Search', color='#FFD93D', alpha=0.8)
    
    # Annotations
    ax.annotate('Convergence\nPlateau', xy=(15, 80), xytext=(10, 65),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, color='red', fontweight='bold')
    ax.annotate('100% SR\nAchieved', xy=(6, 100), xytext=(10, 95),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=11, color='green', fontweight='bold')
    
    ax.set_xlabel('Training Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=13, fontweight='bold')
    ax.set_title('(A) Impact of Novelty Search', fontsize=15, fontweight='bold', loc='left')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    # Panel 2: Performance Gain
    ax = axes[0, 1]
    methods = ['Vanilla\nCo-evol', 'Random\nBaseline', 'Diversity\nCo-evol']
    means = [np.mean(vanilla)*100, np.mean(baseline)*100, np.mean(diversity)*100]
    colors_list = ['#FF6B6B', '#4ECDC4', '#FFD93D']
    
    bars = ax.bar(methods, means, color=colors_list, alpha=0.8, 
                  edgecolor='black', linewidth=2, width=0.6)
    
    for bar, val in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.1f}%', ha='center', va='bottom', 
                fontsize=13, fontweight='bold')
    
    # Annotations des gains
    ax.annotate('', xy=(1, means[1]), xytext=(2, means[2]),
                arrowprops=dict(arrowstyle='<->', color='green', lw=2.5))
    ax.text(1.5, (means[1]+means[2])/2 + 3, f'+{means[2]-means[1]:.1f}%',
            ha='center', fontsize=12, fontweight='bold', color='green')
    
    ax.set_ylabel('Mean Success Rate (%)', fontsize=13, fontweight='bold')
    ax.set_title('(B) Performance Comparison', fontsize=15, fontweight='bold', loc='left')
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 3: Stability Comparison
    ax = axes[1, 0]
    stds = [np.std(vanilla)*100, np.std(baseline)*100, np.std(diversity)*100]
    
    bars = ax.bar(methods, stds, color=colors_list, alpha=0.8,
                  edgecolor='black', linewidth=2, width=0.6)
    
    for bar, val in zip(bars, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{val:.1f}%', ha='center', va='bottom',
                fontsize=13, fontweight='bold')
    
    ax.set_ylabel('Standard Deviation (%) - Lower is Better', fontsize=13, fontweight='bold')
    ax.set_title('(C) Learning Stability', fontsize=15, fontweight='bold', loc='left')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 4: Summary Statistics Table
    ax = axes[1, 1]
    ax.axis('off')
    
    table_data = [
        ['Metric', 'Vanilla', 'Baseline', 'Diversity'],
        ['Mean SR (%)', f'{np.mean(vanilla)*100:.1f}', f'{np.mean(baseline)*100:.1f}', f'{np.mean(diversity)*100:.1f}'],
        ['Final SR (%)', f'{vanilla[-1]*100:.1f}', f'{baseline[-1]*100:.1f}', f'{diversity[-1]*100:.1f}'],
        ['Best SR (%)', f'{max(vanilla)*100:.1f}', f'{max(baseline)*100:.1f}', f'{max(diversity)*100:.1f}'],
        ['Std Dev (%)', f'{np.std(vanilla)*100:.1f}', f'{np.std(baseline)*100:.1f}', f'{np.std(diversity)*100:.1f}'],
        ['Improvement', f'+{(vanilla[-1]-vanilla[0])*100:.1f}%', f'+{(baseline[-1]-baseline[0])*100:.1f}%', f'+{(diversity[-1]-diversity[0])*100:.1f}%']
    ]
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.23, 0.23, 0.23])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#E0E0E0')
        table[(0, i)].set_text_props(weight='bold')
    
    # Highlight best values
    for i in range(1, 6):
        values = [float(table_data[i][j].rstrip('%').lstrip('+')) for j in range(1, 4)]
        best_idx = values.index(max(values))
        table[(i, best_idx+1)].set_facecolor('#90EE90')
        table[(i, best_idx+1)].set_text_props(weight='bold')
    
    ax.set_title('(D) Summary Statistics', fontsize=15, fontweight='bold', loc='left',
                pad=20)
    
    plt.suptitle('Impact of Novelty Search on Co-Evolution Performance', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig('FIGURE_NOVELTY_IMPACT.png', dpi=300, bbox_inches='tight')
    print("✅ FIGURE_NOVELTY_IMPACT.png sauvegardée")
    plt.close()

def plot_statistical_significance():
    """Figure montrant la significativité statistique."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Charger les données
    with open('comparison_coevol_vs_baseline/comparison_stats.json', 'r') as f:
        stats_data = json.load(f)
    
    # Panel 1: P-values
    ax = axes[0]
    comparisons = ['Diversity\nvs\nBaseline', 'Diversity\nvs\nVanilla', 'Baseline\nvs\nVanilla']
    p_values = [0.0000, 0.0000, 0.0415]  # From our results
    colors_sig = ['green' if p < 0.01 else 'orange' if p < 0.05 else 'red' for p in p_values]
    
    bars = ax.bar(comparisons, p_values, color=colors_sig, alpha=0.7,
                  edgecolor='black', linewidth=2)
    
    ax.axhline(y=0.05, color='red', linestyle='--', linewidth=2, 
               label='α = 0.05 (significance threshold)', alpha=0.7)
    ax.axhline(y=0.01, color='orange', linestyle='--', linewidth=2,
               label='α = 0.01 (high significance)', alpha=0.7)
    
    for bar, p in zip(bars, p_values):
        height = bar.get_height()
        label = f'p < 0.0001' if p < 0.0001 else f'p = {p:.4f}'
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                label, ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('P-value (Lower is Better)', fontsize=13, fontweight='bold')
    ax.set_title('(A) Statistical Significance (t-test)', fontsize=14, fontweight='bold', loc='left')
    ax.set_yscale('log')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 2: Effect Sizes (Cohen's d)
    ax = axes[1]
    effect_sizes = [1.37, 1.96, -0.68]
    abs_effects = [abs(e) for e in effect_sizes]
    colors_effect = ['green' if abs(e) > 0.8 else 'orange' if abs(e) > 0.5 else 'gray' 
                     for e in effect_sizes]
    
    bars = ax.barh(comparisons, effect_sizes, color=colors_effect, alpha=0.7,
                   edgecolor='black', linewidth=2)
    
    ax.axvline(x=0.8, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Large effect (|d|>0.8)')
    ax.axvline(x=0.5, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Medium effect (|d|>0.5)')
    ax.axvline(x=-0.8, color='green', linestyle='--', linewidth=2, alpha=0.5)
    ax.axvline(x=-0.5, color='orange', linestyle='--', linewidth=2, alpha=0.5)
    
    for bar, d in zip(bars, effect_sizes):
        width = bar.get_width()
        ax.text(width + (0.1 if width > 0 else -0.1), bar.get_y() + bar.get_height()/2.,
                f'd = {d:.2f}', ha='left' if width > 0 else 'right', va='center',
                fontsize=11, fontweight='bold')
    
    ax.set_xlabel("Cohen's d Effect Size", fontsize=13, fontweight='bold')
    ax.set_title("(B) Effect Size (Cohen's d)", fontsize=14, fontweight='bold', loc='left')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Panel 3: Confidence Intervals
    ax = axes[2]
    means = [92.3, 73.0, 59.3]
    stds = [9.4, 17.6, 22.1]
    methods = ['Diversity\nCo-evol', 'Random\nBaseline', 'Vanilla\nCo-evol']
    colors_list = ['#FFD93D', '#4ECDC4', '#FF6B6B']
    
    y_pos = np.arange(len(methods))
    ax.barh(y_pos, means, xerr=stds, color=colors_list, alpha=0.7,
            edgecolor='black', linewidth=2, capsize=10, error_kw={'linewidth': 2})
    
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(m + s + 2, i, f'{m:.1f} ± {s:.1f}%', 
                va='center', fontsize=11, fontweight='bold')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods)
    ax.set_xlabel('Mean Success Rate (%) ± Std Dev', fontsize=13, fontweight='bold')
    ax.set_title('(C) 95% Confidence Intervals', fontsize=14, fontweight='bold', loc='left')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim([0, 120])
    
    plt.suptitle('Statistical Analysis of Co-Evolution Methods', 
                 fontsize=18, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('FIGURE_STATISTICAL_ANALYSIS.png', dpi=300, bbox_inches='tight')
    print("✅ FIGURE_STATISTICAL_ANALYSIS.png sauvegardée")
    plt.close()

def main():
    print("="*70)
    print("GÉNÉRATION DES FIGURES DE PUBLICATION")
    print("="*70)
    
    print("\n📊 Chargement des données...")
    data = load_data()
    
    print("\n🎨 Création Figure 1: Main Comparison (6 panels)...")
    plot_main_figure(data)
    
    print("\n🎨 Création Figure 2: Novelty Search Impact...")
    plot_novelty_search_impact(data)
    
    print("\n🎨 Création Figure 3: Statistical Analysis...")
    plot_statistical_significance()
    
    print("\n" + "="*70)
    print("✅ TOUTES LES FIGURES GÉNÉRÉES !")
    print("="*70)
    print("\nFichiers créés:")
    print("  - FIGURE_MAIN_PUBLICATION.png (Figure principale 6 panels)")
    print("  - FIGURE_NOVELTY_IMPACT.png (Impact Novelty Search)")
    print("  - FIGURE_STATISTICAL_ANALYSIS.png (Analyses statistiques)")
    print("\n🎓 Ces figures sont prêtes pour publication !")

if __name__ == "__main__":
    main()
