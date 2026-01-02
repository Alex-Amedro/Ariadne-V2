"""
Comparaison complète : Vanilla Co-evol vs Baseline vs Diversity Co-evol
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def load_history(run_path):
    """Charge l'historique d'un run."""
    history_path = f"{run_path}/logs/history.json"
    with open(history_path, 'r') as f:
        return json.load(f)

def extract_success_rates(history):
    """Extrait les success rates."""
    if 'agent_performance' in history:
        return [ep['success_rate'] for ep in history['agent_performance']]
    elif 'success_rates' in history:
        return history['success_rates']
    return []

def main():
    # Chemins des runs
    runs = {
        'Vanilla Co-evol': 'runs/run_20251229_035928',
        'Baseline (Random)': 'runs/baseline_20260101_151704',
        'Diversity Co-evol': 'runs/diversity_20260102_043337'
    }
    
    print("="*70)
    print("COMPARAISON DES 3 MÉTHODES")
    print("="*70)
    
    # Charger les données
    data = {}
    for name, path in runs.items():
        try:
            history = load_history(path)
            sr = extract_success_rates(history)
            data[name] = sr
            print(f"\n{name}:")
            print(f"  Epochs: {len(sr)}")
            print(f"  Initial SR: {sr[0]*100:.1f}%")
            print(f"  Final SR: {sr[-1]*100:.1f}%")
            print(f"  Best SR: {max(sr)*100:.1f}%")
            print(f"  Mean SR: {np.mean(sr)*100:.1f}%")
            print(f"  Std SR: {np.std(sr)*100:.1f}%")
        except Exception as e:
            print(f"\n❌ Erreur pour {name}: {e}")
    
    # Statistiques comparatives
    print("\n" + "="*70)
    print("TESTS STATISTIQUES")
    print("="*70)
    
    # T-tests
    if len(data) == 3:
        methods = list(data.keys())
        
        # Vanilla vs Baseline
        t1, p1 = stats.ttest_ind(data[methods[0]], data[methods[1]])
        print(f"\n{methods[0]} vs {methods[1]}:")
        print(f"  t-statistic: {t1:.3f}")
        print(f"  p-value: {p1:.4f}")
        print(f"  Significatif: {'✅' if p1 < 0.05 else '❌'}")
        
        # Diversity vs Baseline
        t2, p2 = stats.ttest_ind(data[methods[2]], data[methods[1]])
        print(f"\n{methods[2]} vs {methods[1]}:")
        print(f"  t-statistic: {t2:.3f}")
        print(f"  p-value: {p2:.4f}")
        print(f"  Significatif: {'✅' if p2 < 0.05 else '❌'}")
        
        # Diversity vs Vanilla
        t3, p3 = stats.ttest_ind(data[methods[2]], data[methods[0]])
        print(f"\n{methods[2]} vs {methods[0]}:")
        print(f"  t-statistic: {t3:.3f}")
        print(f"  p-value: {p3:.4f}")
        print(f"  Significatif: {'✅' if p3 < 0.05 else '❌'}")
    
    # Visualisations
    create_comparison_plots(data)
    
    print("\n" + "="*70)
    print("✅ Graphiques sauvegardés: comparison_3methods.png")
    print("="*70)

def create_comparison_plots(data):
    """Crée des visualisations comparatives."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comparaison des 3 Méthodes : Co-Evolution Systems', 
                 fontsize=18, fontweight='bold')
    
    colors = {
        'Vanilla Co-evol': '#FF6B6B',
        'Baseline (Random)': '#4ECDC4',
        'Diversity Co-evol': '#95E1D3'
    }
    
    # 1. Learning curves
    ax = axes[0, 0]
    for name, sr_list in data.items():
        epochs = range(1, len(sr_list) + 1)
        ax.plot(epochs, [s*100 for s in sr_list], 
                marker='o', linewidth=2.5, markersize=6,
                label=name, color=colors[name], alpha=0.9)
    
    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=13, fontweight='bold')
    ax.set_title('Learning Curves Comparison', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0, 105])
    
    # 2. Box plots
    ax = axes[0, 1]
    box_data = [[s*100 for s in sr_list] for sr_list in data.values()]
    bp = ax.boxplot(box_data, labels=data.keys(), patch_artist=True)
    
    for patch, name in zip(bp['boxes'], data.keys()):
        patch.set_facecolor(colors[name])
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Success Rate (%)', fontsize=13, fontweight='bold')
    ax.set_title('Distribution Comparison', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    # 3. Bar chart - Final SR
    ax = axes[1, 0]
    final_srs = {name: sr_list[-1]*100 for name, sr_list in data.items()}
    bars = ax.bar(final_srs.keys(), final_srs.values(), 
                   color=[colors[name] for name in final_srs.keys()],
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Final Success Rate (%)', fontsize=13, fontweight='bold')
    ax.set_title('Final Performance Comparison', fontsize=15, fontweight='bold')
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    # 4. Improvement over time
    ax = axes[1, 1]
    for name, sr_list in data.items():
        if len(sr_list) > 1:
            improvement = [(sr_list[i] - sr_list[0])*100 for i in range(len(sr_list))]
            epochs = range(1, len(sr_list) + 1)
            ax.plot(epochs, improvement, 
                    marker='o', linewidth=2.5, markersize=6,
                    label=name, color=colors[name], alpha=0.9)
    
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Improvement from Initial (%)', fontsize=13, fontweight='bold')
    ax.set_title('Learning Progress', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('comparison_3methods.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
