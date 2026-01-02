"""
Compare les résultats entre co-évolution et baseline.
Génère des graphiques de comparaison pour le papier.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from scipy import stats


def load_run(run_dir):
    """Charge un run (co-évolution ou baseline)."""
    history_path = Path(run_dir) / "logs" / "history.json"
    config_path = Path(run_dir) / "config.json"
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return history, config


def compare_performance(coevol_dir, baseline_dir, save_dir):
    """Compare les performances et génère les graphiques."""
    
    # Charger les données
    coevol_hist, coevol_conf = load_run(coevol_dir)
    baseline_hist, baseline_conf = load_run(baseline_dir)
    
    # Extraire les métriques
    coevol_epochs = coevol_hist['epochs']
    coevol_sr = [p['success_rate'] for p in coevol_hist['agent_performance']]
    coevol_reward = [p['mean_reward'] for p in coevol_hist['agent_performance']]
    
    baseline_epochs = baseline_hist['epochs']
    baseline_sr = [p['success_rate'] for p in baseline_hist['agent_performance']]
    baseline_reward = [p['mean_reward'] for p in baseline_hist['agent_performance']]
    
    # Créer le dossier de sauvegarde
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # ===== FIGURE 1: Comparaison Success Rate =====
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('CO-EVOLUTION vs BASELINE COMPARISON', fontsize=18, fontweight='bold')
    
    # 1. Success Rate Evolution
    ax = axes[0, 0]
    ax.plot(coevol_epochs, coevol_sr, 'o-', linewidth=3, markersize=7, 
            color='#2E86AB', label='Co-Evolution', alpha=0.8)
    ax.plot(baseline_epochs, baseline_sr, 's-', linewidth=3, markersize=7,
            color='#E63946', label='Baseline (Random)', alpha=0.8)
    ax.fill_between(coevol_epochs, coevol_sr, alpha=0.2, color='#2E86AB')
    ax.fill_between(baseline_epochs, baseline_sr, alpha=0.2, color='#E63946')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Target (50%)')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Success Rate', fontsize=12, fontweight='bold')
    ax.set_title('Success Rate Over Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Annoter la différence finale
    diff_final = coevol_sr[-1] - baseline_sr[-1]
    color = 'green' if diff_final > 0 else 'red'
    ax.text(0.98, 0.02, f'Δ Final: {diff_final:+.1%}', 
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            color=color, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. Reward Evolution
    ax = axes[0, 1]
    ax.plot(coevol_epochs, coevol_reward, 'o-', linewidth=3, markersize=7,
            color='#06A77D', label='Co-Evolution', alpha=0.8)
    ax.plot(baseline_epochs, baseline_reward, 's-', linewidth=3, markersize=7,
            color='#F77F00', label='Baseline (Random)', alpha=0.8)
    ax.fill_between(coevol_epochs, coevol_reward, alpha=0.2, color='#06A77D')
    ax.fill_between(baseline_epochs, baseline_reward, alpha=0.2, color='#F77F00')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Reward', fontsize=12, fontweight='bold')
    ax.set_title('Reward Progression', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # 3. Distribution Comparison
    ax = axes[1, 0]
    ax.hist(coevol_sr, bins=15, alpha=0.6, color='#2E86AB', 
            edgecolor='black', linewidth=1.5, label='Co-Evolution')
    ax.hist(baseline_sr, bins=15, alpha=0.6, color='#E63946',
            edgecolor='black', linewidth=1.5, label='Baseline')
    ax.axvline(x=np.mean(coevol_sr), color='#2E86AB', linestyle='--', 
               linewidth=2, label=f'CoEvol Mean ({np.mean(coevol_sr):.2f})')
    ax.axvline(x=np.mean(baseline_sr), color='#E63946', linestyle='--',
               linewidth=2, label=f'Baseline Mean ({np.mean(baseline_sr):.2f})')
    ax.set_xlabel('Success Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Success Rate Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Statistical Comparison (Box Plot)
    ax = axes[1, 1]
    data_to_plot = [coevol_sr, baseline_sr]
    bp = ax.boxplot(data_to_plot, labels=['Co-Evolution', 'Baseline'],
                    patch_artist=True, widths=0.6)
    colors = ['#2E86AB', '#E63946']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Test statistique
    t_stat, p_value = stats.ttest_ind(coevol_sr, baseline_sr)
    
    ax.set_ylabel('Success Rate', fontsize=12, fontweight='bold')
    ax.set_title('Statistical Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Annoter le test statistique
    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
    ax.text(0.5, 0.98, f't-test: p={p_value:.4f} {significance}',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=11, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(save_path / "comparison_main.png", dpi=300, bbox_inches='tight')
    print(f"[SAVED] {save_path / 'comparison_main.png'}")
    plt.close()
    
    # ===== FIGURE 2: Amélioration relative =====
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('IMPROVEMENT ANALYSIS', fontsize=18, fontweight='bold')
    
    # 1. Amélioration cumulée
    ax = axes[0]
    coevol_improvement = np.array(coevol_sr) - coevol_sr[0]
    baseline_improvement = np.array(baseline_sr) - baseline_sr[0]
    
    ax.plot(coevol_epochs, coevol_improvement, 'o-', linewidth=3, markersize=7,
            color='#2E86AB', label='Co-Evolution')
    ax.plot(baseline_epochs, baseline_improvement, 's-', linewidth=3, markersize=7,
            color='#E63946', label='Baseline')
    ax.fill_between(coevol_epochs, 0, coevol_improvement, alpha=0.2, color='#2E86AB')
    ax.fill_between(baseline_epochs, 0, baseline_improvement, alpha=0.2, color='#E63946')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Improvement from Initial', fontsize=12, fontweight='bold')
    ax.set_title('Cumulative Improvement', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 2. Taux d'apprentissage (dérivée)
    ax = axes[1]
    coevol_learning_rate = np.diff(coevol_sr)
    baseline_learning_rate = np.diff(baseline_sr)
    
    ax.bar(np.array(coevol_epochs[1:]) - 0.2, coevol_learning_rate, width=0.4,
           alpha=0.7, color='#2E86AB', label='Co-Evolution', edgecolor='black')
    ax.bar(np.array(baseline_epochs[1:]) + 0.2, baseline_learning_rate, width=0.4,
           alpha=0.7, color='#E63946', label='Baseline', edgecolor='black')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Change in Success Rate', fontsize=12, fontweight='bold')
    ax.set_title('Learning Rate (Epoch-to-Epoch)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path / "comparison_improvement.png", dpi=300, bbox_inches='tight')
    print(f"[SAVED] {save_path / 'comparison_improvement.png'}")
    plt.close()
    
    # ===== STATISTIQUES =====
    stats_dict = {
        'co_evolution': {
            'mean_sr': float(np.mean(coevol_sr)),
            'std_sr': float(np.std(coevol_sr)),
            'final_sr': float(coevol_sr[-1]),
            'best_sr': float(np.max(coevol_sr)),
            'improvement': float(coevol_sr[-1] - coevol_sr[0]),
            'mean_reward': float(np.mean(coevol_reward)),
        },
        'baseline': {
            'mean_sr': float(np.mean(baseline_sr)),
            'std_sr': float(np.std(baseline_sr)),
            'final_sr': float(baseline_sr[-1]),
            'best_sr': float(np.max(baseline_sr)),
            'improvement': float(baseline_sr[-1] - baseline_sr[0]),
            'mean_reward': float(np.mean(baseline_reward)),
        },
        'comparison': {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05),
            'effect_size_cohen_d': float((np.mean(coevol_sr) - np.mean(baseline_sr)) / 
                                        np.sqrt((np.std(coevol_sr)**2 + np.std(baseline_sr)**2) / 2)),
            'mean_difference': float(np.mean(coevol_sr) - np.mean(baseline_sr)),
            'final_difference': float(coevol_sr[-1] - baseline_sr[-1]),
        }
    }
    
    with open(save_path / "comparison_stats.json", 'w') as f:
        json.dump(stats_dict, f, indent=2)
    print(f"[SAVED] {save_path / 'comparison_stats.json'}")
    
    # ===== RAPPORT TEXTE =====
    report = []
    report.append("="*80)
    report.append("RAPPORT DE COMPARAISON CO-EVOLUTION vs BASELINE")
    report.append("="*80)
    report.append("")
    
    report.append("CO-EVOLUTION")
    report.append("-"*80)
    report.append(f"  Success Rate Moyen : {stats_dict['co_evolution']['mean_sr']:.3f} ± {stats_dict['co_evolution']['std_sr']:.3f}")
    report.append(f"  Success Rate Final : {stats_dict['co_evolution']['final_sr']:.3f}")
    report.append(f"  Meilleur Score : {stats_dict['co_evolution']['best_sr']:.3f}")
    report.append(f"  Amélioration : {stats_dict['co_evolution']['improvement']:+.3f}")
    report.append("")
    
    report.append("BASELINE (Random Levels)")
    report.append("-"*80)
    report.append(f"  Success Rate Moyen : {stats_dict['baseline']['mean_sr']:.3f} ± {stats_dict['baseline']['std_sr']:.3f}")
    report.append(f"  Success Rate Final : {stats_dict['baseline']['final_sr']:.3f}")
    report.append(f"  Meilleur Score : {stats_dict['baseline']['best_sr']:.3f}")
    report.append(f"  Amélioration : {stats_dict['baseline']['improvement']:+.3f}")
    report.append("")
    
    report.append("COMPARAISON")
    report.append("-"*80)
    report.append(f"  Différence Moyenne : {stats_dict['comparison']['mean_difference']:+.3f}")
    report.append(f"  Différence Finale : {stats_dict['comparison']['final_difference']:+.3f}")
    report.append(f"  Effect Size (Cohen's d) : {stats_dict['comparison']['effect_size_cohen_d']:.3f}")
    report.append(f"  Test-t : t={stats_dict['comparison']['t_statistic']:.3f}, p={stats_dict['comparison']['p_value']:.4f}")
    
    if stats_dict['comparison']['significant']:
        report.append(f"  ✅ Différence statistiquement SIGNIFICATIVE (p < 0.05)")
    else:
        report.append(f"  ❌ Différence NON significative (p ≥ 0.05)")
    report.append("")
    
    # Interprétation Cohen's d
    d = abs(stats_dict['comparison']['effect_size_cohen_d'])
    if d < 0.2:
        effect = "NÉGLIGEABLE"
    elif d < 0.5:
        effect = "PETIT"
    elif d < 0.8:
        effect = "MOYEN"
    else:
        effect = "LARGE"
    report.append(f"  Effect Size Interprétation : {effect}")
    report.append("")
    
    report.append("CONCLUSION")
    report.append("-"*80)
    
    if stats_dict['comparison']['mean_difference'] > 0.1 and stats_dict['comparison']['significant']:
        report.append("  🎯 La CO-ÉVOLUTION est SIGNIFICATIVEMENT MEILLEURE que la baseline.")
        report.append(f"     L'agent co-évolué performe {stats_dict['comparison']['mean_difference']:.1%} mieux en moyenne.")
    elif stats_dict['comparison']['mean_difference'] > 0:
        report.append("  ⚠️  La co-évolution semble meilleure, mais la différence n'est pas statistiquement significative.")
    else:
        report.append("  ❌ La baseline performe aussi bien ou mieux que la co-évolution.")
        report.append("     Le système de co-évolution n'apporte pas d'avantage mesurable.")
    
    report.append("")
    report.append("="*80)
    
    report_text = '\n'.join(report)
    with open(save_path / "comparison_report.txt", 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"[SAVED] {save_path / 'comparison_report.txt'}")
    
    return report_text


def main():
    parser = argparse.ArgumentParser(description="Compare co-évolution et baseline")
    parser.add_argument("--coevol", type=str, required=True, help="Dossier du run co-évolution")
    parser.add_argument("--baseline", type=str, required=True, help="Dossier du run baseline")
    parser.add_argument("--output", type=str, default="comparison", help="Dossier de sortie")
    
    args = parser.parse_args()
    
    print("="*80)
    print("COMPARAISON CO-EVOLUTION vs BASELINE")
    print("="*80)
    print(f"\nCo-Evolution: {args.coevol}")
    print(f"Baseline: {args.baseline}")
    print(f"Output: {args.output}")
    print()
    
    report = compare_performance(args.coevol, args.baseline, args.output)
    
    print("\n" + "="*80)
    print("COMPARAISON TERMINÉE!")
    print("="*80)
    print()
    print("Fichiers générés:")
    print(f"  - comparison_main.png")
    print(f"  - comparison_improvement.png")
    print(f"  - comparison_stats.json")
    print(f"  - comparison_report.txt")
    print()
    print("RÉSUMÉ:")
    print("-"*80)
    print(report)


if __name__ == "__main__":
    main()
