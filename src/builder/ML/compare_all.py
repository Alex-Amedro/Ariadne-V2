"""
COMPARAISON GLOBALE : Compare tous les runs ensemble.
Génère un mega-rapport avec tous les résultats.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import seaborn as sns


class GlobalComparison:
    """Compare tous les runs d'entraînement."""
    
    def __init__(self, runs_dict, save_dir='global_comparison'):
        """
        Args:
            runs_dict: Dict {name: run_dir}
                Exemple: {
                    'Co-Evolution': 'runs/run_20251229_035928',
                    'Baseline': 'runs/baseline_XXXXX',
                    'No Reward Shaping': 'runs/ablation_no_reward_shaping_XXXXX'
                }
            save_dir: Dossier de sortie
        """
        self.runs_dict = runs_dict
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.runs_data = {}
        self._load_all_runs()
        
        print(f"[GlobalComp] {len(self.runs_data)} runs chargés")
        print(f"[GlobalComp] Sauvegarde dans: {save_dir}")
    
    def _load_all_runs(self):
        """Charge tous les runs."""
        for name, run_dir in self.runs_dict.items():
            history_path = Path(run_dir) / "logs" / "history.json"
            config_path = Path(run_dir) / "config.json"
            
            if not history_path.exists():
                print(f"[WARNING] {name}: history.json introuvable")
                continue
            
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            config = {}
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
            
            self.runs_data[name] = {
                'history': history,
                'config': config,
                'epochs': history['epochs'],
                'success_rates': [p['success_rate'] for p in history['agent_performance']]
            }
            
            print(f"  ✅ {name}: {len(history['epochs'])} epochs")
    
    def plot_all_success_rates(self):
        """Graphique: toutes les success rates ensemble."""
        fig, ax = plt.subplots(figsize=(16, 10))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.runs_data)))
        
        for (name, data), color in zip(self.runs_data.items(), colors):
            epochs = data['epochs']
            sr = data['success_rates']
            
            ax.plot(epochs, sr, 'o-', linewidth=3, markersize=7,
                   label=name, color=color, alpha=0.8)
            ax.fill_between(epochs, 0, sr, alpha=0.1, color=color)
        
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=2, 
                  alpha=0.5, label='Target (50%)')
        
        ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
        ax.set_ylabel('Success Rate', fontsize=14, fontweight='bold')
        ax.set_title('Success Rate Comparison - All Methods', 
                    fontsize=16, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        save_path = self.save_dir / "all_success_rates.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] {save_path}")
        plt.close()
    
    def plot_final_comparison(self):
        """Bar plot: comparaison des performances finales."""
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle('Final Performance Comparison', fontsize=18, fontweight='bold')
        
        names = list(self.runs_data.keys())
        
        # Métriques
        final_sr = [data['success_rates'][-1] for data in self.runs_data.values()]
        mean_sr = [np.mean(data['success_rates']) for data in self.runs_data.values()]
        best_sr = [np.max(data['success_rates']) for data in self.runs_data.values()]
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
        
        # 1. Final SR
        ax = axes[0]
        bars = ax.bar(range(len(names)), final_sr, color=colors, 
                     alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel('Success Rate', fontsize=12, fontweight='bold')
        ax.set_title('Final Success Rate', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Annoter
        for i, (bar, sr) in enumerate(zip(bars, final_sr)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.03,
                   f'{sr*100:.1f}%', ha='center', va='bottom', 
                   fontweight='bold', fontsize=11)
        
        # 2. Mean SR
        ax = axes[1]
        bars = ax.bar(range(len(names)), mean_sr, color=colors,
                     alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel('Success Rate', fontsize=12, fontweight='bold')
        ax.set_title('Mean Success Rate', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, (bar, sr) in enumerate(zip(bars, mean_sr)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.03,
                   f'{sr*100:.1f}%', ha='center', va='bottom',
                   fontweight='bold', fontsize=11)
        
        # 3. Best SR
        ax = axes[2]
        bars = ax.bar(range(len(names)), best_sr, color=colors,
                     alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel('Success Rate', fontsize=12, fontweight='bold')
        ax.set_title('Best Success Rate', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, (bar, sr) in enumerate(zip(bars, best_sr)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.03,
                   f'{sr*100:.1f}%', ha='center', va='bottom',
                   fontweight='bold', fontsize=11)
        
        plt.tight_layout()
        save_path = self.save_dir / "final_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] {save_path}")
        plt.close()
    
    def plot_statistical_tests(self):
        """Tests statistiques entre toutes les paires."""
        names = list(self.runs_data.keys())
        n = len(names)
        
        # Matrice de p-values
        p_matrix = np.zeros((n, n))
        effect_matrix = np.zeros((n, n))
        
        for i, name1 in enumerate(names):
            for j, name2 in enumerate(names):
                if i == j:
                    p_matrix[i, j] = 1.0
                    effect_matrix[i, j] = 0.0
                else:
                    sr1 = self.runs_data[name1]['success_rates']
                    sr2 = self.runs_data[name2]['success_rates']
                    
                    # t-test
                    t_stat, p_val = stats.ttest_ind(sr1, sr2)
                    p_matrix[i, j] = p_val
                    
                    # Cohen's d
                    pooled_std = np.sqrt((np.std(sr1)**2 + np.std(sr2)**2) / 2)
                    if pooled_std > 0:
                        cohens_d = (np.mean(sr1) - np.mean(sr2)) / pooled_std
                    else:
                        cohens_d = 0.0
                    effect_matrix[i, j] = cohens_d
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle('Statistical Significance Tests', fontsize=18, fontweight='bold')
        
        # 1. P-values
        ax = axes[0]
        im = ax.imshow(p_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=0.1)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_yticklabels(names)
        ax.set_title('P-values (t-test)', fontsize=14, fontweight='bold')
        
        # Annoter
        for i in range(n):
            for j in range(n):
                if i != j:
                    text = f'{p_matrix[i, j]:.3f}'
                    color = 'white' if p_matrix[i, j] < 0.05 else 'black'
                    ax.text(j, i, text, ha='center', va='center', 
                           color=color, fontweight='bold', fontsize=9)
        
        plt.colorbar(im, ax=ax, label='p-value')
        
        # 2. Effect sizes
        ax = axes[1]
        im = ax.imshow(effect_matrix, cmap='RdBu', aspect='auto', vmin=-2, vmax=2)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_yticklabels(names)
        ax.set_title("Cohen's d (Effect Size)", fontsize=14, fontweight='bold')
        
        # Annoter
        for i in range(n):
            for j in range(n):
                if i != j:
                    text = f'{effect_matrix[i, j]:.2f}'
                    color = 'white' if abs(effect_matrix[i, j]) > 0.5 else 'black'
                    ax.text(j, i, text, ha='center', va='center',
                           color=color, fontweight='bold', fontsize=9)
        
        plt.colorbar(im, ax=ax, label="Cohen's d")
        
        plt.tight_layout()
        save_path = self.save_dir / "statistical_tests.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] {save_path}")
        plt.close()
        
        return p_matrix, effect_matrix
    
    def generate_report(self):
        """Génère un rapport textuel complet."""
        report = []
        report.append("="*80)
        report.append("RAPPORT DE COMPARAISON GLOBALE")
        report.append("="*80)
        report.append("")
        
        # Sommaire
        report.append("RUNS COMPARÉS")
        report.append("-"*80)
        for name, data in self.runs_data.items():
            report.append(f"  {name}:")
            report.append(f"    Epochs: {len(data['epochs'])}")
            report.append(f"    Final SR: {data['success_rates'][-1]:.3f}")
            report.append(f"    Mean SR: {np.mean(data['success_rates']):.3f} ± {np.std(data['success_rates']):.3f}")
            report.append(f"    Best SR: {np.max(data['success_rates']):.3f}")
            report.append("")
        
        # Classement
        report.append("CLASSEMENT (par Final SR)")
        report.append("-"*80)
        ranking = sorted(self.runs_data.items(), 
                        key=lambda x: x[1]['success_rates'][-1],
                        reverse=True)
        
        for rank, (name, data) in enumerate(ranking, 1):
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
            report.append(f"  {medal} {name}: {data['success_rates'][-1]*100:.1f}%")
        report.append("")
        
        # Meilleur par critère
        report.append("MEILLEURS PAR CRITÈRE")
        report.append("-"*80)
        
        best_final = max(self.runs_data.items(), 
                        key=lambda x: x[1]['success_rates'][-1])
        report.append(f"  Best Final SR: {best_final[0]} ({best_final[1]['success_rates'][-1]*100:.1f}%)")
        
        best_mean = max(self.runs_data.items(),
                       key=lambda x: np.mean(x[1]['success_rates']))
        report.append(f"  Best Mean SR: {best_mean[0]} ({np.mean(best_mean[1]['success_rates'])*100:.1f}%)")
        
        best_peak = max(self.runs_data.items(),
                       key=lambda x: np.max(x[1]['success_rates']))
        report.append(f"  Best Peak SR: {best_peak[0]} ({np.max(best_peak[1]['success_rates'])*100:.1f}%)")
        
        most_stable = min(self.runs_data.items(),
                         key=lambda x: np.std(x[1]['success_rates']))
        report.append(f"  Most Stable: {most_stable[0]} (std={np.std(most_stable[1]['success_rates']):.3f})")
        
        report.append("")
        report.append("="*80)
        
        report_text = '\n'.join(report)
        
        with open(self.save_dir / "global_report.txt", 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"[SAVED] {self.save_dir / 'global_report.txt'}")
        
        return report_text
    
    def compare_all(self):
        """Lance toutes les comparaisons."""
        print("="*80)
        print("COMPARAISON GLOBALE")
        print("="*80)
        
        self.plot_all_success_rates()
        self.plot_final_comparison()
        p_matrix, effect_matrix = self.plot_statistical_tests()
        report = self.generate_report()
        
        # Sauvegarder stats
        stats_dict = {}
        for name, data in self.runs_data.items():
            stats_dict[name] = {
                'final_sr': float(data['success_rates'][-1]),
                'mean_sr': float(np.mean(data['success_rates'])),
                'std_sr': float(np.std(data['success_rates'])),
                'best_sr': float(np.max(data['success_rates'])),
                'epochs': len(data['epochs'])
            }
        
        with open(self.save_dir / "global_stats.json", 'w') as f:
            json.dump(stats_dict, f, indent=2)
        print(f"[SAVED] {self.save_dir / 'global_stats.json'}")
        
        print("\n" + "="*80)
        print("COMPARAISON GLOBALE TERMINÉE!")
        print("="*80)
        print(f"\nFichiers dans: {self.save_dir}/")
        print()
        print(report)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Comparaison globale de tous les runs")
    parser.add_argument("--runs", type=str, nargs='+', required=True,
                       help="Format: name1:path1 name2:path2 ...")
    parser.add_argument("--output", type=str, default="global_comparison",
                       help="Dossier de sortie")
    
    args = parser.parse_args()
    
    # Parser les runs
    runs_dict = {}
    for run_spec in args.runs:
        name, path = run_spec.split(':', 1)
        runs_dict[name] = path
    
    comparison = GlobalComparison(runs_dict, args.output)
    comparison.compare_all()
