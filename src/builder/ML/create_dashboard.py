"""
Crée et affiche un dashboard interactif avec tous les résultats.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle

def create_dashboard(run_dir):
    """Crée un dashboard visuel complet sur une seule figure."""
    
    # Charger données
    history_path = Path(run_dir) / "logs" / "history.json"
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = history['epochs']
    success_rates = [p['success_rate'] for p in history['agent_performance']]
    mean_rewards = [p['mean_reward'] for p in history['agent_performance']]
    diversity = history['generator_diversity']
    
    # Créer figure avec layout custom
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Titre principal
    fig.suptitle(f'CO-EVOLUTION DASHBOARD - {Path(run_dir).name}', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # ===== ROW 1: Métriques principales =====
    
    # 1. Success Rate Timeline
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(epochs, success_rates, 'o-', linewidth=3, markersize=8, color='#2E86AB', alpha=0.8)
    ax1.fill_between(epochs, success_rates, alpha=0.3, color='#2E86AB')
    ax1.axhline(y=0.5, color='#E63946', linestyle='--', linewidth=2, label='Target (50%)')
    ax1.fill_between(epochs, 0.4, 0.6, alpha=0.2, color='#E63946')
    ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Success Rate', fontsize=11, fontweight='bold')
    ax1.set_title('Agent Performance Evolution', fontsize=13, fontweight='bold', pad=10)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Annoter progression
    improvement = success_rates[-1] - success_rates[0]
    ax1.annotate(f'+{improvement:.1%}', 
                xy=(epochs[-1], success_rates[-1]), 
                xytext=(epochs[-1]-2, success_rates[-1]+0.1),
                fontsize=12, fontweight='bold', color='green',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    # 2. KPIs (Key Performance Indicators)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    
    # Calculer KPIs
    final_sr = success_rates[-1]
    best_sr = max(success_rates)
    mean_sr = np.mean(success_rates)
    final_div = diversity[-1]
    
    kpis = [
        ('Final Success Rate', f'{final_sr:.1%}', '#2E86AB'),
        ('Best Success Rate', f'{best_sr:.1%}', '#06A77D'),
        ('Average Success Rate', f'{mean_sr:.1%}', '#F77F00'),
        ('Final Diversity', f'{final_div:.1%}', '#9B5DE5'),
    ]
    
    y_pos = 0.85
    for label, value, color in kpis:
        ax2.text(0.05, y_pos, label, fontsize=11, fontweight='normal', va='center')
        ax2.text(0.95, y_pos, value, fontsize=14, fontweight='bold', va='center', ha='right', color=color)
        y_pos -= 0.22
    
    ax2.set_title('Key Metrics', fontsize=13, fontweight='bold', pad=10, loc='left')
    
    # ===== ROW 2: Analyses détaillées =====
    
    # 3. Distribution Success Rate
    ax3 = fig.add_subplot(gs[1, 0])
    counts, bins, patches = ax3.hist(success_rates, bins=12, alpha=0.7, color='#2E86AB', edgecolor='black', linewidth=1.5)
    # Colorer selon zone
    for i, patch in enumerate(patches):
        if bins[i] >= 0.4 and bins[i] <= 0.6:
            patch.set_facecolor('#06A77D')
    ax3.axvline(x=0.5, color='#E63946', linestyle='--', linewidth=2, label='Target')
    ax3.axvline(x=mean_sr, color='#F77F00', linestyle='--', linewidth=2, label=f'Mean ({mean_sr:.2f})')
    ax3.set_xlabel('Success Rate', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax3.set_title('Success Rate Distribution', fontsize=13, fontweight='bold', pad=10)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Reward Evolution
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(epochs, mean_rewards, 's-', linewidth=2, markersize=6, color='#06A77D', alpha=0.8)
    ax4.fill_between(epochs, mean_rewards, alpha=0.3, color='#06A77D')
    ax4.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Mean Reward', fontsize=11, fontweight='bold')
    ax4.set_title('Reward Progression', fontsize=13, fontweight='bold', pad=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])
    
    # 5. Diversity Timeline
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(epochs, diversity, '^-', linewidth=2, markersize=7, color='#9B5DE5', alpha=0.8)
    ax5.fill_between(epochs, diversity, alpha=0.3, color='#9B5DE5')
    ax5.axhline(y=0.7, color='#F77F00', linestyle='--', linewidth=2, alpha=0.7, label='Good (≥70%)')
    ax5.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Diversity Score', fontsize=11, fontweight='bold')
    ax5.set_title('Generator Diversity', fontsize=13, fontweight='bold', pad=10)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([0, 1])
    
    # ===== ROW 3: Analyses avancées =====
    
    # 6. Convergence vers target
    ax6 = fig.add_subplot(gs[2, 0])
    distances = [abs(sr - 0.5) for sr in success_rates]
    colors_dist = ['#06A77D' if d < 0.1 else '#F77F00' if d < 0.2 else '#E63946' for d in distances]
    ax6.bar(epochs, distances, width=0.8, alpha=0.7, color=colors_dist, edgecolor='black', linewidth=1)
    ax6.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label='Excellent (<0.1)')
    ax6.axhline(y=0.2, color='orange', linestyle='--', alpha=0.5, label='Good (<0.2)')
    ax6.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Distance to Target', fontsize=11, fontweight='bold')
    ax6.set_title('Convergence Quality', fontsize=13, fontweight='bold', pad=10)
    ax6.legend(fontsize=8, loc='upper right')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. Success vs Diversity Scatter
    ax7 = fig.add_subplot(gs[2, 1])
    scatter = ax7.scatter(diversity, success_rates, c=epochs, cmap='plasma', 
                         s=150, alpha=0.8, edgecolors='black', linewidth=1.5)
    # Régression
    z = np.polyfit(diversity, success_rates, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(diversity), max(diversity), 100)
    ax7.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.8)
    ax7.set_xlabel('Diversity', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Success Rate', fontsize=11, fontweight='bold')
    ax7.set_title('Performance vs Diversity', fontsize=13, fontweight='bold', pad=10)
    cbar = plt.colorbar(scatter, ax=ax7)
    cbar.set_label('Epoch', fontsize=10)
    ax7.grid(True, alpha=0.3)
    
    # 8. Summary Stats
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    # Stats textuelles
    stats_text = [
        ('PERFORMANCE', ''),
        ('  Initial SR', f'{success_rates[0]:.1%}'),
        ('  Final SR', f'{success_rates[-1]:.1%}'),
        ('  Improvement', f'+{improvement:.1%}'),
        ('', ''),
        ('STABILITY', ''),
        ('  Std Dev', f'{np.std(success_rates):.3f}'),
        ('  CV', f'{np.std(success_rates)/np.mean(success_rates):.2f}'),
        ('', ''),
        ('GENERATOR', ''),
        ('  Avg Diversity', f'{np.mean(diversity):.1%}'),
        ('  High Div Epochs', f'{sum([1 for d in diversity if d >= 0.7])}/20'),
    ]
    
    y_pos = 0.95
    for label, value in stats_text:
        if label == '':
            y_pos -= 0.05
            continue
        if label in ['PERFORMANCE', 'STABILITY', 'GENERATOR']:
            ax8.text(0.05, y_pos, label, fontsize=12, fontweight='bold', va='top', color='#2E86AB')
        else:
            ax8.text(0.1, y_pos, label, fontsize=10, va='top')
            ax8.text(0.95, y_pos, value, fontsize=10, va='top', ha='right', fontweight='bold')
        y_pos -= 0.07
    
    ax8.set_title('Summary Statistics', fontsize=13, fontweight='bold', pad=10, loc='left')
    
    # Sauvegarder
    output_path = Path(run_dir) / "analysis" / "00_DASHBOARD.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[SAVED] {output_path}")
    
    # Afficher
    plt.show()

def main():
    # Trouver le run le plus récent
    runs_dir = Path("runs")
    latest_run = max(runs_dir.glob("run_*"), key=lambda p: p.stat().st_mtime)
    
    print("="*80)
    print(f"CRÉATION DU DASHBOARD: {latest_run.name}")
    print("="*80)
    
    create_dashboard(latest_run)
    
    print()
    print("Dashboard créé et sauvegardé!")
    print(f"Fichier: {latest_run}/analysis/00_DASHBOARD.png")

if __name__ == "__main__":
    main()
