"""
Analyse complète des résultats d'entraînement de co-évolution.
Génère tous les graphiques et statistiques exploitables.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from scipy import stats
from scipy.signal import savgol_filter
import os

# Style matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_run_data(run_dir):
    """Charge les données d'un run."""
    history_path = Path(run_dir) / "logs" / "history.json"
    config_path = Path(run_dir) / "config.json"
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return history, config

def smooth_curve(data, window=5):
    """Lisse une courbe avec filtre Savitzky-Golay."""
    if len(data) < window:
        return data
    return savgol_filter(data, window_length=min(window, len(data)), polyorder=2)

def plot_training_curves(history, save_path):
    """Graphique principal : courbes d'apprentissage."""
    epochs = history['epochs']
    success_rates = [p['success_rate'] for p in history['agent_performance']]
    mean_rewards = [p['mean_reward'] for p in history['agent_performance']]
    diversity = history['generator_diversity']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Success Rate avec tendance
    ax = axes[0, 0]
    ax.plot(epochs, success_rates, 'o-', linewidth=2, markersize=6, alpha=0.6, label='Raw')
    if len(success_rates) >= 5:
        smoothed = smooth_curve(success_rates, window=5)
        ax.plot(epochs, smoothed, '-', linewidth=3, label='Smoothed (SG filter)')
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Target (50%)', alpha=0.7)
    ax.fill_between(epochs, 0.4, 0.6, alpha=0.2, color='red', label='Target zone')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Success Rate', fontsize=12, fontweight='bold')
    ax.set_title('Agent Success Rate Evolution', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # 2. Mean Reward
    ax = axes[0, 1]
    ax.plot(epochs, mean_rewards, 's-', linewidth=2, markersize=6, color='green', alpha=0.6, label='Raw')
    if len(mean_rewards) >= 5:
        smoothed = smooth_curve(mean_rewards, window=5)
        ax.plot(epochs, smoothed, '-', linewidth=3, color='darkgreen', label='Smoothed')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Reward', fontsize=12, fontweight='bold')
    ax.set_title('Agent Mean Reward Evolution', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # 3. Generator Diversity
    ax = axes[1, 0]
    ax.plot(epochs, diversity, '^-', linewidth=2, markersize=6, color='purple', alpha=0.6, label='Raw')
    if len(diversity) >= 5:
        smoothed = smooth_curve(diversity, window=5)
        ax.plot(epochs, smoothed, '-', linewidth=3, color='indigo', label='Smoothed')
    ax.axhline(y=0.7, color='orange', linestyle='--', linewidth=2, label='Good diversity (>70%)', alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Diversity Score', fontsize=12, fontweight='bold')
    ax.set_title('Generator Level Diversity', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # 4. Correlation Success Rate vs Diversity
    ax = axes[1, 1]
    scatter = ax.scatter(diversity, success_rates, c=epochs, cmap='viridis', 
                         s=100, alpha=0.7, edgecolors='black', linewidth=1)
    # Régression linéaire
    z = np.polyfit(diversity, success_rates, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(diversity), max(diversity), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.8, label=f'Linear fit (slope={z[0]:.2f})')
    
    ax.set_xlabel('Generator Diversity', fontsize=12, fontweight='bold')
    ax.set_ylabel('Success Rate', fontsize=12, fontweight='bold')
    ax.set_title('Success Rate vs Diversity Correlation', fontsize=14, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Epoch', fontsize=10)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[SAVED] {save_path}")
    plt.close()

def plot_learning_phases(history, save_path):
    """Identifie et visualise les phases d'apprentissage."""
    epochs = history['epochs']
    success_rates = [p['success_rate'] for p in history['agent_performance']]
    
    # Détecter les phases basées sur le taux de changement
    changes = np.diff(success_rates)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 1. Success Rate avec phases colorées
    ax = axes[0]
    ax.plot(epochs, success_rates, 'o-', linewidth=3, markersize=8, color='blue')
    
    # Identifier exploration (changements importants) vs exploitation (stable)
    threshold = 0.1
    for i in range(len(changes)):
        if abs(changes[i]) > threshold:
            ax.axvspan(epochs[i], epochs[i+1], alpha=0.3, color='orange', label='Exploration' if i == 0 else '')
        else:
            ax.axvspan(epochs[i], epochs[i+1], alpha=0.2, color='green', label='Exploitation' if i == 0 else '')
    
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Target')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Success Rate', fontsize=12, fontweight='bold')
    ax.set_title('Learning Phases: Exploration vs Exploitation', fontsize=14, fontweight='bold')
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # 2. Gradient de changement
    ax = axes[1]
    ax.bar(epochs[1:], changes, width=0.6, alpha=0.7, color=['green' if abs(c) <= threshold else 'orange' for c in changes])
    ax.axhline(y=0, color='black', linewidth=1)
    ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.5, label=f'Threshold (±{threshold})')
    ax.axhline(y=-threshold, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Change in Success Rate', fontsize=12, fontweight='bold')
    ax.set_title('Learning Rate: Epoch-to-Epoch Changes', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[SAVED] {save_path}")
    plt.close()

def plot_convergence_analysis(history, save_path):
    """Analyse de convergence et stabilité."""
    epochs = history['epochs']
    success_rates = [p['success_rate'] for p in history['agent_performance']]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Convergence vers la cible (50%)
    ax = axes[0, 0]
    target = 0.5
    distances = [abs(sr - target) for sr in success_rates]
    ax.plot(epochs, distances, 'o-', linewidth=2, markersize=6, color='red')
    ax.fill_between(epochs, 0, distances, alpha=0.3, color='red')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Distance to Target (|SR - 0.5|)', fontsize=12, fontweight='bold')
    ax.set_title('Convergence to Target Success Rate', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 2. Variance glissante (stabilité)
    ax = axes[0, 1]
    window = 5
    rolling_std = []
    for i in range(len(success_rates)):
        start = max(0, i - window + 1)
        window_data = success_rates[start:i+1]
        rolling_std.append(np.std(window_data))
    
    ax.plot(epochs, rolling_std, 's-', linewidth=2, markersize=6, color='purple')
    ax.fill_between(epochs, 0, rolling_std, alpha=0.3, color='purple')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Rolling Std (window={window})', fontsize=12, fontweight='bold')
    ax.set_title('Training Stability Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 3. Distribution des Success Rates
    ax = axes[1, 0]
    ax.hist(success_rates, bins=15, alpha=0.7, color='skyblue', edgecolor='black', linewidth=1.5)
    ax.axvline(x=target, color='red', linestyle='--', linewidth=3, label='Target (50%)')
    ax.axvline(x=np.mean(success_rates), color='green', linestyle='--', linewidth=2, label=f'Mean ({np.mean(success_rates):.2f})')
    ax.axvline(x=np.median(success_rates), color='orange', linestyle='--', linewidth=2, label=f'Median ({np.median(success_rates):.2f})')
    ax.set_xlabel('Success Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Success Rates', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Cumulative average (tendance long terme)
    ax = axes[1, 1]
    cumulative_avg = np.cumsum(success_rates) / np.arange(1, len(success_rates) + 1)
    ax.plot(epochs, cumulative_avg, '-', linewidth=3, color='teal', label='Cumulative Average')
    ax.plot(epochs, success_rates, 'o', alpha=0.4, markersize=4, color='gray', label='Individual Epochs')
    ax.axhline(y=target, color='red', linestyle='--', linewidth=2, label='Target')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Success Rate', fontsize=12, fontweight='bold')
    ax.set_title('Cumulative Average Success Rate', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[SAVED] {save_path}")
    plt.close()

def plot_generator_analysis(history, save_path):
    """Analyse spécifique du générateur."""
    epochs = history['epochs']
    diversity = history['generator_diversity']
    success_rates = [p['success_rate'] for p in history['agent_performance']]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Évolution de la diversité
    ax = axes[0, 0]
    ax.plot(epochs, diversity, '^-', linewidth=2, markersize=8, color='purple')
    ax.fill_between(epochs, diversity, alpha=0.3, color='purple')
    ax.axhline(y=np.mean(diversity), color='orange', linestyle='--', linewidth=2, label=f'Mean ({np.mean(diversity):.3f})')
    ax.axhline(y=0.7, color='green', linestyle='--', linewidth=2, label='Good threshold (0.7)', alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Diversity Score', fontsize=12, fontweight='bold')
    ax.set_title('Generator Diversity Evolution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # 2. Distribution de la diversité
    ax = axes[0, 1]
    ax.hist(diversity, bins=10, alpha=0.7, color='mediumpurple', edgecolor='black', linewidth=1.5)
    ax.axvline(x=np.mean(diversity), color='red', linestyle='--', linewidth=2, label=f'Mean ({np.mean(diversity):.3f})')
    ax.axvline(x=np.median(diversity), color='green', linestyle='--', linewidth=2, label=f'Median ({np.median(diversity):.3f})')
    ax.set_xlabel('Diversity Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Diversity Score Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Heatmap de corrélation
    ax = axes[1, 0]
    corr, p_value = stats.pearsonr(diversity, success_rates)
    scatter = ax.scatter(diversity, success_rates, c=epochs, cmap='plasma', s=120, alpha=0.8, edgecolors='black', linewidth=1)
    ax.set_xlabel('Diversity', fontsize=12, fontweight='bold')
    ax.set_ylabel('Success Rate', fontsize=12, fontweight='bold')
    ax.set_title(f'Diversity vs Success Rate\nPearson r={corr:.3f}, p={p_value:.3f}', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Epoch')
    ax.grid(True, alpha=0.3)
    
    # 4. Autocorrélation de la diversité (mémoire du générateur)
    ax = axes[1, 1]
    lags = range(1, min(10, len(diversity)))
    autocorr = [np.corrcoef(diversity[:-lag], diversity[lag:])[0, 1] for lag in lags]
    ax.bar(lags, autocorr, width=0.6, alpha=0.7, color='teal', edgecolor='black')
    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_xlabel('Lag (epochs)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Autocorrelation', fontsize=12, fontweight='bold')
    ax.set_title('Generator Memory: Diversity Autocorrelation', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[SAVED] {save_path}")
    plt.close()

def compute_statistics(history):
    """Calcule toutes les statistiques importantes."""
    success_rates = [p['success_rate'] for p in history['agent_performance']]
    mean_rewards = [p['mean_reward'] for p in history['agent_performance']]
    diversity = history['generator_diversity']
    
    stats_dict = {
        'success_rate': {
            'mean': np.mean(success_rates),
            'std': np.std(success_rates),
            'min': np.min(success_rates),
            'max': np.max(success_rates),
            'median': np.median(success_rates),
            'q1': np.percentile(success_rates, 25),
            'q3': np.percentile(success_rates, 75),
            'final': success_rates[-1],
            'best': np.max(success_rates),
            'improvement': success_rates[-1] - success_rates[0],
        },
        'mean_reward': {
            'mean': np.mean(mean_rewards),
            'std': np.std(mean_rewards),
            'min': np.min(mean_rewards),
            'max': np.max(mean_rewards),
            'final': mean_rewards[-1],
            'improvement': mean_rewards[-1] - mean_rewards[0],
        },
        'diversity': {
            'mean': np.mean(diversity),
            'std': np.std(diversity),
            'min': np.min(diversity),
            'max': np.max(diversity),
            'median': np.median(diversity),
            'final': diversity[-1],
        },
        'correlation': {
            'diversity_vs_success': stats.pearsonr(diversity, success_rates)[0],
            'diversity_vs_success_pvalue': stats.pearsonr(diversity, success_rates)[1],
        },
        'convergence': {
            'distance_to_target_mean': np.mean([abs(sr - 0.5) for sr in success_rates]),
            'distance_to_target_final': abs(success_rates[-1] - 0.5),
            'epochs_above_40pct': sum([1 for sr in success_rates if sr >= 0.4]),
            'epochs_in_target_zone': sum([1 for sr in success_rates if 0.4 <= sr <= 0.6]),
        }
    }
    
    return stats_dict

def generate_report(history, config, stats, save_path):
    """Génère un rapport texte détaillé."""
    report = []
    report.append("="*80)
    report.append("RAPPORT D'ANALYSE - CO-ÉVOLUTION AGENT-GÉNÉRATEUR")
    report.append("="*80)
    report.append("")
    
    # Configuration
    report.append("CONFIGURATION DE L'ENTRAÎNEMENT")
    report.append("-"*80)
    report.append(f"  Époques totales : {len(history['epochs'])}")
    report.append(f"  Timesteps/epoch : {config['agent_timesteps_per_epoch']}")
    report.append(f"  Batch size : {config['batch_size']}")
    report.append(f"  Target success rate : {config['target_success_rate']}")
    report.append(f"  Date de création : {config.get('created_at', 'N/A')}")
    report.append("")
    
    # Success Rate
    report.append("SUCCESS RATE (Agent)")
    report.append("-"*80)
    report.append(f"  Mean : {stats['success_rate']['mean']:.3f} ± {stats['success_rate']['std']:.3f}")
    report.append(f"  Median : {stats['success_rate']['median']:.3f}")
    report.append(f"  Range : [{stats['success_rate']['min']:.3f}, {stats['success_rate']['max']:.3f}]")
    report.append(f"  Q1-Q3 : [{stats['success_rate']['q1']:.3f}, {stats['success_rate']['q3']:.3f}]")
    report.append(f"  Initial : {stats['success_rate']['mean'] - stats['success_rate']['improvement']:.3f}")
    report.append(f"  Final : {stats['success_rate']['final']:.3f}")
    report.append(f"  Best : {stats['success_rate']['best']:.3f}")
    report.append(f"  Improvement : {stats['success_rate']['improvement']:+.3f}")
    report.append("")
    
    # Reward
    report.append("MEAN REWARD (Agent)")
    report.append("-"*80)
    report.append(f"  Mean : {stats['mean_reward']['mean']:.3f} ± {stats['mean_reward']['std']:.3f}")
    report.append(f"  Range : [{stats['mean_reward']['min']:.3f}, {stats['mean_reward']['max']:.3f}]")
    report.append(f"  Final : {stats['mean_reward']['final']:.3f}")
    report.append(f"  Improvement : {stats['mean_reward']['improvement']:+.3f}")
    report.append("")
    
    # Diversity
    report.append("DIVERSITY (Générateur)")
    report.append("-"*80)
    report.append(f"  Mean : {stats['diversity']['mean']:.3f} ± {stats['diversity']['std']:.3f}")
    report.append(f"  Median : {stats['diversity']['median']:.3f}")
    report.append(f"  Range : [{stats['diversity']['min']:.3f}, {stats['diversity']['max']:.3f}]")
    report.append(f"  Final : {stats['diversity']['final']:.3f}")
    high_div = sum([1 for d in history['generator_diversity'] if d >= 0.7])
    report.append(f"  Epochs avec diversité ≥70% : {high_div}/{len(history['epochs'])} ({100*high_div/len(history['epochs']):.1f}%)")
    report.append("")
    
    # Corrélations
    report.append("CORRÉLATIONS")
    report.append("-"*80)
    report.append(f"  Diversity vs Success Rate : r={stats['correlation']['diversity_vs_success']:.3f}, p={stats['correlation']['diversity_vs_success_pvalue']:.4f}")
    if stats['correlation']['diversity_vs_success_pvalue'] < 0.05:
        report.append(f"    → Corrélation statistiquement significative (p < 0.05)")
    else:
        report.append(f"    → Corrélation non significative (p ≥ 0.05)")
    report.append("")
    
    # Convergence
    report.append("CONVERGENCE VERS LA CIBLE (50%)")
    report.append("-"*80)
    report.append(f"  Distance moyenne à la cible : {stats['convergence']['distance_to_target_mean']:.3f}")
    report.append(f"  Distance finale à la cible : {stats['convergence']['distance_to_target_final']:.3f}")
    report.append(f"  Époques avec SR ≥ 40% : {stats['convergence']['epochs_above_40pct']}/{len(history['epochs'])}")
    report.append(f"  Époques dans zone cible (40-60%) : {stats['convergence']['epochs_in_target_zone']}/{len(history['epochs'])}")
    report.append("")
    
    # Analyse qualitative
    report.append("ANALYSE QUALITATIVE")
    report.append("-"*80)
    
    # Performance générale
    final_sr = stats['success_rate']['final']
    if final_sr >= 0.7:
        perf = "EXCELLENTE"
    elif final_sr >= 0.5:
        perf = "BONNE"
    elif final_sr >= 0.3:
        perf = "MOYENNE"
    else:
        perf = "FAIBLE"
    report.append(f"  Performance finale : {perf} ({final_sr:.1%})")
    
    # Tendance
    if stats['success_rate']['improvement'] > 0.2:
        trend = "Forte progression"
    elif stats['success_rate']['improvement'] > 0:
        trend = "Progression modérée"
    elif stats['success_rate']['improvement'] > -0.1:
        trend = "Stable"
    else:
        trend = "Régression"
    report.append(f"  Tendance : {trend}")
    
    # Stabilité
    if stats['success_rate']['std'] < 0.1:
        stability = "Très stable"
    elif stats['success_rate']['std'] < 0.2:
        stability = "Stable"
    else:
        stability = "Variable"
    report.append(f"  Stabilité : {stability} (std={stats['success_rate']['std']:.3f})")
    
    # Diversité
    if stats['diversity']['mean'] >= 0.75:
        div_quality = "EXCELLENTE"
    elif stats['diversity']['mean'] >= 0.6:
        div_quality = "BONNE"
    else:
        div_quality = "MOYENNE"
    report.append(f"  Qualité de la diversité : {div_quality} (mean={stats['diversity']['mean']:.3f})")
    
    report.append("")
    report.append("="*80)
    
    # Sauvegarder
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"[SAVED] {save_path}")
    return '\n'.join(report)

def main():
    # Trouver le run le plus récent
    runs_dir = Path("runs")
    latest_run = max(runs_dir.glob("run_*"), key=lambda p: p.stat().st_mtime)
    
    print("="*80)
    print(f"ANALYSE DU RUN: {latest_run.name}")
    print("="*80)
    print()
    
    # Charger les données
    history, config = load_run_data(latest_run)
    
    # Créer dossier d'analyse
    analysis_dir = latest_run / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    
    # Générer tous les graphiques
    print("Génération des graphiques...")
    print("-"*80)
    
    plot_training_curves(history, analysis_dir / "01_training_curves.png")
    plot_learning_phases(history, analysis_dir / "02_learning_phases.png")
    plot_convergence_analysis(history, analysis_dir / "03_convergence.png")
    plot_generator_analysis(history, analysis_dir / "04_generator_analysis.png")
    
    # Calculer statistiques
    print()
    print("Calcul des statistiques...")
    print("-"*80)
    stats = compute_statistics(history)
    
    # Générer rapport
    report = generate_report(history, config, stats, analysis_dir / "REPORT.txt")
    
    # Sauvegarder stats en JSON
    with open(analysis_dir / "statistics.json", 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"[SAVED] {analysis_dir / 'statistics.json'}")
    
    print()
    print("="*80)
    print("ANALYSE TERMINÉE!")
    print("="*80)
    print()
    print(f"Tous les fichiers sauvegardés dans : {analysis_dir}")
    print()
    print("Fichiers générés:")
    print(f"  - 01_training_curves.png : Courbes d'apprentissage principales")
    print(f"  - 02_learning_phases.png : Phases exploration/exploitation")
    print(f"  - 03_convergence.png : Analyse de convergence et stabilité")
    print(f"  - 04_generator_analysis.png : Analyse du générateur")
    print(f"  - REPORT.txt : Rapport textuel détaillé")
    print(f"  - statistics.json : Statistiques en format JSON")
    print()
    
    # Afficher résumé
    print("RÉSUMÉ RAPIDE:")
    print("-"*80)
    print(report)

if __name__ == "__main__":
    main()
