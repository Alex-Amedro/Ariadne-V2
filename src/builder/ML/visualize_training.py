"""
Programme de visualisation des données d'entraînement.

Usage:
    python visualize_training.py --run runs/run_20231111_143022
    python visualize_training.py --compare runs/run_1 runs/run_2 runs/run_3
"""

import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_history(run_dir):
    """Charge l'historique d'un run."""
    history_path = f"{run_dir}/logs/history.json"
    
    if not os.path.exists(history_path):
        print(f"[ERREUR] Fichier d'historique introuvable: {history_path}")
        return None
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    # Charger aussi la config
    config_path = f"{run_dir}/config.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        history['config'] = config
    
    return history


def plot_single_run(run_dir, save_fig=True):
    """Visualise les métriques d'un seul run."""
    history = load_history(run_dir)
    
    if history is None:
        return
    
    run_name = Path(run_dir).name
    
    epochs = history['epochs']
    success_rates = [perf['success_rate'] for perf in history['agent_performance']]
    mean_rewards = [perf['mean_reward'] for perf in history['agent_performance']]
    diversity = history['generator_diversity']
    
    # Créer la figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Co-Evolution Training: {run_name}', fontsize=16)
    
    # 1. Success Rate
    axes[0, 0].plot(epochs, success_rates, 'b-', linewidth=2, marker='o')
    axes[0, 0].axhline(y=0.5, color='r', linestyle='--', label='Target (50%)')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Success Rate')
    axes[0, 0].set_title('Agent Success Rate over Time')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].set_ylim([0, 1])
    
    # 2. Mean Reward
    axes[0, 1].plot(epochs, mean_rewards, 'g-', linewidth=2, marker='s')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Mean Reward')
    axes[0, 1].set_title('Agent Mean Reward over Time')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Generator Diversity
    axes[1, 0].plot(epochs, diversity, 'm-', linewidth=2, marker='^')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Diversity Score')
    axes[1, 0].set_title('Generator Level Diversity')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])
    
    # 4. Distribution des Success Rates
    axes[1, 1].hist(success_rates, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(x=0.5, color='r', linestyle='--', linewidth=2, label='Target')
    axes[1, 1].set_xlabel('Success Rate')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Success Rates')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_fig:
        output_path = f"{run_dir}/logs/training_curves.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Figure sauvegardée: {output_path}")
    
    plt.show()


def compare_runs(run_dirs, metric='success_rate'):
    """Compare plusieurs runs."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(run_dirs)))
    
    for i, run_dir in enumerate(run_dirs):
        history = load_history(run_dir)
        if history is None:
            continue
        
        run_name = Path(run_dir).name
        epochs = history['epochs']
        
        if metric == 'success_rate':
            values = [perf['success_rate'] for perf in history['agent_performance']]
            ylabel = 'Success Rate'
            title = 'Comparison: Success Rate'
        elif metric == 'mean_reward':
            values = [perf['mean_reward'] for perf in history['agent_performance']]
            ylabel = 'Mean Reward'
            title = 'Comparison: Mean Reward'
        elif metric == 'diversity':
            values = history['generator_diversity']
            ylabel = 'Diversity Score'
            title = 'Comparison: Generator Diversity'
        else:
            print(f"[ERREUR] Métrique inconnue: {metric}")
            return
        
        ax.plot(epochs, values, color=colors[i], linewidth=2, marker='o', label=run_name)
    
    if metric == 'success_rate':
        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, label='Target (50%)', alpha=0.7)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()


def print_summary(run_dir):
    """Affiche un résumé des métriques."""
    history = load_history(run_dir)
    
    if history is None:
        return
    
    run_name = Path(run_dir).name
    
    print("\n" + "="*60)
    print(f"RÉSUMÉ: {run_name}")
    print("="*60)
    
    if 'config' in history:
        print("\nConfiguration:")
        config = history['config']
        for key, value in config.items():
            if key != 'created_at':
                print(f"  {key}: {value}")
        if 'created_at' in config:
            print(f"  Créé le: {config['created_at']}")
    
    print("\nStatistiques d'entraînement:")
    print(f"  Nombre d'époques: {len(history['epochs'])}")
    
    success_rates = [perf['success_rate'] for perf in history['agent_performance']]
    mean_rewards = [perf['mean_reward'] for perf in history['agent_performance']]
    diversity = history['generator_diversity']
    
    print(f"\nSuccess Rate:")
    print(f"  Initial: {success_rates[0]*100:.1f}%")
    print(f"  Final: {success_rates[-1]*100:.1f}%")
    print(f"  Best: {max(success_rates)*100:.1f}%")
    print(f"  Mean: {np.mean(success_rates)*100:.1f}%")
    
    print(f"\nMean Reward:")
    print(f"  Initial: {mean_rewards[0]:.3f}")
    print(f"  Final: {mean_rewards[-1]:.3f}")
    print(f"  Best: {max(mean_rewards):.3f}")
    print(f"  Mean: {np.mean(mean_rewards):.3f}")
    
    print(f"\nGenerator Diversity:")
    print(f"  Initial: {diversity[0]:.3f}")
    print(f"  Final: {diversity[-1]:.3f}")
    print(f"  Mean: {np.mean(diversity):.3f}")
    
    print("\n" + "="*60)


def list_available_runs():
    """Liste tous les runs disponibles."""
    runs_dir = "runs"
    
    if not os.path.exists(runs_dir):
        print("[INFO] Aucun dossier 'runs' trouvé")
        return []
    
    runs = []
    for item in os.listdir(runs_dir):
        run_path = os.path.join(runs_dir, item)
        if os.path.isdir(run_path):
            history_path = f"{run_path}/logs/history.json"
            if os.path.exists(history_path):
                runs.append(run_path)
    
    if not runs:
        print("[INFO] Aucun run trouvé dans 'runs/'")
        return []
    
    print("\n" + "="*60)
    print("RUNS DISPONIBLES")
    print("="*60)
    
    for i, run_path in enumerate(runs, 1):
        run_name = Path(run_path).name
        
        # Charger les infos basiques
        history = load_history(run_path)
        if history:
            num_epochs = len(history['epochs'])
            final_sr = history['agent_performance'][-1]['success_rate']
            print(f"{i}. {run_name}")
            print(f"   - Époques: {num_epochs}")
            print(f"   - Success Rate final: {final_sr*100:.1f}%")
    
    print("="*60 + "\n")
    
    return runs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualisation des données d'entraînement")
    parser.add_argument("--run", type=str, help="Chemin vers un run à visualiser")
    parser.add_argument("--compare", nargs='+', help="Comparer plusieurs runs")
    parser.add_argument("--metric", type=str, default='success_rate', 
                       choices=['success_rate', 'mean_reward', 'diversity'],
                       help="Métrique pour la comparaison")
    parser.add_argument("--list", action='store_true', help="Lister tous les runs disponibles")
    parser.add_argument("--summary", action='store_true', help="Afficher juste le résumé")
    
    args = parser.parse_args()
    
    if args.list:
        list_available_runs()
    
    elif args.compare:
        print(f"[INFO] Comparaison de {len(args.compare)} runs...")
        compare_runs(args.compare, metric=args.metric)
    
    elif args.run:
        if args.summary:
            print_summary(args.run)
        else:
            print(f"[INFO] Visualisation de: {args.run}")
            print_summary(args.run)
            plot_single_run(args.run, save_fig=True)
    
    else:
        # Mode interactif
        runs = list_available_runs()
        
        if runs:
            print("Entrez le numéro du run à visualiser (ou 'q' pour quitter): ", end='')
            choice = input()
            
            if choice.lower() != 'q' and choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(runs):
                    print_summary(runs[idx])
                    plot_single_run(runs[idx], save_fig=True)
                else:
                    print("[ERREUR] Numéro invalide")
