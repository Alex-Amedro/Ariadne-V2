"""
Test de TRANSFER LEARNING : Évalue l'agent entraîné par co-évolution
sur d'autres environnements MiniGrid standards.
"""

import torch
import numpy as np
from stable_baselines3 import PPO
import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
import json
from pathlib import Path
import matplotlib.pyplot as plt


# Environnements MiniGrid standards pour transfer
TRANSFER_ENVS = {
    'MiniGrid-Empty-5x5-v0': {'name': 'Empty 5x5', 'difficulty': 'Easy'},
    'MiniGrid-Empty-8x8-v0': {'name': 'Empty 8x8', 'difficulty': 'Easy'},
    'MiniGrid-DoorKey-5x5-v0': {'name': 'DoorKey 5x5', 'difficulty': 'Medium'},
    'MiniGrid-DoorKey-6x6-v0': {'name': 'DoorKey 6x6', 'difficulty': 'Medium'},
    'MiniGrid-DoorKey-8x8-v0': {'name': 'DoorKey 8x8', 'difficulty': 'Hard'},
    'MiniGrid-MultiRoom-N2-S4-v0': {'name': 'MultiRoom N2', 'difficulty': 'Medium'},
    'MiniGrid-MultiRoom-N4-S5-v0': {'name': 'MultiRoom N4', 'difficulty': 'Hard'},
}


def load_agent(run_dir, model_name='agent_final.zip'):
    """Charge un agent entraîné."""
    model_path = Path(run_dir) / "models" / model_name
    
    if not model_path.exists():
        model_path = Path(run_dir) / "models" / "agent_best.zip"
    
    agent = PPO.load(model_path)
    print(f"[LOAD] Agent chargé: {model_path}")
    return agent


def evaluate_on_env(agent, env_id, num_episodes=20):
    """Évalue un agent sur un environnement."""
    try:
        env = gym.make(env_id)
        env = FlatObsWrapper(env)
        
        rewards = []
        successes = []
        steps_list = []
        
        for _ in range(num_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            steps = 0
            max_steps = getattr(env.unwrapped, 'max_steps', 1000)
            
            while not done and steps < max_steps:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                steps += 1
                done = terminated or truncated
            
            rewards.append(episode_reward)
            successes.append(1 if episode_reward > 0 else 0)
            steps_list.append(steps)
        
        env.close()
        
        return {
            'success_rate': np.mean(successes),
            'std_success': np.std(successes),
            'mean_reward': np.mean(rewards),
            'mean_steps': np.mean(steps_list),
            'error': None
        }
    
    except Exception as e:
        return {
            'success_rate': 0.0,
            'std_success': 0.0,
            'mean_reward': 0.0,
            'mean_steps': 0.0,
            'error': str(e)
        }


def test_transfer(run_dir, num_episodes=20, save_dir='transfer_results'):
    """Teste le transfer sur tous les environnements."""
    
    print("="*80)
    print("TEST DE TRANSFER LEARNING")
    print("="*80)
    print()
    
    # Charger l'agent
    agent = load_agent(run_dir)
    
    # Créer dossier de sauvegarde
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Tester sur chaque environnement
    results = {}
    
    print("\nÉvaluation sur environnements standards...")
    print("-"*80)
    
    for env_id, env_info in TRANSFER_ENVS.items():
        print(f"\n[{env_info['name']}] ({env_info['difficulty']})")
        print(f"  Env: {env_id}")
        print(f"  Episodes: {num_episodes}")
        
        metrics = evaluate_on_env(agent, env_id, num_episodes)
        
        if metrics['error']:
            print(f"  ❌ ERREUR: {metrics['error']}")
        else:
            print(f"  ✅ Success Rate: {metrics['success_rate']*100:.1f}% ± {metrics['std_success']*100:.1f}%")
            print(f"     Mean Reward: {metrics['mean_reward']:.3f}")
            print(f"     Mean Steps: {metrics['mean_steps']:.1f}")
        
        results[env_id] = {
            'name': env_info['name'],
            'difficulty': env_info['difficulty'],
            **metrics
        }
    
    # Sauvegarder résultats
    with open(save_path / "transfer_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] {save_path / 'transfer_results.json'}")
    
    # Créer visualisations
    create_transfer_plots(results, save_path)
    
    # Rapport
    create_transfer_report(results, save_path, run_dir)
    
    return results


def create_transfer_plots(results, save_path):
    """Crée les graphiques de transfer."""
    
    # Extraire les données
    env_names = []
    success_rates = []
    std_errors = []
    difficulties = []
    colors_map = {'Easy': '#06A77D', 'Medium': '#F77F00', 'Hard': '#E63946'}
    
    for env_id, data in results.items():
        if data['error'] is None:
            env_names.append(data['name'])
            success_rates.append(data['success_rate'])
            std_errors.append(data['std_success'])
            difficulties.append(data['difficulty'])
    
    # Trier par difficulté
    sorted_indices = sorted(range(len(difficulties)), 
                           key=lambda i: (difficulties[i], env_names[i]))
    
    env_names = [env_names[i] for i in sorted_indices]
    success_rates = [success_rates[i] for i in sorted_indices]
    std_errors = [std_errors[i] for i in sorted_indices]
    difficulties = [difficulties[i] for i in sorted_indices]
    
    colors = [colors_map[d] for d in difficulties]
    
    # ===== FIGURE 1: Bar plot avec error bars =====
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('TRANSFER LEARNING RESULTS', fontsize=18, fontweight='bold')
    
    # 1. Success Rates
    ax = axes[0]
    x = np.arange(len(env_names))
    bars = ax.bar(x, success_rates, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.errorbar(x, success_rates, yerr=std_errors, fmt='none', color='black', 
                capsize=5, capthick=2, linewidth=1.5)
    
    # Ligne de référence
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='50% Target')
    
    ax.set_xlabel('Environment', fontsize=12, fontweight='bold')
    ax.set_ylabel('Success Rate', fontsize=12, fontweight='bold')
    ax.set_title('Zero-Shot Performance on Standard MiniGrid Environments', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(env_names, rotation=45, ha='right')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    # Annoter les valeurs
    for i, (bar, sr, diff) in enumerate(zip(bars, success_rates, difficulties)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std_errors[i] + 0.03,
                f'{sr*100:.0f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
        ax.text(bar.get_x() + bar.get_width()/2., -0.08,
                diff, ha='center', va='top', fontsize=9, style='italic',
                transform=ax.get_xaxis_transform())
    
    # 2. Par difficulté (moyennes)
    ax = axes[1]
    
    difficulty_order = ['Easy', 'Medium', 'Hard']
    difficulty_means = []
    difficulty_stds = []
    difficulty_counts = []
    
    for diff in difficulty_order:
        indices = [i for i, d in enumerate(difficulties) if d == diff]
        if indices:
            mean_sr = np.mean([success_rates[i] for i in indices])
            std_sr = np.std([success_rates[i] for i in indices])
            difficulty_means.append(mean_sr)
            difficulty_stds.append(std_sr)
            difficulty_counts.append(len(indices))
        else:
            difficulty_means.append(0)
            difficulty_stds.append(0)
            difficulty_counts.append(0)
    
    x_diff = np.arange(len(difficulty_order))
    bars = ax.bar(x_diff, difficulty_means, 
                   color=[colors_map[d] for d in difficulty_order],
                   alpha=0.7, edgecolor='black', linewidth=2)
    ax.errorbar(x_diff, difficulty_means, yerr=difficulty_stds, 
                fmt='none', color='black', capsize=8, capthick=2, linewidth=2)
    
    ax.set_xlabel('Difficulty Level', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Success Rate', fontsize=12, fontweight='bold')
    ax.set_title('Performance by Difficulty', fontsize=14, fontweight='bold')
    ax.set_xticks(x_diff)
    ax.set_xticklabels([f'{d}\n(n={c})' for d, c in zip(difficulty_order, difficulty_counts)])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Annoter
    for i, (bar, mean, std) in enumerate(zip(bars, difficulty_means, difficulty_stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.03,
                f'{mean*100:.1f}%', ha='center', va='bottom', 
                fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path / "transfer_performance.png", dpi=300, bbox_inches='tight')
    print(f"[SAVED] {save_path / 'transfer_performance.png'}")
    plt.close()


def create_transfer_report(results, save_path, run_dir):
    """Crée le rapport de transfer."""
    
    report = []
    report.append("="*80)
    report.append("RAPPORT DE TRANSFER LEARNING")
    report.append("="*80)
    report.append(f"\nAgent source: {run_dir}")
    report.append("")
    
    # Statistiques globales
    all_sr = [r['success_rate'] for r in results.values() if r['error'] is None]
    
    report.append("STATISTIQUES GLOBALES")
    report.append("-"*80)
    report.append(f"  Nombre d'environnements testés: {len(results)}")
    report.append(f"  Succès: {len(all_sr)}/{len(results)}")
    if all_sr:
        report.append(f"  Success Rate Moyen: {np.mean(all_sr):.3f} ± {np.std(all_sr):.3f}")
        report.append(f"  Meilleur: {np.max(all_sr):.3f}")
        report.append(f"  Pire: {np.min(all_sr):.3f}")
    report.append("")
    
    # Par difficulté
    for difficulty in ['Easy', 'Medium', 'Hard']:
        diff_results = {k: v for k, v in results.items() 
                       if v['difficulty'] == difficulty and v['error'] is None}
        
        if diff_results:
            report.append(f"{difficulty.upper()} ENVIRONMENTS")
            report.append("-"*80)
            
            for env_id, data in diff_results.items():
                report.append(f"  {data['name']} ({env_id})")
                report.append(f"    Success Rate: {data['success_rate']:.3f} ± {data['std_success']:.3f}")
                report.append(f"    Mean Reward: {data['mean_reward']:.3f}")
                report.append(f"    Mean Steps: {data['mean_steps']:.1f}")
            
            sr_list = [d['success_rate'] for d in diff_results.values()]
            report.append(f"  Moyenne {difficulty}: {np.mean(sr_list):.3f}")
            report.append("")
    
    # Analyse
    report.append("ANALYSE")
    report.append("-"*80)
    
    if all_sr:
        if np.mean(all_sr) > 0.5:
            report.append("  ✅ EXCELLENT transfer: L'agent généralise bien (>50% en moyenne)")
        elif np.mean(all_sr) > 0.3:
            report.append("  ✅ BON transfer: L'agent s'adapte correctement (>30% en moyenne)")
        elif np.mean(all_sr) > 0.1:
            report.append("  ⚠️  MOYEN transfer: Certaines compétences se transfèrent")
        else:
            report.append("  ❌ FAIBLE transfer: L'agent ne généralise pas bien")
        
        # Analyse par difficulté
        easy_sr = [r['success_rate'] for r in results.values() 
                  if r['difficulty'] == 'Easy' and r['error'] is None]
        hard_sr = [r['success_rate'] for r in results.values()
                  if r['difficulty'] == 'Hard' and r['error'] is None]
        
        if easy_sr and hard_sr:
            degradation = np.mean(easy_sr) - np.mean(hard_sr)
            report.append(f"  Dégradation Easy→Hard: {degradation:.3f}")
            
            if degradation < 0.2:
                report.append("    → Robustesse EXCELLENTE à la difficulté")
            elif degradation < 0.4:
                report.append("    → Robustesse BONNE")
            else:
                report.append("    → Sensible à la difficulté")
    
    report.append("")
    report.append("="*80)
    
    report_text = '\n'.join(report)
    with open(save_path / "transfer_report.txt", 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"[SAVED] {save_path / 'transfer_report.txt'}")
    
    print()
    print("="*80)
    print("TRANSFER LEARNING TEST TERMINÉ!")
    print("="*80)
    print()
    print(report_text)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test de transfer learning")
    parser.add_argument("--run", type=str, required=True, help="Dossier du run source")
    parser.add_argument("--episodes", type=int, default=20, help="Nombre d'épisodes par env")
    parser.add_argument("--output", type=str, default="transfer_results", help="Dossier de sortie")
    
    args = parser.parse_args()
    
    test_transfer(args.run, args.episodes, args.output)
