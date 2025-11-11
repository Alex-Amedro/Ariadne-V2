"""
Test 4 : Tester l'agent sur différents environnements MiniGrid
Pour voir comment il généralise
"""

import gymnasium as gym
import minigrid
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt

print("=" * 50)
print("TEST 4 : Test sur différents environnements")
print("=" * 50)

# Charger le modèle
print("\n[1] Chargement du modèle...")
try:
    model = PPO.load("ppo_minigrid_baseline")
    print("✅ Modèle chargé!")
except Exception as e:
    print(f"❌ Erreur: {e}")
    print("   Lance d'abord test_ppo_baseline.py!")
    exit(1)

# Liste d'environnements à tester
environments = [
    "MiniGrid-Empty-5x5-v0",
    "MiniGrid-Empty-8x8-v0",
    "MiniGrid-Empty-16x16-v0",
    "MiniGrid-DoorKey-5x5-v0",
    "MiniGrid-DoorKey-8x8-v0",
    "MiniGrid-MultiRoom-N2-S4-v0",
]

print("\n[2] Test sur chaque environnement (10 épisodes par env)...")

results = {}

for env_name in environments:
    print(f"\n  Testing {env_name}...")
    
    try:
        env = gym.make(env_name, render_mode="rgb_array")
        env = FlatObsWrapper(env)
        
        rewards = []
        success = []
        steps_list = []
        
        for episode in range(10):
            obs, info = env.reset()
            done = False
            episode_reward = 0
            steps = 0
            
            while not done and steps < 200:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated
                steps += 1
            
            rewards.append(episode_reward)
            success.append(1 if episode_reward > 0 else 0)
            steps_list.append(steps)
        
        env.close()
        
        results[env_name] = {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'success_rate': np.mean(success) * 100,
            'mean_steps': np.mean(steps_list),
        }
        
        print(f"    Reward: {results[env_name]['mean_reward']:.2f} ± {results[env_name]['std_reward']:.2f}")
        print(f"    Success: {results[env_name]['success_rate']:.1f}%")
        print(f"    Steps: {results[env_name]['mean_steps']:.1f}")
        
    except Exception as e:
        print(f"    ❌ Erreur: {e}")
        results[env_name] = None

# Créer un graphique des résultats
print("\n[3] Création des visualisations...")

valid_results = {k: v for k, v in results.items() if v is not None}

if len(valid_results) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Graph 1: Success rate
    env_names = [k.replace('MiniGrid-', '').replace('-v0', '') for k in valid_results.keys()]
    success_rates = [v['success_rate'] for v in valid_results.values()]
    
    axes[0].bar(range(len(env_names)), success_rates, color='steelblue')
    axes[0].set_xlabel('Environment')
    axes[0].set_ylabel('Success Rate (%)')
    axes[0].set_title('Success Rate par Environnement')
    axes[0].set_xticks(range(len(env_names)))
    axes[0].set_xticklabels(env_names, rotation=45, ha='right')
    axes[0].axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% threshold')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Graph 2: Mean reward
    mean_rewards = [v['mean_reward'] for v in valid_results.values()]
    std_rewards = [v['std_reward'] for v in valid_results.values()]
    
    axes[1].bar(range(len(env_names)), mean_rewards, yerr=std_rewards, 
                color='coral', capsize=5)
    axes[1].set_xlabel('Environment')
    axes[1].set_ylabel('Mean Reward')
    axes[1].set_title('Reward Moyen par Environnement')
    axes[1].set_xticks(range(len(env_names)))
    axes[1].set_xticklabels(env_names, rotation=45, ha='right')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('generalization_results.png', dpi=150, bbox_inches='tight')
    print("✅ Graphique sauvegardé: generalization_results.png")
    plt.close()

print("\n" + "=" * 50)
print("✅ TESTS TERMINÉS!")
print("=" * 50)

# Résumé
print("\n[RÉSUMÉ]")
for env_name, result in results.items():
    if result:
        env_short = env_name.replace('MiniGrid-', '').replace('-v0', '')
        print(f"  {env_short:30s}: {result['success_rate']:5.1f}% success")

print("\n💡 Observations:")
print("  - Si l'agent performe bien sur Empty mais mal sur DoorKey/MultiRoom:")
print("    → C'est normal! Il n'a été entraîné que sur Empty")
print("  - C'est exactement pour ça qu'on va créer le générateur de niveaux")
print("  - Le générateur va créer des niveaux variés pour améliorer la généralisation")