"""
Test 3 : Visualiser l'agent entraîné qui joue
Ce script charge le modèle entraîné et affiche l'agent en action
"""

import gymnasium as gym
import minigrid
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

print("=" * 50)
print("TEST 3 : Visualisation de l'agent")
print("=" * 50)

# Charger le modèle
print("\n[1] Chargement du modèle...")
try:
    model = PPO.load("ppo_minigrid_baseline")
    print("[OK] Modèle chargé!")
except Exception as e:
    print(f"[ERREUR]: {e}")
    print("   Lance d'abord test_ppo.py pour entraîner un modèle!")
    exit(1)

# Créer l'environnement avec render
ENV_NAME = "MiniGrid-Empty-8x8-v0"
env = gym.make(ENV_NAME, render_mode="rgb_array")
env = FlatObsWrapper(env)

print("\n[2] Exécution de 3 épisodes avec capture d'images...")

for episode in range(3):
    print(f"\n--- Episode {episode + 1} ---")
    obs, info = env.reset()
    done = False
    steps = 0
    episode_reward = 0
    frames = []
    
    while not done and steps < 100:
        # Rendre l'image
        frame = env.render()
        frames.append(frame)
        
        # Prédire l'action
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        episode_reward += reward
        done = terminated or truncated
        steps += 1
    
    print(f"  Reward: {episode_reward:.2f}, Steps: {steps}, Success: {episode_reward > 0}")
    
    # Afficher quelques frames clés
    if len(frames) > 0:
        fig, axes = plt.subplots(1, min(5, len(frames)), figsize=(15, 3))
        if len(frames) == 1:
            axes = [axes]
        
        # Sélectionner des frames uniformément espacées
        indices = np.linspace(0, len(frames)-1, min(5, len(frames)), dtype=int)
        
        for idx, frame_idx in enumerate(indices):
            if len(frames) > 1:
                axes[idx].imshow(frames[frame_idx])
                axes[idx].set_title(f"Step {frame_idx}")
                axes[idx].axis('off')
            else:
                axes[0].imshow(frames[0])
                axes[0].set_title("Step 0")
                axes[0].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"episode_{episode+1}_visualization.png", dpi=100, bbox_inches='tight')
        print(f"  [OK] Sauvegardé: episode_{episode+1}_visualization.png")
        plt.close()

env.close()

print("\n" + "=" * 50)
print("[OK] VISUALISATIONS CRÉÉES!")
print("=" * 50)
print("\nFichiers créés:")
print("  - episode_1_visualization.png")
print("  - episode_2_visualization.png")
print("  - episode_3_visualization.png")
print("\nRegarde ces images pour voir comment ton agent se comporte!")