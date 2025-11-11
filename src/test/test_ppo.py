"""
Test 2 : Entraîner un agent PPO sur MiniGrid et tester ses performances
Ce script entraîne rapidement un agent pour vérifier que tout marche
"""

import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper, FlatObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np

print("=" * 50)
print("TEST 2 : Entraînement PPO baseline")
print("=" * 50)

# Configuration
ENV_NAME = "MiniGrid-Empty-8x8-v0"  # Environnement simple pour test rapide
TOTAL_TIMESTEPS = 50000  # Court pour le test (augmente à 200k+ pour vrai training)
N_ENVS = 4  # Entraînement parallèle

print(f"\n[Config]")
print(f"  Environnement: {ENV_NAME}")
print(f"  Timesteps: {TOTAL_TIMESTEPS}")
print(f"  Envs parallèles: {N_ENVS}")

# Wrapper pour rendre l'observation compatible avec PPO
def make_env(rank=0, seed=0):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        env = gym.make(ENV_NAME, render_mode="rgb_array")
        env = FlatObsWrapper(env)  # Flatten l'observation pour PPO
        return env
    return _init

# Créer les environnements
print("\n[1] Création des environnements vectorisés...")
env = make_vec_env(lambda: FlatObsWrapper(gym.make(ENV_NAME, render_mode="rgb_array")), n_envs=N_ENVS)
eval_env = make_vec_env(lambda: FlatObsWrapper(gym.make(ENV_NAME, render_mode="rgb_array")), n_envs=1)
print("[OK] Environnements créés!")

# Créer l'agent PPO
print("\n[2] Création de l'agent PPO...")
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=128,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    tensorboard_log="./ppo_minigrid_tensorboard/",
)
print("[OK] Agent créé!")

# Callback pour évaluation
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./models/",
    log_path="./logs/",
    eval_freq=5000,
    deterministic=True,
    render=False,
    n_eval_episodes=10,
)

# Entraînement
print(f"\n[3] Entraînement de l'agent ({TOTAL_TIMESTEPS} timesteps)...")
print("    (Cela peut prendre 2-5 minutes)")
try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
        progress_bar=True,
    )
    print("[OK] Entraînement terminé!")
except Exception as e:
    print(f"[ERREUR] Erreur pendant l'entraînement: {e}")
    exit(1)

# Sauvegarder le modèle
print("\n[4] Sauvegarde du modèle...")
model.save("ppo_minigrid_baseline")
print("[OK] Modèle sauvegardé: ppo_minigrid_baseline.zip")

# Test de l'agent entraîné
print("\n[5] Test de l'agent entraîné (5 épisodes)...")
test_env = gym.make(ENV_NAME, render_mode="rgb_array")
test_env = FlatObsWrapper(test_env)

rewards = []
success = []

for episode in range(5):
    obs, info = test_env.reset()
    episode_reward = 0
    done = False
    steps = 0
    
    while not done and steps < 100:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        episode_reward += reward
        done = terminated or truncated
        steps += 1
    
    rewards.append(episode_reward)
    success.append(1 if episode_reward > 0 else 0)
    print(f"  Episode {episode+1}: reward={episode_reward:.2f}, steps={steps}, success={episode_reward > 0}")

test_env.close()

print("\n[Résultats]")
print(f"  Reward moyen: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
print(f"  Taux de succès: {np.mean(success)*100:.1f}%")

print("\n" + "=" * 50)      
print("[OK] TEST TERMINÉ!")
print("=" * 50)
print("\nProchaines étapes:")
print("  1. Si le taux de succès > 50%, ton setup marche!")
print("  2. Tu peux maintenant passer au générateur de niveaux")
print("  3. Augmente TOTAL_TIMESTEPS à 200k+ pour un vrai training")