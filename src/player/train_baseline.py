import gymnasium as gym
# CORRECTION 1: On importe le paquet 'minigrid' moderne
import minigrid 
# CORRECTION 1: Le wrapper est dans 'minigrid.wrappers'
from minigrid.wrappers import ImgObsWrapper

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed

# CORRECTION 2: Le chemin d'import a changé dans SB3 !
from stable_baselines3.common.features_extractor import CnnFeaturesExtractor

import torch.nn as nn
import os

# --- Configuration ---
ENV_NAME = "MiniGrid-DoorKey-8x8-v0"
LOG_DIR = "analysis/logs"
MODEL_DIR = "player/models"
RUN_ID = "baseline_ppo_doorkey"
MODEL_SAVE_NAME = f"{RUN_ID}.zip"
TOTAL_TIMESTEPS = 200_000

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def make_env(env_id, seed):
    def _init():
        # On utilise gym.make, qui fonctionne grâce à 'import minigrid'
        env = gym.make(env_id)
        # On utilise le ImgObsWrapper de 'minigrid.wrappers'
        env = ImgObsWrapper(env)
        env.reset(seed=seed)
        return env
    
    set_random_seed(seed)
    return _init

# --- Exécution principale ---
if __name__ == "__main__":
    
    print(f"--- Démarrage de l'entraînement Baseline ---")
    print(f"Environnement: {ENV_NAME}")
    print(f"Steps: {TOTAL_TIMESTEPS}")

    seed = 0
    vec_env = DummyVecEnv([make_env(ENV_NAME, seed)])

    # On utilise toujours le CNN personnalisé car l'image est en 7x7
    policy_kwargs = dict(
        features_extractor_class=CnnFeaturesExtractor,
        features_extractor_kwargs=dict(
            cnn_layers=[
                dict(filters=16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                dict(filters=32, kernel_size=2, stride=1, padding=0),
                nn.ReLU(),
            ],
        ),
    )

    # --- Définition du Modèle PPO ---
    model = PPO(
        "CnnPolicy",
        vec_env,
        policy_kwargs=policy_kwargs, 
        verbose=1,
        tensor_log=LOG_DIR  # Note: c'est 'tensorboard_log', pas 'tensor_log'
    )

    # --- Lancement de l'entraînement ---
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        tb_log_name=RUN_ID
    )

    print("--- Entraînement Baseline Terminé ---")

    # --- Sauvegarde du modèle ---
    save_path = os.path.join(MODEL_DIR, MODEL_SAVE_NAME)
    model.save(save_path)
    
    print(f"Modèle sauvegardé à: {save_path}")
    print(f"Logs TensorBoard disponibles dans: {LOG_DIR}/{RUN_ID}")