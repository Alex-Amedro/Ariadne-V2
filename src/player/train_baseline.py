import gymnasium as gym
# Doit être importé pour enregistrer les envs, mais on l'utilise aussi pour le wrapper
import gymnasium_minigrid 
# 1. CORRECTION DE L'IMPORT :
# On utilise le wrapper qui vient de gymnasium_minigrid
from gymnasium_minigrid.wrappers import ImgObsWrapper

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.torch_layers import CnnFeaturesExtractor
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
        env = gym.make(env_id)
        # On utilise le ImgObsWrapper
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

    # 2. CORRECTION DU CNN (LE CŒUR DU PROBLÈME)
    # L'architecture CNN par défaut de SB3 (pour Atari) est trop grande (kernel 8x8)
    # pour nos observations 7x7. Nous devons en définir une plus petite.
    
    # Définition d'un dictionnaire pour les "policy_kwargs"
    # C'est ici qu'on dit à PPO : "N'utilise pas ton CNN par défaut !"
    policy_kwargs = dict(
        features_extractor_class=CnnFeaturesExtractor,
        features_extractor_kwargs=dict(
            # On définit un réseau de convolution simple et petit
            # adapté à nos images 7x7
            cnn_layers=[
                # Couche 1: 16 filtres, kernel 3x3, stride 1, padding 1
                dict(filters=16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                # Couche 2: 32 filtres, kernel 2x2, stride 1, padding 0
                dict(filters=32, kernel_size=2, stride=1, padding=0),
                nn.ReLU(),
            ],
            # SB3 s'occupera de la couche "Flatten" et du MLP final
        ),
    )

    # --- Définition du Modèle PPO ---
    model = PPO(
        "CnnPolicy",
        vec_env,
        policy_kwargs=policy_kwargs, # <--- On passe notre CNN personnalisé ici
        verbose=1,
        tensorboard_log=LOG_DIR
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
