import gymnasium as gym
import gymnasium_minigrid  
from gymnasium_minigrid.wrappers import ImgObsWrapper

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed

import os

ENV_NAME = "MiniGrid-DoorKey-8x8-v0"
LOG_DIR = "analysis/logs"
MODEL_DIR = "player/models"

RUN_ID = "baseline_ppo_doorkey"
MODEL_SAVE_NAME = f"{RUN_ID}.zip"

TOTAL_TIMESTEPS = 200_000

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def make_env(env_id, seed):
    """
    Fonction utilitaire pour créer et wrapper l'environnement.
    """
    def _init():

        env = gym.make(env_id)
        
        env = ImgObsWrapper(env)
        
        env.reset(seed=seed)
        return env
    
    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    
    print(f"--- Démarrage de l'entraînement Baseline ---")
    print(f"Environnement: {ENV_NAME}")
    print(f"Steps: {TOTAL_TIMESTEPS}")

    seed = 0
    vec_env = DummyVecEnv([make_env(ENV_NAME, seed)])


    model = PPO(
        "CnnPolicy",
        vec_env,
        verbose=1,              
        tensorboard_log=LOG_DIR 
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        tb_log_name=RUN_ID
    )

    print("--- Entraînement Baseline Terminé ---")

    save_path = os.path.join(MODEL_DIR, MODEL_SAVE_NAME)
    model.save(save_path)
    
    print(f"Modèle sauvegardé à: {save_path}")
    print(f"Logs TensorBoard disponibles dans: {LOG_DIR}/{RUN_ID}")
