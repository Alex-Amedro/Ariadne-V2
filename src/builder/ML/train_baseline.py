"""
Entraînement BASELINE : Agent sur niveaux ALÉATOIRES (sans générateur neural).
Compare avec la co-évolution pour montrer l'avantage du système.
"""

import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from minigrid.wrappers import FlatObsWrapper
import os
import json
from datetime import datetime
import random

from parametric_minigrid import ParametricMiniGridEnv


class BaselineTrainer:
    """Entraîne un agent sur des niveaux aléatoires (sans générateur neural)."""
    
    def __init__(
        self,
        save_dir=None,
        agent_timesteps_per_epoch=50000,
        batch_size=16,
        num_eval_episodes=3
    ):
        if save_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = f"runs/baseline_{timestamp}"
        
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(f"{self.save_dir}/models", exist_ok=True)
        os.makedirs(f"{self.save_dir}/logs", exist_ok=True)
        
        self.agent_timesteps_per_epoch = agent_timesteps_per_epoch
        self.batch_size = batch_size
        self.num_eval_episodes = num_eval_episodes
        
        self.agent = None
        
        self.history = {
            'epochs': [],
            'agent_performance': [],
            'level_configs': [],  # Pour tracking des configs utilisées
            'best_success_rate': 0.0
        }
        
        config = {
            'type': 'BASELINE (Random Levels)',
            'agent_timesteps_per_epoch': agent_timesteps_per_epoch,
            'batch_size': batch_size,
            'created_at': datetime.now().isoformat()
        }
        
        with open(f"{self.save_dir}/config.json", 'w') as f:
            json.dump(config, indent=2, fp=f)
        
        print(f"[Baseline] Sauvegarde dans: {self.save_dir}")
        print(f"[Baseline] Mode: RANDOM LEVELS (pas de générateur neural)")
    
    def generate_random_level(self):
        """Génère un niveau avec des paramètres aléatoires."""
        # Distribution similaire à ce que le générateur pourrait produire
        grid_size = random.randint(6, 12)
        max_obstacles = (grid_size - 4) * 2  # Formule heuristique
        num_obstacles = random.randint(0, max_obstacles)
        num_doors = random.randint(0, min(3, grid_size // 3))
        # IMPORTANT: num_keys doit être >= num_doors
        num_keys = max(num_doors, random.randint(0, 2))
        
        return {
            'grid_size': grid_size,
            'num_obstacles': num_obstacles,
            'num_doors': num_doors,
            'num_keys': num_keys
        }
    
    def create_env_from_params(self, params):
        """Crée un environnement à partir des paramètres."""
        env = ParametricMiniGridEnv(
            grid_size=params['grid_size'],
            num_obstacles=params['num_obstacles'],
            num_doors=params['num_doors'],
            num_keys=params['num_keys'],
            render_mode="rgb_array"
        )
        return FlatObsWrapper(env)
    
    def make_vec_env(self, level_params_list):
        """Crée un environnement vectorisé."""
        def make_env(params):
            def _init():
                return self.create_env_from_params(params)
            return _init
        
        env_fns = [make_env(params) for params in level_params_list]
        return DummyVecEnv(env_fns)
    
    def evaluate_agent_on_level(self, level_params):
        """Évalue l'agent sur un niveau spécifique."""
        env = self.create_env_from_params(level_params)
        
        rewards = []
        successes = []
        steps_list = []
        
        for _ in range(self.num_eval_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            steps = 0
            max_steps = getattr(env.unwrapped, 'max_steps', 1000)
            
            while not done and steps < max_steps:
                action, _ = self.agent.predict(obs, deterministic=True)
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
            'mean_reward': np.mean(rewards),
            'mean_steps': np.mean(steps_list)
        }
    
    def train(self, num_epochs=20, initial_training_timesteps=100000):
        """Entraînement baseline avec niveaux aléatoires."""
        print("\n" + "="*60)
        print("BASELINE TRAINING (Random Levels)")
        print("="*60)
        
        # Phase 0: Entraînement initial
        print(f"\n[Phase 0] Entraînement initial de l'agent...")
        print(f"  Timesteps: {initial_training_timesteps}")
        
        initial_levels = [self.generate_random_level() for _ in range(self.batch_size)]
        initial_env = self.make_vec_env(initial_levels)
        
        self.agent = PPO(
            "MlpPolicy",
            initial_env,
            verbose=0,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=128
        )
        
        self.agent.learn(total_timesteps=initial_training_timesteps, progress_bar=True)
        initial_env.close()
        
        print("  [OK] Agent initial entraîné")
        self.agent.save(f"{self.save_dir}/models/agent_initial.zip")
        
        # Boucle d'entraînement
        for epoch in range(num_epochs):
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch+1}/{num_epochs}")
            print(f"{'='*60}")
            
            # 1. Générer de nouveaux niveaux ALÉATOIRES
            print(f"\n[1] Génération de {self.batch_size} niveaux ALÉATOIRES...")
            new_levels = [self.generate_random_level() for _ in range(self.batch_size)]
            
            # Calculer diversité (pour comparaison équitable)
            unique_configs = set()
            for level in new_levels:
                config = (level['grid_size'], level['num_obstacles'], level['num_doors'])
                unique_configs.add(config)
            diversity = len(unique_configs) / len(new_levels)
            print(f"  Diversité: {diversity:.4f}")
            
            # Stocker les configs pour analyse
            self.history['level_configs'].append(new_levels[:5])  # Stocker quelques exemples
            
            # 2. Entraîner l'agent sur ces niveaux
            print(f"\n[2] Entraînement de l'agent...")
            print(f"  Timesteps: {self.agent_timesteps_per_epoch}")
            
            train_env = self.make_vec_env(new_levels)
            self.agent.set_env(train_env)
            self.agent.learn(total_timesteps=self.agent_timesteps_per_epoch, progress_bar=True)
            train_env.close()
            
            # 3. Évaluer l'agent
            print(f"\n[3] Évaluation de l'agent...")
            
            eval_metrics = []
            for level in new_levels[:5]:
                try:
                    metrics = self.evaluate_agent_on_level(level)
                    eval_metrics.append(metrics)
                except:
                    pass
            
            if eval_metrics:
                avg_sr = np.mean([m['success_rate'] for m in eval_metrics])
                avg_reward = np.mean([m['mean_reward'] for m in eval_metrics])
                print(f"  Success rate moyen: {avg_sr*100:.1f}%")
                print(f"  Reward moyen: {avg_reward:.2f}")
            else:
                avg_sr = 0.0
                avg_reward = 0.0
            
            # 4. Sauvegarder
            self.history['epochs'].append(epoch + 1)
            self.history['agent_performance'].append({
                'success_rate': float(avg_sr),
                'mean_reward': float(avg_reward),
                'diversity': float(diversity)  # Pour comparaison
            })
            
            if avg_sr > self.history['best_success_rate']:
                self.history['best_success_rate'] = float(avg_sr)
                self.agent.save(f"{self.save_dir}/models/agent_best.zip")
                print(f"  [BEST] Nouveau meilleur score: {avg_sr*100:.1f}%")
            
            # Sauvegarde périodique
            if (epoch + 1) % 5 == 0:
                self.agent.save(f"{self.save_dir}/models/agent_epoch_{epoch+1}.zip")
            
            # Sauvegarder l'historique
            with open(f"{self.save_dir}/logs/history.json", 'w') as f:
                json.dump(self.history, f, indent=2)
        
        # Sauvegarde finale
        print(f"\n{'='*60}")
        print("[OK] BASELINE TRAINING TERMINÉ!")
        print(f"{'='*60}")
        
        self.agent.save(f"{self.save_dir}/models/agent_final.zip")
        
        print(f"\nModèles sauvegardés dans: {self.save_dir}/models/")
        print(f"Logs sauvegardés dans: {self.save_dir}/logs/")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Baseline training avec niveaux aléatoires")
    parser.add_argument("--epochs", type=int, default=20, help="Nombre d'époques")
    parser.add_argument("--timesteps", type=int, default=50000, help="Timesteps par époque")
    parser.add_argument("--initial-timesteps", type=int, default=100000, help="Timesteps initiaux")
    
    args = parser.parse_args()
    
    trainer = BaselineTrainer(
        agent_timesteps_per_epoch=args.timesteps
    )
    
    trainer.train(
        num_epochs=args.epochs,
        initial_training_timesteps=args.initial_timesteps
    )
    
    print("\n" + "="*60)
    print("BASELINE PRÊT POUR COMPARAISON!")
    print("="*60)
    print("\nPour comparer avec co-évolution:")
    print("  python compare_results.py --coevol runs/run_XXXXXX --baseline runs/baseline_XXXXXX")
