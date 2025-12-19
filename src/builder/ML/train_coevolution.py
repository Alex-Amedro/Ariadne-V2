"""
Training principal pour la co-évolution Agent <-> Générateur de niveaux.

Usage:
    python train_coevolution.py --epochs 50 --timesteps 100000
    python train_coevolution.py --load runs/run_20231111_143022 --epochs 10
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
import os
import json
import argparse
from datetime import datetime
import time
import matplotlib.pyplot as plt

from parametric_minigrid import ParametricMiniGridEnv
from generator import LevelGenerator


class CoEvolutionTrainer:
    """
    Entraîne un agent et un générateur de niveaux en co-évolution.
    """
    
    def __init__(
        self,
        save_dir=None,
        agent_timesteps_per_epoch=50000,
        generator_updates_per_epoch=10,
        batch_size=16,
        target_success_rate=0.5,
        num_eval_episodes=3  # Réduit de 20 à 3 pour accélérer
    ):
        # Créer un dossier de sauvegarde unique avec timestamp
        if save_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = f"runs/run_{timestamp}"
        
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(f"{self.save_dir}/models", exist_ok=True)
        os.makedirs(f"{self.save_dir}/logs", exist_ok=True)
        
        # Hyperparamètres
        self.agent_timesteps_per_epoch = agent_timesteps_per_epoch
        self.generator_updates = generator_updates_per_epoch
        self.batch_size = batch_size
        self.target_success_rate = target_success_rate
        self.num_eval_episodes = num_eval_episodes
        
        # Initialisation
        self.generator = LevelGenerator(latent_dim=16, hidden_dim=64)
        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=0.0003)
        
        self.agent = None
        self.level_buffer = []
        
        # Historique pour tracking
        self.history = {
            'epochs': [],
            'agent_performance': [],
            'generator_diversity': [],
            'best_success_rate': 0.0
        }
        
        # Sauvegarder la config
        config = {
            'agent_timesteps_per_epoch': agent_timesteps_per_epoch,
            'generator_updates_per_epoch': generator_updates_per_epoch,
            'batch_size': batch_size,
            'target_success_rate': target_success_rate,
            'created_at': datetime.now().isoformat()
        }
        
        with open(f"{self.save_dir}/config.json", 'w') as f:
            json.dump(config, indent=2, fp=f)
        
        print(f"[CoEvolution] Sauvegarde dans: {self.save_dir}")
        print(f"[CoEvolution] Timesteps/epoch: {agent_timesteps_per_epoch}")
        print(f"[CoEvolution] Generator updates/epoch: {generator_updates_per_epoch}")
    
    def create_env_from_params(self, params):
        """Crée un environnement à partir des paramètres."""
        env = ParametricMiniGridEnv(
            grid_size=params['grid_size'],
            num_obstacles=params['num_obstacles'],
            num_doors=params['num_doors'],
            num_keys=params['num_keys'],
            goal_position=params.get('goal_position'),
            render_mode="rgb_array"
        )
        return FlatObsWrapper(env)
    
    def make_vec_env(self, level_params_list):
        """Crée un environnement vectorisé à partir d'une liste de niveaux."""
        def make_env(params):
            def _init():
                return self.create_env_from_params(params)
            return _init
        
        # Créer plusieurs copies de chaque niveau pour parallélisation
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
            max_steps = getattr(env, 'max_steps', None) or getattr(env.unwrapped, 'max_steps', 1000)
            
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
    
    def compute_generator_reward(self, metrics):
        """Calcule le reward pour le générateur."""
        sr = metrics['success_rate']
        
        if sr < 0.1:
            return 0.0
        elif sr > 0.9:
            return 0.0
        else:
            distance = abs(sr - self.target_success_rate)
            reward = 1.0 - (distance / 0.5)
            return max(0.0, reward)
    
    def train_generator(self, current_agent):
        """Entraîne le générateur."""
        total_loss = 0.0
        
        # Réduire le batch_size pour l'entraînement du générateur (trop lent sinon)
        gen_batch_size = min(8, self.batch_size)
        
        for update in range(self.generator_updates):
            z_batch = torch.randn(gen_batch_size, self.generator.latent_dim, requires_grad=False)
            
            level_params_tensors = []
            rewards = []
            
            print(f"    Update {update+1}/{self.generator_updates}: Évaluation de {gen_batch_size} niveaux...", end='', flush=True)
            
            for i in range(gen_batch_size):
                output_dict = self.generator(z_batch[i].unsqueeze(0))
                
                params_tensor = torch.stack([
                    output_dict['grid_size'].squeeze(),
                    output_dict['num_obstacles'].squeeze(),
                    output_dict['num_doors'].squeeze(),
                    output_dict['num_keys'].squeeze(),
                ])
                level_params_tensors.append(params_tensor)
                
                with torch.no_grad():
                    params_dict = {
                        'grid_size': int(output_dict['grid_size'].item()),
                        'num_obstacles': int(output_dict['num_obstacles'].item()),
                        'num_doors': int(output_dict['num_doors'].item()),
                        'num_keys': int(output_dict['num_keys'].item()),
                    }
                    
                    try:
                        metrics = self.evaluate_agent_on_level(params_dict)
                        reward = self.compute_generator_reward(metrics)
                        rewards.append(reward)
                    except:
                        rewards.append(0.0)
            
            print(" [OK]")
            
            stacked_params = torch.stack(level_params_tensors)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32, requires_grad=False)
            
            if rewards_tensor.std() > 0:
                normalized_rewards = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
            else:
                normalized_rewards = rewards_tensor
            
            target = stacked_params.detach().clone()
            # Appliquer les modifications seulement sur les éléments évalués (gen_batch_size)
            for i in range(gen_batch_size):
                if rewards[i] < 0.3:
                    target[i] = target[i] + torch.randn_like(target[i]) * 0.2
            
            loss = nn.MSELoss()(stacked_params, target.detach())
            
            self.generator_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
            self.generator_optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / self.generator_updates
    
    def train(self, num_epochs=10, initial_training_timesteps=100000):
        """Entraînement principal."""
        print("\n" + "="*60)
        print("DÉMARRAGE DE LA CO-ÉVOLUTION")
        print("="*60)
        
        # Phase 0: Entraînement initial de l'agent
        print(f"\n[Phase 0] Entraînement initial de l'agent...")
        print(f"  Timesteps: {initial_training_timesteps}")
        
        initial_levels = self.generator.generate_batch(batch_size=self.batch_size)
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
        
        # Sauvegarder l'agent initial
        self.agent.save(f"{self.save_dir}/models/agent_initial.zip")
        
        # Boucle de co-évolution
        for epoch in range(num_epochs):
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch+1}/{num_epochs}")
            print(f"{'='*60}")
            
            # 1. Générer de nouveaux niveaux
            print(f"\n[1] Génération de {self.batch_size} niveaux...")
            new_levels = self.generator.generate_batch(batch_size=self.batch_size)
            
            # Calculer la diversité
            unique_configs = set()
            for level in new_levels:
                config = (level['grid_size'], level['num_obstacles'], level['num_doors'])
                unique_configs.add(config)
            diversity = len(unique_configs) / len(new_levels)
            print(f"  Diversité: {diversity:.4f}")
            
            # 2. Entraîner l'agent sur les nouveaux niveaux
            print(f"\n[2] Entraînement de l'agent sur les nouveaux niveaux...")
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
            
            # 4. Entraîner le générateur
            print(f"\n[4] Entraînement du générateur...")
            generator_loss = self.train_generator(self.agent)
            print(f"  Loss: {generator_loss:.4f}")
            
            # 5. Sauvegarder
            self.history['epochs'].append(epoch + 1)
            self.history['agent_performance'].append({
                'success_rate': float(avg_sr),
                'mean_reward': float(avg_reward)
            })
            self.history['generator_diversity'].append(float(diversity))
            
            if avg_sr > self.history['best_success_rate']:
                self.history['best_success_rate'] = float(avg_sr)
                self.agent.save(f"{self.save_dir}/models/agent_best.zip")
                torch.save(self.generator.state_dict(), f"{self.save_dir}/models/generator_best.pth")
                print(f"  [BEST] Nouveau meilleur score: {avg_sr*100:.1f}%")
            
            # Sauvegarde périodique
            if (epoch + 1) % 5 == 0:
                self.agent.save(f"{self.save_dir}/models/agent_epoch_{epoch+1}.zip")
                torch.save(self.generator.state_dict(), f"{self.save_dir}/models/generator_epoch_{epoch+1}.pth")
            
            # Sauvegarder l'historique
            with open(f"{self.save_dir}/logs/history.json", 'w') as f:
                json.dump(self.history, f, indent=2)
        
        # Sauvegarde finale
        print(f"\n{'='*60}")
        print("[OK] CO-ÉVOLUTION TERMINÉE!")
        print(f"{'='*60}")
        
        self.agent.save(f"{self.save_dir}/models/agent_final.zip")
        torch.save(self.generator.state_dict(), f"{self.save_dir}/models/generator_final.pth")
        
        print(f"\nModèles sauvegardés dans: {self.save_dir}/models/")
        print(f"Logs sauvegardés dans: {self.save_dir}/logs/")


def load_models(run_dir):
    """Charge les modèles depuis un run précédent."""
    # Charger le générateur
    generator = LevelGenerator(latent_dim=16, hidden_dim=64)
    generator_path = f"{run_dir}/models/generator_final.pth"
    if not os.path.exists(generator_path):
        generator_path = f"{run_dir}/models/generator_best.pth"
    
    generator.load_state_dict(torch.load(generator_path))
    print(f"[LOAD] Générateur chargé: {generator_path}")
    
    # Charger l'agent
    agent_path = f"{run_dir}/models/agent_final.zip"
    if not os.path.exists(agent_path):
        agent_path = f"{run_dir}/models/agent_best.zip"
    
    agent = PPO.load(agent_path)
    print(f"[LOAD] Agent chargé: {agent_path}")
    
    return agent, generator


def visualize_agent(agent, generator, num_levels=5, save_dir="visualization"):
    """Visualise l'agent jouant sur des niveaux générés."""
    print(f"\n{'='*60}")
    print("VISUALISATION DE L'AGENT")
    print(f"{'='*60}")
    
    os.makedirs(save_dir, exist_ok=True)

    # Générer des niveaux
    levels = generator.generate_batch(batch_size=num_levels)

    # Affichage via matplotlib en mode interactif pour éviter de bloquer
    plt.ion()
    fig, ax = plt.subplots()
    im = None

    for i, level_params in enumerate(levels):
        print(f"\n[Niveau {i+1}/{num_levels}]")
        print(f"  Config: grid={level_params['grid_size']}, "
              f"obstacles={level_params['num_obstacles']}, "
              f"doors={level_params['num_doors']}")

        # Créer l'environnement en mode rgb_array (plus portable que 'human')
        env = ParametricMiniGridEnv(
            grid_size=level_params['grid_size'],
            num_obstacles=level_params['num_obstacles'],
            num_doors=level_params['num_doors'],
            num_keys=level_params['num_keys'],
            goal_position=level_params.get('goal_position'),
            render_mode="rgb_array"
        )
        env = FlatObsWrapper(env)

        obs, _ = env.reset()
        done = False
        steps = 0
        total_reward = 0
        max_steps = getattr(env.unwrapped, 'max_steps', 1000)

        try:
            while not done and steps < max_steps:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                steps += 1
                done = terminated or truncated

                # Récupérer l'image RGB et l'afficher via matplotlib
                frame = env.render()
                if frame is None:
                    # Fallback vers human si rgb_array n'est pas supporté
                    env.render(mode='human')
                else:
                    if im is None:
                        im = ax.imshow(frame)
                        ax.axis('off')
                    else:
                        im.set_data(frame)
                    fig.canvas.draw_idle()
                    plt.pause(0.001)

                time.sleep(0.05)  # Slow motion

        except Exception as e:
            print(f"  [WARNING] Visualisation interrompue: {e}")

        print(f"  Résultat: {'SUCCESS' if total_reward > 0 else 'FAIL'} "
              f"(reward={total_reward:.2f}, steps={steps})")

        env.close()
        plt.pause(0.5)

    plt.ioff()
    plt.close(fig)

    print(f"\n{'='*60}")
    print("[OK] VISUALISATION TERMINÉE")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Co-évolution Agent-Générateur")
    parser.add_argument("--epochs", type=int, default=20, help="Nombre d'époques")
    parser.add_argument("--timesteps", type=int, default=50000, help="Timesteps par époque")
    parser.add_argument("--initial-timesteps", type=int, default=100000, help="Timesteps pour l'entraînement initial")
    parser.add_argument("--load", type=str, default=None, help="Chemin vers un run à charger")
    parser.add_argument("--visualize", action="store_true", help="Visualiser après l'entraînement")
    parser.add_argument("--visualize-only", action="store_true", help="Juste visualiser (pas d'entraînement)")
    
    args = parser.parse_args()
    
    if args.visualize_only and args.load:
        # Mode visualisation uniquement
        agent, generator = load_models(args.load)
        visualize_agent(agent, generator, num_levels=5)
    
    elif args.load:
        # Continuer l'entraînement depuis un checkpoint
        print(f"[INFO] Chargement depuis: {args.load}")
        agent, generator = load_models(args.load)
        
        trainer = CoEvolutionTrainer(
            save_dir=args.load,
            agent_timesteps_per_epoch=args.timesteps
        )
        trainer.agent = agent
        trainer.generator = generator
        
        # Charger l'historique
        history_path = f"{args.load}/logs/history.json"
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                trainer.history = json.load(f)
        
        trainer.train(num_epochs=args.epochs, initial_training_timesteps=0)
        
        if args.visualize:
            visualize_agent(trainer.agent, trainer.generator, num_levels=5)
    
    else:
        # Nouvel entraînement
        trainer = CoEvolutionTrainer(
            agent_timesteps_per_epoch=args.timesteps
        )
        
        trainer.train(
            num_epochs=args.epochs,
            initial_training_timesteps=args.initial_timesteps
        )
        
        if args.visualize:
            visualize_agent(trainer.agent, trainer.generator, num_levels=5)
