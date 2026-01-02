"""
DIVERSITY OBJECTIVE : Entraînement avec objectif de diversité.
Utilise Novelty Search pour éviter que le générateur crée toujours les mêmes niveaux.
"""

import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from minigrid.wrappers import FlatObsWrapper
import os
import json
from datetime import datetime
from scipy.spatial.distance import pdist, squareform

from parametric_minigrid import ParametricMiniGridEnv
from generator import LevelGenerator


class DiversityTrainer:
    """Co-évolution avec objectif de diversité (Novelty Search)."""
    
    def __init__(
        self,
        save_dir=None,
        agent_timesteps_per_epoch=50000,
        batch_size=16,
        num_eval_episodes=3,
        diversity_weight=0.3,  # Poids du terme de diversité
        archive_size=100  # Taille de l'archive pour novelty
    ):
        if save_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = f"runs/diversity_{timestamp}"
        
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(f"{self.save_dir}/models", exist_ok=True)
        os.makedirs(f"{self.save_dir}/logs", exist_ok=True)
        
        self.agent_timesteps_per_epoch = agent_timesteps_per_epoch
        self.batch_size = batch_size
        self.num_eval_episodes = num_eval_episodes
        self.diversity_weight = diversity_weight
        
        self.agent = None
        self.generator = LevelGenerator(latent_dim=8)
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=1e-4)
        
        # Archive pour Novelty Search
        self.archive = []
        self.archive_size = archive_size
        
        self.history = {
            'epochs': [],
            'agent_performance': [],
            'generator_diversity': [],
            'novelty_scores': [],
            'best_success_rate': 0.0
        }
        
        config = {
            'type': 'DIVERSITY (Novelty Search)',
            'diversity_weight': diversity_weight,
            'archive_size': archive_size,
            'agent_timesteps_per_epoch': agent_timesteps_per_epoch,
            'batch_size': batch_size,
            'created_at': datetime.now().isoformat()
        }
        
        with open(f"{self.save_dir}/config.json", 'w') as f:
            json.dump(config, indent=2, fp=f)
        
        print(f"[Diversity] Sauvegarde dans: {self.save_dir}")
        print(f"[Diversity] Diversity weight: {diversity_weight}")
        print(f"[Diversity] Archive size: {archive_size}")
    
    def generate_level_params(self):
        """Génère un niveau avec le générateur."""
        z = torch.randn(8)
        return self.generator.generate_level_params(z)
    
    def level_to_vector(self, level_params):
        """Convertit un niveau en vecteur pour calcul de distance."""
        return np.array([
            level_params['grid_size'],
            level_params['num_obstacles'],
            level_params['num_doors'],
            level_params['num_keys']
        ])
    
    def compute_novelty(self, level_params, k=15):
        """
        Calcule le score de nouveauté d'un niveau.
        Moyenne des distances aux k plus proches voisins dans l'archive.
        """
        if len(self.archive) < k:
            return 1.0  # Maximum novelty si pas assez de samples
        
        level_vec = self.level_to_vector(level_params)
        
        # Distances à tous les niveaux de l'archive
        distances = []
        for archived_level in self.archive:
            archived_vec = self.level_to_vector(archived_level)
            dist = np.linalg.norm(level_vec - archived_vec)
            distances.append(dist)
        
        # Moyenne des k plus proches
        distances.sort()
        novelty = np.mean(distances[:k])
        
        return novelty
    
    def update_archive(self, level_params):
        """Ajoute un niveau à l'archive."""
        self.archive.append(level_params.copy())
        
        # Limiter la taille de l'archive
        if len(self.archive) > self.archive_size:
            # Retirer aléatoirement (ou garder les plus divers)
            self.archive.pop(0)
    
    def compute_diversity_batch(self, levels):
        """Calcule la diversité d'un batch de niveaux."""
        if len(levels) < 2:
            return 0.0
        
        # Convertir en matrice
        vectors = np.array([self.level_to_vector(l) for l in levels])
        
        # Distances par paire
        distances = pdist(vectors, metric='euclidean')
        
        # Moyenne des distances = diversité
        diversity = np.mean(distances)
        
        return float(diversity)
    
    def create_env_from_params(self, params):
        """Crée un environnement."""
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
        """Évalue l'agent sur un niveau."""
        env = self.create_env_from_params(level_params)
        
        rewards = []
        successes = []
        
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
        
        env.close()
        
        return {
            'success_rate': np.mean(successes),
            'mean_reward': np.mean(rewards)
        }
    
    def train(self, num_epochs=20, initial_training_timesteps=100000):
        """Entraînement avec diversité."""
        print("\n" + "="*60)
        print("DIVERSITY-BASED CO-EVOLUTION")
        print("="*60)
        
        # Phase 0: Entraînement initial
        print(f"\n[Phase 0] Entraînement initial...")
        initial_levels = [self.generate_level_params() for _ in range(self.batch_size)]
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
        
        # Initialiser l'archive avec les niveaux initiaux
        for level in initial_levels:
            self.update_archive(level)
        
        print("  [OK] Agent initial entraîné")
        self.agent.save(f"{self.save_dir}/models/agent_initial.zip")
        
        # Boucle d'entraînement
        for epoch in range(num_epochs):
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch+1}/{num_epochs}")
            print(f"{'='*60}")
            
            # 1. Générer niveaux
            print(f"\n[1] Génération de {self.batch_size} niveaux...")
            new_levels = [self.generate_level_params() for _ in range(self.batch_size)]
            
            # Calculer diversité et novelty
            diversity = self.compute_diversity_batch(new_levels)
            novelty_scores = [self.compute_novelty(l) for l in new_levels]
            avg_novelty = np.mean(novelty_scores)
            
            print(f"  Diversité batch: {diversity:.4f}")
            print(f"  Novelty moyenne: {avg_novelty:.4f}")
            
            # 2. Entraîner l'agent
            print(f"\n[2] Entraînement de l'agent...")
            train_env = self.make_vec_env(new_levels)
            self.agent.set_env(train_env)
            self.agent.learn(total_timesteps=self.agent_timesteps_per_epoch, progress_bar=True)
            train_env.close()
            
            # 3. Entraîner le générateur avec DIVERSITY OBJECTIVE
            print(f"\n[3] Entraînement du générateur (avec diversity)...")
            
            total_performance_loss = 0
            total_diversity_loss = 0
            num_iterations = 20
            
            for i in range(num_iterations):
                self.generator_optimizer.zero_grad()
                
                # Générer un batch de z
                z_batch = torch.randn(self.batch_size, 8, requires_grad=True)
                
                # Forward pass du générateur (retourne dict de tenseurs)
                raw_params_batch = self.generator(z_batch)
                
                # Loss 1: Performance de l'agent (comme avant)
                performance_losses = []
                generated_levels_iter = []
                
                for j in range(min(5, self.batch_size)):
                    # Extraire les paramètres pour ce niveau et convertir en entiers
                    params = {
                        'grid_size': int(raw_params_batch['grid_size'][j].item()),
                        'num_obstacles': int(raw_params_batch['num_obstacles'][j].item()),
                        'num_doors': int(raw_params_batch['num_doors'][j].item()),
                        'num_keys': int(raw_params_batch['num_keys'][j].item())
                    }
                    generated_levels_iter.append(params)
                    
                    try:
                        metrics = self.evaluate_agent_on_level(params)
                        loss = -metrics['success_rate']  # Négatif car on veut max la SR
                        performance_losses.append(loss)
                    except:
                        performance_losses.append(1.0)
                
                performance_loss = torch.tensor(performance_losses, requires_grad=True).mean()
                
                # Loss 2: Diversité (on veut MAXIMISER, donc MINIMISER -diversity)
                if len(generated_levels_iter) >= 2:
                    batch_diversity = self.compute_diversity_batch(generated_levels_iter)
                    diversity_loss = -batch_diversity  # Négatif pour maximiser
                else:
                    diversity_loss = 0.0
                
                diversity_loss_tensor = torch.tensor(diversity_loss, requires_grad=True)
                
                # Loss totale: combinaison pondérée
                total_loss = performance_loss + self.diversity_weight * diversity_loss_tensor
                
                total_loss.backward()
                self.generator_optimizer.step()
                
                total_performance_loss += performance_loss.item()
                total_diversity_loss += diversity_loss
            
            avg_perf_loss = total_performance_loss / num_iterations
            avg_div_loss = total_diversity_loss / num_iterations
            
            print(f"  Performance loss: {avg_perf_loss:.4f}")
            print(f"  Diversity loss: {avg_div_loss:.4f}")
            
            # Mettre à jour l'archive avec les nouveaux niveaux
            for level in new_levels:
                self.update_archive(level)
            
            # 4. Évaluer
            print(f"\n[4] Évaluation...")
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
                print(f"  Success rate: {avg_sr*100:.1f}%")
                print(f"  Reward: {avg_reward:.2f}")
            else:
                avg_sr = 0.0
                avg_reward = 0.0
            
            # 5. Sauvegarder
            self.history['epochs'].append(epoch + 1)
            self.history['agent_performance'].append({
                'success_rate': float(avg_sr),
                'mean_reward': float(avg_reward)
            })
            self.history['generator_diversity'].append(float(diversity))
            self.history['novelty_scores'].append(float(avg_novelty))
            
            if avg_sr > self.history['best_success_rate']:
                self.history['best_success_rate'] = float(avg_sr)
                self.agent.save(f"{self.save_dir}/models/agent_best.zip")
                torch.save(self.generator.state_dict(), 
                          f"{self.save_dir}/models/generator_best.pth")
                print(f"  [BEST] {avg_sr*100:.1f}%")
            
            if (epoch + 1) % 5 == 0:
                self.agent.save(f"{self.save_dir}/models/agent_epoch_{epoch+1}.zip")
                torch.save(self.generator.state_dict(),
                          f"{self.save_dir}/models/generator_epoch_{epoch+1}.pth")
            
            with open(f"{self.save_dir}/logs/history.json", 'w') as f:
                json.dump(self.history, f, indent=2)
        
        # Sauvegarde finale
        print(f"\n{'='*60}")
        print("[OK] DIVERSITY TRAINING TERMINÉ!")
        print(f"{'='*60}")
        
        self.agent.save(f"{self.save_dir}/models/agent_final.zip")
        torch.save(self.generator.state_dict(),
                  f"{self.save_dir}/models/generator_final.pth")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Training avec diversity objective")
    parser.add_argument("--epochs", type=int, default=20, help="Nombre d'époques")
    parser.add_argument("--timesteps", type=int, default=50000, help="Timesteps par époque")
    parser.add_argument("--diversity-weight", type=float, default=0.3, 
                       help="Poids du terme de diversité")
    parser.add_argument("--archive-size", type=int, default=100,
                       help="Taille de l'archive pour novelty")
    
    args = parser.parse_args()
    
    trainer = DiversityTrainer(
        agent_timesteps_per_epoch=args.timesteps,
        diversity_weight=args.diversity_weight,
        archive_size=args.archive_size
    )
    
    trainer.train(num_epochs=args.epochs)
    
    print("\n" + "="*60)
    print("DIVERSITY TRAINING PRÊT!")
    print("="*60)
