"""
CURRICULUM PACING : Contrôle adaptatif de la difficulté.
Ajuste la vitesse de progression selon les performances de l'agent.
"""

import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from minigrid.wrappers import FlatObsWrapper
import os
import json
from datetime import datetime

from parametric_minigrid import ParametricMiniGridEnv
from generator import LevelGenerator


class CurriculumTrainer:
    """Co-évolution avec curriculum pacing adaptatif."""
    
    def __init__(
        self,
        save_dir=None,
        agent_timesteps_per_epoch=50000,
        batch_size=16,
        num_eval_episodes=3,
        pacing_strategy='adaptive',  # 'adaptive' ou 'staged'
        target_success_rate=0.5  # SR cible pour adaptive pacing
    ):
        if save_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = f"runs/curriculum_{pacing_strategy}_{timestamp}"
        
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(f"{self.save_dir}/models", exist_ok=True)
        os.makedirs(f"{self.save_dir}/logs", exist_ok=True)
        
        self.agent_timesteps_per_epoch = agent_timesteps_per_epoch
        self.batch_size = batch_size
        self.num_eval_episodes = num_eval_episodes
        self.pacing_strategy = pacing_strategy
        self.target_success_rate = target_success_rate
        
        self.agent = None
        self.generator = LevelGenerator(latent_dim=8)
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=1e-4)
        
        # Paramètres de curriculum
        self.difficulty_weight = 1.0  # Poids de la difficulté
        self.min_difficulty = 0.3
        self.max_difficulty = 2.0
        
        self.history = {
            'epochs': [],
            'agent_performance': [],
            'difficulty_weights': [],
            'generator_diversity': [],
            'best_success_rate': 0.0
        }
        
        config = {
            'type': f'CURRICULUM ({pacing_strategy})',
            'pacing_strategy': pacing_strategy,
            'target_success_rate': target_success_rate,
            'agent_timesteps_per_epoch': agent_timesteps_per_epoch,
            'batch_size': batch_size,
            'created_at': datetime.now().isoformat()
        }
        
        with open(f"{self.save_dir}/config.json", 'w') as f:
            json.dump(config, indent=2, fp=f)
        
        print(f"[Curriculum] Stratégie: {pacing_strategy}")
        print(f"[Curriculum] Target SR: {target_success_rate}")
        print(f"[Curriculum] Sauvegarde dans: {self.save_dir}")
    
    def adjust_difficulty_adaptive(self, current_sr):
        """
        Adaptive pacing: ajuste la difficulté selon la performance.
        Si SR trop faible → diminuer difficulté
        Si SR trop élevée → augmenter difficulté
        """
        sr_diff = current_sr - self.target_success_rate
        
        # Ajustement proportionnel
        if sr_diff < -0.2:  # Trop dur (SR << target)
            self.difficulty_weight *= 0.85
            status = "TROP DUR - Diminution"
        elif sr_diff < -0.1:
            self.difficulty_weight *= 0.93
            status = "Un peu dur - Légère diminution"
        elif sr_diff > 0.2:  # Trop facile (SR >> target)
            self.difficulty_weight *= 1.15
            status = "TROP FACILE - Augmentation"
        elif sr_diff > 0.1:
            self.difficulty_weight *= 1.07
            status = "Un peu facile - Légère augmentation"
        else:
            status = "OK - Maintien"
        
        # Bornes
        self.difficulty_weight = np.clip(
            self.difficulty_weight, 
            self.min_difficulty, 
            self.max_difficulty
        )
        
        return status
    
    def get_difficulty_staged(self, epoch):
        """
        Staged curriculum: difficulté augmente par paliers.
        """
        if epoch < 5:
            difficulty = 0.5  # Easy
            stage = "STAGE 1 (Easy)"
        elif epoch < 10:
            difficulty = 0.8  # Medium-Easy
            stage = "STAGE 2 (Medium-Easy)"
        elif epoch < 15:
            difficulty = 1.0  # Medium
            stage = "STAGE 3 (Medium)"
        elif epoch < 20:
            difficulty = 1.3  # Medium-Hard
            stage = "STAGE 4 (Medium-Hard)"
        else:
            difficulty = 1.5  # Hard
            stage = "STAGE 5 (Hard)"
        
        self.difficulty_weight = difficulty
        return stage
    
    def generate_level_params(self, difficulty_modifier=1.0):
        """Génère un niveau avec contrôle de difficulté."""
        z = torch.randn(8)
        params = self.generator.generate_level_params(z)
        
        # Appliquer le modificateur de difficulté
        if difficulty_modifier != 1.0:
            params['num_obstacles'] = int(params['num_obstacles'] * difficulty_modifier)
            params['num_doors'] = int(params['num_doors'] * min(difficulty_modifier, 1.2))
            params['num_keys'] = max(params['num_keys'], params['num_doors'])
            
            # Bornes
            params['num_obstacles'] = np.clip(params['num_obstacles'], 0, params['grid_size'] * 3)
            params['num_doors'] = np.clip(params['num_doors'], 0, 3)
        
        return params
    
    def compute_diversity(self, levels):
        """Calcule la diversité des niveaux."""
        if len(levels) < 2:
            return 0.0
        
        params_array = np.array([
            [l['grid_size'], l['num_obstacles'], l['num_doors'], l['num_keys']]
            for l in levels
        ])
        
        variance = np.mean(np.var(params_array, axis=0))
        return float(variance)
    
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
        """Entraînement avec curriculum pacing."""
        print("\n" + "="*60)
        print(f"CURRICULUM PACING ({self.pacing_strategy.upper()})")
        print("="*60)
        
        # Phase 0: Entraînement initial (facile)
        print(f"\n[Phase 0] Entraînement initial (facile)...")
        initial_levels = [self.generate_level_params(difficulty_modifier=0.5) 
                         for _ in range(self.batch_size)]
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
            
            # Ajuster la difficulté selon la stratégie
            if self.pacing_strategy == 'staged':
                stage_info = self.get_difficulty_staged(epoch)
                print(f"\n[Curriculum] {stage_info}")
            
            print(f"[Curriculum] Difficulty weight: {self.difficulty_weight:.3f}")
            
            # 1. Générer niveaux avec difficulté adaptée
            print(f"\n[1] Génération de {self.batch_size} niveaux...")
            new_levels = [self.generate_level_params(difficulty_modifier=self.difficulty_weight) 
                         for _ in range(self.batch_size)]
            
            diversity = self.compute_diversity(new_levels)
            avg_obstacles = np.mean([l['num_obstacles'] for l in new_levels])
            avg_grid = np.mean([l['grid_size'] for l in new_levels])
            
            print(f"  Diversité: {diversity:.4f}")
            print(f"  Grid size moyen: {avg_grid:.1f}")
            print(f"  Obstacles moyen: {avg_obstacles:.1f}")
            
            # 2. Entraîner l'agent
            print(f"\n[2] Entraînement de l'agent...")
            train_env = self.make_vec_env(new_levels)
            self.agent.set_env(train_env)
            self.agent.learn(total_timesteps=self.agent_timesteps_per_epoch, progress_bar=True)
            train_env.close()
            
            # 3. Entraîner le générateur
            print(f"\n[3] Entraînement du générateur...")
            
            total_generator_loss = 0
            num_iterations = 20
            
            for i in range(num_iterations):
                self.generator_optimizer.zero_grad()
                
                z = torch.randn(self.batch_size, 8, requires_grad=True)
                params_normalized = self.generator(z)
                
                losses = []
                for j in range(min(5, self.batch_size)):
                    params = {
                        'grid_size': int(params_normalized[j, 0].item() * 6 + 6),
                        'num_obstacles': int(params_normalized[j, 1].item() * params_normalized[j, 0].item() * 12 * self.difficulty_weight),
                        'num_doors': int(params_normalized[j, 2].item() * 3 * min(self.difficulty_weight, 1.2)),
                        'num_keys': max(int(params_normalized[j, 3].item() * 2), int(params_normalized[j, 2].item() * 3))
                    }
                    
                    try:
                        metrics = self.evaluate_agent_on_level(params)
                        loss = -metrics['success_rate']
                        losses.append(loss)
                    except:
                        losses.append(1.0)
                
                loss = torch.tensor(losses, requires_grad=True).mean()
                loss.backward()
                self.generator_optimizer.step()
                
                total_generator_loss += loss.item()
            
            avg_gen_loss = total_generator_loss / num_iterations
            print(f"  Loss moyen: {avg_gen_loss:.4f}")
            
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
            
            # 5. Ajuster la difficulté (adaptive seulement)
            if self.pacing_strategy == 'adaptive':
                status = self.adjust_difficulty_adaptive(avg_sr)
                print(f"\n[Curriculum] Ajustement: {status}")
                print(f"[Curriculum] Nouvelle difficulty: {self.difficulty_weight:.3f}")
            
            # 6. Sauvegarder
            self.history['epochs'].append(epoch + 1)
            self.history['agent_performance'].append({
                'success_rate': float(avg_sr),
                'mean_reward': float(avg_reward)
            })
            self.history['difficulty_weights'].append(float(self.difficulty_weight))
            self.history['generator_diversity'].append(float(diversity))
            
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
        print("[OK] CURRICULUM TRAINING TERMINÉ!")
        print(f"{'='*60}")
        
        self.agent.save(f"{self.save_dir}/models/agent_final.zip")
        torch.save(self.generator.state_dict(),
                  f"{self.save_dir}/models/generator_final.pth")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Training avec curriculum pacing")
    parser.add_argument("--epochs", type=int, default=20, help="Nombre d'époques")
    parser.add_argument("--timesteps", type=int, default=50000, help="Timesteps par époque")
    parser.add_argument("--strategy", type=str, choices=['adaptive', 'staged'], 
                       default='adaptive', help="Stratégie de pacing")
    parser.add_argument("--target-sr", type=float, default=0.5,
                       help="Success rate cible (adaptive seulement)")
    
    args = parser.parse_args()
    
    trainer = CurriculumTrainer(
        agent_timesteps_per_epoch=args.timesteps,
        pacing_strategy=args.strategy,
        target_success_rate=args.target_sr
    )
    
    trainer.train(num_epochs=args.epochs)
    
    print("\n" + "="*60)
    print("CURRICULUM TRAINING PRÊT!")
    print("="*60)
