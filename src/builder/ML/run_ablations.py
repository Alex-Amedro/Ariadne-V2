"""
ABLATION STUDIES : Teste l'importance de chaque composant.

Ablations implémentées :
1. Sans reward shaping (reward binaire seulement)
2. Sans gradient générateur (générateur fixe/aléatoire)
3. Architecture générateur différente (plus simple/plus complexe)
"""

import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from minigrid.wrappers import FlatObsWrapper
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path

from parametric_minigrid import ParametricMiniGridEnv
from generator import LevelGenerator


class AblationTrainer:
    """Entraîneur pour études d'ablation."""
    
    def __init__(
        self,
        ablation_type='no_reward_shaping',
        save_dir=None,
        agent_timesteps_per_epoch=50000,
        batch_size=16,
        num_eval_episodes=3
    ):
        """
        Args:
            ablation_type: Type d'ablation
                - 'no_reward_shaping': Reward binaire seulement
                - 'no_generator_gradient': Générateur fixe (pas d'update)
                - 'simple_generator': Architecture plus simple
                - 'complex_generator': Architecture plus complexe
        """
        self.ablation_type = ablation_type
        
        if save_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = f"runs/ablation_{ablation_type}_{timestamp}"
        
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(f"{self.save_dir}/models", exist_ok=True)
        os.makedirs(f"{self.save_dir}/logs", exist_ok=True)
        
        self.agent_timesteps_per_epoch = agent_timesteps_per_epoch
        self.batch_size = batch_size
        self.num_eval_episodes = num_eval_episodes
        
        self.agent = None
        self.generator = None
        
        self.history = {
            'epochs': [],
            'agent_performance': [],
            'generator_diversity': [],
            'best_success_rate': 0.0,
            'ablation_type': ablation_type
        }
        
        # Créer le générateur selon le type d'ablation
        self._create_generator()
        
        config = {
            'type': f'ABLATION - {ablation_type}',
            'ablation_type': ablation_type,
            'agent_timesteps_per_epoch': agent_timesteps_per_epoch,
            'batch_size': batch_size,
            'created_at': datetime.now().isoformat()
        }
        
        with open(f"{self.save_dir}/config.json", 'w') as f:
            json.dump(config, indent=2, fp=f)
        
        print(f"[Ablation] Type: {ablation_type}")
        print(f"[Ablation] Sauvegarde dans: {self.save_dir}")
    
    def _create_generator(self):
        """Crée le générateur selon le type d'ablation."""
        if self.ablation_type == 'simple_generator':
            # Architecture plus simple : 8 → 32 → 4
            class SimpleGenerator(torch.nn.Module):
                def __init__(self, latent_dim=8, output_dim=4):
                    super().__init__()
                    self.network = torch.nn.Sequential(
                        torch.nn.Linear(latent_dim, 32),
                        torch.nn.ReLU(),
                        torch.nn.Linear(32, output_dim),
                        torch.nn.Sigmoid()
                    )
                
                def forward(self, z):
                    return self.network(z)
            
            self.generator = SimpleGenerator()
            print("[Generator] Architecture SIMPLE : 8→32→4")
        
        elif self.ablation_type == 'complex_generator':
            # Architecture plus complexe : 8 → 128 → 128 → 64 → 4
            class ComplexGenerator(torch.nn.Module):
                def __init__(self, latent_dim=8, output_dim=4):
                    super().__init__()
                    self.network = torch.nn.Sequential(
                        torch.nn.Linear(latent_dim, 128),
                        torch.nn.ReLU(),
                        torch.nn.Linear(128, 128),
                        torch.nn.ReLU(),
                        torch.nn.Linear(128, 64),
                        torch.nn.ReLU(),
                        torch.nn.Linear(64, output_dim),
                        torch.nn.Sigmoid()
                    )
                
                def forward(self, z):
                    return self.network(z)
            
            self.generator = ComplexGenerator()
            print("[Generator] Architecture COMPLEXE : 8→128→128→64→4")
        
        else:
            # Architecture standard pour autres ablations
            self.generator = LevelGenerator()
            print("[Generator] Architecture STANDARD : 8→64→64→4")
    
    def generate_level_params(self):
        """Génère un niveau avec le générateur."""
        z = torch.randn(1, 8)
        
        with torch.no_grad():
            params_normalized = self.generator(z).squeeze().numpy()
        
        grid_size = int(params_normalized[0] * 6 + 6)
        num_obstacles = int(params_normalized[1] * grid_size * 2)
        num_doors = int(params_normalized[2] * 3)
        num_keys = min(int(params_normalized[3] * 2), num_doors)
        
        return {
            'grid_size': grid_size,
            'num_obstacles': num_obstacles,
            'num_doors': num_doors,
            'num_keys': num_keys
        }
    
    def create_env_from_params(self, params):
        """Crée un environnement."""
        # Modifier selon le type d'ablation
        if self.ablation_type == 'no_reward_shaping':
            # Reward binaire seulement (désactiver reward shaping)
            env = ParametricMiniGridEnv(
                grid_size=params['grid_size'],
                num_obstacles=params['num_obstacles'],
                num_doors=params['num_doors'],
                num_keys=params['num_keys'],
                render_mode="rgb_array",
                # Désactiver tous les bonus
                enable_reward_shaping=False
            )
        else:
            # Reward shaping normal
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
    
    def compute_diversity(self, levels):
        """Calcule la diversité des niveaux générés."""
        if len(levels) < 2:
            return 0.0
        
        params_array = np.array([
            [l['grid_size'], l['num_obstacles'], l['num_doors'], l['num_keys']]
            for l in levels
        ])
        
        variance = np.mean(np.var(params_array, axis=0))
        return float(variance)
    
    def train(self, num_epochs=20, initial_training_timesteps=100000):
        """Entraînement avec ablation."""
        print("\n" + "="*60)
        print(f"ABLATION STUDY: {self.ablation_type.upper()}")
        print("="*60)
        
        # Phase 0: Entraînement initial
        print(f"\n[Phase 0] Entraînement initial de l'agent...")
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
            diversity = self.compute_diversity(new_levels)
            print(f"  Diversité: {diversity:.4f}")
            
            # 2. Entraîner l'agent
            print(f"\n[2] Entraînement de l'agent...")
            train_env = self.make_vec_env(new_levels)
            self.agent.set_env(train_env)
            self.agent.learn(total_timesteps=self.agent_timesteps_per_epoch, progress_bar=True)
            train_env.close()
            
            # 3. Entraîner le générateur (SAUF si no_generator_gradient)
            if self.ablation_type != 'no_generator_gradient':
                print(f"\n[3] Entraînement du générateur...")
                
                total_generator_loss = 0
                num_iterations = 20
                
                for i in range(num_iterations):
                    z = torch.randn(self.batch_size, 8)
                    params_normalized = self.generator(z)
                    
                    losses = []
                    for j in range(min(5, self.batch_size)):
                        params = {
                            'grid_size': int(params_normalized[j, 0].item() * 6 + 6),
                            'num_obstacles': int(params_normalized[j, 1].item() * params_normalized[j, 0].item() * 6 * 2),
                            'num_doors': int(params_normalized[j, 2].item() * 3),
                            'num_keys': min(int(params_normalized[j, 3].item() * 2), int(params_normalized[j, 2].item() * 3))
                        }
                        
                        try:
                            metrics = self.evaluate_agent_on_level(params)
                            loss = -metrics['success_rate']
                            losses.append(loss)
                        except:
                            losses.append(1.0)
                    
                    loss = torch.tensor(losses).mean()
                    
                    self.generator.zero_grad()
                    loss.backward()
                    
                    optimizer = torch.optim.Adam(self.generator.parameters(), lr=1e-4)
                    optimizer.step()
                    
                    total_generator_loss += loss.item()
                
                avg_gen_loss = total_generator_loss / num_iterations
                print(f"  Loss moyen: {avg_gen_loss:.4f}")
            else:
                print(f"\n[3] Générateur FIXE (pas d'entraînement)")
            
            # 4. Évaluer
            print(f"\n[4] Évaluation de l'agent...")
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
        print(f"[OK] ABLATION '{self.ablation_type}' TERMINÉE!")
        print(f"{'='*60}")
        
        self.agent.save(f"{self.save_dir}/models/agent_final.zip")
        torch.save(self.generator.state_dict(),
                  f"{self.save_dir}/models/generator_final.pth")


def run_all_ablations(epochs=15, timesteps=50000):
    """Lance toutes les ablations."""
    
    ablations = [
        'no_reward_shaping',
        'no_generator_gradient',
        'simple_generator',
        'complex_generator'
    ]
    
    results = {}
    
    for ablation in ablations:
        print("\n" + "="*80)
        print(f"LANCEMENT ABLATION: {ablation}")
        print("="*80)
        
        trainer = AblationTrainer(
            ablation_type=ablation,
            agent_timesteps_per_epoch=timesteps
        )
        
        trainer.train(num_epochs=epochs)
        
        results[ablation] = {
            'save_dir': trainer.save_dir,
            'best_sr': trainer.history['best_success_rate'],
            'final_sr': trainer.history['agent_performance'][-1]['success_rate']
        }
    
    # Sauvegarder résumé
    summary_path = Path("ablations_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("TOUTES LES ABLATIONS TERMINÉES!")
    print("="*80)
    print("\nRÉSULTATS:")
    for ablation, data in results.items():
        print(f"\n{ablation}:")
        print(f"  Best SR: {data['best_sr']*100:.1f}%")
        print(f"  Final SR: {data['final_sr']*100:.1f}%")
        print(f"  Dir: {data['save_dir']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ablation studies")
    parser.add_argument("--ablation", type=str, 
                       choices=['no_reward_shaping', 'no_generator_gradient', 
                               'simple_generator', 'complex_generator', 'all'],
                       default='no_reward_shaping',
                       help="Type d'ablation à lancer")
    parser.add_argument("--epochs", type=int, default=15, help="Nombre d'époques")
    parser.add_argument("--timesteps", type=int, default=50000, help="Timesteps par époque")
    
    args = parser.parse_args()
    
    if args.ablation == 'all':
        run_all_ablations(epochs=args.epochs, timesteps=args.timesteps)
    else:
        trainer = AblationTrainer(
            ablation_type=args.ablation,
            agent_timesteps_per_epoch=args.timesteps
        )
        
        trainer.train(num_epochs=args.epochs)
        
        print("\n" + "="*60)
        print("ABLATION PRÊTE POUR ANALYSE!")
        print("="*60)
        print(f"\nRésultats: {trainer.save_dir}")
        print(f"Best SR: {trainer.history['best_success_rate']*100:.1f}%")
