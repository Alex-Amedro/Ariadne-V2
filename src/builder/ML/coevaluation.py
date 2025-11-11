import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
import json
import os
from datetime import datetime


class CoEvolutionTrainer:
    """
    Gère le training loop de co-évolution entre le générateur et l'agent.
    """
    
    def __init__(
        self,
        env_class,
        generator,
        save_dir="./coevolution_results",
        target_success_rate=0.5,
        generator_lr=1e-3,
        agent_timesteps_per_epoch=50000,
        generator_updates_per_epoch=20,
        batch_size_generator=32,
        num_eval_episodes=10
    ):
        """
        Args:
            env_class: classe de l'environnement (ParametricMiniGridEnv)
            generator: instance de LevelGenerator
            save_dir: dossier pour sauvegarder les résultats
            target_success_rate: taux de succès cible pour le reward du générateur
            generator_lr: learning rate du générateur
            agent_timesteps_per_epoch: timesteps d'entraînement PPO par epoch
            generator_updates_per_epoch: nombre d'updates du générateur par epoch
            batch_size_generator: taille de batch pour entraîner le générateur
            num_eval_episodes: nombre d'épisodes pour évaluer chaque niveau
        """
        self.env_class = env_class
        self.generator = generator
        self.save_dir = save_dir
        self.target_success_rate = target_success_rate
        self.agent_timesteps = agent_timesteps_per_epoch
        self.generator_updates = generator_updates_per_epoch
        self.batch_size = batch_size_generator
        self.num_eval_episodes = num_eval_episodes
        
        # Créer le dossier de sauvegarde
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(f"{save_dir}/models", exist_ok=True)
        os.makedirs(f"{save_dir}/logs", exist_ok=True)
        
        # Optimizer pour le générateur
        self.generator_optimizer = optim.Adam(generator.parameters(), lr=generator_lr)
        
        # Agent PPO (sera créé au début du training)
        self.agent = None
        
        # Historique des métriques
        self.history = {
            'epoch': [],
            'agent_success_rate': [],
            'generator_loss': [],
            'mean_difficulty': [],
            'diversity_score': [],
            'best_agent_reward': []
        }
        
        print(f"[CoEvolution] Initialisé avec:")
        print(f"  - Agent timesteps/epoch: {self.agent_timesteps}")
        print(f"  - Generator updates/epoch: {self.generator_updates}")
        print(f"  - Batch size: {self.batch_size}")
        print(f"  - Save dir: {self.save_dir}")
    
    def create_agent(self, initial_levels=None):
        """
        Crée un nouvel agent PPO.
        
        Args:
            initial_levels: liste de paramètres de niveaux pour l'entraînement initial
        """
        if initial_levels is None:
            # Créer quelques niveaux simples par défaut
            initial_levels = [
                {'grid_size': 8, 'num_obstacles': 2, 'num_doors': 0, 'num_keys': 0}
                for _ in range(4)
            ]
        
        # Créer des environnements vectorisés
        def make_env(level_params):
            def _init():
                env = self.env_class(**level_params, render_mode="rgb_array")
                env = FlatObsWrapper(env)
                return env
            return _init
        
        # Créer 4 envs en parallèle avec le premier niveau
        envs = [make_env(initial_levels[0]) for _ in range(4)]
        vec_env = DummyVecEnv(envs)
        
        # Créer l'agent PPO
        self.agent = PPO(
            "MlpPolicy",
            vec_env,
            verbose=0,
            learning_rate=3e-4,
            n_steps=128,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
        )
        
        print("[CoEvolution] Agent PPO créé")
        
        return self.agent
    
    def evaluate_agent_on_level(self, level_params):
        """
        Évalue l'agent sur un niveau et retourne les métriques.
        """
        env = self.env_class(**level_params, render_mode="rgb_array")
        env = FlatObsWrapper(env)
        
        rewards = []
        successes = []
        steps_list = []
        
        for _ in range(self.num_eval_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            steps = 0
            
            while not done and steps < env.max_steps:
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
        """
        Calcule le reward pour le générateur basé sur les métriques du niveau.
        
        On veut des niveaux challengeants mais pas impossibles.
        """
        sr = metrics['success_rate']
        
        # Reward = distance au target (inversée)
        if sr < 0.1:  # Trop dur
            return 0.0
        elif sr > 0.9:  # Trop facile
            return 0.0
        else:
            # Maximum reward quand sr = target_success_rate
            distance = abs(sr - self.target_success_rate)
            reward = 1.0 - (distance / 0.5)  # Normalise entre 0 et 1
            return max(0.0, reward)
    
    def train_generator(self, current_agent):
        """
        Entraîne le générateur pour créer des niveaux challengeants.
        """
        total_loss = 0.0
        
        for update in range(self.generator_updates):
            # 1. Générer un batch de niveaux
            z_batch = torch.randn(self.batch_size, self.generator.latent_dim)
            
            # 2. Obtenir les paramètres
            levels_params = []
            for i in range(self.batch_size):
                params = self.generator.generate_level_params(z_batch[i])
                levels_params.append(params)
            
            # 3. Évaluer chaque niveau avec l'agent actuel
            rewards = []
            for params in levels_params:
                try:
                    metrics = self.evaluate_agent_on_level(params)
                    reward = self.compute_generator_reward(metrics)
                    rewards.append(reward)
                except:
                    # Si le niveau est invalide, reward = 0
                    rewards.append(0.0)
            
            # 4. Calculer la loss (on veut maximiser le reward)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
            loss = -rewards_tensor.mean()  # Gradient ascent sur le reward
            
            # 5. Backpropagation
            self.generator_optimizer.zero_grad()
            loss.backward()
            self.generator_optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / self.generator_updates
        return avg_loss
    
    def train(self, num_epochs=10, initial_training_timesteps=100000):
        """
        Training loop principal de co-évolution.
        
        Args:
            num_epochs: nombre d'époques de co-évolution
            initial_training_timesteps: timesteps d'entraînement initial de l'agent
        """
        print("\n" + "="*60)
        print("DÉMARRAGE DE LA CO-ÉVOLUTION")
        print("="*60)
        
        # PHASE 0: Entraînement initial de l'agent sur des niveaux simples
        print("\n[Phase 0] Entraînement initial de l'agent...")
        print(f"  Timesteps: {initial_training_timesteps}")
        
        simple_levels = [
            {'grid_size': 8, 'num_obstacles': 2, 'num_doors': 0, 'num_keys': 0}
        ]
        self.create_agent(simple_levels)
        self.agent.learn(total_timesteps=initial_training_timesteps, progress_bar=True)
        
        print("  ✅ Agent initial entraîné")
        
        # PHASE 1-N: Co-évolution
        for epoch in range(num_epochs):
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch + 1}/{num_epochs}")
            print(f"{'='*60}")
            
            # Étape 1: Générer des niveaux avec le générateur actuel
            print(f"\n[1] Génération de {self.batch_size} niveaux...")
            generated_levels = self.generator.generate_batch(self.batch_size)
            
            # Évaluer la diversité
            from evaluator import compute_diversity_score
            diversity = compute_diversity_score(generated_levels)
            print(f"  Diversité: {diversity:.4f}")
            
            # Étape 2: Entraîner l'agent sur ces niveaux
            print(f"\n[2] Entraînement de l'agent sur les nouveaux niveaux...")
            print(f"  Timesteps: {self.agent_timesteps}")
            
            # Créer des environnements avec les nouveaux niveaux
            def make_env(level_params):
                def _init():
                    env = self.env_class(**level_params, render_mode="rgb_array")
                    env = FlatObsWrapper(env)
                    return env
                return _init
            
            # Utiliser plusieurs niveaux différents
            selected_levels = generated_levels[:4]
            envs = [make_env(level) for level in selected_levels]
            vec_env = DummyVecEnv(envs)
            self.agent.set_env(vec_env)
            
            self.agent.learn(total_timesteps=self.agent_timesteps, progress_bar=True, reset_num_timesteps=False)
            
            # Étape 3: Évaluer l'agent sur les niveaux générés
            print(f"\n[3] Évaluation de l'agent...")
            eval_metrics = []
            for i, level in enumerate(generated_levels[:10]):  # Évaluer sur 10 niveaux
                metrics = self.evaluate_agent_on_level(level)
                eval_metrics.append(metrics)
            
            avg_success = np.mean([m['success_rate'] for m in eval_metrics])
            avg_reward = np.mean([m['mean_reward'] for m in eval_metrics])
            
            print(f"  Success rate moyen: {avg_success*100:.1f}%")
            print(f"  Reward moyen: {avg_reward:.2f}")
            
            # Étape 4: Entraîner le générateur
            print(f"\n[4] Entraînement du générateur...")
            generator_loss = self.train_generator(self.agent)
            print(f"  Generator loss: {generator_loss:.4f}")
            
            # Étape 5: Logger les métriques
            self.history['epoch'].append(epoch + 1)
            self.history['agent_success_rate'].append(avg_success)
            self.history['generator_loss'].append(generator_loss)
            self.history['mean_difficulty'].append(1.0 - avg_success)  # Approximation
            self.history['diversity_score'].append(diversity)
            self.history['best_agent_reward'].append(avg_reward)
            
            # Sauvegarder les modèles
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch + 1)
            
            # Sauvegarder l'historique
            self.save_history()
        
        print("\n" + "="*60)
        print("✅ CO-ÉVOLUTION TERMINÉE!")
        print("="*60)
        
        # Sauvegarde finale
        self.save_checkpoint("final")
    
    def save_checkpoint(self, epoch):
        """Sauvegarde l'état actuel du training."""
        checkpoint_path = f"{self.save_dir}/models/epoch_{epoch}"
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Sauvegarder le générateur
        torch.save(
            self.generator.state_dict(),
            f"{checkpoint_path}/generator.pth"
        )
        
        # Sauvegarder l'agent
        self.agent.save(f"{checkpoint_path}/agent")
        
        print(f"  💾 Checkpoint sauvegardé: {checkpoint_path}")
    
    def save_history(self):
        """Sauvegarde l'historique des métriques."""
        history_path = f"{self.save_dir}/logs/history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)


# --- TEST DE BASE ---
if __name__ == "__main__":
    print("="*60)
    print("TEST: CoEvolutionTrainer")
    print("="*60)
    
    print("\n⚠️  Ce test nécessite:")
    print("  - parametric_minigrid.py")
    print("  - generator.py")
    print("  - evaluator.py")
    
    try:
        from parametric_minigrid import ParametricMiniGridEnv
        from generator import LevelGenerator
        print("\n✅ Imports réussis")
    except ImportError as e:
        print(f"\n❌ Erreur d'import: {e}")
        print("Assure-toi que tous les fichiers sont dans le même dossier")
        exit(1)
    
    print("\n[Test] Création du trainer...")
    
    # Créer le générateur
    generator = LevelGenerator(latent_dim=16, hidden_dim=64)
    
    # Créer le trainer
    trainer = CoEvolutionTrainer(
        env_class=ParametricMiniGridEnv,
        generator=generator,
        save_dir="./test_coevolution",
        agent_timesteps_per_epoch=5000,  # Court pour le test
        generator_updates_per_epoch=5,
        batch_size_generator=8,
        num_eval_episodes=3
    )
    
    print("✅ Trainer créé")
    
    print("\n[Test] Lancement d'une époque de test...")
    print("(Cela va prendre ~2-3 minutes)")
    
    try:
        trainer.train(num_epochs=1, initial_training_timesteps=10000)
        print("\n✅ TEST RÉUSSI!")
    except Exception as e:
        print(f"\n❌ Erreur pendant le training: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("Si le test a réussi, tu peux maintenant:")
    print("  1. Lancer un vrai training avec plus d'epochs")
    print("  2. Créer des scripts de visualisation")
    print("  3. Analyser les résultats")
    print("="*60)
