"""
ENREGISTREMENT VIDÉOS : Capture des trajectoires de l'agent.
Génère des vidéos MP4 pour visualiser le comportement.
"""

import torch
import numpy as np
from stable_baselines3 import PPO
import imageio
from pathlib import Path
from minigrid.wrappers import FlatObsWrapper, RGBImgObsWrapper
import json

from parametric_minigrid import ParametricMiniGridEnv
from generator import LevelGenerator


class AgentRecorder:
    """Enregistre des vidéos de l'agent."""
    
    def __init__(self, agent_path, generator_path=None, save_dir='videos'):
        """
        Args:
            agent_path: Chemin vers l'agent PPO (.zip)
            generator_path: Chemin vers le générateur (.pth), ou None pour random
            save_dir: Dossier de sauvegarde
        """
        self.agent = PPO.load(agent_path)
        
        if generator_path:
            self.generator = LevelGenerator()
            self.generator.load_state_dict(torch.load(generator_path))
            self.generator.eval()
            print(f"[Recorder] Générateur chargé: {generator_path}")
        else:
            self.generator = None
            print(f"[Recorder] Mode random (pas de générateur)")
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[Recorder] Agent chargé: {agent_path}")
        print(f"[Recorder] Sauvegarde dans: {save_dir}")
    
    def generate_level_params(self):
        """Génère un niveau (avec générateur ou random)."""
        if self.generator:
            z = torch.randn(1, 8)
            with torch.no_grad():
                params_normalized = self.generator(z).squeeze().numpy()
            
            grid_size = int(params_normalized[0] * 6 + 6)
            num_obstacles = int(params_normalized[1] * grid_size * 2)
            num_doors = int(params_normalized[2] * 3)
            num_keys = max(int(params_normalized[3] * 2), num_doors)
        else:
            # Random
            grid_size = np.random.randint(6, 12)
            num_obstacles = np.random.randint(0, grid_size * 2)
            num_doors = np.random.randint(0, 3)
            num_keys = max(num_doors, np.random.randint(0, 2))
        
        return {
            'grid_size': grid_size,
            'num_obstacles': num_obstacles,
            'num_doors': num_doors,
            'num_keys': num_keys
        }
    
    def record_episode(self, level_params, filename, max_steps=500, fps=10):
        """
        Enregistre une vidéo d'un épisode.
        
        Args:
            level_params: Paramètres du niveau
            filename: Nom du fichier (sans extension)
            max_steps: Nombre max de steps
            fps: Frames per second
        """
        # Créer l'environnement avec render
        env = ParametricMiniGridEnv(
            grid_size=level_params['grid_size'],
            num_obstacles=level_params['num_obstacles'],
            num_doors=level_params['num_doors'],
            num_keys=level_params['num_keys'],
            render_mode="rgb_array"
        )
        env = FlatObsWrapper(env)
        
        # Reset
        obs, info = env.reset()
        
        frames = []
        done = False
        steps = 0
        total_reward = 0
        
        print(f"\n[Recording] {filename}")
        print(f"  Level: grid={level_params['grid_size']}, "
              f"obstacles={level_params['num_obstacles']}, "
              f"doors={level_params['num_doors']}, "
              f"keys={level_params['num_keys']}")
        
        while not done and steps < max_steps:
            # Capturer la frame
            frame = env.unwrapped.render()
            frames.append(frame)
            
            # Action de l'agent
            action, _ = self.agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            steps += 1
            done = terminated or truncated
        
        env.close()
        
        # Sauvegarder la vidéo
        video_path = self.save_dir / f"{filename}.mp4"
        imageio.mimsave(video_path, frames, fps=fps)
        
        # Sauvegarder métadonnées
        metadata = {
            'level_params': level_params,
            'steps': steps,
            'total_reward': float(total_reward),
            'success': total_reward > 0,
            'fps': fps
        }
        
        metadata_path = self.save_dir / f"{filename}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Steps: {steps}")
        print(f"  Reward: {total_reward:.2f}")
        print(f"  Success: {'✅' if total_reward > 0 else '❌'}")
        print(f"  [SAVED] {video_path}")
        
        return metadata
    
    def record_multiple(self, num_episodes=10, max_steps=500, fps=10):
        """Enregistre plusieurs épisodes."""
        print("="*60)
        print(f"ENREGISTREMENT DE {num_episodes} VIDÉOS")
        print("="*60)
        
        successes = 0
        
        for i in range(num_episodes):
            level_params = self.generate_level_params()
            filename = f"episode_{i+1:03d}"
            
            metadata = self.record_episode(level_params, filename, max_steps, fps)
            
            if metadata['success']:
                successes += 1
        
        print("\n" + "="*60)
        print("ENREGISTREMENT TERMINÉ!")
        print("="*60)
        print(f"\nSuccess rate: {successes}/{num_episodes} ({successes/num_episodes*100:.1f}%)")
        print(f"Vidéos sauvegardées dans: {self.save_dir}/")
    
    def record_showcase(self, difficulties=['easy', 'medium', 'hard'], 
                       episodes_per_difficulty=3, max_steps=500, fps=10):
        """Enregistre des vidéos organisées par difficulté."""
        print("="*60)
        print("SHOWCASE RECORDING (Easy/Medium/Hard)")
        print("="*60)
        
        difficulty_configs = {
            'easy': {'grid_range': (6, 8), 'obstacle_mult': 0.5, 'doors': 0},
            'medium': {'grid_range': (8, 10), 'obstacle_mult': 1.0, 'doors': 1},
            'hard': {'grid_range': (10, 12), 'obstacle_mult': 1.5, 'doors': 2}
        }
        
        for difficulty in difficulties:
            print(f"\n{'='*60}")
            print(f"DIFFICULTY: {difficulty.upper()}")
            print(f"{'='*60}")
            
            config = difficulty_configs[difficulty]
            
            for i in range(episodes_per_difficulty):
                # Générer niveau avec difficulté contrôlée
                if self.generator:
                    level_params = self.generate_level_params()
                    # Ajuster selon la difficulté
                    grid_min, grid_max = config['grid_range']
                    level_params['grid_size'] = np.clip(level_params['grid_size'], grid_min, grid_max)
                    level_params['num_obstacles'] = int(level_params['num_obstacles'] * config['obstacle_mult'])
                    level_params['num_doors'] = config['doors']
                    level_params['num_keys'] = max(config['doors'], level_params['num_keys'])
                else:
                    grid_size = np.random.randint(*config['grid_range'])
                    level_params = {
                        'grid_size': grid_size,
                        'num_obstacles': int(grid_size * config['obstacle_mult']),
                        'num_doors': config['doors'],
                        'num_keys': config['doors']
                    }
                
                filename = f"showcase_{difficulty}_{i+1:02d}"
                self.record_episode(level_params, filename, max_steps, fps)
        
        print("\n" + "="*60)
        print("SHOWCASE TERMINÉ!")
        print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enregistrement de vidéos")
    parser.add_argument("--agent", type=str, required=True,
                       help="Chemin vers l'agent (.zip)")
    parser.add_argument("--generator", type=str, default=None,
                       help="Chemin vers le générateur (.pth)")
    parser.add_argument("--output", type=str, default="videos",
                       help="Dossier de sortie")
    parser.add_argument("--num-episodes", type=int, default=10,
                       help="Nombre d'épisodes à enregistrer")
    parser.add_argument("--max-steps", type=int, default=500,
                       help="Steps max par épisode")
    parser.add_argument("--fps", type=int, default=10,
                       help="Frames per second")
    parser.add_argument("--showcase", action='store_true',
                       help="Mode showcase (easy/medium/hard)")
    
    args = parser.parse_args()
    
    recorder = AgentRecorder(args.agent, args.generator, args.output)
    
    if args.showcase:
        recorder.record_showcase(
            episodes_per_difficulty=args.num_episodes // 3,
            max_steps=args.max_steps,
            fps=args.fps
        )
    else:
        recorder.record_multiple(
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            fps=args.fps
        )
