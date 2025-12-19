"""
Test rapide du système d'entraînement - 1 epoch seulement
"""

import sys
import os

# Ajouter le chemin pour les imports
sys.path.insert(0, os.path.dirname(__file__))

from train_coevolution import CoEvolutionTrainer

if __name__ == "__main__":
    print("="*60)
    print("TEST RAPIDE - 1 EPOCH")
    print("="*60)
    
    # Configuration ultra-rapide
    trainer = CoEvolutionTrainer(
        agent_timesteps_per_epoch=5000,  # Très réduit
        generator_updates_per_epoch=3,    # Réduit de 10 à 3
        batch_size=8,                     # Réduit de 16 à 8
        target_success_rate=0.5,
        num_eval_episodes=2               # Réduit à 2
    )
    
    # 1 seule epoch avec peu de timesteps initiaux
    trainer.train(
        num_epochs=1,
        initial_training_timesteps=5000
    )
    
    print("\n" + "="*60)
    print("[OK] Test terminé!")
    print("="*60)
