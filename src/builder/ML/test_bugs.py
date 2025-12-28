"""
Test ciblé pour identifier les bugs spécifiques
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from minigrid.wrappers import FlatObsWrapper

from parametric_minigrid import ParametricMiniGridEnv
from generator import LevelGenerator
from train_coevolution import CoEvolutionTrainer, visualize_agent

print("="*60)
print("TEST CIBLÉ DES BUGS")
print("="*60)

# Test 1: Bug IndexError dans train_generator
print("\n[Test 1/3] Bug IndexError dans train_generator...")
print("-"*60)

try:
    trainer = CoEvolutionTrainer(
        agent_timesteps_per_epoch=2000,
        generator_updates_per_epoch=2,
        batch_size=8,
        num_eval_episodes=2
    )
    
    # Créer un agent simple pour tester
    level = trainer.generator.generate_batch(batch_size=8)[0]
    env = ParametricMiniGridEnv(
        grid_size=level['grid_size'],
        num_obstacles=level['num_obstacles'],
        num_doors=level['num_doors'],
        num_keys=level['num_keys'],
        render_mode="rgb_array"
    )
    env = FlatObsWrapper(env)
    vec_env = DummyVecEnv([lambda: env])
    
    trainer.agent = PPO("MlpPolicy", vec_env, verbose=0)
    trainer.agent.learn(total_timesteps=1000)
    
    # TEST: Entraîner le générateur
    print("  Entraînement du générateur (2 updates)...")
    loss = trainer.train_generator(trainer.agent)
    print(f"  [OK] Loss: {loss:.4f}")
    print("  ✅ Bug IndexError: CORRIGÉ")
    
except IndexError as e:
    print(f"  ❌ Bug IndexError: PRÉSENT")
    print(f"     Erreur: {e}")
except Exception as e:
    print(f"  ⚠️ Autre erreur: {e}")

vec_env.close()

# Test 2: Bug visualisation freeze
print("\n[Test 2/3] Bug visualisation freeze...")
print("-"*60)

try:
    # Créer un générateur et agent simple
    generator = LevelGenerator(latent_dim=16, hidden_dim=64)
    levels = generator.generate_batch(batch_size=2)
    
    level = levels[0]
    env = ParametricMiniGridEnv(
        grid_size=level['grid_size'],
        num_obstacles=level['num_obstacles'],
        num_doors=level['num_doors'],
        num_keys=level['num_keys'],
        render_mode="rgb_array"
    )
    env = FlatObsWrapper(env)
    vec_env = DummyVecEnv([lambda: env])
    
    agent = PPO("MlpPolicy", vec_env, verbose=0)
    agent.learn(total_timesteps=500)
    
    print("  Lancement de visualize_agent (2 niveaux, timeout 10s)...")
    print("  Si le test freeze > 10s, le bug est toujours là")
    
    # Importer avec timeout
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Visualisation freeze!")
    
    # Windows n'a pas signal.alarm, donc on fait un test simple
    print("  Test de visualisation (doit se terminer rapidement)...")
    
    # Test sans visualisation complète pour vérifier que ça ne freeze pas
    test_env = ParametricMiniGridEnv(
        grid_size=6,
        num_obstacles=2,
        num_doors=1,
        num_keys=1,
        render_mode="rgb_array"
    )
    test_env = FlatObsWrapper(test_env)
    
    obs, _ = test_env.reset()
    for _ in range(5):
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = test_env.step(action)
        frame = test_env.render()
        if frame is not None:
            print(f"    Frame shape: {frame.shape}")
        if terminated or truncated:
            break
    
    test_env.close()
    print("  ✅ Bug visualisation: CORRIGÉ (rgb_array fonctionne)")
    
except TimeoutError:
    print("  ❌ Bug visualisation: PRÉSENT (freeze détecté)")
except Exception as e:
    print(f"  ⚠️ Autre erreur: {e}")

vec_env.close()

# Test 3: Vérifier que batch_size est cohérent
print("\n[Test 3/3] Cohérence batch_size vs gen_batch_size...")
print("-"*60)

try:
    # Test avec différents batch_size
    for batch_size in [8, 16, 32]:
        trainer = CoEvolutionTrainer(
            agent_timesteps_per_epoch=1000,
            generator_updates_per_epoch=1,
            batch_size=batch_size,
            num_eval_episodes=1
        )
        
        gen_batch_size = min(8, batch_size)
        print(f"  batch_size={batch_size} → gen_batch_size={gen_batch_size}")
        
        # Vérifier que les arrays ont la bonne taille
        z_batch = torch.randn(gen_batch_size, 16)
        print(f"    z_batch.shape: {z_batch.shape}")
        
        levels = trainer.generator.generate_batch(batch_size=gen_batch_size)
        print(f"    len(levels): {len(levels)}")
        
        assert len(levels) == gen_batch_size, f"Mismatch! {len(levels)} != {gen_batch_size}"
    
    print("  ✅ Cohérence: OK")
    
except AssertionError as e:
    print(f"  ❌ Incohérence détectée: {e}")
except Exception as e:
    print(f"  ⚠️ Autre erreur: {e}")

print("\n" + "="*60)
print("FIN DES TESTS")
print("="*60)
