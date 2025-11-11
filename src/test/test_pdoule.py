"""
Test 1 : Vérifier que MiniGrid s'installe et tourne correctement
Lance ce script en premier pour vérifier ton setup
"""

import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper

print("=" * 50)
print("TEST 1 : Installation et environnement de base")
print("=" * 50)

# Test 1: Créer un environnement simple
print("\n[1] Création d'un environnement MiniGrid-Empty-5x5-v0...")
try:
    env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="rgb_array")
    print("[OK] Environnement créé avec succès!")
except Exception as e:
    print(f"[ERREUR] : {e}")
    exit(1)

# Test 2: Reset de l'environnement
print("\n[2] Reset de l'environnement...")
try:
    obs, info = env.reset()
    print(f"[OK] Reset OK! Observation shape: {obs['image'].shape}")
    print(f"   Mission: {obs['mission']}")
except Exception as e:
    print(f"[ERREUR] : {e}")
    exit(1)

# Test 3: Prendre des actions aléatoires
print("\n[3] Test de 10 actions aléatoires...")
try:
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"   Step {i+1}: action={action}, reward={reward:.2f}, done={terminated or truncated}")
        
        if terminated or truncated:
            print("   Episode terminé, reset...")
            obs, info = env.reset()
    print("[OK] Actions OK!")
except Exception as e:
    print(f"[ERREUR] : {e}")
    exit(1)

# Test 4: Tester différents environnements
print("\n[4] Test de différents environnements MiniGrid...")
envs_to_test = [
    "MiniGrid-Empty-5x5-v0",
    "MiniGrid-DoorKey-5x5-v0",
    "MiniGrid-MultiRoom-N2-S4-v0",
]

for env_name in envs_to_test:
    try:
        test_env = gym.make(env_name, render_mode="rgb_array")
        test_obs, _ = test_env.reset()
        test_env.close()
        print(f"[OK] {env_name} : OK")
    except Exception as e:
        print(f"[ERREUR] {env_name} : {e}")

env.close()

print("\n" + "=" * 50)
print("[OK] TOUS LES TESTS PASSED! Tu es prêt à continuer.")
print("=" * 50)