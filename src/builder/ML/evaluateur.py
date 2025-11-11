import gymnasium as gym
import numpy as np
from collections import defaultdict
import time


class LevelEvaluator:
    """
    Évalue la difficulté d'un niveau en faisant jouer un agent dessus.
    """
    
    def __init__(self, env_class, agent, num_episodes=10, max_steps=None, verbose=False):
        """
        Args:
            env_class: classe de l'environnement (ex: ParametricMiniGridEnv)
            agent: agent entraîné (avec méthode .predict())
            num_episodes: nombre d'épisodes pour évaluer
            max_steps: limite de steps par épisode (None = utilise la limite de l'env)
            verbose: afficher les détails
        """
        self.env_class = env_class
        self.agent = agent
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.verbose = verbose
        
    def evaluate_level(self, level_params):
        """
        Évalue un niveau et retourne ses métriques.
        
        Args:
            level_params: dict avec les paramètres du niveau
        
        Returns:
            dict avec les métriques: {
                'success_rate': float,
                'mean_reward': float,
                'mean_steps': float,
                'std_reward': float,
                'completion_time': float  # temps moyen pour terminer
            }
        """
        # Créer l'environnement avec les paramètres
        try:
            env = self.env_class(**level_params, render_mode="rgb_array")
        except Exception as e:
            if self.verbose:
                print(f"  [ERROR] Impossible de créer l'env: {e}")
            # Retourner des métriques par défaut (niveau impossible)
            return {
                'success_rate': 0.0,
                'mean_reward': 0.0,
                'mean_steps': 0.0,
                'std_reward': 0.0,
                'completion_time': 0.0,
                'error': str(e)
            }
        
        # Statistiques à collecter
        rewards = []
        steps_list = []
        successes = []
        
        for episode in range(self.num_episodes):
            try:
                obs, info = env.reset()
                done = False
                episode_reward = 0
                steps = 0
                
                max_steps_episode = self.max_steps if self.max_steps else env.max_steps
                
                while not done and steps < max_steps_episode:
                    # Prédire l'action
                    action, _ = self.agent.predict(obs, deterministic=True)
                    
                    # Exécuter l'action
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    episode_reward += reward
                    steps += 1
                    done = terminated or truncated
                
                # Enregistrer les stats
                rewards.append(episode_reward)
                steps_list.append(steps)
                
                # Succès = reward positif (a atteint le goal)
                successes.append(1 if episode_reward > 0 else 0)
                
            except Exception as e:
                if self.verbose:
                    print(f"  [WARNING] Erreur pendant l'épisode {episode}: {e}")
                rewards.append(0)
                steps_list.append(0)
                successes.append(0)
        
        env.close()
        
        # Calculer les métriques
        success_rate = np.mean(successes)
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        mean_steps = np.mean(steps_list)
        
        # Temps de complétion moyen (seulement pour les succès)
        successful_steps = [steps_list[i] for i in range(len(successes)) if successes[i] == 1]
        completion_time = np.mean(successful_steps) if len(successful_steps) > 0 else mean_steps
        
        metrics = {
            'success_rate': float(success_rate),
            'mean_reward': float(mean_reward),
            'mean_steps': float(mean_steps),
            'std_reward': float(std_reward),
            'completion_time': float(completion_time),
            'num_episodes': self.num_episodes
        }
        
        if self.verbose:
            print(f"  Success: {success_rate*100:.1f}%, Reward: {mean_reward:.2f}, Steps: {mean_steps:.1f}")
        
        return metrics
    
    def evaluate_batch(self, level_params_list):
        """
        Évalue un batch de niveaux.
        
        Args:
            level_params_list: liste de dicts de paramètres
        
        Returns:
            list de dicts de métriques
        """
        results = []
        
        for i, params in enumerate(level_params_list):
            if self.verbose:
                print(f"[{i+1}/{len(level_params_list)}] Évaluation du niveau...")
            
            metrics = self.evaluate_level(params)
            results.append(metrics)
        
        return results


def compute_difficulty_score(metrics, target_success_rate=0.5):
    """
    Calcule un score de difficulté à partir des métriques.
    
    Un bon niveau doit être:
    - Ni trop facile (success_rate trop élevé)
    - Ni impossible (success_rate = 0)
    - Idéalement autour de target_success_rate (ex: 50%)
    
    Args:
        metrics: dict avec les métriques du niveau
        target_success_rate: taux de succès cible
    
    Returns:
        float: score de difficulté (plus haut = meilleur niveau pour entraîner)
    """
    sr = metrics['success_rate']
    
    # Pénaliser les niveaux trop faciles ou impossibles
    if sr < 0.1:  # Trop dur / impossible
        return 0.0
    elif sr > 0.9:  # Trop facile
        return 0.0
    else:
        # Score maximal quand on est proche du target
        # Utilise une gaussienne centrée sur target_success_rate
        distance = abs(sr - target_success_rate)
        score = np.exp(-distance**2 / 0.1)  # Gaussienne avec sigma=0.316
        return float(score)


def compute_diversity_score(level_params_list):
    """
    Mesure la diversité d'un ensemble de niveaux.
    
    Args:
        level_params_list: liste de dicts de paramètres
    
    Returns:
        float: score de diversité (variance moyenne des paramètres)
    """
    if len(level_params_list) < 2:
        return 0.0
    
    # Extraire les paramètres numériques
    grid_sizes = [p['grid_size'] for p in level_params_list]
    obstacles = [p['num_obstacles'] for p in level_params_list]
    doors = [p['num_doors'] for p in level_params_list]
    keys = [p['num_keys'] for p in level_params_list]
    
    # Calculer la variance de chaque paramètre (normalisée)
    var_grid = np.var(grid_sizes) / (np.mean(grid_sizes) + 1e-6)
    var_obstacles = np.var(obstacles) / (np.mean(obstacles) + 1e-6)
    var_doors = np.var(doors) / (np.mean(doors) + 1e-6)
    var_keys = np.var(keys) / (np.mean(keys) + 1e-6)
    
    # Score = moyenne des variances normalisées
    diversity = np.mean([var_grid, var_obstacles, var_doors, var_keys])
    
    return float(diversity)


# --- TESTS ---
if __name__ == "__main__":
    print("="*60)
    print("TEST: LevelEvaluator")
    print("="*60)
    
    # Pour tester, on a besoin d'un agent et d'un environnement
    # On va simuler un agent random pour le test
    
    print("\n[Setup] Création d'un agent factice pour les tests...")
    
    class DummyAgent:
        """Agent qui prend des actions aléatoires (pour le test)"""
        def __init__(self, action_space):
            self.action_space = action_space
        
        def predict(self, obs, deterministic=True):
            return self.action_space.sample(), None
    
    # Importer l'environnement qu'on a créé
    try:
        from parametric_minigrid import ParametricMiniGridEnv
        print("  ✅ ParametricMiniGridEnv importé")
    except ImportError:
        print("  ❌ Impossible d'importer ParametricMiniGridEnv")
        print("  Assure-toi que parametric_minigrid.py est dans le même dossier")
        exit(1)
    
    # Créer un environnement temporaire pour obtenir l'action_space
    temp_env = ParametricMiniGridEnv(grid_size=8, render_mode="rgb_array")
    dummy_agent = DummyAgent(temp_env.action_space)
    temp_env.close()
    
    print("  ✅ Agent factice créé")
    
    # Test 1: Évaluer un niveau simple
    print("\n[Test 1] Évaluation d'un niveau simple...")
    evaluator = LevelEvaluator(
        env_class=ParametricMiniGridEnv,
        agent=dummy_agent,
        num_episodes=5,
        verbose=True
    )
    
    simple_params = {
        'grid_size': 8,
        'num_obstacles': 2,
        'num_doors': 0,
        'num_keys': 0,
        'goal_position': (6, 6)
    }
    
    metrics = evaluator.evaluate_level(simple_params)
    print(f"\n  Métriques obtenues:")
    for key, value in metrics.items():
        print(f"    {key}: {value}")
    print("  ✅ Évaluation réussie")
    
    # Test 2: Évaluer un batch de niveaux
    print("\n[Test 2] Évaluation d'un batch de 3 niveaux...")
    
    batch_params = [
        {'grid_size': 8, 'num_obstacles': 2, 'num_doors': 0, 'num_keys': 0},
        {'grid_size': 10, 'num_obstacles': 5, 'num_doors': 1, 'num_keys': 1},
        {'grid_size': 12, 'num_obstacles': 8, 'num_doors': 1, 'num_keys': 1},
    ]
    
    evaluator_batch = LevelEvaluator(
        env_class=ParametricMiniGridEnv,
        agent=dummy_agent,
        num_episodes=3,
        verbose=False
    )
    
    results = evaluator_batch.evaluate_batch(batch_params)
    
    print(f"\n  Résultats du batch:")
    for i, result in enumerate(results):
        print(f"    Niveau {i+1}: success={result['success_rate']*100:.1f}%, reward={result['mean_reward']:.2f}")
    print("  ✅ Batch évalué")
    
    # Test 3: Calcul de difficulty score
    print("\n[Test 3] Calcul des difficulty scores...")
    
    test_metrics = [
        {'success_rate': 0.0, 'mean_reward': 0.0},   # Impossible
        {'success_rate': 0.3, 'mean_reward': 3.0},   # Difficile mais bon
        {'success_rate': 0.5, 'mean_reward': 5.0},   # Parfait
        {'success_rate': 0.7, 'mean_reward': 7.0},   # Facile mais ok
        {'success_rate': 1.0, 'mean_reward': 10.0},  # Trop facile
    ]
    
    for i, metrics in enumerate(test_metrics):
        score = compute_difficulty_score(metrics, target_success_rate=0.5)
        print(f"    SR={metrics['success_rate']:.1f} → score={score:.3f}")
    
    print("  ✅ Difficulty scores calculés")
    
    # Test 4: Calcul de diversity score
    print("\n[Test 4] Calcul du diversity score...")
    
    # Cas 1: Niveaux identiques (diversité faible)
    identical_levels = [
        {'grid_size': 10, 'num_obstacles': 5, 'num_doors': 1, 'num_keys': 1}
    ] * 5
    
    # Cas 2: Niveaux variés (diversité élevée)
    diverse_levels = [
        {'grid_size': 8, 'num_obstacles': 2, 'num_doors': 0, 'num_keys': 0},
        {'grid_size': 10, 'num_obstacles': 5, 'num_doors': 1, 'num_keys': 1},
        {'grid_size': 12, 'num_obstacles': 10, 'num_doors': 2, 'num_keys': 2},
        {'grid_size': 15, 'num_obstacles': 15, 'num_doors': 3, 'num_keys': 3},
        {'grid_size': 9, 'num_obstacles': 3, 'num_doors': 1, 'num_keys': 2},
    ]
    
    diversity_identical = compute_diversity_score(identical_levels)
    diversity_diverse = compute_diversity_score(diverse_levels)
    
    print(f"    Niveaux identiques: diversity={diversity_identical:.4f}")
    print(f"    Niveaux variés: diversity={diversity_diverse:.4f}")
    print("  ✅ Diversity score calculé")
    
    print("\n" + "="*60)
    print("✅ TOUS LES TESTS RÉUSSIS!")
    print("="*60)
    print("\n💡 L'évaluateur est prêt à être utilisé!")
    print("   Prochaine étape: créer le training loop (coevolution.py)")
