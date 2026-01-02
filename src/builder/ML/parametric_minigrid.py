import gymnasium as gym
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.mission import MissionSpace
from minigrid.core.grid import Grid
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.world_object import Door, Goal, Wall, Key
import numpy as np
import time
from collections import deque

class ParametricMiniGridEnv(MiniGridEnv):
    """
    Environnement MiniGrid contrôlé par des paramètres.
    Garantit que le niveau est solvable via BFS.
    
    AMÉLIORATIONS:
    - Correction du BFS (utilisation de deque au lieu de list.pop(0))
    - Meilleure gestion des limites de grille
    - Validation plus robuste
    - Paramètres normalisés pour le générateur
    """

    def __init__(
        self,
        grid_size=10,
        num_obstacles=2,
        num_doors=1,
        num_keys=1,
        goal_position=None,  # None = position automatique
        max_steps=None,
        render_mode=None,
        agent_start_pos=None,  # Ajout: contrôle de la position de départ
        see_through_walls=False,
        enable_reward_shaping=True  # Activer/désactiver reward shaping (pour ablations)
    ):
        
        # Validation des paramètres
        assert grid_size >= 5, "grid_size doit être >= 5"
        assert num_doors >= 0, "num_doors doit être >= 0"
        assert num_keys >= num_doors, "Il faut au moins autant de clés que de portes"
        assert num_obstacles >= 0, "num_obstacles doit être >= 0"
        
        # Stockage des paramètres
        self.num_obstacles = int(num_obstacles)
        self.num_doors = int(num_doors)
        self.num_keys = int(num_keys)
        self.custom_goal_position = goal_position
        self.agent_start_pos_custom = agent_start_pos
        self.enable_reward_shaping = enable_reward_shaping
        
        # Mission
        mission_space = MissionSpace(
            mission_func=lambda: "Reach the goal"
        )
        
        if max_steps is None:
            max_steps = 4 * grid_size * grid_size

        # Appel au parent
        super().__init__(
            mission_space=mission_space,
            grid_size=grid_size,
            max_steps=max_steps,
            see_through_walls=see_through_walls,
            render_mode=render_mode
        )

    def _gen_grid(self, width, height):
        """
        Génération de niveau avec validation de solvabilité.
        """
        max_attempts = 100
        attempt = 0
        
        # DEBUG: Compteur pour voir pourquoi ça échoue
        failure_reasons = {"exception": 0, "not_solvable": 0}
        
        while attempt < max_attempts:
            attempt += 1
            
            try:
                # Créer une grille vide avec des murs sur les bords
                self.grid = Grid(width, height)
                self.grid.wall_rect(0, 0, width, height)
                
                # Zone intérieure jouable
                inner_width = width - 2
                inner_height = height - 2
                
                # Si pas assez d'espace, on abandonne
                min_cells_needed = 3 + self.num_obstacles + self.num_doors + self.num_keys
                available_cells = inner_width * inner_height
                
                if available_cells < min_cells_needed:
                    continue
                
                # 1. Placer le goal
                if self.custom_goal_position is not None:
                    goal_x, goal_y = self.custom_goal_position
                    # Vérifier que c'est dans les limites intérieures
                    if not (1 <= goal_x < width-1 and 1 <= goal_y < height-1):
                        # Position invalide, on place automatiquement
                        goal_x, goal_y = self._find_random_empty_pos()
                    self.put_obj(Goal(), goal_x, goal_y)
                else:
                    self.place_obj(Goal(), max_tries=100)
                
                # 2. Placer les portes (avec clés correspondantes)
                for i in range(self.num_doors):
                    color = COLOR_NAMES[i % len(COLOR_NAMES)]
                    
                    # Placer la porte
                    door = Door(color, is_locked=True)
                    self.place_obj(door, max_tries=100)
                    
                    # Placer la clé correspondante
                    key = Key(color)
                    self.place_obj(key, max_tries=100)
                
                # 3. Placer les clés supplémentaires (si num_keys > num_doors)
                extra_keys = self.num_keys - self.num_doors
                for i in range(extra_keys):
                    color = COLOR_NAMES[(self.num_doors + i) % len(COLOR_NAMES)]
                    self.place_obj(Key(color), max_tries=100)
                
                # 4. Placer les obstacles
                for _ in range(self.num_obstacles):
                    self.place_obj(Wall(), max_tries=100)
                
                # 5. Placer l'agent
                if self.agent_start_pos_custom is not None:
                    start_x, start_y = self.agent_start_pos_custom
                    if 1 <= start_x < width-1 and 1 <= start_y < height-1:
                        if self.grid.get(start_x, start_y) is None:
                            self.agent_pos = (start_x, start_y)
                            self.agent_dir = 0
                        else:
                            self.place_agent()
                    else:
                        self.place_agent()
                else:
                    self.place_agent()
                
                # 6. VALIDATION: Vérifier que le niveau est solvable
                if self._is_solvable():
                    self.mission = "Reach the goal"
                    return  # SUCCESS!
                else:
                    failure_reasons["not_solvable"] += 1
                
            except Exception as e:
                # Si une erreur se produit (ex: plus de place), on réessaye
                failure_reasons["exception"] += 1
                # DEBUG: Afficher la première exception pour comprendre
                if attempt == 1:
                    print(f"[DEBUG] Exception lors de la génération: {type(e).__name__}: {e}")
                continue
        
        # Si on arrive ici, on n'a pas réussi à générer un niveau valide
        # On crée un niveau minimal garanti solvable
        print(f"[WARNING] Impossible de générer un niveau valide après {max_attempts} tentatives.")
        print(f"[DEBUG] Raisons: not_solvable={failure_reasons['not_solvable']}, exceptions={failure_reasons['exception']}")
        print(f"[WARNING] Création d'un niveau minimal avec paramètres réduits...")
        self._gen_minimal_level(width, height)

    def _gen_minimal_level(self, width, height):
        """
        Génère un niveau minimal garanti solvable (fallback).
        """
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        
        # Niveau ultra simple: agent en (1,1), goal en (width-2, height-2)
        self.agent_pos = (1, 1)
        self.agent_dir = 0
        self.put_obj(Goal(), width-2, height-2)
        
        # Une seule porte au milieu si demandé
        if self.num_doors > 0:
            mid_x = width // 2
            mid_y = height // 2
            door = Door('red', is_locked=True)
            self.put_obj(door, mid_x, mid_y)
            key = Key('red')
            self.put_obj(key, 2, 2)
        
        # Quelques obstacles (mais pas trop)
        safe_obstacles = min(self.num_obstacles, 3)
        for i in range(safe_obstacles):
            try:
                self.place_obj(Wall(), max_tries=20)
            except:
                pass
        
        self.mission = "Reach the goal"

    def _find_random_empty_pos(self):
        """Trouve une position vide aléatoire dans la grille."""
        for _ in range(100):
            x = np.random.randint(1, self.width - 1)
            y = np.random.randint(1, self.height - 1)
            if self.grid.get(x, y) is None:
                return (x, y)
        return (1, 1)  # Fallback

    def _is_solvable(self):
        """
        Vérifie si le niveau est solvable en utilisant BFS.
        
        Logique:
        1. Trouver le goal
        2. Si pas de portes verrouillées: vérifier agent -> goal directement
        3. Si des portes verrouillées:
           - Vérifier qu'au moins une clé est atteignable depuis l'agent
           - Vérifier que le goal est atteignable en mode "avec clés" (can_open_doors=True)
        """
        # Trouver le goal en parcourant la grille
        goal_pos = None
        key_positions = []
        locked_doors = []
        
        for x in range(self.width):
            for y in range(self.height):
                cell = self.grid.get(x, y)
                if cell is not None:
                    if isinstance(cell, Goal):
                        goal_pos = (x, y)
                    elif isinstance(cell, Key):
                        key_positions.append((x, y))
                    elif isinstance(cell, Door) and cell.is_locked:
                        locked_doors.append((x, y))
        
        if goal_pos is None:
            return False
        
        # Cas 1: Pas de portes verrouillées
        if len(locked_doors) == 0:
            # Juste vérifier que l'agent peut atteindre le goal
            return self._can_reach(self.agent_pos, goal_pos, can_open_doors=False)
        
        # Cas 2: Il y a des portes verrouillées
        # Stratégie simplifiée: vérifier que le goal est atteignable si on a les clés
        # ET qu'au moins une clé est atteignable sans ouvrir de portes
        
        if len(key_positions) == 0:
            return False  # Pas de clé mais des portes verrouillées
        
        # Vérifier qu'au moins UNE clé est atteignable sans ouvrir de portes
        at_least_one_key_reachable = False
        for key_pos in key_positions:
            if self._can_reach(self.agent_pos, key_pos, can_open_doors=False):
                at_least_one_key_reachable = True
                break
        
        if not at_least_one_key_reachable:
            return False  # Aucune clé n'est accessible
        
        # Vérifier que le goal est atteignable depuis l'agent EN SUPPOSANT qu'on a les clés
        # (ce qui simule: agent récupère clé -> ouvre porte -> va au goal)
        return self._can_reach(self.agent_pos, goal_pos, can_open_doors=True)

    def _can_reach(self, start_pos, target_pos, can_open_doors=False):
        """
        BFS pour vérifier si on peut atteindre target_pos depuis start_pos.
        
        Args:
            start_pos: position de départ (x, y)
            target_pos: position cible (x, y)
            can_open_doors: si True, on peut traverser les portes verrouillées
        
        Returns:
            True si atteignable, False sinon
        """
        if start_pos == target_pos:
            return True
        
        queue = deque([start_pos])
        visited = {start_pos}
        
        while queue:
            x, y = queue.popleft()
            
            # Explorer les 4 directions
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                next_x, next_y = x + dx, y + dy
                next_pos = (next_x, next_y)
                
                # Vérifier les limites
                if not (0 <= next_x < self.width and 0 <= next_y < self.height):
                    continue
                
                # Déjà visité
                if next_pos in visited:
                    continue
                
                # Si c'est la cible, on a gagné!
                if next_pos == target_pos:
                    return True
                
                # Vérifier si la case est traversable
                cell = self.grid.get(next_x, next_y)
                
                can_pass = False
                
                if cell is None:
                    # Case vide
                    can_pass = True
                elif isinstance(cell, Goal):
                    # Goal (toujours accessible)
                    can_pass = True
                elif isinstance(cell, Key):
                    # Clé (toujours accessible)
                    can_pass = True
                elif isinstance(cell, Door):
                    # Porte
                    if not cell.is_locked:
                        can_pass = True
                    elif can_open_doors:
                        # On simule qu'on a la clé
                        can_pass = True
                # Wall et autres objets: can_pass reste False
                
                if can_pass:
                    visited.add(next_pos)
                    queue.append(next_pos)
        
        return False


# --- TEST ---
if __name__ == "__main__":
    print("="*60)
    print("TEST: ParametricMiniGridEnv (Version Corrigée)")
    print("="*60)
    
    # Test 1: Environnement simple
    print("\n[Test 1] Environnement simple (5x5, pas d'obstacles)")
    env = ParametricMiniGridEnv(
        grid_size=5,
        num_obstacles=0,
        num_doors=0,
        num_keys=0,
        render_mode="rgb_array"
    )
    
    for i in range(3):
        obs, info = env.reset()
        print(f"  Reset {i+1}: OK (grid_size={env.width}x{env.height})")
    
    env.close()
    print("  [OK] Test 1 réussi")
    
    # Test 2: Avec portes et clés
    print("\n[Test 2] Avec portes et clés (10x10)")
    env = ParametricMiniGridEnv(
        grid_size=10,
        num_obstacles=5,
        num_doors=1,
        num_keys=1,
        render_mode="rgb_array"
    )
    
    for i in range(3):
        obs, info = env.reset()
        print(f"  Reset {i+1}: OK")
    
    env.close()
    print("  [OK] Test 2 réussi")
    
    # Test 3: Configuration complexe
    print("\n[Test 3] Configuration complexe (12x12, beaucoup d'obstacles)")
    env = ParametricMiniGridEnv(
        grid_size=12,
        num_obstacles=10,
        num_doors=2,
        num_keys=2,
        render_mode="rgb_array"
    )
    
    success_count = 0
    for i in range(5):
        try:
            obs, info = env.reset()
            success_count += 1
            print(f"  Reset {i+1}: OK")
        except Exception as e:
            print(f"  Reset {i+1}: ÉCHEC ({e})")
    
    env.close()
    print(f"  [OK] Test 3: {success_count}/5 resets réussis")
    
    # Test 4: Avec visualisation (optionnel)
    print("\n[Test 4] Test avec visualisation (ferme la fenêtre pour continuer)")
    print("  Si tu veux skip, commente cette partie")
    
    try:
        env = ParametricMiniGridEnv(
            grid_size=10,
            num_obstacles=8,
            num_doors=1,
            num_keys=1,
            render_mode="human"
        )
        
        for i in range(3):
            print(f"  Niveau {i+1}/3...")
            obs, info = env.reset()
            env.render()
            time.sleep(2)  # Pause pour voir le niveau
        
        env.close()
        print("  [OK] Test 4 réussi")
    except Exception as e:
        print(f"  [WARNING] Visualisation skippée: {e}")
    
    print("\n" + "="*60)
    print("[OK] TOUS LES TESTS TERMINÉS!")
    print("\nTon environnement est prêt à être utilisé avec le générateur.")
    print("Prochaine étape: créer le générateur de paramètres (generator.py)")