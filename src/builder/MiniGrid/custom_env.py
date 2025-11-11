import gymnasium as gym
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.mission import MissionSpace
from minigrid.core.grid import Grid
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.world_object import Door, Goal, Wall, Key
import numpy as np
import time
import random

class ParametricMiniGridEnv(MiniGridEnv):
    """
    Environnement MiniGrid contrôlé par des paramètres d'Architecte.
    Garantit que le niveau est solutionnable via BFS.
    """

    def __init__(
        self,
        grid_size=10,
        num_obstacles=2,
        num_doors=1,
        num_keys=1,
        goal_position=(8, 8), 
        max_steps=None,
        render_mode=None
    ):
        
        # 1. Stockage des paramètres
        self.num_obstacles = num_obstacles
        self.num_doors = num_doors
        self.num_keys = num_keys
        self.goal_position = goal_position
        
        # 2. Définition de la Mission
        mission_space = MissionSpace(mission_func=lambda: "Trouver la clé, ouvrir la porte, et atteindre la case verte")
        
        if max_steps is None:
            max_steps = 4 * grid_size * grid_size

        # 3. Appel au parent (initialisation de l'environnement)
        super().__init__(
            mission_space=mission_space,
            grid_size=grid_size,
            max_steps=max_steps,
            render_mode=render_mode
        )

    def _gen_grid(self, width, height):
        """
        La fonction de génération de niveau avec la boucle de Contrôle Qualité.
        """
        is_valid = False
        max_attempts = 100 
        attempt_count = 0
        
        # Boucle de validation (garantit la jouabilité)
        while not is_valid and attempt_count < max_attempts:
            
            # RAZ de la grille à chaque tentative
            self.grid = Grid(width, height)
            self.grid.wall_rect(0, 0, width, height)
            
            goal_x, goal_y = self.goal_position
            
            try:
                # 1. Placer le But (on laisse la possibilité de planter)
                self.put_obj(Goal(), goal_x, goal_y)
                
                colors = COLOR_NAMES
                
                # 2. Placer les Portes Verrouillées
                for i in range(self.num_doors):
                    color = colors[i % len(colors)]
                    self.place_obj(Door(color, is_locked=True))

                # 3. Placer les Clés
                for i in range(self.num_keys):
                    color = colors[i % len(colors)]
                    self.place_obj(Key(color))

                # 4. Placer les obstacles
                for _ in range(self.num_obstacles):
                    self.place_obj(Wall())

                # 5. Placer l'agent
                self.place_agent()
                
                # 6. Contrôle Qualité : VÉRIFIER SI C'EST JOUABLE
                is_valid = self._is_solvable()
            
            except Exception:
                # Si put_obj ou place_agent échouent (ex: hors limites), on rejette
                is_valid = False 

            attempt_count += 1
        
        if not is_valid:
            raise Exception("Génération de niveau impossible après 100 tentatives. Les paramètres sont trop stricts.")

        self.mission = "Trouver la clé, ouvrir la porte, et atteindre la case verte"


    # --- Outils de Pathfinding (BFS) ---

    def _is_reachable(self, start_pos, target_type):
        """
        Vérifie si un objet de type 'target_type' est atteignable depuis 'start_pos' (BFS).
        """
        queue = [start_pos]
        visited = {start_pos}

        while queue:
            curr_pos = queue.pop(0)
            x, y = curr_pos

            cell = self.grid.get(x, y)
            if cell is not None and isinstance(cell, target_type):
                return True

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                next_pos = (x + dx, y + dy)
                
                if next_pos not in visited and \
                   self.grid.is_valid(*next_pos) and \
                   next_pos[0] > 0 and next_pos[1] > 0 and \
                   next_pos[0] < self.width - 1 and next_pos[1] < self.height - 1:
                    
                    neighbor_cell = self.grid.get(*next_pos)
                    
                    # On peut passer si c'est vide, le but, une clé, ou une porte OUVERTE
                    if neighbor_cell is None or \
                       isinstance(neighbor_cell, Goal) or \
                       isinstance(neighbor_cell, Key) or \
                       (isinstance(neighbor_cell, Door) and not neighbor_cell.is_locked):
                        
                        visited.add(next_pos)
                        queue.append(next_pos)
                        
        return False

    def _is_solvable(self):
        """
        Vérifie la chaîne de solution: Agent -> Clé -> Porte -> But
        """
        # Assurez-vous d'avoir assez de clés/portes
        key_pos = self.grid.find_objects(Key)
        door_pos = self.grid.find_objects(Door, lambda x: x.is_locked)
        
        if len(key_pos) != self.num_keys or len(door_pos) != self.num_doors:
            return False 

        # A. Agent -> Clé
        start_agent = self.agent_pos
        key_reachable = self._is_reachable(start_agent, Key)

        # B. Clé -> Porte (On suppose qu'on a la clé)
        start_key = key_pos[0][0] 
        door_reachable = self._is_reachable(start_key, Door)

        # C. Porte -> But (On suppose qu'on a ouvert la porte)
        start_door = door_pos[0][0] 
        goal_reachable = self._is_reachable(start_door, Goal)
        
        return key_reachable and door_reachable and goal_reachable


# --- Brique 4 : Le Test de Conduite (Test manuel) ---

if __name__ == "__main__":
    
    print("--- Démarrage du Test de Conduite (Garantie Jouable) ---")

    env = ParametricMiniGridEnv(
        grid_size=12,
        num_obstacles=10,
        num_doors=1,
        num_keys=1,
        goal_position=(10, 10),
        render_mode="human"
    )

    print("[OK] Environnement créé (Garantie de solution activée)")
    print("Lancement de 5 niveaux uniques (fermez la fenêtre pour arrêter)...")

    for i in range(5):
        print(f"Épisode {i+1}...")
        
        # Le reset appelle _gen_grid, qui appelle la boucle de validation
        obs, info = env.reset() 
        
        try:
            env.render() 
            time.sleep(1) 
            
        except Exception as e:
            print(f"\nFenêtre fermée ou erreur de rendu : {e}")
            break 
    
    print("--- Test de Conduite Terminé ---")
    env.close()