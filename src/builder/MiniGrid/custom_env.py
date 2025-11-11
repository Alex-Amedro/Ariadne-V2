import gymnasium as gym # <--- On importe gym en haut
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.mission import MissionSpace
from minigrid.core.grid import Grid
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.world_object import Door, Goal, Wall, Key

import numpy as np
import time # <--- On importe time en haut
import random

class ParametricMiniGridEnv(MiniGridEnv):
    """
    Notre environnement custom, mis à jour (encore)
    """

    def __init__(
        self,
        grid_size=10,
        num_obstacles=2,
        num_doors=1,
        num_keys=1,
        goal_position=(8, 8), 
        max_steps=None,
        render_mode=None # <--- CORRECTION : On accepte le render_mode
    ):
        
        self.num_obstacles = num_obstacles
        self.num_doors = num_doors
        self.num_keys = num_keys
        self.goal_position = goal_position
        
        mission_space = MissionSpace(mission_func=lambda: "Trouver la clé, ouvrir la porte, et atteindre la case verte")
        
        if max_steps is None:
            max_steps = 4 * grid_size * grid_size

        super().__init__(
            mission_space=mission_space,
            grid_size=grid_size,
            max_steps=max_steps,
            render_mode=render_mode # <--- CORRECTION : On le passe au parent
        )

    def _gen_grid(self, width, height):
        
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        goal_x, goal_y = self.goal_position
        
        # On laisse le 'try...except' pour la Brique 3, pour l'instant
        # on fait confiance à l'appelant.
        self.put_obj(Goal(), goal_x, goal_y)
        
        colors = COLOR_NAMES
        
        for i in range(self.num_doors):
            color = colors[i % len(colors)]
            self.place_obj(Door(color, is_locked=True))

        for i in range(self.num_keys):
            color = colors[i % len(colors)]
            self.place_obj(Key(color))

        for _ in range(self.num_obstacles):
            self.place_obj(Wall())

        self.place_agent()
        
        self.mission = "Trouver la clé, ouvrir la porte, et atteindre la case verte"


    def _is_solvable(self):
        # (Sera implémenté dans la Brique 3)
        return True


# --- Brique 4 : Le Test de Conduite (MÉTHODE MODERNE) ---

if __name__ == "__main__":
    
    import gymnasium as gym # <--- On importe gym en haut
    import time             # <--- On importe time en haut

    print("--- Démarrage du Test de Conduite (Méthode Try/Except) ---")

    env = ParametricMiniGridEnv(
        grid_size=12,
        num_obstacles=10,
        num_doors=1,
        num_keys=1,
        goal_position=(10, 10),
        render_mode="human"
    )

    print("[OK] Environnement créé (grid_size=12, 1 porte, 1 clé)")
    print("Lancement de 10 épisodes de test (fermez la fenêtre pour arrêter)...")

    for i in range(10):
        print(f"Épisode {i+1}...")
        obs, info = env.reset()
        
        # --- LA CORRECTION ---
        # On essaie de 'render'. Si la fenêtre est fermée,
        # PyGame lèvera une erreur, et on l'attrapera.
        try:
            env.render() 
            time.sleep(3) # Pause de 1s pour voir
            
        except Exception as e:
            # L'exception est souvent 'pygame.error: display Surface quit'
            print(f"\nFenêtre fermée ou erreur de rendu détectée : {e}")
            break # On sort proprement de la boucle for
    
    print("--- Test de Conduite Terminé ---")
    env.close()