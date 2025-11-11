# --- Imports : Nos "Briques LEGO" ---
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.mission import MissionSpace
from minigrid.core.grid import Grid
from minigrid.core.constants import COLOR_NAMES

# <--- CORRECTION (Point 3) : On importe 'Key'
from minigrid.core.world_object import Door, Goal, Wall, Key

import numpy as np
import random # On importe 'random' pour la Brique 3


class ParametricMiniGridEnv(MiniGridEnv):
    """
    Notre environnement custom, mis à jour avec vos corrections.
    """

    def __init__(
        self,
        grid_size=10,
        num_obstacles=2,
        num_doors=1,
        # <--- CORRECTION (Point 3) : Ajout de 'num_keys'
        num_keys=1,
        goal_position=(8, 8), 
        max_steps=None
    ):
        
        # On stocke tous les paramètres
        self.num_obstacles = num_obstacles
        self.num_doors = num_doors
        self.num_keys = num_keys # <--- CORRECTION (Point 3)
        self.goal_position = goal_position
        
        # On met à jour la mission pour refléter la nouvelle complexité
        mission_space = MissionSpace(mission_func=lambda: "Trouver la clé, ouvrir la porte, et atteindre la case verte")
        
        if max_steps is None:
            max_steps = 4 * grid_size * grid_size

        super().__init__(
            mission_space=mission_space,
            grid_size=grid_size,
            max_steps=max_steps
        )

    # <--- CORRECTION (Syntaxe) : La fonction _gen_grid DOIT être
    # indentée À L'INTÉRIEUR de la classe pour fonctionner.
    def _gen_grid(self, width, height):
        """
        La "Planche à Dessin", version corrigée.
        """
        
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        goal_x, goal_y = self.goal_position
        
        # <--- CORRECTION (Point 1) : Suppression du "patch"
        # On fait confiance à l'Architecte. S'il place le but
        # hors des limites, self.put_obj plantera.
        # Plus tard (Brique 3), on attrapera ce plantage.
        self.put_obj(Goal(), goal_x, goal_y)
        
        
        # --- CORRECTION (Point 3) : Logique Clé/Porte ---
        
        # On prépare un cycle de couleurs pour nos clés/portes
        colors = COLOR_NAMES
        
        # 1. Placer les PORTES VERROUILLÉES
        for i in range(self.num_doors):
            color = colors[i % len(colors)]
            # is_locked=True, comme vous l'avez demandé !
            self.place_obj(Door(color, is_locked=True))

        # 2. Placer les CLÉS correspondantes
        for i in range(self.num_keys):
            color = colors[i % len(colors)]
            self.place_obj(Key(color))

        # 3. Placer les obstacles (murs)
        # (Logique inchangée, Point 4 : l'IA choisit 'num_obstacles',
        # 'place_obj' choisit la position aléatoire)
        for _ in range(self.num_obstacles):
            self.place_obj(Wall())

        # 4. Placer l'agent
        self.place_agent()
        
        # Mettre à jour la mission textuelle
        self.mission = "Trouver la clé, ouvrir la porte, et atteindre la case verte"


    # --- Brique 3 (à venir) ---
    def _is_solvable(self):
        """
        Vérifie si le niveau généré est solutionnable.
        (Sera implémenté dans la prochaine étape)
        """
        # Pour l'instant, on dit que c'est toujours bon
        return True


# --- Brique 4 : Le Test de Conduite (Mis à jour) ---

if __name__ == "__main__":
    
    import gymnasium as gym
    from minigrid.utils.window import Window
    import time

    print("--- Démarrage du Test de Conduite (Corrigé) ---")
    
    # On enregistre notre environnement
    # 'entry_point' est le nom_du_fichier:Nom_de_la_Classe
    gym.register(
        id='MiniGrid-Parametric-v0',
        entry_point='envs.custom_env:ParametricMiniGridEnv',
        # On peut mettre une limite de steps ici aussi
        max_episode_steps=100 
    )
    
    print("[OK] Environnement enregistré sous 'MiniGrid-Parametric-v0'")

    env = gym.make(
        'MiniGrid-Parametric-v0', 
        grid_size=12,
        num_obstacles=10,
        # <--- CORRECTION (Point 3) : Test avec 1 porte et 1 clé
        num_doors=1,
        num_keys=1,
        goal_position=(10, 10)
    )

    print("[OK] Environnement créé (grid_size=12, num_obstacles=10, 1 porte, 1 clé)")

    window = Window("MiniGrid - Test de Conduite")
    window.show(block=False) 

    print("Lancement de 10 épisodes de test...")
    for i in range(10):
        print(f"Épisode {i+1}...")
        obs, info = env.reset()
        
        window.set_image(env.get_frame())
        window.set_caption(f"Épisode {i+1}")
        
        # J'ai gardé votre 'sleep(3)'
        time.sleep(3) 
        
        if window.window.closed:
            break
    
    print("--- Test de Conduite Terminé ---")
    window.close()
