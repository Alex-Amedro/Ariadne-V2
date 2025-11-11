import torch
import torch.nn as nn
import numpy as np


class LevelGenerator(nn.Module):
    
    def __init__(self, latent_dim=16, hidden_dim=64):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Architecture du générateur
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 6)  # 6 paramètres de sortie
        )
        
        # Paramètres pour normaliser les outputs
        self.min_grid_size = 8
        self.max_grid_size = 15
        self.max_obstacles = 15
        self.max_doors = 3
        self.max_keys = 3
        
    def forward(self, z):
        # Passer dans le réseau
        raw_output = self.network(z)
        
        # Appliquer des activations appropriées pour chaque paramètre
        # On utilise sigmoid pour normaliser entre 0 et 1, puis on scale
        
        grid_size = torch.sigmoid(raw_output[..., 0])  # [0, 1]
        grid_size = self.min_grid_size + grid_size * (self.max_grid_size - self.min_grid_size)
        
        num_obstacles = torch.sigmoid(raw_output[..., 1]) * self.max_obstacles
        num_doors = torch.sigmoid(raw_output[..., 2]) * self.max_doors
        num_keys = torch.sigmoid(raw_output[..., 3]) * self.max_keys
        
        # Goal position (normalisé entre 0 et 1, sera scalé selon grid_size plus tard)
        goal_x = torch.sigmoid(raw_output[..., 4])
        goal_y = torch.sigmoid(raw_output[..., 5])
        
        return {
            'grid_size': grid_size,
            'num_obstacles': num_obstacles,
            'num_doors': num_doors,
            'num_keys': num_keys,
            'goal_x': goal_x,
            'goal_y': goal_y
        }
    
    def generate_level_params(self, z=None, deterministic=False):
        if z is None:
            if deterministic:
                z = torch.zeros(self.latent_dim)
            else:
                z = torch.randn(self.latent_dim)
        
        with torch.no_grad():
            raw_params = self.forward(z)
            
            # Convertir en valeurs utilisables
            grid_size = int(raw_params['grid_size'].item())
            num_obstacles = int(raw_params['num_obstacles'].item())
            num_doors = int(raw_params['num_doors'].item())
            num_keys = max(int(raw_params['num_keys'].item()), num_doors)  # Au moins autant de clés que de portes
            
            # Goal position (relatif à la taille de la grille)
            goal_x = int(1 + raw_params['goal_x'].item() * (grid_size - 3))
            goal_y = int(1 + raw_params['goal_y'].item() * (grid_size - 3))
            
            # Assurer que goal est dans les limites
            goal_x = max(1, min(goal_x, grid_size - 2))
            goal_y = max(1, min(goal_y, grid_size - 2))
            
            return {
                'grid_size': grid_size,
                'num_obstacles': num_obstacles,
                'num_doors': num_doors,
                'num_keys': num_keys,
                'goal_position': (goal_x, goal_y)
            }
    
    def generate_batch(self, batch_size=32):
        """
        Génère un batch de paramètres de niveaux.
        
        Returns:
            list de dicts de paramètres
        """
        z_batch = torch.randn(batch_size, self.latent_dim)
        
        levels = []
        for i in range(batch_size):
            params = self.generate_level_params(z_batch[i])
            levels.append(params)
        
        return levels


class LevelBuffer:
    """
    Buffer pour stocker les niveaux générés et leurs métriques.
    Utile pour l'entraînement du générateur.
    """
    
    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.buffer = []
    
    def add(self, level_params, metrics):
        """
        Ajoute un niveau et ses métriques au buffer.
        
        Args:
            level_params: dict des paramètres du niveau
            metrics: dict avec 'success_rate', 'mean_reward', 'mean_steps', etc.
        """
        self.buffer.append({
            'params': level_params,
            'metrics': metrics
        })
        
        # Limiter la taille du buffer
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)
    
    def get_best_levels(self, k=10, criterion='difficulty'):
        """
        Récupère les k meilleurs niveaux selon un critère.
        
        Args:
            k: nombre de niveaux à retourner
            criterion: 'difficulty', 'diversity', etc.
        
        Returns:
            list de dicts
        """
        if len(self.buffer) == 0:
            return []
        
        if criterion == 'difficulty':
            # Trier par difficulté (success_rate proche de 0.3-0.7 = optimal)
            def difficulty_score(item):
                sr = item['metrics'].get('success_rate', 0.5)
                # On veut des niveaux challengeants mais pas impossibles
                return -abs(sr - 0.5)  # Négatif pour trier en ordre décroissant
            
            sorted_buffer = sorted(self.buffer, key=difficulty_score, reverse=True)
        else:
            sorted_buffer = self.buffer
        
        return sorted_buffer[:k]
    
    def get_statistics(self):
        """
        Calcule des statistiques sur les niveaux dans le buffer.
        """
        if len(self.buffer) == 0:
            return {}
        
        success_rates = [item['metrics']['success_rate'] for item in self.buffer]
        rewards = [item['metrics']['mean_reward'] for item in self.buffer]
        
        return {
            'count': len(self.buffer),
            'mean_success_rate': np.mean(success_rates),
            'std_success_rate': np.std(success_rates),
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards)
        }


# --- TESTS ---
if __name__ == "__main__":
    print("="*60)
    print("TEST: LevelGenerator (Architecte)")
    print("="*60)
    
    # Test 1: Créer le générateur
    print("\n[Test 1] Création du générateur...")
    generator = LevelGenerator(latent_dim=16, hidden_dim=64)
    print(f"  ✅ Générateur créé: {sum(p.numel() for p in generator.parameters())} paramètres")
    
    # Test 2: Générer un niveau
    print("\n[Test 2] Génération d'un niveau...")
    z = torch.randn(16)
    params = generator.generate_level_params(z)
    print(f"  Paramètres générés:")
    for key, value in params.items():
        print(f"    {key}: {value}")
    print("  ✅ Niveau généré avec succès")
    
    # Test 3: Générer un batch
    print("\n[Test 3] Génération d'un batch de 10 niveaux...")
    batch = generator.generate_batch(batch_size=10)
    print(f"  ✅ {len(batch)} niveaux générés")
    print(f"  Exemple de variation:")
    print(f"    Grid sizes: {[l['grid_size'] for l in batch[:5]]}")
    print(f"    Obstacles: {[l['num_obstacles'] for l in batch[:5]]}")
    
    # Test 4: Forward pass
    print("\n[Test 4] Forward pass (pour entraînement)...")
    z_batch = torch.randn(8, 16)
    output = generator(z_batch)
    print(f"  ✅ Output shape: grid_size={output['grid_size'].shape}")
    print(f"     Values range: [{output['grid_size'].min().item():.2f}, {output['grid_size'].max().item():.2f}]")
    
    # Test 5: LevelBuffer
    print("\n[Test 5] Test du LevelBuffer...")
    buffer = LevelBuffer(max_size=100)
    
    # Ajouter quelques niveaux fictifs
    for i in range(20):
        fake_params = generator.generate_level_params()
        fake_metrics = {
            'success_rate': np.random.rand(),
            'mean_reward': np.random.rand() * 10,
            'mean_steps': np.random.randint(10, 100)
        }
        buffer.add(fake_params, fake_metrics)
    
    stats = buffer.get_statistics()
    print(f"  Buffer contient {stats['count']} niveaux")
    print(f"  Success rate moyen: {stats['mean_success_rate']:.3f} ± {stats['std_success_rate']:.3f}")
    print(f"  ✅ Buffer fonctionne")
    
    # Test 6: Sauvegarder/Charger le modèle
    print("\n[Test 6] Sauvegarde/Chargement du modèle...")
    torch.save(generator.state_dict(), 'generator_test.pth')
    print("  ✅ Modèle sauvegardé: generator_test.pth")
    
    generator2 = LevelGenerator(latent_dim=16, hidden_dim=64)
    generator2.load_state_dict(torch.load('generator_test.pth'))
    print("  ✅ Modèle chargé avec succès")
    
    # Vérifier que les deux produisent le même output
    z_test = torch.randn(16)
    out1 = generator.generate_level_params(z_test)
    out2 = generator2.generate_level_params(z_test)
    
    assert out1['grid_size'] == out2['grid_size'], "Les modèles ne sont pas identiques!"
    print("  ✅ Vérification: les deux modèles sont identiques")
    
    print("\n" + "="*60)
    print("✅ TOUS LES TESTS RÉUSSIS!")
    print("="*60)
    print("\nProchaine étape: évaluateur de niveaux (evaluator.py)")
