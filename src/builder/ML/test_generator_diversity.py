"""
Script de test pour vérifier la diversité des niveaux générés par LevelGenerator.
"""

import torch
from generator import LevelGenerator

print("="*60)
print("TEST: Diversité du LevelGenerator")
print("="*60)

# Créer le générateur
generator = LevelGenerator(latent_dim=16, hidden_dim=64)

print("\n[Test 1] Vérification de l'initialisation du réseau")
print("  Poids du premier layer:")
first_layer_weights = generator.network[0].weight
print(f"    Shape: {first_layer_weights.shape}")
print(f"    Mean: {first_layer_weights.mean().item():.4f}")
print(f"    Std: {first_layer_weights.std().item():.4f}")
print(f"    Min: {first_layer_weights.min().item():.4f}")
print(f"    Max: {first_layer_weights.max().item():.4f}")

if first_layer_weights.std().item() < 0.01:
    print("  [WARNING] Les poids ont une variance très faible!")
else:
    print("  [OK] Les poids semblent bien initialisés")

print("\n[Test 2] Génération de 5 z aléatoires différents")
z_samples = []
for i in range(5):
    z = torch.randn(16)
    z_samples.append(z)
    print(f"  z[{i}][:5] = {z[:5].tolist()}")

print("\n[Test 3] Forward pass avec ces z")
outputs = []
for i, z in enumerate(z_samples):
    with torch.no_grad():
        output = generator.forward(z.unsqueeze(0))
    outputs.append(output)
    print(f"  Output {i}:")
    print(f"    grid_size: {output['grid_size'].item():.2f}")
    print(f"    num_obstacles: {output['num_obstacles'].item():.2f}")
    print(f"    num_doors: {output['num_doors'].item():.2f}")

# Vérifier si les outputs sont identiques
print("\n[Test 4] Vérification de la diversité des outputs")
all_same = True
first_output = outputs[0]
for i in range(1, len(outputs)):
    diff_grid = abs(outputs[i]['grid_size'].item() - first_output['grid_size'].item())
    diff_obs = abs(outputs[i]['num_obstacles'].item() - first_output['num_obstacles'].item())
    diff_doors = abs(outputs[i]['num_doors'].item() - first_output['num_doors'].item())
    
    if diff_grid > 0.1 or diff_obs > 0.1 or diff_doors > 0.1:
        all_same = False
        break

if all_same:
    print("  [PROBLEME] Tous les outputs sont identiques!")
    print("  Le réseau semble être 'mort' ou mal initialisé")
else:
    print("  [OK] Les outputs sont différents")

print("\n[Test 5] Génération d'un batch de 8 niveaux avec debug")
levels = generator.generate_batch(batch_size=8, debug=True)

print("\n[Test 6] Vérification de l'unicité des niveaux générés")
unique_configs = set()
for i, level in enumerate(levels):
    config = (level['grid_size'], level['num_obstacles'], level['num_doors'], level['num_keys'])
    unique_configs.add(config)
    if i < 5:
        print(f"  Niveau {i}: {config}")

print(f"\n  Total de configurations uniques: {len(unique_configs)}/8")
if len(unique_configs) == 1:
    print("  [PROBLEME MAJEUR] Tous les niveaux sont identiques!")
elif len(unique_configs) < 4:
    print("  [WARNING] Peu de diversité dans les niveaux")
else:
    print("  [OK] Bonne diversité")

print("\n" + "="*60)
print("FIN DU TEST")
print("="*60)
