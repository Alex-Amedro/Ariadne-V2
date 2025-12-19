"""
Test rapide du système de co-évolution.
Lance 2 époques pour vérifier que tout fonctionne.
"""

import subprocess
import sys

print("="*60)
print("TEST RAPIDE DU SYSTÈME DE CO-ÉVOLUTION")
print("="*60)

print("\n[1/3] Test avec 2 époques...")
print("Commande: python train_coevolution.py --epochs 2 --timesteps 5000 --initial-timesteps 10000")

result = subprocess.run([
    sys.executable, 
    "train_coevolution.py",
    "--epochs", "2",
    "--timesteps", "5000",
    "--initial-timesteps", "10000"
], capture_output=False)

if result.returncode != 0:
    print("\n[ERREUR] Le test a échoué!")
    sys.exit(1)

print("\n[OK] Entraînement terminé!")

print("\n[2/3] Test de la visualisation des données...")
print("Commande: python visualize_training.py --list")

result = subprocess.run([
    sys.executable,
    "visualize_training.py",
    "--list"
], capture_output=False)

print("\n[3/3] Résumé du dernier run...")

# Trouver le dernier run
import os
runs_dir = "runs"
if os.path.exists(runs_dir):
    runs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
    if runs:
        latest_run = sorted(runs)[-1]
        run_path = f"runs/{latest_run}"
        
        print(f"Commande: python visualize_training.py --run {run_path} --summary")
        
        result = subprocess.run([
            sys.executable,
            "visualize_training.py",
            "--run", run_path,
            "--summary"
        ], capture_output=False)

print("\n" + "="*60)
print("TEST TERMINÉ AVEC SUCCÈS!")
print("="*60)
print("\nProchaines étapes:")
print("  1. Lance un vrai entraînement: python train_coevolution.py --epochs 20")
print("  2. Visualise les résultats: python visualize_training.py --list")
print("  3. Regarde l'agent jouer: python train_coevolution.py --load runs/XXXXX --visualize-only")
