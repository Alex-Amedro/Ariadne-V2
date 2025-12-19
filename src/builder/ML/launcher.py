"""
Launcher interactif pour le système de co-évolution.
Interface simple pour lancer les différentes fonctionnalités.
"""

import os
import sys
import subprocess
from pathlib import Path


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    print("="*60)
    print(" "*15 + "CO-ÉVOLUTION ARIADNE")
    print("="*60)
    print()


def list_runs():
    """Liste les runs disponibles."""
    runs_dir = "runs"
    if not os.path.exists(runs_dir):
        return []
    
    runs = []
    for item in os.listdir(runs_dir):
        run_path = os.path.join(runs_dir, item)
        if os.path.isdir(run_path) and os.path.exists(f"{run_path}/logs/history.json"):
            runs.append(item)
    
    return sorted(runs, reverse=True)


def main_menu():
    while True:
        clear_screen()
        print_header()
        
        print("MENU PRINCIPAL")
        print("-" * 60)
        print("1. Nouvel entraînement")
        print("2. Continuer un entraînement")
        print("3. Visualiser les données")
        print("4. Démo de l'agent (slow motion)")
        print("5. Test rapide du système")
        print("6. Quitter")
        print("-" * 60)
        
        choice = input("\nChoix: ")
        
        if choice == "1":
            new_training()
        elif choice == "2":
            continue_training()
        elif choice == "3":
            visualize_data()
        elif choice == "4":
            demo_agent()
        elif choice == "5":
            quick_test()
        elif choice == "6":
            print("\nAu revoir!")
            break
        else:
            print("\nChoix invalide!")
            input("Appuyez sur Entrée pour continuer...")


def new_training():
    clear_screen()
    print_header()
    print("NOUVEL ENTRAÎNEMENT")
    print("-" * 60)
    
    print("\nPrésets disponibles:")
    print("1. Rapide (5 époques, 10k timesteps) - ~5 min")
    print("2. Normal (20 époques, 50k timesteps) - ~30 min")
    print("3. Intensif (50 époques, 100k timesteps) - ~2h")
    print("4. Personnalisé")
    print("5. Retour")
    
    choice = input("\nChoix: ")
    
    if choice == "1":
        epochs, timesteps, initial = 5, 10000, 20000
    elif choice == "2":
        epochs, timesteps, initial = 20, 50000, 100000
    elif choice == "3":
        epochs, timesteps, initial = 50, 100000, 200000
    elif choice == "4":
        try:
            epochs = int(input("Nombre d'époques: "))
            timesteps = int(input("Timesteps par époque: "))
            initial = int(input("Timesteps initiaux: "))
        except ValueError:
            print("\n[ERREUR] Valeurs invalides!")
            input("Appuyez sur Entrée...")
            return
    else:
        return
    
    visualize = input("\nVisualiser à la fin? (o/n): ").lower() == 'o'
    
    cmd = [
        sys.executable,
        "train_coevolution.py",
        "--epochs", str(epochs),
        "--timesteps", str(timesteps),
        "--initial-timesteps", str(initial)
    ]
    
    if visualize:
        cmd.append("--visualize")
    
    print(f"\n[INFO] Lancement de l'entraînement...")
    print(f"Commande: {' '.join(cmd)}\n")
    
    subprocess.run(cmd)
    
    input("\nAppuyez sur Entrée pour continuer...")


def continue_training():
    clear_screen()
    print_header()
    print("CONTINUER UN ENTRAÎNEMENT")
    print("-" * 60)
    
    runs = list_runs()
    
    if not runs:
        print("\n[INFO] Aucun run trouvé!")
        input("Appuyez sur Entrée pour continuer...")
        return
    
    print("\nRuns disponibles:")
    for i, run in enumerate(runs, 1):
        print(f"{i}. {run}")
    
    print(f"{len(runs)+1}. Retour")
    
    try:
        choice = int(input("\nChoix: "))
        if choice == len(runs) + 1:
            return
        if 1 <= choice <= len(runs):
            run_path = f"runs/{runs[choice-1]}"
        else:
            print("\n[ERREUR] Choix invalide!")
            input("Appuyez sur Entrée...")
            return
    except ValueError:
        print("\n[ERREUR] Choix invalide!")
        input("Appuyez sur Entrée...")
        return
    
    try:
        epochs = int(input("\nNombre d'époques supplémentaires: "))
    except ValueError:
        print("\n[ERREUR] Valeur invalide!")
        input("Appuyez sur Entrée...")
        return
    
    visualize = input("Visualiser à la fin? (o/n): ").lower() == 'o'
    
    cmd = [
        sys.executable,
        "train_coevolution.py",
        "--load", run_path,
        "--epochs", str(epochs)
    ]
    
    if visualize:
        cmd.append("--visualize")
    
    print(f"\n[INFO] Lancement...")
    subprocess.run(cmd)
    
    input("\nAppuyez sur Entrée pour continuer...")


def visualize_data():
    clear_screen()
    print_header()
    print("VISUALISATION DES DONNÉES")
    print("-" * 60)
    
    runs = list_runs()
    
    if not runs:
        print("\n[INFO] Aucun run trouvé!")
        input("Appuyez sur Entrée pour continuer...")
        return
    
    print("\nRuns disponibles:")
    for i, run in enumerate(runs, 1):
        print(f"{i}. {run}")
    
    print(f"{len(runs)+1}. Comparer plusieurs runs")
    print(f"{len(runs)+2}. Retour")
    
    try:
        choice = int(input("\nChoix: "))
        
        if choice == len(runs) + 2:
            return
        
        elif choice == len(runs) + 1:
            # Mode comparaison
            print("\nEntrez les numéros des runs à comparer (séparés par des espaces):")
            indices = input("> ").split()
            
            run_paths = []
            for idx in indices:
                i = int(idx) - 1
                if 0 <= i < len(runs):
                    run_paths.append(f"runs/{runs[i]}")
            
            if len(run_paths) < 2:
                print("\n[ERREUR] Au moins 2 runs nécessaires!")
                input("Appuyez sur Entrée...")
                return
            
            print("\nMétrique à comparer:")
            print("1. Success Rate")
            print("2. Mean Reward")
            print("3. Diversity")
            
            metric_choice = input("Choix: ")
            metrics = {'1': 'success_rate', '2': 'mean_reward', '3': 'diversity'}
            metric = metrics.get(metric_choice, 'success_rate')
            
            cmd = [
                sys.executable,
                "visualize_training.py",
                "--compare"
            ] + run_paths + ["--metric", metric]
            
        elif 1 <= choice <= len(runs):
            run_path = f"runs/{runs[choice-1]}"
            
            print("\n1. Résumé seulement")
            print("2. Résumé + Graphiques")
            
            viz_choice = input("Choix: ")
            
            cmd = [
                sys.executable,
                "visualize_training.py",
                "--run", run_path
            ]
            
            if viz_choice == "1":
                cmd.append("--summary")
        
        else:
            print("\n[ERREUR] Choix invalide!")
            input("Appuyez sur Entrée...")
            return
        
        print(f"\n[INFO] Lancement de la visualisation...")
        subprocess.run(cmd)
    
    except (ValueError, IndexError):
        print("\n[ERREUR] Choix invalide!")
    
    input("\nAppuyez sur Entrée pour continuer...")


def demo_agent():
    clear_screen()
    print_header()
    print("DÉMO DE L'AGENT (SLOW MOTION)")
    print("-" * 60)
    
    runs = list_runs()
    
    if not runs:
        print("\n[INFO] Aucun run trouvé!")
        input("Appuyez sur Entrée pour continuer...")
        return
    
    print("\nRuns disponibles:")
    for i, run in enumerate(runs, 1):
        print(f"{i}. {run}")
    
    print(f"{len(runs)+1}. Retour")
    
    try:
        choice = int(input("\nChoix: "))
        if choice == len(runs) + 1:
            return
        if 1 <= choice <= len(runs):
            run_path = f"runs/{runs[choice-1]}"
        else:
            print("\n[ERREUR] Choix invalide!")
            input("Appuyez sur Entrée...")
            return
    except ValueError:
        print("\n[ERREUR] Choix invalide!")
        input("Appuyez sur Entrée...")
        return
    
    cmd = [
        sys.executable,
        "train_coevolution.py",
        "--load", run_path,
        "--visualize-only"
    ]
    
    print(f"\n[INFO] Lancement de la démo...")
    print("[INFO] L'agent va jouer sur 5 niveaux générés aléatoirement")
    print("[INFO] Fermez les fenêtres pour passer au niveau suivant")
    
    subprocess.run(cmd)
    
    input("\nAppuyez sur Entrée pour continuer...")


def quick_test():
    clear_screen()
    print_header()
    print("TEST RAPIDE DU SYSTÈME")
    print("-" * 60)
    
    print("\n[INFO] Ce test va:")
    print("  1. Lancer 2 époques d'entraînement (~2 min)")
    print("  2. Vérifier que la visualisation fonctionne")
    print("  3. Afficher un résumé")
    
    confirm = input("\nContinuer? (o/n): ").lower()
    
    if confirm == 'o':
        cmd = [sys.executable, "test_quick.py"]
        subprocess.run(cmd)
    
    input("\nAppuyez sur Entrée pour continuer...")


if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\nInterruption détectée. Au revoir!")
