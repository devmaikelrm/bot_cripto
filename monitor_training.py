import time
import os
import re

def monitor():
    log_path = "logs/training.log"
    if not os.path.exists(log_path):
        return
    
    with open(log_path, "r") as f:
        content = f.read()
    
    epochs = re.findall(r"Epoch (\d+):", content)
    if not epochs:
        print("Entrenamiento en fase de inicializacion...")
        return

    last_epoch = int(epochs[-1])
    losses = re.findall(r"val_loss=([\d.e+-]+)", content)
    
    percent = ((last_epoch + 1) / 50) * 100
    print(f"--- STATUS ---")
    print(f"Epoca actual: {last_epoch + 1} / 50")
    print(f"Progreso total: {percent:.1f}%")
    if losses:
        print(f"Mejor Val Loss: {losses[-1]}")
    
    if "entrenamiento_completado" in content:
        print("ESTADO: COMPLETADO")
    else:
        print("ESTADO: EN CURSO")

if __name__ == "__main__":
    monitor()
