import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import re


# Dati dei punti per tipo di file

INPUT_FILES = [
    "20D",
    "100D2"
]

INPUT_AMOUNTS = {
    "20D": 10000,
    "100D2": 100000,
}



# Funzione per convertire il tempo in secondi
def time_to_seconds(time_str):
    if time_str == "-1":
        return 1e10  # Gestiamo i casi di errore con np.inf
    time_obj = datetime.strptime(time_str, "%H:%M:%S.%f")
    return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1e6

DATA = [
    ("20D", 10, 1.03, 0.025, 0.023),
    ("100D2", 10, 2.29, 1.683, 14.754),
    ("20D", 100, 0.21, 0.013, 0.126),
    ("100D2", 100, 3.2, 4.04, 10.7),
    ("20D", 1000, 0.141, 0.017, 1.77),
    ("100D2", 1000, 1.73, 6.96, 215.09),
    ("20D", 10000, 0.155, 0.04, 0.649),
    ("100D2", 10000, 3.798, 25.629, 77.903)
]

# Creazione del DataFrame
df = pd.DataFrame(DATA, columns=["FILE", "# Cluster", "OMP/MPI (30 Core)", "CUDA", "Seq (1 core)"])

# Calcolo dello Speedup
df["Speedup"] = df["Seq (1 core)"] / df["CUDA"]
df["Speedup_omp"] = df["Seq (1 core)"] / df["OMP/MPI (30 Core)"]

# Creazione dei grafici
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

for i, file_type in enumerate(sorted(df["FILE"].unique())):
    ax_time = axes[i, 0]
    ax_speedup = axes[i, 1]
    subset = df[df["FILE"] == file_type]
    
    ax_time.plot(subset["# Cluster"], subset["OMP/MPI (30 Core)"], marker='o', linestyle='-', label="OMP/MPI (30 Core)")
    ax_time.plot(subset["# Cluster"], subset["Seq (1 core)"], marker='o', linestyle='-', label="Seq (1 core)")
    ax_time.plot(subset["# Cluster"], subset["CUDA"], marker='o', linestyle='-', label="CUDA")
    
    ax_time.set_yscale("log")  
    ax_time.set_xscale("log")
    ax_time.set_xlabel("Numero di Cluster")
    ax_time.set_ylabel("Tempo di Esecuzione (s)")
    ax_time.set_title(f"Tempo di Esecuzione per {file_type}")
    ax_time.legend()
    ax_time.grid(True, which="both", linestyle="--", linewidth=0.5)
    
    # Speedup
    ax_speedup.plot(subset["# Cluster"], subset["Speedup"], marker='o', linestyle='-', label="Speedup CUDA")
    ax_speedup.plot(subset["# Cluster"], subset["Speedup_omp"], marker='o', linestyle='-', label="Speedup MP/MPI (30 Core)")
    ax_speedup.set_xscale("log")
    ax_speedup.set_xlabel("Numero di Cluster")
    ax_speedup.set_ylabel("Speedup")
    ax_speedup.set_title(f"Speedup rispetto a Seq (1 Core) per {file_type}")
    ax_speedup.legend()
    ax_speedup.grid(True, linestyle="--", linewidth=0.5)

fig.subplots_adjust(hspace=0.4)
plt.tight_layout()
plt.show()
