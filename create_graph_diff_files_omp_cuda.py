import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import re


# Dati dei punti per tipo di file

INPUT_FILES = [
    "2D",
    "20D",
    "100D2"
]

INPUT_AMOUNTS = {
    "2D": 5000,
    "20D": 10000,
    "100D2": 100000,
}



# Funzione per convertire il tempo in secondi
def time_to_seconds(time_str):
    if time_str == "-1":
        return 1e10  # Gestiamo i casi di errore con np.inf
    time_obj = datetime.strptime(time_str, "%H:%M:%S.%f")
    return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1e6

# Caricamento dei dati dal CSV
df = pd.read_csv("omp_cuda_omp_diff_files_output.csv")
df["Time_seconds"] = df["Time"].apply(time_to_seconds)
df["FILE"] = df["Type"].apply(lambda x: re.search(r"input(\w+)\.inp", x).group(1))
df["Method"] = df["Type"].apply(lambda x: "CUDA" if "CUDA" in x else "OMP/MPI")

# Pivot dei dati per avere colonne separate per CUDA e OMP/MPI
pivot_df = df.pivot(index=["FILE", "# Cluster"], columns="Method", values="Time_seconds").reset_index()

# Calcolo dello Speedup
pivot_df["Speedup"] = pivot_df["OMP/MPI"] / pivot_df["CUDA"]

# Salvataggio in un nuovo CSV
pivot_df.to_csv("omp_cuda_comparison_results.csv", index=False)

# Creazione dei grafici
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12))

for i, file_type in enumerate(sorted(df["FILE"].unique())):
    ax_time = axes[i, 0]
    ax_speedup = axes[i, 1]
    subset = pivot_df[pivot_df["FILE"] == file_type]
    
    ax_time.plot(subset["# Cluster"], subset["OMP/MPI"], marker='o', linestyle='-', label="OMP/MPI")
    ax_time.plot(subset["# Cluster"], subset["CUDA"], marker='o', linestyle='-', label="CUDA")
    
    ax_time.set_yscale("log")  
    ax_time.set_xscale("log")
    ax_time.set_xlabel("Numero di Cluster")
    ax_time.set_ylabel("Tempo di Esecuzione (s)")
    ax_time.set_title(f"Tempo di Esecuzione per {INPUT_AMOUNTS[file_type]} punti")
    ax_time.legend()
    ax_time.grid(True, which="both", linestyle="--", linewidth=0.5)
    
    # Speedup
    ax_speedup.plot(subset["# Cluster"], subset["Speedup"], marker='o', linestyle='-', label="Speedup CUDA vs OMP/MPI")
    ax_speedup.set_xscale("log")
    ax_speedup.set_xlabel("Numero di Cluster")
    ax_speedup.set_ylabel("Speedup")
    ax_speedup.set_title(f"Speedup CUDA vs OMP/MPI per {INPUT_AMOUNTS[file_type]} punti")
    ax_speedup.legend()
    ax_speedup.grid(True, linestyle="--", linewidth=0.5)

fig.subplots_adjust(hspace=0.4)
plt.tight_layout()
plt.show()
