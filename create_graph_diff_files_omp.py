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
df = pd.read_csv("omp_cores_diff_files_output.csv")
df["Cores"] = df["# Cores"].apply(lambda x: int(re.search(r"\d+", x).group()))
df["FILE"] = df["# Cores"].apply(lambda x: x.split("File: test_files/")[1])
df["Time_seconds"] = df["Time"].apply(time_to_seconds)

fig,axes = plt.subplots(nrows=3, ncols=2, figsize=(10,6 * 3))

for i in range(0, 3):
    ax_time = axes[i, 0]
    ax_speedup = axes[i, 1]
    subset = df[df["FILE"] == "input" + INPUT_FILES[i] + ".inp)"]

    cores = sorted(subset["Cores"].unique())
    clusters = sorted(subset["# Cluster"].unique())

    # Prendi il tempo con 1 core per il calcolo dello speedup
    base_times = subset[subset["Cores"] == 1].set_index("# Cluster")["Time_seconds"]


    for core in sorted(subset["Cores"].unique()):
        subset_core = subset[subset["Cores"] == core]
        ax_time.plot(subset_core["# Cluster"], subset_core["Time_seconds"], marker='o', linestyle='-', label=f"{core} Cores")
        
        # Calcola Speedup
        speedup = base_times.loc[subset_core["# Cluster"].values] / subset_core["Time_seconds"].values
        ax_speedup.plot(subset_core["# Cluster"], speedup, marker='o', linestyle='-', label=f"{core} Cores")




    # Configurazione grafico Tempo
    ax_time.set_yscale("log")  
    ax_time.set_xscale("log")
    ax_time.set_xticks(clusters)
    ax_time.set_xlabel("Numero di Cluster")
    ax_time.set_ylabel("Tempo di Esecuzione (s)")
    ax_time.set_title(f"Tempo per {INPUT_AMOUNTS[INPUT_FILES[i]]} Punti")
    ax_time.legend(title="Numero di Core", loc="upper right", bbox_to_anchor=(1.2, 1))
    ax_time.grid(True, which="both", linestyle="--", linewidth=0.5)
    

    # Configurazione grafico Speedup
   # ax_speedup.plot(cores, cores, 'k--', label="Speedup Ideale")  # Linea Speedup ideale
    ax_speedup.set_yscale("log")  
    ax_speedup.set_xscale("log")
    ax_speedup.set_xticks(clusters)
    ax_speedup.set_xlabel("Numero di Cluster")
    ax_speedup.set_ylabel("Speedup (Rispetto a #1 Core)")
    ax_speedup.set_title(f"Speedup per {INPUT_AMOUNTS[INPUT_FILES[i]]} Punti")
    ax_speedup.legend(title="Numero di Core", loc="upper right", bbox_to_anchor=(1.2, 1))
    ax_speedup.grid(True, linestyle="--", linewidth=0.5)

fig.subplots_adjust(hspace=0.4) 
plt.tight_layout(pad=6)
plt.show()