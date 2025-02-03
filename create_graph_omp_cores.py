import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import re

# Leggere il file CSV
df = pd.read_csv("omp_cores_output_input100D2.csv")

# Convertire la colonna "Time" in secondi
def time_to_seconds(time_str):
    if time_str == "-1":
        return 1e10
    time_obj = datetime.datetime.strptime(time_str, "%H:%M:%S.%f")
    return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1e6

df["Time_seconds"] = df["Time"].apply(time_to_seconds)

# Estrarre il numero di core
df["Cores"] = df["# Cores"].apply(lambda x: int(re.search(r"\d+", x).group()))

df = df[df["Cores"] == df["Cores"].astype(int)]

# Ordinare i dati per numero di cluster e numero di core
df = df.sort_values(by=["# Cluster", "Cores"])

# Creare il grafico
plt.figure(figsize=(10, 6))

# Ordinare i dati per cluster e core
df_sorted = df.sort_values(by=["# Cluster", "Cores"])

# Tracciare una linea per ogni numero di core
for core in sorted(df["Cores"].unique()):
    subset = df_sorted[df_sorted["Cores"] == core]
    plt.plot(subset["# Cluster"], subset["Time_seconds"], marker='o', linestyle='-', label=f"{core} Cores")

plt.xticks(df["# Cluster"].unique())
# Aggiungere etichette e titolo
plt.xlabel("Numero di Cluster")
plt.ylabel("Tempo di Esecuzione (s)")
plt.title("Prestazioni K-Means Parallelo")
plt.legend(title="Numero di Core")
plt.grid()
plt.show()