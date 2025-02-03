import pandas as pd
import matplotlib.pyplot as plt
import datetime
import re

# Leggere il file CSV
df = pd.read_csv("omp_cores_output_input100D.csv")

# Convertire la colonna "Time" in secondi
def time_to_seconds(time_str):
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
for cluster in sorted(df["# Cluster"].unique()):
    subset = df[df["# Cluster"] == cluster]
    plt.plot(subset["Cores"], subset["Time_seconds"], marker='o', linestyle='-', label=f"{cluster} Clusters")

plt.xticks(range(int(df["Cores"].min()), int(df["Cores"].max()) + 1))
plt.xlabel("Numero di Core")
plt.ylabel("Tempo di Esecuzione (s)")
plt.title("Prestazioni K-Means Parallelo")
plt.legend()
plt.grid()
plt.show()
