import matplotlib.pyplot as plt
import numpy as np

# Dati
clusters = [1, 2, 5, 10, 100, 1000]
seq_values =np.array([
     [0.061998, 1.68, 3.6, 5.5, 31.7, 43.4],
      [0.22, 0.7, 1.4, 1.7, 2.7, 1.2] # 30 core
])
values = np.array([
    [0.014,	0.25,	0.54,	0.086,	3.900,	7.00],
])

# Creazione del grafico
plt.figure(figsize=(10, 6))


speed_up_cuda = seq_values[0] / values[0]
speed_up_seq = seq_values[0] / seq_values[1]

plt.plot(clusters, speed_up_seq, label="Speedup MPI/OMP (30 core)", marker='o')
plt.plot(clusters, speed_up_cuda, label="Speedup CUDA", marker='o')

plt.xlabel('Numero di Cluster')
plt.ylabel('Valore')
plt.title('Speedup (input100D2)')
plt.grid(True)
plt.legend(title='Versione')
plt.xticks(clusters)

# Mostra il grafico
plt.show()



