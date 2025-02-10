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

plt.plot(clusters, seq_values[0], label="Sequenziale", marker='o')
plt.plot(clusters, seq_values[1], label="MPI/OMP (30 core)", marker='o')
plt.plot(clusters, values[0], label="CUDA", marker='o')

plt.xlabel('Numero di Cluster')
plt.ylabel('Tempo di esecuzione (sec)')
plt.title('Tempo di esecuzione (input100D2)')
plt.grid(True)
plt.legend(title='Versione')
plt.xticks(clusters)

# Mostra il grafico
plt.show()



