import matplotlib.pyplot as plt
import numpy as np

# Dati
cores = [1, 2, 5, 10, 20, 30]
clusters = [1, 2, 5, 10, 100, 1000]

values = np.array([
    [0.061998, 1.68, 3.6, 5.5, 31.7, 43.4],
    [0.05, 1.3, 3.5, 4.2, 31.515664, 22.05],
    [0.04, 0.53, 1.4, 2, 5.6, 8.352053],
    [0.079784, 0.650639, 1.2, 2.1, 3.5, 3.8],
    [0.068149, 0.56, 1.195563, 1.8, 2.8, 1.6],
    [0.22, 0.7, 1.4, 1.7, 2.7, 1.2]
])

# Creazione del grafico
plt.figure(figsize=(10, 6))

for i, core in enumerate(cores):
    plt.plot(clusters, values[i], label=f'{core} Core', marker='o')

plt.xlabel('Numero di Cluster')
plt.ylabel('Tempo di esecuzione (sec)')
plt.title('Tempo di esecuzione (input100D2)')
plt.legend(title='Numero di Core')
plt.grid(True)
plt.xticks(clusters)

# Mostra il grafico
plt.show()