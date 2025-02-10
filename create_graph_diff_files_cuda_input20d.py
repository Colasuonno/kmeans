import matplotlib.pyplot as plt
import numpy as np

# Dati
clusters = [1, 2, 5, 10, 100, 1000]
seq_values =np.array([
     [0.00895, 0.22765, 0.2531593333, 0.3334533333, 0.3029853333, 0.554527],
     [0.016126, 0.214344, 0.138687, 0.290288, 0.094207, 0.093051] # 5 core
])
values = np.array([
    [0.000746, 0.008, 0.009, 0.15, 0.011, 0.017],
])

# Creazione del grafico
plt.figure(figsize=(10, 6))

plt.plot(clusters, seq_values[0], label="Sequenziale", marker='o')
plt.plot(clusters, seq_values[1], label="MPI/OMP (5 core)", marker='o')
plt.plot(clusters, values[0], label="CUDA", marker='o')

plt.xlabel('Numero di Cluster')
plt.ylabel('Tempo di esecuzione (sec)')
plt.title('Tempo di esecuzione (input20D)')
plt.grid(True)
plt.legend(title='Versione')
plt.xticks(clusters)

# Mostra il grafico
plt.show()



