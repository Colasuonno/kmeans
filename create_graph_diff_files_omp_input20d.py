import matplotlib.pyplot as plt
import numpy as np

# Dati
cores = [1, 2, 5]
clusters = [1, 2, 5, 10, 100, 1000]
values = np.array([
    [0.00895, 0.22765, 0.2531593333, 0.3334533333, 0.3029853333, 0.554527],
    [0.005034333333, 0.1618856667, 0.1181876667, 0.3503086667, 0.2272616667, 0.4045],
    [0.016126, 0.214344, 0.138687, 0.290288, 0.094207, 0.093051]
])

# Creazione del grafico
plt.figure(figsize=(10, 6))

for i, core in enumerate(cores):
    plt.plot(clusters, values[i], label=f'{core} Core', marker='o')

plt.xlabel('Numero di Cluster')
plt.ylabel('Tempo di esecuzione (sec)')
plt.title('Tempo di esecuzione (input20D)')
plt.legend(title='Numero di Core')
plt.grid(True)
plt.xticks(clusters)

# Mostra il grafico
plt.show()



