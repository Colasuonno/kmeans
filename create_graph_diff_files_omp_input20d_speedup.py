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

all_values = []

# Traccia ogni linea
for i, core in enumerate(cores):
    y_values = values[0] / values[i]
    all_values.append(y_values)  # Aggiungi i valori per il calcolo globale
    plt.plot(clusters, y_values, marker='o', linestyle='-', label=f'{core} Core')

# Unisci tutti i valori per trovare il massimo e minimo globali
all_values = np.concatenate(all_values)

# Trova il massimo e minimo globale
global_max = np.max(all_values)
global_min = np.min(all_values)

# Trova le posizioni di massimo e minimo globale
max_index = np.argmax(all_values)
min_index = np.argmin(all_values)

max_x, max_y = clusters[max_index % len(clusters)], all_values[max_index]
min_x, min_y = clusters[min_index % len(clusters)], all_values[min_index]

# Evidenzia i punti massimo e minimo globali
plt.scatter([max_x, min_x], [max_y, min_y], color='red', zorder=5)
plt.text(max_x, max_y, f'Max ({max_x}, {max_y:.2f})', fontsize=12, color='black', ha='center')
plt.text(min_x, min_y, f'Min ({min_x}, {min_y:.2f})', fontsize=12, color='black', ha='center')




plt.xlabel('Numero di Cluster')
plt.ylabel('Valore')
plt.title('Speedup (input20D)')
plt.legend(title='Numero di Core')
plt.grid(True)
plt.xticks(clusters)

# Mostra il grafico
plt.show()



