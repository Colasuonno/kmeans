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
plt.title('Speedup (input100D2)')
plt.legend(title='Numero di Core')
plt.grid(True)
plt.xticks(clusters)

# Mostra il grafico
plt.show()



