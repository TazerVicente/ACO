import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from tabulate import tabulate

# Criação do grafo com mais nós e arestas
G = nx.Graph()

# Adicionando mais nós e arestas de forma mais dinâmica (20 nós)
nodes = list(range(1, 26))  # Nós de 1 a 20
arestas = []

# Gerar arestas aleatórias entre nós
np.random.seed(82)  # Semente para resultados consistentes
for i in range(1, 26):
    for j in range(i + 1, 26):
        if np.random.rand() < 0.3:  # Probabilidade de adicionar uma aresta
            weight = np.random.randint(5, 300)  # Peso aleatório entre 5 e 30
            arestas.append((i, j, weight))

G.add_weighted_edges_from(arestas)

# Parâmetros do algoritmo ACO
alpha = 1.0       # Influência do feromônio
beta = 1.0        # Influência da distância
evaporation_rate = 0.01  # Taxa de evaporação do feromônio
Q = 10            # Constante para depósito de feromônio
num_iterations = 100

# Inicializa o feromônio em cada aresta com valor inicial 0.1
pheromone = {edge: 0.1 for edge in G.edges}

# Função para calcular as probabilidades de transição
def calculate_transition_probabilities(current_node, unvisited, pheromone, G):
    probabilities = []
    data_table = []  # Tabela para armazenar os dados
    for next_node in unvisited:
        if G.has_edge(current_node, next_node):
            tau = pheromone.get((current_node, next_node), pheromone.get((next_node, current_node), 0.1))
            eta = 1 / G[current_node][next_node]['weight']
            prob = (tau ** alpha) * (eta ** beta)
        else:
            tau, eta, prob = 0, 0, 0
        
        probabilities.append(prob)
        data_table.append([f"{current_node} -> {next_node}", G[current_node][next_node]['weight'], tau, eta, prob, f"{prob * 100:.2f}%"])
    
    total_prob = sum(probabilities)
    probabilities = [p / total_prob if total_prob > 0 else 0 for p in probabilities]

    # Exibe a tabela de dados
    print(tabulate(data_table, headers=["Rota", "Distância", "Tau", "Eta", "Probabilidade", "Probabilidade (%)"], tablefmt="fancy_grid"))
    return probabilities

# Função para selecionar o próximo nó usando o método da roleta
def roulette_wheel_selection(probabilities):
    r = np.random.rand()
    cumulative_sum = 0.0
    for i, probability in enumerate(probabilities):
        cumulative_sum += probability
        if r <= cumulative_sum:
            return i
    return len(probabilities) - 1

# Função para simular o percurso de uma formiga
def simulate_ant(start_node, pheromone, G):
    path = []
    visited = set([start_node])
    current_node = start_node
    total_distance = 0

    while len(visited) < len(nodes):
        unvisited = [node for node in nodes if node not in visited and G.has_edge(current_node, node)]
        if not unvisited:
            return None, float('inf')

        probabilities = calculate_transition_probabilities(current_node, unvisited, pheromone, G)
        next_index = roulette_wheel_selection(probabilities)
        next_node = unvisited[next_index]

        path.append((current_node, next_node))
        total_distance += G[current_node][next_node]['weight']
        visited.add(next_node)
        current_node = next_node

    if G.has_edge(current_node, start_node):
        path.append((current_node, start_node))
        total_distance += G[current_node][start_node]['weight']
        return path, total_distance
    else:
        return None, float('inf')

# Função para atualizar os feromônios
def update_pheromone(pheromone, all_paths, evaporation_rate):
    # Evaporação do feromônio
    for edge in pheromone:
        pheromone[edge] *= (1 - evaporation_rate)
    
    # Reforço de feromônio baseado nas melhores rotas
    for path, total_distance in all_paths:
        if path and total_distance > 0:  # Evita divisão por zero
            pheromone_deposit = Q / total_distance  # Mais feromônio para rotas curtas
            for edge in path:
                if edge in pheromone:
                    pheromone[edge] += pheromone_deposit
                elif (edge[1], edge[0]) in pheromone:
                    pheromone[(edge[1], edge[0])] += pheromone_deposit

# Algoritmo ACO
best_path = None
best_distance = float('inf')

print("Iniciando o algoritmo ACO...\n")

for iteration in range(num_iterations):
    all_paths = []
    iteration_best_path = None
    iteration_best_distance = float('inf')

    # Simulação de cada formiga
    for start_node in nodes:
        print(f"\nSimulando formiga iniciando no nó {start_node}")
        path, total_distance = simulate_ant(start_node, pheromone, G)
        if path and total_distance < float('inf'):
            all_paths.append((path, total_distance))
            if total_distance < iteration_best_distance:
                iteration_best_path = path
                iteration_best_distance = total_distance

            if total_distance < best_distance:
                best_path = path
                best_distance = total_distance

    # Atualização dos feromônios
    update_pheromone(pheromone, all_paths, evaporation_rate)

    print(f"\nIteração {iteration + 1}: Melhor distância = {iteration_best_distance}")

print("\nMelhor caminho encontrado globalmente:", best_path)
print("Distância total do melhor caminho global:", best_distance)




#%%

# Visualização do grafo inicial
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, seed=42)  # Layout fixo para consistência

# Normalizar os pesos para mapear as cores
weights = [G[u][v]['weight'] for u, v in G.edges]
norm = mcolors.Normalize(vmin=min(weights), vmax=max(weights))
cmap = cm.get_cmap('viridis')

# Desenhar os nós
nx.draw_networkx_nodes(G, pos, node_size=700, node_color='black')

# Desenhar as arestas com cores baseadas nos pesos
edge_colors = [cmap(norm(G[u][v]['weight'])) for u, v in G.edges]
nx.draw_networkx_edges(G, pos, width=2, edge_color=edge_colors)

# Desenhar os rótulos dos nós
nx.draw_networkx_labels(G, pos, font_size=10, font_color='white')

# Desenhar os rótulos dos pesos das arestas
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='blue')

plt.title("Grafo Inicial com Arestas Coloridas Baseadas nos Pesos", fontsize=14)
plt.axis('off')
plt.show()

# Visualização do grafo com a rota ótima
plt.figure(figsize=(12, 12))
nx.draw_networkx_nodes(G, pos, node_size=700, node_color='black')
nx.draw_networkx_labels(G, pos, font_size=10, font_color='white')

if best_path:
    edge_colors = ['red' if (u, v) in best_path or (v, u) in best_path else 'gray' for u, v in G.edges]
    widths = [2.5 if (u, v) in best_path or (v, u) in best_path else 1 for u, v in G.edges]
else:
    edge_colors = ['gray' for _ in G.edges]
    widths = [1 for _ in G.edges]

nx.draw_networkx_edges(G, pos, edgelist=G.edges, edge_color=edge_colors, width=widths)

edge_labels_best = {edge: G[edge[0]][edge[1]]['weight'] for edge in best_path} if best_path else {}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels_best, font_color='red')

plt.title("Grafo com a Rota Ótima Destacada", fontsize=14)
plt.axis('off')
plt.show()
