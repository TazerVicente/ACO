import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

# Parâmetros
alpha = 1.0       # Influência do feromônio
beta = 1.0        # Influência da distância
evaporation_rate = 0.1  # Taxa de evaporação do feromônio
Q = 10            # Constante para depósito de feromônio
num_iterations = 10
min_pheromone = 0.01
max_pheromone = 10

# Lista de arestas (nós conectados e seus pesos)
arestas = [
    (1, 2, 5), (1, 3, 10), (1, 6, 20),
    (2, 6, 25), (2, 4, 25),
    (6, 4, 16), (6, 5, 15), (5, 3, 10), (6, 3, 10), (4, 5, 2),
    (7, 3, 20), (7, 2, 20)
]

# Criação do grafo
G = nx.Graph()
G.add_weighted_edges_from(arestas)

# Lista de nós
nodes = list(G.nodes)

# Inicialização dos feromônios em cada aresta
pheromone = {edge: min_pheromone for edge in G.edges}
for u, v in G.edges:
    pheromone[(v, u)] = pheromone[(u, v)]  # Garante simetria

# Função para calcular as probabilidades de transição e tabelar os dados
def calculate_transition_probabilities(current_node, unvisited, pheromone, G, iteration):
    data = []
    probabilities = []
    for next_node in unvisited:
        if G.has_edge(current_node, next_node):
            tau = pheromone.get((current_node, next_node), pheromone.get((next_node, current_node), min_pheromone))
            distance = G[current_node][next_node]['weight']
            eta = 1 / distance  # Inverso da distância
            prob = (tau ** alpha) * (eta ** beta)
            data.append({
                'Iteration': iteration,
                'Route': f"{current_node} -> {next_node}",
                'Distance': distance,
                'Eta (1/distance)': eta,
                'Tau': tau,
                'Probability (numeric)': prob,
                'Probability (%)': 0  # Será calculado depois
            })
            probabilities.append(prob)
        else:
            probabilities.append(0)
    
    total_prob = sum(probabilities)
    probabilities = [p / total_prob if total_prob > 0 else 0 for p in probabilities]
    
    # Atualiza as probabilidades em %
    for i, prob in enumerate(probabilities):
        if i < len(data):
            data[i]['Probability (%)'] = prob * 100
    
    return probabilities, data

# Método da roleta para selecionar o próximo nó
def roulette_wheel_selection(probabilities):
    r = np.random.rand()
    cumulative_sum = 0.0
    for i, probability in enumerate(probabilities):
        cumulative_sum += probability
        if r <= cumulative_sum:
            return i
    return len(probabilities) - 1

# Simula o percurso de uma formiga
def simulate_ant(start_node, pheromone, G, iteration):
    path = []
    visited = set([start_node])
    current_node = start_node
    total_distance = 0
    data = []
    
    while len(visited) < len(nodes):
        unvisited = [node for node in nodes if node not in visited and G.has_edge(current_node, node)]
        if not unvisited:
            return None, float('inf'), data  # Caminho inválido
        
        probabilities, prob_data = calculate_transition_probabilities(current_node, unvisited, pheromone, G, iteration)
        data.extend(prob_data)
        next_index = roulette_wheel_selection(probabilities)
        next_node = unvisited[next_index]
        
        path.append((current_node, next_node))
        total_distance += G[current_node][next_node]['weight']
        visited.add(next_node)
        current_node = next_node

    if G.has_edge(current_node, start_node):
        path.append((current_node, start_node))
        total_distance += G[current_node][start_node]['weight']
        return path, total_distance, data
    else:
        return None, float('inf'), data

# Atualiza os feromônios
def update_pheromone(pheromone, all_paths, evaporation_rate):
    for edge in list(pheromone.keys()):
        pheromone[edge] *= (1 - evaporation_rate)
        pheromone[edge] = max(min_pheromone, pheromone[edge])
    
    for path, total_distance in all_paths:
        if path and total_distance > 0:
            pheromone_deposit = Q / total_distance
            for edge in path:
                if edge in pheromone:
                    pheromone[edge] += pheromone_deposit
                elif (edge[1], edge[0]) in pheromone:
                    pheromone[(edge[1], edge[0])] += pheromone_deposit
                else:
                    pheromone[edge] = pheromone_deposit
                pheromone[edge] = min(max_pheromone, max(min_pheromone, pheromone[edge]))

# Algoritmo ACO
best_path = None
best_distance = float('inf')
all_data = []

for iteration in range(num_iterations):
    all_paths = []
    for start_node in nodes:
        path, total_distance, data = simulate_ant(start_node, pheromone, G, iteration)
        all_data.extend(data)
        if path and total_distance < float('inf'):
            all_paths.append((path, total_distance))
            if total_distance < best_distance:
                best_path = path
                best_distance = total_distance
    update_pheromone(pheromone, all_paths, evaporation_rate)

# Exibindo os dados em tabela
df = pd.DataFrame(all_data)
print(df)

# Resultado
print("\nMelhor caminho encontrado:", best_path)
print("Distância total do melhor caminho:", best_distance)

# Visualização do grafo e melhor caminho
pos = {
    1: (0.5, 1.5),
    2: (0, 1),
    3: (1, 1),
    4: (0, 0),
    5: (1, 0),
    6: (0.5, 0.6),
    7: (0.5, 1.9)
}
plt.figure(figsize=(6, 6))
nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='black')
nx.draw_networkx_edges(G, pos, width=2, edge_color='gray')
nx.draw_networkx_labels(G, pos, font_size=20, font_color='white')

# Destaca o melhor caminho
if best_path:
    edge_colors = ['red' if (u, v) in best_path or (v, u) in best_path else 'black' for u, v in G.edges]
else:
    edge_colors = ['black' for _ in G.edges]
nx.draw_networkx_edges(G, pos, edgelist=G.edges, edge_color=edge_colors, width=2)

# Adiciona os pesos das arestas
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black')

plt.axis('off')
plt.show()
