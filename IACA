import numpy as np

# Função de custo do tanque (exemplo genérico)
# Substitua por sua própria função
def volume_tanque(altura, raio):
    return np.pi * raio**2 * altura

# Parâmetros do algoritmo IACA
def custo_tanque(altura, raio):
    """
    Calcula o custo de um tanque com base em sua altura e raio.
    A fórmula pode incluir custos de material, volume, etc.
    """
    c_l = 2 # Custo para fazer a lateral do tanque
    c_b = 2 # Custo para fazer a base do tanque
    volume_desejado = 2 # Volume desejado
    if volume_desejado >= 15:
      tx = volume_desejado
    else: # Para volumes abaixo de 15 o código estava dando erro, logo, fiz esse artíficie para tentar fazê-lo funcionar
      tx = 15
    volume_tanque = np.pi * raio**2 * altura # Cálculo do volume do tanque
    custo_material = 2 * (c_l * np.pi * raio * altura +  c_b * np.pi * raio**2) # Cálculo do custo para fazer o tanque com tais especificações
    penalidade_volume = abs(volume_tanque - volume_desejado) * tx # Esse fator de penalidade é para forçar as formigas encontrarem a solução ótima, talvez utilizar valores maiores, melhore a solução
    if volume_desejado == 0:
        return 1
    else:
      return custo_material + penalidade_volume # O retorno é o custo do material + a penalidade do volume

# Parâmetros do algoritmo IACA
num_formigas = 5000 # Número de formigas
num_iteracoes = 2500 # Número de Iterações
rho = 0.1  # Taxa de evaporação do feromônio
Q = 1.0    # Contribuição de feromônio
bounds = {"altura": (0.0, 5), "raio": (0.0, 5)}  # Limites inferiores e superiores para altura e raio

# Inicialização das trilhas de feromônio
tau_altura = np.ones(15000)  # Discretização da altura em intervalos, seus valores começam com 1, para dizer que daca espaço já possui feromônio
tau_raio = np.ones(15000)    # Discretização do raio em intervalos, seus valores começam com 1, para dizer que daca espaço já possui feromônio

# Função para mapear valor para o espaço contínuo
def mapear_intervalo(index, bounds, steps): # Sua principal função é converter um indexador da discretização em um número real dentro da região viável
    low, high = bounds
    return low + (high - low) * index / steps

# Execução do algoritmo IACA
melhor_solucao = None
melhor_custo = float('inf')

for iteracao in range(num_iteracoes):
    solucoes = []
    custos = []

    # Cada formiga gera uma solução
    for _ in range(num_formigas):
        # Escolha proporcional ao feromônio
        prob_altura = tau_altura / tau_altura.sum()
        prob_raio = tau_raio / tau_raio.sum()

        idx_altura = np.random.choice(len(tau_altura), p=prob_altura)
        idx_raio = np.random.choice(len(tau_raio), p=prob_raio)

        altura = mapear_intervalo(idx_altura, bounds["altura"], len(tau_altura))
        raio = mapear_intervalo(idx_raio, bounds["raio"], len(tau_raio))

        custo = custo_tanque(altura, raio)
        solucoes.append((altura, raio))
        custos.append(custo)

        # Atualização local de feromônio
        tau_altura[idx_altura] *= (1 - rho)
        tau_altura[idx_altura] += rho * Q / custo
        tau_raio[idx_raio] *= (1 - rho)
        tau_raio[idx_raio] += rho * Q / custo

    # Atualização global de feromônio
    for (altura, raio), custo in zip(solucoes, custos):
        idx_altura = int((altura - bounds["altura"][0]) / (bounds["altura"][1] - bounds["altura"][0]) * len(tau_altura))
        idx_raio = int((raio - bounds["raio"][0]) / (bounds["raio"][1] - bounds["raio"][0]) * len(tau_raio))
        tau_altura[idx_altura] += Q / custo
        tau_raio[idx_raio] += Q / custo

    # Melhor solução na iteração
    iter_melhor_custo = min(custos)
    iter_melhor_solucao = solucoes[np.argmin(custos)]

    # Atualizar a melhor solução global
    if iter_melhor_custo < melhor_custo:
        melhor_custo = iter_melhor_custo
        melhor_solucao = iter_melhor_solucao

    print(f"Iteração {iteracao + 1}: Melhor Custo = {melhor_custo}, Melhor Solução = {melhor_solucao}, Volume = {volume_tanque(melhor_solucao[0],melhor_solucao[1])}")

# Resultado final
print("\nMelhor solução encontrada:")
print(f"Altura: {melhor_solucao[0]}, Raio: {melhor_solucao[1]}, Custo: {melhor_custo:.2f}, ")
