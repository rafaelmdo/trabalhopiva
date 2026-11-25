"""Demonstração de otimização de entregas com ACO."""

import argparse
import math
import random
from pathlib import Path

import networkx as nx
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def cadastro(clientes=None):
    """Devolve o CD e os clientes no formato (código, endereço, (x, y))."""
    locais = (
        ("Rua Ipê", 2, 18), ("Rua Cedro", 18, 3), ("Rua Angico", 4, 4),
        ("Rua Jatobá", 19, 16), ("Rua Jacarandá", 9, 19),
        ("Rua Peroba", 13, 2), ("Rua Aroeira", 2, 10),
        ("Rua Araucária", 17, 10), ("Rua Canela", 5, 15),
        ("Rua Pau-Brasil", 15, 6), ("Rua Jequitibá", 6, 7),
        ("Rua Cambuci", 16, 19), ("Rua Manacá", 3, 2),
        ("Rua Pitanga", 14, 13), ("Rua Quaresmeira", 8, 12),
        ("Rua Embaúba", 10, 5), ("Rua Imbuia", 12, 17),
        ("Rua Guapuruvu", 7, 2), ("Rua Sapucaia", 1, 14),
        ("Rua Jabuticaba", 20, 7), ("Rua Santos", -1, 4),
    )

    if clientes is None:
        clientes = len(locais)

    if not 1 <= clientes <= len(locais):
        raise ValueError(f"Informe entre 1 e {len(locais)} clientes.")

    pontos = [("CD", "Avenida das Entregas, 100", (10, 10))]
    for numero, (rua, x, y) in enumerate(locais[:clientes], start=1):
        pontos.append((f"C{numero:02}", f"{rua}, {numero * 10}", (x, y)))
    return pontos


def matriz_distancias(coordenadas):
    """Monta a matriz euclidiana de distâncias em quilômetros."""
    if not coordenadas:
        raise ValueError("Informe pares de coordenadas finitas.")
    for ponto in coordenadas:
        if len(ponto) != 2 or not all(math.isfinite(valor) for valor in ponto):
            raise ValueError("Informe pares de coordenadas finitas.")

    matriz = []
    for x_origem, y_origem in coordenadas:
        linha = []
        for x_destino, y_destino in coordenadas:
            linha.append(math.hypot(x_destino - x_origem, y_destino - y_origem))
        matriz.append(linha)
    return matriz


def distancia_rota(matriz, rota):
    """Soma a distância entre cada par de pontos consecutivos."""
    total = 0.0
    for origem, destino in zip(rota, rota[1:]):
        total += matriz[origem][destino]
    return total


def validar_parametros(formigas, iteracoes, alfa, beta, evaporacao, semente):
    """Rejeita configurações inválidas antes do primeiro sorteio."""
    for quantidade in (formigas, iteracoes):
        if type(quantidade) is not int or quantidade < 1:
            raise ValueError("Formigas e iterações devem ser inteiros positivos.")
    for peso in (alfa, beta):
        if not math.isfinite(peso) or peso < 0:
            raise ValueError("Alfa e beta devem ser finitos e não negativos.")
    if not 0 < evaporacao < 1:
        raise ValueError("A evaporação deve ser maior que zero e menor que um.")
    if type(semente) is not int or semente < 0:
        raise ValueError("A semente deve ser um inteiro não negativo.")


def validar_matriz(matriz):
    """Confere as distâncias exigidas pelo TSP simétrico."""
    tamanho = len(matriz)
    if tamanho < 2 or any(len(linha) != tamanho for linha in matriz):
        raise ValueError("A matriz deve ser quadrada com pelo menos dois pontos.")
    for origem in range(tamanho):
        for destino in range(tamanho):
            distancia = matriz[origem][destino]
            if origem == destino and distancia == 0:
                continue
            if (origem == destino or not math.isfinite(distancia)
                    or distancia <= 0 or distancia != matriz[destino][origem]):
                raise ValueError(
                    "A matriz deve ser simétrica, com diagonal zero e demais distâncias positivas."
                )


def escolher_destino(atual, restantes, matriz, feromonios, alfa, beta, sorteio):
    """Sorteia um cliente com base no feromônio e na proximidade."""
    # Em log, o produto feromônio^alfa × (1/distância)^beta vira uma soma.
    # A escala e a subtração do maior escore evitam pesos grandes demais.
    escala = max(1.0, alfa, beta)
    escores = []
    for destino in restantes:
        feromonio = feromonios[atual][destino]
        distancia = matriz[atual][destino]
        escore = (alfa / escala) * math.log(feromonio) - (beta / escala) * math.log(distancia)
        escores.append(escore)

    maior_escore = max(escores)
    pesos = []
    for escore in escores:
        pesos.append(math.exp((escore - maior_escore) * escala))
    # A probabilidade de cada cliente é seu peso dividido pela soma dos pesos.
    return sorteio.choices(restantes, weights=pesos)[0]


def construir_rota(matriz, feromonios, alfa, beta, sorteio):
    """Monta o circuito de uma formiga partindo do CD."""
    restantes = list(range(1, len(matriz)))
    rota = [0]
    while restantes:
        destino = escolher_destino(rota[-1], restantes, matriz, feromonios, alfa, beta, sorteio)
        rota.append(destino)
        restantes.remove(destino)
    rota.append(0)
    return rota


def atualizar_feromonios(feromonios, depositos, evaporacao):
    """Evapora parte do feromônio anterior e soma os depósitos da iteração."""
    for origem in range(len(feromonios)):
        for destino in range(len(feromonios)):
            restante = (1 - evaporacao) * feromonios[origem][destino]
            # O piso mantém o feromônio positivo para o próximo logaritmo.
            feromonios[origem][destino] = max(restante + depositos[origem][destino], 1e-12)


def otimizar(matriz, formigas=25, iteracoes=100, alfa=1.0, beta=3.0,
             evaporacao=0.35, semente=42):
    """Devolve a melhor rota, sua distância e o histórico por iteração."""
    validar_parametros(formigas, iteracoes, alfa, beta, evaporacao, semente)
    validar_matriz(matriz)
    sorteio = random.Random(semente)
    tamanho = len(matriz)
    feromonios = []
    for _ in range(tamanho):
        feromonios.append([1.0] * tamanho)

    melhor_rota = []
    melhor_distancia = math.inf
    historico = []

    for _ in range(iteracoes):
        depositos = []
        for _ in range(tamanho):
            depositos.append([0.0] * tamanho)

        for _ in range(formigas):
            rota = construir_rota(matriz, feromonios, alfa, beta, sorteio)
            distancia = distancia_rota(matriz, rota)
            if distancia < melhor_distancia:
                melhor_rota = rota
                melhor_distancia = distancia

            deposito = 1 / distancia
            for origem, destino in zip(rota, rota[1:]):
                depositos[origem][destino] += deposito
                depositos[destino][origem] += deposito

        # As formigas da mesma iteração usam o mesmo estado de feromônio.
        atualizar_feromonios(feromonios, depositos, evaporacao)
        historico.append(melhor_distancia)

    return melhor_rota, melhor_distancia, historico


def salvar_graficos(
    pontos, inicial, distancia_inicial, melhor_rota, melhor_distancia, historico, destino
) -> None:
    """Desenha as duas rotas na mesma escala e o melhor valor por iteração."""
    destino.mkdir(parents=True, exist_ok=True)
    posicoes = {}
    rotulos = {}
    for indice, (codigo, _, coordenadas) in enumerate(pontos):
        posicoes[indice] = coordenadas
        rotulos[indice] = codigo
    figura = Figure(figsize=(12, 6), layout="constrained")
    FigureCanvasAgg(figura)
    eixos = figura.subplots(1, 2, sharex=True, sharey=True)
    for eixo, rota, titulo, distancia, cor in (
        (eixos[0], inicial, "Ordem de cadastro", distancia_inicial, "#825128"),
        (eixos[1], melhor_rota, "Melhor rota ACO", melhor_distancia, "#17675c"),
    ):
        grafo = nx.DiGraph()
        grafo.add_nodes_from(posicoes)
        grafo.add_edges_from(zip(rota[:-1], rota[1:]))
        nx.draw_networkx_edges(grafo, posicoes, ax=eixo, edge_color=cor, arrowsize=12,
                               node_size=430, alpha=0.72)
        nx.draw_networkx_nodes(grafo, posicoes, ax=eixo, node_size=430,
                               node_color=["#172535" if no == 0 else cor for no in grafo])
        nx.draw_networkx_labels(grafo, posicoes, labels=rotulos, ax=eixo,
                                font_size=8, font_color="white")
        eixo.set_title(f"{titulo} · {distancia:.2f} km")
        eixo.set(xlim=(-1, 22), ylim=(0, 21), aspect="equal", #tam grafico
                 xlabel="Coordenada X (km)", ylabel="Coordenada Y (km)")
        eixo.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        eixo.grid(alpha=0.17)
        eixo.set_axisbelow(True)
    figura.savefig(destino / "rotas.png", dpi=150)

    figura = Figure(figsize=(9, 4), layout="constrained")
    FigureCanvasAgg(figura)
    eixo = figura.subplots()
    eixo.plot(range(1, len(historico) + 1), historico,
              color="#17675c", drawstyle="steps-post", marker=".", markersize=3,
              label="Melhor distância acumulada")
    eixo.axhline(distancia_inicial, color="#825128", linestyle="--", label="Rota inicial")
    eixo.set(title="Evolução da colônia", xlabel="Iteração", ylabel="Distância (km)")
    eixo.grid(alpha=0.2)
    eixo.legend()
    figura.savefig(destino / "evolucao.png", dpi=150)


def main() -> None:
    """Valida os parâmetros, executa o ACO e apresenta a comparação."""
    parser = argparse.ArgumentParser(description="Otimização de rotas de entrega com ACO.")
    parser.add_argument("--clientes", type=int, help="Quantidade de clientes. Por padrão, usa todos.")
    parser.add_argument("--formigas", type=int, default=25, help="Formigas por iteração")
    parser.add_argument("--iteracoes", type=int, default=100, help="Total de iterações")
    parser.add_argument("--alfa", type=float, default=1.0, help="Influência do feromônio")
    parser.add_argument("--beta", type=float, default=3.0, help="Influência da proximidade")
    parser.add_argument("--evaporacao", type=float, default=0.35,
                        help="Fração de feromônio evaporada a cada iteração")
    parser.add_argument("--semente", type=int, default=42, help="Semente aleatória")
    parser.add_argument("--velocidade", type=float, default=35.0, help="Velocidade média em km/h")
    parser.add_argument("--custo-km", type=float, default=1.50, help="Custo variável em R$/km")
    parser.add_argument("--saida", type=Path, default=Path("resultados"), help="Pasta dos gráficos")
    args = parser.parse_args()
    try:
        pontos = cadastro(args.clientes)
        validar_parametros(args.formigas, args.iteracoes, args.alfa, args.beta,
                           args.evaporacao, args.semente)
        if not math.isfinite(args.velocidade) or args.velocidade <= 0:
            raise ValueError("A velocidade deve ser finita e positiva.")
        if not math.isfinite(args.custo_km) or args.custo_km < 0:
            raise ValueError("O custo por km deve ser finito e não negativo.")
    except ValueError as erro:
        parser.error(str(erro))

    coordenadas = []
    for _, _, posicao in pontos:
        coordenadas.append(posicao)
    matriz = matriz_distancias(coordenadas)
    inicial = list(range(len(pontos)))
    inicial.append(0)
    distancia_inicial = distancia_rota(matriz, inicial)
    melhor_rota, melhor_distancia, historico = otimizar(
        matriz, args.formigas, args.iteracoes, args.alfa, args.beta,
        args.evaporacao, args.semente
    )
    economia = distancia_inicial - melhor_distancia
    percentual = 100 * economia / distancia_inicial
    salvar_graficos(pontos, inicial, distancia_inicial, melhor_rota,
                    melhor_distancia, historico, args.saida)

    print("Cadastro fictício")
    for codigo, endereco, posicao in pontos:
        print(f"  {codigo}  {endereco}  {posicao}")
    for titulo, rota, distancia in (
        ("Rota inicial", inicial, distancia_inicial),
        ("Melhor rota ACO", melhor_rota, melhor_distancia),
    ):
        sequencia = " -> ".join(pontos[indice][0] for indice in rota)
        print(f"\n{titulo}\n  {sequencia}")
        print(f"  {distancia:.2f} km | {60 * (distancia / args.velocidade):.1f} min estimados"
              f" | R$ {distancia * args.custo_km:.2f} de custo variável estimado")
    metrica = "Economia" if economia >= 0 else "Aumento"
    print(f"\n{metrica} de distância: {abs(economia):.2f} km ({abs(percentual):.2f}%)")
    print(f"Gráficos salvos em {args.saida.resolve()}")


if __name__ == "__main__":
    main()
