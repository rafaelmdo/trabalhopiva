"""Demonstração de otimização de entregas com ACO."""

import argparse
import subprocess
import sys
import unittest
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory

import networkx as nx
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from numpy.typing import NDArray



@dataclass(frozen=True)
class Ponto:
    codigo: str
    endereco: str
    coordenadas: tuple[float, float]


def cadastro(clientes: int = 15) -> tuple[Ponto, ...]:
    """Carrega o depósito e de 10 a 20 clientes em ordem de cadastro."""
    if not 10 <= clientes <= 20:
        raise ValueError("Informe entre 10 e 20 clientes.")
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
        ("Rua Jabuticaba", 20, 7),
    )
    deposito = Ponto("CD", "Avenida das Entregas, 100", (10, 10))
    return (deposito, *(
        Ponto(f"C{indice:02}", f"{rua}, {indice * 10}", (x, y))
        for indice, (rua, x, y) in enumerate(locais[:clientes], 1)
    ))


def matriz_distancias(coordenadas: Sequence[tuple[float, float]]) -> NDArray[np.float64]:
    """Calcula a distância euclidiana entre cada par de coordenadas em km."""
    pontos = np.asarray(coordenadas, dtype=float)
    if pontos.ndim != 2 or pontos.shape[1] != 2 or not np.isfinite(pontos).all():
        raise ValueError("Informe pares de coordenadas finitas.")
    diferencas = pontos[:, None] - pontos[None, :]
    return np.hypot(diferencas[..., 0], diferencas[..., 1])


def distancia_rota(matriz: NDArray[np.float64], rota: tuple[int, ...]) -> float:
    """Soma os trechos consecutivos da rota, incluindo o retorno explícito."""
    return float(matriz[list(rota[:-1]), list(rota[1:])].sum())


@dataclass(frozen=True)
class Config:
    formigas: int = 25
    iteracoes: int = 100
    alfa: float = 1.0
    beta: float = 3.0
    evaporacao: float = 0.35
    semente: int = 42

    def __post_init__(self) -> None:
        if any(type(valor) is not int or valor < 1 for valor in (self.formigas, self.iteracoes)):
            raise ValueError("Formigas e iterações devem ser inteiros positivos.")
        if any(not isfinite(valor) or valor < 0 for valor in (self.alfa, self.beta)):
            raise ValueError("Alfa e beta devem ser finitos e não negativos.")
        if not 0 < self.evaporacao < 1:
            raise ValueError("A evaporação deve ser maior que zero e menor que um.")
        if type(self.semente) is not int or self.semente < 0:
            raise ValueError("A semente deve ser um inteiro não negativo.")


@dataclass(frozen=True)
class Resultado:
    rota: tuple[int, ...]
    distancia: float
    historico: tuple[float, ...]


def _construir_rota(
    atratividade: NDArray[np.float64], rng: np.random.Generator, escala: float
) -> tuple[int, ...]:
    """Sorteia apenas clientes ainda não visitados e fecha o circuito no CD."""
    restantes = list(range(1, len(atratividade)))
    rota = [0]
    while restantes:
        escores = atratividade[rota[-1], restantes]
        with np.errstate(over="ignore", under="ignore"):
            pesos = np.exp((escores - escores.max()) * escala)
        indice = int(rng.choice(len(restantes), p=pesos / pesos.sum()))
        rota.append(restantes.pop(indice))
    return (*rota, 0)


def otimizar(matriz: NDArray[np.float64], config: Config) -> Resultado:
    """Retorna a melhor rota amostrada e o melhor valor acumulado por iteração."""
    matriz = np.asarray(matriz, dtype=float)
    if matriz.ndim != 2 or matriz.shape[0] != matriz.shape[1] or len(matriz) < 2:
        raise ValueError("A matriz deve ser quadrada com pelo menos dois pontos.")
    conexoes = ~np.eye(len(matriz), dtype=bool)
    if (not np.isfinite(matriz).all() or not np.array_equal(matriz, matriz.T)
            or np.any(matriz[conexoes] <= 0) or np.any(matriz.diagonal() != 0)):
        raise ValueError("A matriz deve ser simétrica, com diagonal zero e demais distâncias positivas.")

    rng = np.random.default_rng(config.semente)
    feromonio = np.ones_like(matriz)
    log_visibilidade = np.zeros_like(matriz)
    log_visibilidade[conexoes] = -np.log(matriz[conexoes])
    # Escalar pelo maior expoente preserva a distribuição ao reverter a escala
    # depois de subtrair o máximo, evitando overflow nos escores intermediários.
    escala = max(1.0, config.alfa, config.beta)
    melhor_rota: tuple[int, ...] = ()
    melhor_distancia = float("inf")
    historico: list[float] = []

    for _ in range(config.iteracoes):
        atratividade = ((config.alfa / escala) * np.log(feromonio)
                        + (config.beta / escala) * log_visibilidade)
        deposito = np.zeros_like(matriz)
        for _ in range(config.formigas):
            rota = _construir_rota(atratividade, rng, escala)
            distancia = distancia_rota(matriz, rota)
            if distancia < melhor_distancia:
                melhor_rota, melhor_distancia = rota, distancia
            inicio, fim = list(rota[:-1]), list(rota[1:])
            np.add.at(deposito, (inicio, fim), 1.0 / distancia)
            np.add.at(deposito, (fim, inicio), 1.0 / distancia)
        feromonio = np.maximum((1 - config.evaporacao) * feromonio + deposito, 1e-12)
        historico.append(melhor_distancia)

    return Resultado(melhor_rota, melhor_distancia, tuple(historico))


def salvar_graficos(
    pontos: tuple[Ponto, ...], inicial: tuple[int, ...],
    distancia_inicial: float, resultado: Resultado, destino: Path
) -> None:
    """Desenha as duas rotas na mesma escala e o melhor valor por iteração."""
    destino.mkdir(parents=True, exist_ok=True)
    posicoes = {indice: ponto.coordenadas for indice, ponto in enumerate(pontos)}
    rotulos = {indice: ponto.codigo for indice, ponto in enumerate(pontos)}
    figura = Figure(figsize=(12, 6), layout="constrained")
    FigureCanvasAgg(figura)
    eixos = figura.subplots(1, 2, sharex=True, sharey=True)
    for eixo, rota, titulo, distancia, cor in (
        (eixos[0], inicial, "Ordem de cadastro", distancia_inicial, "#825128"),
        (eixos[1], resultado.rota, "Melhor rota ACO", resultado.distancia, "#17675c"),
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
        eixo.set(xlim=(-1, 22), ylim=(0, 21), aspect="equal",
                 xlabel="Coordenada X (km)", ylabel="Coordenada Y (km)")
        eixo.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        eixo.grid(alpha=0.17)
        eixo.set_axisbelow(True)
    figura.savefig(destino / "rotas.png", dpi=150)

    figura = Figure(figsize=(9, 4), layout="constrained")
    FigureCanvasAgg(figura)
    eixo = figura.subplots()
    eixo.plot(range(1, len(resultado.historico) + 1), resultado.historico,
              color="#17675c", drawstyle="steps-post", marker=".", markersize=3,
              label="Melhor distância acumulada")
    eixo.axhline(distancia_inicial, color="#825128", linestyle="--", label="Rota inicial")
    eixo.set(title="Evolução da colônia", xlabel="Iteração", ylabel="Distância (km)")
    eixo.grid(alpha=0.2)
    eixo.legend()
    figura.savefig(destino / "evolucao.png", dpi=150)


def main() -> None:
    """Valida os parâmetros, executa o ACO e apresenta a comparação."""
    padrao = Config()
    parser = argparse.ArgumentParser(description="Otimização de rotas de entrega com ACO.")
    parser.add_argument("--clientes", type=int, default=15, help="Quantidade de clientes (10 a 20)")
    parser.add_argument("--formigas", type=int, default=padrao.formigas, help="Formigas por iteração")
    parser.add_argument("--iteracoes", type=int, default=padrao.iteracoes, help="Total de iterações")
    parser.add_argument("--alfa", type=float, default=padrao.alfa, help="Influência do feromônio")
    parser.add_argument("--beta", type=float, default=padrao.beta, help="Influência da proximidade")
    parser.add_argument("--evaporacao", type=float, default=padrao.evaporacao,
                        help="Fração de feromônio evaporada a cada iteração")
    parser.add_argument("--semente", type=int, default=padrao.semente, help="Semente aleatória")
    parser.add_argument("--velocidade", type=float, default=35.0, help="Velocidade média em km/h")
    parser.add_argument("--custo-km", type=float, default=1.50, help="Custo variável em R$/km")
    parser.add_argument("--saida", type=Path, default=Path("resultados"), help="Pasta dos gráficos")
    args = parser.parse_args()
    try:
        pontos = cadastro(args.clientes)
        config = Config(args.formigas, args.iteracoes, args.alfa, args.beta,
                        args.evaporacao, args.semente)
        if not isfinite(args.velocidade) or args.velocidade <= 0:
            raise ValueError("A velocidade deve ser finita e positiva.")
        if not isfinite(args.custo_km) or args.custo_km < 0:
            raise ValueError("O custo por km deve ser finito e não negativo.")
    except ValueError as erro:
        parser.error(str(erro))

    matriz = matriz_distancias([ponto.coordenadas for ponto in pontos])
    inicial = (*range(len(pontos)), 0)
    distancia_inicial = distancia_rota(matriz, inicial)
    resultado = otimizar(matriz, config)
    economia = distancia_inicial - resultado.distancia
    percentual = 100 * economia / distancia_inicial
    salvar_graficos(pontos, inicial, distancia_inicial, resultado, args.saida)

    print("Cadastro fictício")
    for ponto in pontos:
        print(f"  {ponto.codigo}  {ponto.endereco}  {ponto.coordenadas}")
    for titulo, rota, distancia in (
        ("Rota inicial", inicial, distancia_inicial),
        ("Melhor rota ACO", resultado.rota, resultado.distancia),
    ):
        sequencia = " -> ".join(pontos[indice].codigo for indice in rota)
        print(f"\n{titulo}\n  {sequencia}")
        print(f"  {distancia:.2f} km | {60 * (distancia / args.velocidade):.1f} min estimados"
              f" | R$ {distancia * args.custo_km:.2f} de custo variável estimado")
    metrica = "Economia" if economia >= 0 else "Aumento"
    print(f"\n{metrica} de distância: {abs(economia):.2f} km ({abs(percentual):.2f}%)")
    print(f"Gráficos salvos em {args.saida.resolve()}")


class RotasTest(unittest.TestCase):
    def test_matriz_e_fechamento_do_circuito(self):
        matriz = matriz_distancias([(0, 0), (3, 0), (3, 4)])
        np.testing.assert_array_equal(matriz, [[0, 3, 5], [3, 0, 4], [5, 4, 0]])
        self.assertEqual(distancia_rota(matriz, (0, 1, 2, 0)), 12)

    def test_aco_encontra_perimetro_do_quadrado(self):
        matriz = matriz_distancias([(0, 0), (1, 1), (1, 0), (0, 1)])
        config = Config(formigas=8, iteracoes=10, semente=42)
        resultado = otimizar(matriz, config)
        self.assertEqual(resultado.rota[0], 0)
        self.assertEqual(resultado.rota[-1], 0)
        self.assertEqual(sorted(resultado.rota[1:-1]), [1, 2, 3])
        self.assertEqual(resultado.distancia, 4)
        self.assertEqual(resultado.distancia, distancia_rota(matriz, resultado.rota))
        self.assertEqual(len(resultado.historico), 10)
        self.assertEqual(resultado.historico, tuple(sorted(resultado.historico, reverse=True)))
        self.assertEqual(resultado.historico[-1], resultado.distancia)
        self.assertEqual(resultado, otimizar(matriz, config))

    def test_limites_do_cadastro(self):
        self.assertEqual(len(cadastro(10)), 11)
        self.assertEqual(len(cadastro(20)), 21)
        for quantidade in (9, 21):
            with self.subTest(quantidade=quantidade), self.assertRaises(ValueError):
                cadastro(quantidade)

    def test_parametros_invalidos(self):
        for parametro in ({"formigas": 0}, {"iteracoes": 0}, {"alfa": -1},
                          {"beta": float("nan")}, {"evaporacao": 1}, {"semente": -1}):
            with self.subTest(parametro=parametro), self.assertRaises(ValueError):
                Config(**parametro)

    def test_rejeita_distancias_invalidas(self):
        for matriz in ([[0, 0], [0, 0]], [[0, 1], [2, 0]], [[0, 1, 2]]):
            with self.subTest(matriz=matriz), self.assertRaises(ValueError):
                otimizar(np.array(matriz), Config())


class CLITest(unittest.TestCase):
    def test_gera_comparacao_e_graficos(self):
        projeto = Path(__file__).resolve().parent
        with TemporaryDirectory() as pasta:
            execucao = subprocess.run(
                [sys.executable, "-B", "-X", "utf8", "main.py", "--clientes", "10",
                 "--formigas", "4", "--iteracoes", "3", "--saida", pasta],
                cwd=projeto, capture_output=True, encoding="utf-8", timeout=30,
            )
            self.assertEqual(execucao.returncode, 0, execucao.stderr)
            for trecho in ("Rota inicial", "Melhor rota ACO", "min estimados", "R$", "%"):
                self.assertIn(trecho, execucao.stdout)
            for nome in ("rotas.png", "evolucao.png"):
                self.assertTrue((Path(pasta) / nome).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))

    def test_rejeita_cliente_fora_do_intervalo(self):
        projeto = Path(__file__).resolve().parent
        execucao = subprocess.run(
            [sys.executable, "-B", "-X", "utf8", "main.py", "--clientes", "21"],
            cwd=projeto, capture_output=True, encoding="utf-8", timeout=30,
        )
        self.assertEqual(execucao.returncode, 2)
        self.assertIn("Informe entre 10 e 20 clientes.", execucao.stderr)


if __name__ == "__main__":
    if sys.argv[1:] == ["--test"]:
        unittest.main(argv=[sys.argv[0]], verbosity=2)
    else:
        main()
