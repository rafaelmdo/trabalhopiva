# Otimização de rotas de entregas com ACO

O projeto simula um veículo que sai de um centro de distribuição (CD), visita de 10 a 20 clientes fictícios uma vez e volta ao CD. O objetivo do TSP é minimizar a distância euclidiana total. A ordem de cadastro serve de referência para medir a economia.

## Instalação e execução

O projeto foi validado com Python 3.13. Execute na pasta que contém o arquivo `main.py`.

```console
python -m venv .venv
```

No PowerShell, instale as dependências e execute a demonstração.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

No Linux e macOS, use `./.venv/bin/python` nos mesmos comandos. Com as dependências já instaladas, basta executar `python main.py`.

## Parâmetros

| Opção | Padrão | Significado |
|---|---:|---|
| `--clientes` | 15 | Quantidade de clientes, entre 10 e 20 |
| `--formigas` | 25 | Rotas amostradas por iteração |
| `--iteracoes` | 100 | Ciclos de construção e atualização do feromônio |
| `--alfa` | 1,0 | Expoente do feromônio, finito e não negativo |
| `--beta` | 3,0 | Expoente do inverso da distância, finito e não negativo |
| `--evaporacao` | 0,35 | Fração evaporada por iteração, entre 0 e 1 exclusivos |
| `--semente` | 42 | Inteiro não negativo para repetir o sorteio |
| `--velocidade` | 35 | Velocidade média em km/h para estimar tempo em trânsito |
| `--custo-km` | 1,50 | Custo variável estimado em reais por km |
| `--saida` | `resultados` | Diretório de exportação dos PNGs |

Use ponto decimal nos parâmetros de linha de comando.

```console
python main.py --clientes 20 --formigas 35 --iteracoes 120 --alfa 1 --beta 4 --evaporacao 0.25 --semente 8
```

## Como funciona

1. O Python calcula a distância em linha reta entre cada par de pontos com `math.hypot` e guarda os resultados em uma lista de listas. Cada conexão começa com feromônio igual a 1.
2. Cada formiga parte do CD. Entre os clientes ainda não visitados, a chance de escolher uma conexão é proporcional a `feromônio^alfa × (1/distância)^beta`. O sorteio usa `random.choices`. O código calcula os pesos com logaritmos para lidar com expoentes grandes.
3. A formiga retorna ao CD e a distância inclui esse último trecho. Ao final de cada iteração, a colônia multiplica o feromônio por `1 - evaporação` e soma o depósito de todas as formigas. Cada formiga deposita `1/distância total` em cada trecho usado, nos dois sentidos.
4. O algoritmo guarda a melhor rota amostrada e o menor valor acumulado até cada iteração. ACO é uma heurística estocástica, com resultado dependente dos parâmetros e da semente.

## Resultados da demonstração

Com os parâmetros padrão (semente 42) e 15 clientes, a execução produziu os seguintes valores.

| Indicador | Ordem de cadastro | Melhor rota ACO |
|---|---:|---:|
| Distância | 219,99 km | 82,36 km |
| Tempo estimado em trânsito | 377,1 min | 141,2 min |
| Custo variável estimado | R$ 329,99 | R$ 123,53 |

A distância economizada foi **137,64 km (62,56%)**, calculada antes do arredondamento. O percentual é a diferença de distância dividida pela distância inicial, multiplicada por 100. Com poucas amostras, a melhor rota da colônia pode superar a distância de referência e o terminal reporta o aumento. O tempo é distância ÷ velocidade constante. O custo é distância × tarifa fixa por km. As estimativas representam somente deslocamento, com coordenadas fictícias e trajetos em linha reta.

- [Mapas da rota inicial e da melhor rota ACO](resultados/rotas.png)
- [Curva de evolução e linha de referência](resultados/evolucao.png)

## Dependências e estrutura

O algoritmo usa apenas a biblioteca padrão do Python para calcular distâncias e fazer sorteios. NetworkX representa as rotas como grafos e Matplotlib gera os PNGs. Matplotlib instala NumPy como dependência indireta.

O arquivo `main.py` reúne o cadastro, o algoritmo, os gráficos, e a interface de terminal.

A função de otimização recebe uma matriz independente do cadastro e dos gráficos. Uma integração com mapas pode fornecer sua própria matriz simétrica. Para usar tempos ou distâncias viárias assimétricas, será preciso estender a validação e o depósito de feromônio para trechos direcionados. Um futuro solucionador VRP pode reutilizar o cadastro e as operações de distância, com um resultado de múltiplas rotas e regras de capacidade, prioridade e janelas de entrega.
