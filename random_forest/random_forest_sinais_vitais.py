import argparse
import time
from pathlib import Path
import numpy as np


# parametros da floresta (mesma semente do MLP para gerar a mesma divisao treino/teste)
SEMENTE = 10
ARVORES = 100
ATRIBUTOS_SORTEADOS = 2
PROFUNDIDADE_MAXIMA = 20
MINIMO_FOLHA = 1
CLASSES = ("crítico", "instável", "potencialmente estável", "estável")


# le o csv e valida formato, valores finitos, ids inteiros e classes de 1 a 4
def ler_dados(caminho, colunas):
    permitidas = (colunas,) if isinstance(colunas, int) else colunas
    dados = np.loadtxt(caminho, delimiter=",", encoding="utf-8-sig", ndmin=2)
    if dados.size == 0:
        raise ValueError("Arquivo vazio.")
    if dados.shape[1] not in permitidas:
        raise ValueError(f"Esperadas {permitidas} colunas em {caminho}.")
    if not np.isfinite(dados).all():
        raise ValueError("Os dados devem conter apenas números finitos.")
    if not np.equal(dados[:, 0], np.floor(dados[:, 0])).all():
        raise ValueError("Os IDs devem ser inteiros.")
    if dados.shape[1] == 8 and not np.isin(dados[:, 7], [1, 2, 3, 4]).all():
        raise ValueError("As classes devem ser 1, 2, 3 ou 4.")
    return dados


# seleciona qPA, pulso e respiracao (arquivo sem label de 7 colunas ou teste cego de 4)
def entradas_para_previsao(dados):
    return dados[:, 3:6] if dados.shape[1] == 7 else dados[:, 1:4]


# divisao estratificada 80/20 por classe, identica a usada no MLP
def separar_dados(dados):
    sorteio = np.random.default_rng(SEMENTE)
    treino, teste = [], []

    for classe in range(1, 5):
        grupo = dados[dados[:, 7] == classe].copy()

        if len(grupo) < 2:
            raise ValueError("São necessários pelo menos dois exemplos por classe.")

        sorteio.shuffle(grupo)
        quantidade = max(1, round(len(grupo) * 0.2))
        teste.append(grupo[:quantidade])
        treino.append(grupo[quantidade:])

    return np.concatenate(treino), np.concatenate(teste)


# entropia de shannon (em bits) de cada linha de contagens de classes
def entropia(contagens):
    p = contagens / contagens.sum(axis=-1, keepdims=True)
    logs = np.zeros_like(p)
    np.log2(p, out=logs, where=p > 0)
    return -(p * logs).sum(axis=-1)


# melhor limiar binario (x <= limiar) de um atributo continuo
# classificacao: ganho de informacao (C4.5); regressao: reducao da variancia (CART)
def melhor_divisao(x, y, classificacao):
    ordem = np.argsort(x, kind="stable")
    x, y = x[ordem], y[ordem]
    n = len(y)

    # cortes possiveis: entre valores distintos e respeitando o minimo de exemplos por folha
    cortes = np.nonzero(x[:-1] < x[1:])[0] + 1
    cortes = cortes[(cortes >= MINIMO_FOLHA) & (cortes <= n - MINIMO_FOLHA)]
    if len(cortes) == 0:
        return None

    # somas acumuladas permitem avaliar todos os cortes de uma vez
    if classificacao:
        acumulado = np.cumsum(np.eye(4)[y], axis=0)
        esquerda = acumulado[cortes - 1]
        direita = acumulado[-1] - esquerda
        impureza = entropia(acumulado[-1])
        filhos = (cortes * entropia(esquerda) + (n - cortes) * entropia(direita)) / n

    else:
        soma, soma2 = np.cumsum(y), np.cumsum(y ** 2)
        se, se2 = soma[cortes - 1], soma2[cortes - 1]
        sd, sd2 = soma[-1] - se, soma2[-1] - se2
        impureza = soma2[-1] / n - (soma[-1] / n) ** 2
        filhos = (se2 - se ** 2 / cortes + sd2 - sd ** 2 / (n - cortes)) / n

    ganhos = impureza - filhos
    melhor = np.argmax(ganhos)
    corte = cortes[melhor]
    return ganhos[melhor], (x[corte - 1] + x[corte]) / 2, corte / n


# percorre a arvore enviando cada grupo de exemplos para o lado correto
def prever_no(no, entradas):
    if not isinstance(no, tuple):
        return np.full(len(entradas), no, dtype=float)

    atributo, limiar, esquerda, direita = no
    lado = entradas[:, atributo] <= limiar
    saida = np.empty(len(entradas))
    saida[lado] = prever_no(esquerda, entradas[lado])
    saida[~lado] = prever_no(direita, entradas[~lado])
    return saida


class FlorestaAleatoria:

    def __init__(self, classificacao):
        self.classificacao = classificacao
        self.sorteio = np.random.default_rng(SEMENTE)
        self.arvores = []

    # constroi recursivamente uma arvore, folhas guardam a classe majoritaria ou a media
    def construir(self, x, y, profundidade):
        folha = np.bincount(y, minlength=4).argmax() if self.classificacao else y.mean()

        if (profundidade >= PROFUNDIDADE_MAXIMA or len(y) < 2 * MINIMO_FOLHA
                or np.all(y == y[0])):
            return folha

        # avalia somente um subconjunto aleatorio de atributos neste no
        melhor = None
        for atributo in self.sorteio.choice(3, ATRIBUTOS_SORTEADOS, replace=False):
            divisao = melhor_divisao(x[:, atributo], y, self.classificacao)
            if divisao is None:
                continue

            # C4.5 compara atributos pela razao de ganho (ganho / informacao da divisao)
            ganho, limiar, proporcao = divisao
            criterio = ganho
            if self.classificacao:
                criterio = ganho / entropia(np.array([proporcao, 1 - proporcao]))

            if ganho > 1e-12 and (melhor is None or criterio > melhor[0]):
                melhor = (criterio, ganho, atributo, limiar)

        if melhor is None:
            return folha

        # divide os exemplos pelo melhor limiar e constroi os dois filhos
        _, _, atributo, limiar = melhor
        lado = x[:, atributo] <= limiar
        return (atributo, limiar,
                self.construir(x[lado], y[lado], profundidade + 1),
                self.construir(x[~lado], y[~lado], profundidade + 1))

    def treinar(self, entradas, alvos):
        n = len(entradas)
        nome = "Classificação" if self.classificacao else "Regressão"
        votos_oob = np.zeros((n, 4)) if self.classificacao else np.zeros(n)
        contagem_oob = np.zeros(n)
        historico = []
        inicio = time.perf_counter()

        for indice in range(1, ARVORES + 1):
            # bootstrap: sorteia n exemplos com reposicao, os nao sorteados sao out-of-bag (OOB)
            amostra = self.sorteio.integers(0, n, n)
            fora = np.ones(n, dtype=bool)
            fora[amostra] = False
            arvore = self.construir(entradas[amostra], alvos[amostra], 0)
            self.arvores.append(arvore)

            # erro out-of-bag: cada exemplo é avaliado so pelas arvores que nao o viram
            previsto = prever_no(arvore, entradas[fora])
            if self.classificacao:
                votos_oob[np.nonzero(fora)[0], previsto.astype(int)] += 1
            else:
                votos_oob[fora] += previsto
            contagem_oob[fora] += 1
            vistos = contagem_oob > 0

            if self.classificacao:
                erro = np.mean(votos_oob[vistos].argmax(axis=1) != alvos[vistos])
            else:
                media = votos_oob[vistos] / contagem_oob[vistos]
                erro = np.sqrt(np.mean((media - alvos[vistos]) ** 2))
            historico.append((indice, erro))

            if indice == 1 or indice % 10 == 0 or indice == ARVORES:
                print(f"{nome}: árvore {indice}/{ARVORES} | erro OOB {erro:.4f} | "
                      f"{time.perf_counter() - inicio:.1f}s", flush=True)

        return historico

    # regressao: media das arvores 
    # classificacao: voto majoritario (empate -> classe mais grave)
    def prever(self, entradas):
        previsoes = np.array([prever_no(arvore, entradas) for arvore in self.arvores])

        if self.classificacao:
            votos = (previsoes[:, :, None] == np.arange(4)).sum(axis=0)
            return votos.argmax(axis=1) + 1

        return previsoes.mean(axis=0)


def avaliar(dados, regressao, classificacao):
    entradas = dados[:, 3:6]
    gravidades = regressao.prever(entradas)
    previstas = classificacao.prever(entradas)
    reais = dados[:, 7].astype(int)

    rmse = np.sqrt(np.mean((gravidades - dados[:, 6]) ** 2))
    acuracia = np.mean(previstas == reais)
    matriz = np.zeros((4, 4), dtype=int)

    for real, prevista in zip(reais, previstas):
        matriz[real - 1, prevista - 1] += 1

    print(f"RMSE da gravidade: {rmse:.4f}")
    print(f"Acurácia: {acuracia:.2%}")
    print("Matriz de confusão: linhas = reais; colunas = previstas (1 a 4)")
    print(matriz)
    print("Classe                         Precisão  Recall     F1   Suporte")
    f1_total = 0.0

    for k in range(4):
        vp = matriz[k, k]
        total_previstas = matriz[:, k].sum()
        total_reais = matriz[k, :].sum()
        precisao = vp / total_previstas if total_previstas else 0.0
        recall = vp / total_reais if total_reais else 0.0
        f1 = 2 * precisao * recall / (precisao + recall) if vp else 0.0
        f1_total += f1
        print(f"{k + 1} {CLASSES[k]:28} {precisao:.3f}    {recall:.3f}  "
              f"{f1:.3f}  {total_reais:6d}")

    print(f"F1 macro: {f1_total / 4:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("treino", help="Arquivo com as oito colunas e a classe")
    parser.add_argument("--prever", help="Arquivo sem label de 7 colunas ou teste cego de 4 colunas")
    parser.add_argument("--saida", default="previsoes_rf.csv")
    parser.add_argument("--pasta-resultados", default="resultados_rf",
                        help="Pasta dos CSVs utilizados pelo script de gráficos")
    args = parser.parse_args()
    inicio = time.perf_counter()

    # leitura e divisao treino/teste
    print(f"Lendo {args.treino}", flush=True)
    dados = ler_dados(args.treino, 8)
    novos = ler_dados(args.prever, (7, 4)) if args.prever else None
    pasta = Path(args.pasta_resultados)
    pasta.mkdir(parents=True, exist_ok=True)
    treino, teste = separar_dados(dados)
    entradas = treino[:, 3:6]
    print(f"Treino: {len(treino)} | Teste: {len(teste)}", flush=True)
    print(f"Parâmetros: {ARVORES} árvores | {ATRIBUTOS_SORTEADOS} atributos sorteados por nó | "
          f"profundidade máxima {PROFUNDIDADE_MAXIMA} | mínimo por folha {MINIMO_FOLHA}")

    # treinamento das duas florestas (sem normalizacao, arvores nao dependem de escala)
    regressao = FlorestaAleatoria(classificacao=False)
    classificacao = FlorestaAleatoria(classificacao=True)
    print("\nTreinando a floresta de regressão (gravidade)", flush=True)
    historico_regressao = regressao.treinar(entradas, treino[:, 6])
    print("\nTreinando a floresta de classificação (classe)", flush=True)
    historico_classificacao = classificacao.treinar(entradas, treino[:, 7].astype(int) - 1)

    # avaliacao nos dados de treino e nos dados de teste separados
    print("\nResultados de treinamento")
    avaliar(treino, regressao, classificacao)
    print("\nResultados de teste (exemplos não usados no treinamento)")
    avaliar(teste, regressao, classificacao)

    # csvs usados pelos graficos
    oob = np.column_stack((np.array(historico_regressao),
                           np.array(historico_classificacao)[:, 1]))
    np.savetxt(pasta / "erro_oob.csv", oob, delimiter=",",
               header="arvores,rmse_oob_regressao,erro_oob_classificacao", comments="",
               fmt=["%d", "%.12g", "%.12g"])
    gravidades = np.column_stack((teste[:, 0], teste[:, 6], regressao.prever(teste[:, 3:6])))
    np.savetxt(pasta / "gravidade_teste.csv", gravidades, delimiter=",",
               header="id,gravidade_real,gravidade_prevista", comments="",
               fmt=["%d", "%.12g", "%.12g"])
    #debug
    print(f"\nDados para gráficos salvos em {pasta}")

    # previsao para arquivo novo (id, gravidade, classe)
    if novos is not None:
        entradas = entradas_para_previsao(novos)
        resultado = np.column_stack((novos[:, 0], regressao.prever(entradas),
                                     classificacao.prever(entradas)))
        np.savetxt(args.saida, resultado, delimiter=",", fmt=["%d", "%.6f", "%d"])
        print(f"\nPrevisões salvas em {args.saida}")

    print(f"\nTempo total: {time.perf_counter() - inicio:.1f}s")


if __name__ == "__main__":
    main()
