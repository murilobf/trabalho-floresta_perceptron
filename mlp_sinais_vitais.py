import argparse
import numpy as np


SEMENTE = 10
NEURONIOS = 8
EPOCAS = 3200
TAXA = 0.01
CLASSES = ("crítico", "instável", "potencialmente estável", "estável")


def ler_dados(caminho, colunas):
    """Lê números separados por vírgulas, sem cabeçalho."""
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


def entradas_para_previsao(dados):
    # No arquivo de 7 colunas, ignora ID, pressões e gravidade conhecida.
    return dados[:, 3:6] if dados.shape[1] == 7 else dados[:, 1:4]


def separar_dados(dados):
    """Reserva aproximadamente 20% de cada classe para teste."""
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


def calcular_escala(treino):
    """Calcula a escala somente no treino, evitando vazamento de dados."""
    entradas = treino[:, 3:6]
    medias = entradas.mean(axis=0)
    desvios = entradas.std(axis=0)
    desvios[desvios == 0] = 1.0
    return medias, desvios


def normalizar(entradas, medias, desvios):
    return (entradas - medias) / desvios


class MLP:

    def __init__(self, classificacao):
        self.classificacao = classificacao
        saidas = 4 if classificacao else 1
        sorteio = np.random.default_rng(SEMENTE)
        self.w1 = sorteio.uniform(-0.5, 0.5, size=(3, NEURONIOS))
        self.b1 = np.zeros(NEURONIOS)
        self.w2 = sorteio.uniform(-0.5, 0.5, size=(NEURONIOS, saidas))
        self.b2 = np.zeros(saidas)

    def propagar(self, x):
        oculta = np.tanh(x @ self.w1 + self.b1)
        saida = oculta @ self.w2 + self.b2

        if self.classificacao:

            exponenciais = np.exp(saida - saida.max(axis=-1, keepdims=True))
            saida = exponenciais / exponenciais.sum(axis=-1, keepdims=True)
        return oculta, saida

    def treinar(self, entradas, alvos):
        sorteio = np.random.default_rng(SEMENTE)
        for _ in range(EPOCAS):
            for indice in sorteio.permutation(len(entradas)):
                x, alvo = entradas[indice], alvos[indice]
                oculta, saida = self.propagar(x)

                if self.classificacao:
                    erro_saida = saida.copy()
                    erro_saida[int(alvo)] -= 1.0
                else:
                    erro_saida = saida - alvo

                erro_oculta = (erro_saida @ self.w2.T) * (1 - oculta ** 2)

                self.w2 -= TAXA * np.outer(oculta, erro_saida)
                self.b2 -= TAXA * erro_saida
                self.w1 -= TAXA * np.outer(x, erro_oculta)
                self.b1 -= TAXA * erro_oculta

    def prever(self, entradas):
        _, saida = self.propagar(entradas)
        if self.classificacao:
            return np.argmax(saida, axis=-1) + 1
        return saida[..., 0] * 100.0  # volta pra escala original da gravidade


def avaliar(dados, medias, desvios, regressao, classificacao):
    entradas = normalizar(dados[:, 3:6], medias, desvios)
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
    parser.add_argument("--saida", default="previsoes_mlp.csv")
    args = parser.parse_args()

    dados = ler_dados(args.treino, 8)
    novos = ler_dados(args.prever, (7, 4)) if args.prever else None
    treino, teste = separar_dados(dados)
    medias, desvios = calcular_escala(treino)
    entradas = normalizar(treino[:, 3:6], medias, desvios)

    regressao = MLP(classificacao=False)
    classificacao = MLP(classificacao=True)
    print(f"Treino: {len(treino)} | Teste: {len(teste)}", flush=True)
    print("Treinando as redes", flush=True)
    regressao.treinar(entradas, treino[:, 6] / 100.0)
    classificacao.treinar(entradas, treino[:, 7].astype(int) - 1)

    print("\nResultados de treinamento")
    avaliar(treino, medias, desvios, regressao, classificacao)
    print("\nResultados de teste (exemplos não usados no treinamento)")
    avaliar(teste, medias, desvios, regressao, classificacao)

    if novos is not None:
        entradas = normalizar(entradas_para_previsao(novos), medias, desvios)
        resultado = np.column_stack((novos[:, 0], regressao.prever(entradas),
                                     classificacao.prever(entradas)))
        np.savetxt(args.saida, resultado, delimiter=",", fmt=["%d", "%.6f", "%d"])
        print(f"\nPrevisões salvas em {args.saida}")


if __name__ == "__main__":
    main()  
