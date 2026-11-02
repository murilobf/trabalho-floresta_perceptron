# Código para calcular o erro OOB

import argparse
import contextlib
import io
import time
import numpy as np
import random_forest_sinais_vitais as rf


# variacoes testadas, uma de cada vez, a partir da configuracao padrao do modelo
VARIACOES = {
    "ATRIBUTOS_SORTEADOS": (1, 2, 3),
    "MINIMO_FOLHA": (1, 3, 5, 10),
    "PROFUNDIDADE_MAXIMA": (4, 8, 12, 20),
}


# treina as duas florestas e devolve o erro OOB final (usa apenas o treino)
def erro_oob(entradas, gravidades, classes):
    # silencia os logs por arvore para manter a tabela legivel
    with contextlib.redirect_stdout(io.StringIO()):
        rmse = rf.FlorestaAleatoria(False).treinar(entradas, gravidades)[-1][1]
        erro = rf.FlorestaAleatoria(True).treinar(entradas, classes)[-1][1]
    return rmse, erro


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("treino", help="Arquivo com as oito colunas e a classe")
    args = parser.parse_args()

    # o teste fica guardado, a escolha dos parametros olha somente o erro OOB do treino
    treino, _ = rf.separar_dados(rf.ler_dados(args.treino, 8))
    entradas, gravidades = treino[:, 3:6], treino[:, 6]
    classes = treino[:, 7].astype(int) - 1
    padrao = {nome: getattr(rf, nome) for nome in VARIACOES}
    total = sum(len(valores) for valores in VARIACOES.values())
    print(f"Configuração padrão: {padrao} | {rf.ARVORES} árvores | {total} execuções", flush=True)
    print("Parâmetro               Valor  RMSE OOB  Erro OOB  Tempo", flush=True)
    execucao = 0

    for nome, valores in VARIACOES.items():
        for valor in valores:
            execucao += 1
            setattr(rf, nome, valor)
            inicio = time.perf_counter()
            rmse, erro = erro_oob(entradas, gravidades, classes)
            print(f"{nome:22} {valor:5d}  {rmse:8.4f}  {erro:8.2%}  "
                  f"{time.perf_counter() - inicio:4.1f}s  ({execucao}/{total})", flush=True)

        setattr(rf, nome, padrao[nome])


if __name__ == "__main__":
    main()
