
import argparse
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")  
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def ler_csv(caminho, colunas):

    if not caminho.is_file():
        raise ValueError(f"Arquivo não encontrado: {caminho}. Execute o MLP primeiro.")
    
    with caminho.open(encoding="utf-8-sig") as arquivo:
        cabecalho = arquivo.readline().strip().split(",")

    if cabecalho != colunas:
        raise ValueError(f"Cabeçalho inválido em {caminho}: esperado {colunas}.")
    
    dados = np.loadtxt(caminho, delimiter=",", skiprows=1, ndmin=2)

    if dados.size == 0 or dados.shape[1] != len(colunas):
        raise ValueError(f"Arquivo vazio ou com formato inválido: {caminho}.")
    
    if not np.isfinite(dados).all():
        raise ValueError(f"Há valores não finitos em {caminho}.")
    
    return dados


def grafico_perda(dados, destino):
    epocas, perdas = dados[:, 0], dados[:, 1]

    if (epocas < 1).any() or (epocas != np.floor(epocas)).any():
        raise ValueError("As épocas devem ser números inteiros positivos.")
    
    if (np.diff(epocas) <= 0).any() or (perdas < 0).any():
        raise ValueError("Épocas devem estar em ordem crescente e perdas não negativas.")
    
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(epocas, perdas, color="#236699", linewidth=1.1, label="Treinamento")
    ax.set(title="Perda de classificação por época", xlabel="Época", ylabel="Entropia cruzada média")
    ax.set_ylim(0, max(perdas.max() * 1.08, 0.01))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.savefig(destino, dpi=300)
    plt.close(fig)


def grafico_gravidade(dados, destino):
    reais, previstas = dados[:, 1], dados[:, 2]
    rmse = np.sqrt(np.mean((previstas - reais) ** 2))
    menor = min(reais.min(), previstas.min())
    maior = max(reais.max(), previstas.max())
    margem = max((maior - menor) * 0.05, 1.0)
    limites = (menor - margem, maior + margem)

    fig, ax = plt.subplots(figsize=(6.5, 6.5), constrained_layout=True)
    ax.scatter(reais, previstas, s=24, alpha=0.65, color="#236699", edgecolors="none", label="Exemplos de teste")
    ax.plot(limites, limites, "--", color="#a33a32", linewidth=1.3, label="Previsão perfeita (y = x)")
    ax.set(title="Gravidade real × gravidade prevista", xlabel="Gravidade real",  ylabel="Gravidade prevista", xlim=limites, ylim=limites)
    ax.set_aspect("equal", adjustable="box")
    ax.text(0.04, 0.96, f"Teste: {len(reais)} exemplos\nRMSE: {rmse:.4f}",
            transform=ax.transAxes, va="top",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.savefig(destino, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dados", type=Path, default=Path("resultados"))
    parser.add_argument("--saida", type=Path, default=Path("graficos"))
    args = parser.parse_args()

    try:
        perdas = ler_csv(args.dados / "perda_classificacao.csv",
                         ["epoca", "perda_classificacao"])
        gravidades = ler_csv(args.dados / "gravidade_teste.csv",
                            ["id", "gravidade_real", "gravidade_prevista"])
        args.saida.mkdir(parents=True, exist_ok=True)
        grafico_perda(perdas, args.saida / "perda_classificacao.png")
        grafico_gravidade(gravidades, args.saida / "gravidade_real_prevista.png")
        
    except ValueError as erro:
        parser.error(str(erro))
    print(f"Gráficos salvos em {args.saida.resolve()}")


if __name__ == "__main__":
    main()
