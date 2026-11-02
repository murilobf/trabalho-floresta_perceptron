import argparse
import sys
from pathlib import Path

# permite importar o gerar_graficos.py da raiz do projeto
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gerar_graficos import ler_csv, grafico_gravidade, plt


# curva do erro OOB pelo numero de arvores (equivalente a curva de perda do MLP)
def grafico_oob(dados, destino):
    arvores = dados[:, 0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    ax1.plot(arvores, dados[:, 1], color="#236699", linewidth=1.3)
    ax1.set(title="Regressão: RMSE out-of-bag", xlabel="Número de árvores", ylabel="RMSE da gravidade")
    ax2.plot(arvores, dados[:, 2] * 100, color="#a33a32", linewidth=1.3)
    ax2.set(title="Classificação: erro out-of-bag", xlabel="Número de árvores", ylabel="Erro (%)")

    for ax in (ax1, ax2):
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)

    fig.savefig(destino, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dados", type=Path, default=Path("resultados_rf"))
    parser.add_argument("--saida", type=Path, default=Path("graficos_rf"))
    args = parser.parse_args()

    try:
        print(f"Lendo CSVs de {args.dados}", flush=True)
        oob = ler_csv(args.dados / "erro_oob.csv",
                      ["arvores", "rmse_oob_regressao", "erro_oob_classificacao"])
        gravidades = ler_csv(args.dados / "gravidade_teste.csv",
                             ["id", "gravidade_real", "gravidade_prevista"])
        args.saida.mkdir(parents=True, exist_ok=True)

        # o grafico de gravidade e reaproveitado do gerar_graficos.py
        print("Gerando gráfico de gravidade real x prevista", flush=True)
        grafico_gravidade(gravidades, args.saida / "gravidade_real_prevista.png")
        print("Gerando curva de erro out-of-bag", flush=True)
        grafico_oob(oob, args.saida / "erro_oob.png")

    except ValueError as erro:
        parser.error(str(erro))
    print(f"Gráficos salvos em {args.saida.resolve()}")


if __name__ == "__main__":
    main()
