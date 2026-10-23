"""Questão 2: Elastic Net (alpha = 0,5) e trajetória de coeficientes do LASSO."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import lasso_path

from .config import ENET_ALPHA_FIXO, LAMBDAS
from .data import Split
from .models import fit_enet_cv
from .utils import header, rmse, save_figure, save_table


def trajetoria_lasso(split: Split, lambdas=LAMBDAS):
    """Coeficientes do LASSO ao longo da grade (y centrado, X já padronizado)."""
    y = split.y_tr - split.y_tr.mean()
    lams, coefs, _ = lasso_path(split.X_tr_sc, y, alphas=np.sort(lambdas)[::-1],
                                max_iter=100_000)
    return lams, coefs.T            # (n_lambdas, n_features), lambdas decrescentes


def lambda_de_saida(lams: np.ndarray, coefs: np.ndarray, features: list) -> pd.Series:
    """Maior lambda em que cada variável ainda tem coeficiente não nulo.

    Quanto maior esse valor, mais tempo a variável "sobrevive" no modelo
    à medida que a penalização aumenta.
    """
    saida = {}
    for j, f in enumerate(features):
        nz = np.nonzero(coefs[:, j])[0]
        saida[f] = lams[nz].max() if len(nz) else 0.0
    return pd.Series(saida).sort_values(ascending=False)


def plot_trajetoria(lams, coefs, features, top3, lambda_min, lambda_1se) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.log10(lams)
    for j, f in enumerate(features):
        destaque = f in top3
        ax.plot(x, coefs[:, j], lw=2.4 if destaque else 1,
                alpha=1 if destaque else 0.45,
                color=None if destaque else "grey",
                label=f if destaque else None)
    for f in top3:
        j = features.index(f)
        i = np.nonzero(coefs[:, j])[0][0]
        ax.annotate(f, (x[i], 0), xytext=(x[i] + 0.05, coefs[min(i + 25, len(x) - 1), j]),
                    fontsize=10, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color="grey", lw=0.6))
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(np.log10(lambda_1se), color="grey", ls=":", lw=1.5, label="λ 1-EP")
    ax.set_xlim(-2, x.max())
    ax.set_xlabel("log10(λ)   →  mais regularização")
    ax.set_ylabel("Coeficiente (preditores padronizados)")
    ax.set_title("(b) LASSO: trajetória dos coeficientes (destaque: 3 últimas a sair)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save_figure(fig, "q2_trajetoria_lasso")


def run(split: Split, folds, q1: dict) -> dict:
    header("QUESTÃO 2 - Elastic Net e trajetória do LASSO")

    # (a) Elastic Net com alpha fixo
    enet = fit_enet_cv(split, folds, l1_ratio=ENET_ALPHA_FIXO)
    rmse_enet = rmse(split.y_te, enet.modelo.predict(split.X_te_sc))
    print(f"(a) Elastic Net (α = {ENET_ALPHA_FIXO}): λ = {enet.lambda_min:.5f} | "
          f"RMSE teste = {rmse_enet:.2f} | coef. não nulos = {(enet.modelo.coef_ != 0).sum()}")
    if enet.na_borda:
        print("    Atenção: λ na borda inferior da grade (CV prefere quase nenhuma penalização).")
    for nome, v in q1["rmse"].items():
        print(f"    {nome:<6} RMSE teste = {v:.2f}")

    # (b) Trajetória do LASSO
    lams, coefs = trajetoria_lasso(split)
    saida = lambda_de_saida(lams, coefs, split.features)
    top3 = saida.index[:3].tolist()
    save_table(saida.rename("lambda_de_saida").to_frame(), "q2_lambda_de_saida_lasso")
    print("\n(b) Ordem em que as variáveis saem do LASSO (maior λ em que ainda são não nulas):")
    print(saida.head(8).round(2).to_string())
    print(f"    Três que permanecem por mais tempo: {', '.join(top3)}")
    plot_trajetoria(lams, coefs, split.features, top3,
                    q1["lasso"].lambda_min, q1["lasso"].lambda_1se)

    return {"enet_05": enet, "rmse_enet_05": rmse_enet, "top3_lasso": top3, "saida": saida}
