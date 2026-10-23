"""Análise complementar: a comparação entre modelos se mantém em outras divisões?

Com 234 observações de teste, diferenças de 1-2% no RMSE podem ser só sorte
da divisão. Aqui o processo inteiro (divisão, padronização e CV) é repetido
com várias sementes.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ENET_ALPHA_FIXO, N_REPETICOES, PALETTE
from .data import make_split
from .models import enet_at, fit_enet_cv, fit_lasso_cv, fit_ols, fit_ridge_cv, lasso_at, make_folds
from .utils import header, rmse, save_figure, save_table


def run(df: pd.DataFrame, n_rep: int = N_REPETICOES) -> pd.DataFrame:
    header(f"ANÁLISE COMPLEMENTAR - {n_rep} divisões treino/teste")
    linhas = []
    for seed in range(n_rep):
        s, folds = make_split(df, seed=seed), make_folds(seed)
        lasso = fit_lasso_cv(s, folds)
        enet = fit_enet_cv(s, folds, ENET_ALPHA_FIXO)
        linhas.append({
            "seed": seed,
            "OLS": rmse(s.y_te, fit_ols(s).predict(s.X_te)),
            "Ridge": rmse(s.y_te, fit_ridge_cv(s, folds).modelo.predict(s.X_te_sc)),
            "LASSO": rmse(s.y_te, lasso.modelo.predict(s.X_te_sc)),
            "LASSO (1-EP)": rmse(s.y_te, lasso_at(lasso.lambda_1se, s).predict(s.X_te_sc)),
            "Elastic Net": rmse(s.y_te, enet.modelo.predict(s.X_te_sc)),
            "Elastic Net (1-EP)": rmse(s.y_te, enet_at(enet.lambda_1se, ENET_ALPHA_FIXO, s)
                                       .predict(s.X_te_sc)),
        })
    res = pd.DataFrame(linhas).set_index("seed")
    save_table(res.round(3), "robustez_rmse_por_divisao")

    rel = res.div(res["OLS"], axis=0) - 1     # diferença relativa ao OLS
    resumo = pd.DataFrame({
        "RMSE médio": res.mean(),
        "desvio padrão": res.std(),
        "diferença média vs OLS (%)": 100 * rel.mean(),
        "diferença mediana vs OLS (%)": 100 * rel.median(),
        "% divisões melhor que OLS": 100 * (res.lt(res["OLS"], axis=0)).mean(),
        "% divisões como melhor modelo": 100 * res.idxmin(axis=1).value_counts(normalize=True)
                                              .reindex(res.columns).fillna(0),
    }).round(2)
    resumo.loc["OLS", "% divisões melhor que OLS"] = float("nan")
    save_table(resumo, "robustez_resumo")
    print(resumo.to_string())

    fig, ax = plt.subplots(figsize=(9, 4.8))
    dados = [100 * rel[c] for c in res.columns if c != "OLS"]
    nomes = [c for c in res.columns if c != "OLS"]
    bp = ax.boxplot(dados, tick_labels=nomes, patch_artist=True, widths=0.5)
    for patch, n in zip(bp["boxes"], nomes):
        patch.set_facecolor(PALETTE[n])
        patch.set_alpha(0.6)
    ax.axhline(0, color=PALETTE["OLS"], ls="--", lw=1.2, label="OLS")
    ax.set_ylabel("RMSE teste relativo ao OLS (%)")
    ax.set_title(f"Diferença de RMSE em relação ao OLS em {n_rep} divisões treino/teste")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "robustez_rmse_relativo")
    return resumo
