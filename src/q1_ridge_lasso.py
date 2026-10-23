"""Questão 1: OLS, Ridge e LASSO com lambda escolhido por CV de 10 folds."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .config import PALETTE
from .data import Split
from .models import CVResult, fit_lasso_cv, fit_ols, fit_ridge_cv, lasso_at
from .utils import header, rmse, save_figure, save_table


def plot_cv_curvas(ridge: CVResult, lasso: CVResult) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, res in zip(axes, [ridge, lasso]):
        x = np.log10(res.lambdas)
        rmse_cv = np.sqrt(res.cv_mse)
        # Banda de 1 erro padrão, convertida para a escala do RMSE
        lo = np.sqrt(np.clip(res.cv_mse - res.cv_se, 0, None))
        hi = np.sqrt(res.cv_mse + res.cv_se)
        cor = PALETTE[res.nome]
        ax.plot(x, rmse_cv, color=cor, lw=2)
        ax.fill_between(x, lo, hi, color=cor, alpha=0.2, label="± 1 erro padrão")
        ax.axvline(np.log10(res.lambda_min), color="k", ls="--", lw=1,
                   label=f"λ mín = {res.lambda_min:.4g}")
        ax.axvline(np.log10(res.lambda_1se), color="grey", ls=":", lw=1.5,
                   label=f"λ 1-EP = {res.lambda_1se:.4g}")
        ax.set_title(f"{res.nome}: RMSE de validação cruzada (10 folds)")
        ax.set_xlabel("log10(λ)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("RMSE (CV)")
    fig.tight_layout()
    save_figure(fig, "q1_cv_curvas")


def plot_rmse(tabela: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(tabela.index, tabela["RMSE teste"],
                  color=[PALETTE[n] for n in tabela.index], width=0.55)
    ax.bar_label(bars, fmt="{:.1f}", padding=3)
    ax.set_ylabel("RMSE no teste")
    ax.set_ylim(0, tabela["RMSE teste"].max() * 1.18)
    ax.set_title("(e) RMSE no conjunto de teste")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save_figure(fig, "q1_rmse_teste")


def run(split: Split, folds) -> dict:
    header("QUESTÃO 1 - OLS, Ridge e LASSO")
    print(f"(a) Treino: {split.n_train} obs | Teste: {len(split.y_te)} obs | "
          f"{len(split.features)} preditores (padronizados com média/desvio do treino)")

    # (b) OLS
    ols = fit_ols(split)
    rmse_ols = rmse(split.y_te, ols.predict(split.X_te))
    print(f"\n(b) OLS: RMSE teste = {rmse_ols:.2f}")

    # (c) Ridge
    ridge = fit_ridge_cv(split, folds)
    rmse_ridge = rmse(split.y_te, ridge.modelo.predict(split.X_te_sc))
    print(f"(c) Ridge: λ = {ridge.lambda_min:.5f} (alpha do sklearn = "
          f"{ridge.extra['sklearn_alpha']:.3f}) | RMSE teste = {rmse_ridge:.2f}")

    # (d) LASSO
    lasso = fit_lasso_cv(split, folds)
    coef = lasso.modelo.coef_
    rmse_lasso = rmse(split.y_te, lasso.modelo.predict(split.X_te_sc))
    print(f"(d) LASSO: λ = {lasso.lambda_min:.5f} | RMSE teste = {rmse_lasso:.2f} | "
          f"coeficientes não nulos = {(coef != 0).sum()} de {len(coef)}")
    if lasso.na_borda:
        print("    Atenção: λ escolhido está na borda inferior da grade -> o CV "
              "prefere (quase) nenhuma penalização.")

    # Complemento: regra de 1 erro padrão
    lasso_1se = lasso_at(lasso.lambda_1se, split)
    rmse_1se = rmse(split.y_te, lasso_1se.predict(split.X_te_sc))
    sel = [f for f, c in zip(split.features, lasso_1se.coef_) if c != 0]
    print(f"    LASSO (regra 1-EP): λ = {lasso.lambda_1se:.2f} | RMSE teste = {rmse_1se:.2f}"
          f" | {len(sel)} variáveis: {', '.join(sel)}")

    tabela = pd.DataFrame({
        "λ (escala Elastic Net)": [np.nan, ridge.lambda_min, lasso.lambda_min, lasso.lambda_1se],
        "RMSE CV": [np.nan, ridge.cv_rmse_min, lasso.cv_rmse_min,
                    np.sqrt(lasso.cv_mse[np.isclose(lasso.lambdas, lasso.lambda_1se)][0])],
        "RMSE teste": [rmse_ols, rmse_ridge, rmse_lasso, rmse_1se],
        "coef. não nulos": [len(coef), len(coef), (coef != 0).sum(), len(sel)],
    }, index=["OLS", "Ridge", "LASSO", "LASSO (1-EP)"]).round(4)
    save_table(tabela, "q1_comparacao")
    print("\n(e) Comparação:")
    print(tabela.to_string())

    plot_cv_curvas(ridge, lasso)
    plot_rmse(tabela)

    coefs = pd.DataFrame({
        "OLS (padronizado)": LinearRegression().fit(split.X_tr_sc, split.y_tr).coef_,
        "Ridge": ridge.modelo.coef_,
        "LASSO": coef,
        "LASSO (1-EP)": lasso_1se.coef_,
    }, index=split.features).round(2)
    save_table(coefs, "q1_coeficientes")

    return {"ridge": ridge, "lasso": lasso, "lasso_1se": lasso_1se, "tabela": tabela,
            "rmse": {"OLS": rmse_ols, "Ridge": rmse_ridge, "LASSO": rmse_lasso}}
