"""Questão 3: Elastic Net com grade de alpha, esparsidade e preditores correlacionados.

O item (a), a demonstração algébrica, está em docs/elastic_net_penalidade.md.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, enet_path, lasso_path

from .config import L1_RATIOS, LAMBDAS, SEED
from .data import Split
from .models import enet_at, fit_enet_cv, fit_ridge_cv, lasso_at, ridge_from_lambda
from .utils import header, rmse, save_figure, save_table


def grade_alpha(split: Split, folds) -> tuple[pd.DataFrame, object]:
    """CV conjunta em alpha e lambda. alpha = 0 (Ridge) é ajustado à parte,
    porque o ElasticNetCV não é confiável com l1_ratio = 0."""
    linhas = []
    ridge = fit_ridge_cv(split, folds)
    linhas.append({"alpha": 0.0, "lambda_otimo": ridge.lambda_min, "RMSE_CV": ridge.cv_rmse_min})
    melhor_enet = None
    for a in [r for r in L1_RATIOS if r > 0]:
        res = fit_enet_cv(split, folds, l1_ratio=a)
        linhas.append({"alpha": a, "lambda_otimo": res.lambda_min, "RMSE_CV": res.cv_rmse_min})
        if melhor_enet is None or res.cv_rmse_min < melhor_enet.cv_rmse_min:
            melhor_enet = res
    tabela = pd.DataFrame(linhas).set_index("alpha")
    melhor = ridge if ridge.cv_rmse_min < melhor_enet.cv_rmse_min else melhor_enet
    return tabela, melhor


def zeros_mesmo_lambda(lam: float, alpha_enet: float, split: Split) -> dict:
    return {
        "Ridge (α=0)": int((ridge_from_lambda(lam, split).coef_ == 0).sum()),
        f"Elastic Net (α={alpha_enet:g})": int((enet_at(lam, alpha_enet, split).coef_ == 0).sum()),
        "LASSO (α=1)": int((lasso_at(lam, split).coef_ == 0).sum()),
    }


def curva_zeros(split: Split, alphas=(0.0, 0.25, 0.5, 1.0)) -> pd.DataFrame:
    y = split.y_tr - split.y_tr.mean()
    lams = np.sort(LAMBDAS)[::-1]
    out = {}
    for a in alphas:
        if a == 0:
            out[a] = [int((ridge_from_lambda(l, split).coef_ == 0).sum()) for l in lams]
        else:
            _, coefs, _ = enet_path(split.X_tr_sc, y, l1_ratio=a, alphas=lams, max_iter=100_000)
            out[a] = (coefs == 0).sum(axis=0)
    return pd.DataFrame(out, index=lams)


def plot_curva_zeros(curva: pd.DataFrame, marcas: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    cores = {0.0: "#3498DB", 0.25: "#B07CC6", 0.5: "#9B59B6", 1.0: "#2ECC71"}
    for a in curva.columns:
        nome = {0.0: "Ridge", 1.0: "LASSO"}.get(a, "Elastic Net")
        ax.step(np.log10(curva.index), curva[a], where="post", lw=2,
                color=cores.get(a), label=f"{nome} (α = {a:g})")
    for rotulo, lam in marcas.items():
        ax.axvline(np.log10(lam), color="grey", ls=":", lw=1)
        ax.text(np.log10(lam), 17.4, f"λ {rotulo} do LASSO", ha="center", fontsize=8, color="grey")
    ax.set_xlabel("log10(λ)")
    ax.set_ylabel("Coeficientes zerados (de 17)")
    ax.set_ylim(-0.5, 18)
    ax.set_yticks(range(0, 18, 2))
    ax.set_title("(c) Número de coeficientes zerados para o mesmo λ")
    ax.legend(loc="center right")
    fig.tight_layout()
    save_figure(fig, "q3_zeros_vs_lambda")


def frequencia_selecao(split: Split, lam_lasso: float, lam_enet: float, alpha: float,
                       n_boot: int = 200) -> pd.DataFrame:
    """Proporção de amostras bootstrap em que cada variável fica com coef. não nulo.

    Cada método usa o próprio λ (regra de 1 erro padrão da sua CV).
    """
    rng = np.random.default_rng(SEED)
    n = split.n_train
    y_c = split.y_tr - split.y_tr.mean()
    nome_enet = f"Elastic Net (α={alpha:g})"
    freq = {"LASSO": np.zeros(len(split.features)), nome_enet: np.zeros(len(split.features))}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        X, y = split.X_tr_sc[idx], y_c[idx]
        _, c1, _ = lasso_path(X, y, alphas=[lam_lasso], max_iter=100_000)
        _, c2, _ = enet_path(X, y, l1_ratio=alpha, alphas=[lam_enet], max_iter=100_000)
        freq["LASSO"] += c1[:, 0] != 0
        freq[nome_enet] += c2[:, 0] != 0
    return pd.DataFrame(freq, index=split.features) / n_boot


def plot_correlacao_grupo(df_corr: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(df_corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(df_corr)), df_corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(df_corr)), df_corr.columns)
    for i in range(len(df_corr)):
        for j in range(len(df_corr)):
            ax.text(j, i, f"{df_corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax)
    ax.set_title("Correlação: alvo e preditores de 'escala'")
    fig.tight_layout()
    save_figure(fig, "q3_correlacao_escala")


def run(split: Split, folds, df: pd.DataFrame, q1: dict) -> dict:
    header("QUESTÃO 3 - Elastic Net: penalização e seleção de variáveis")
    print("(a) Demonstração algébrica: ver docs/elastic_net_penalidade.md")

    # (b) Grade de alpha
    tabela_alpha, melhor = grade_alpha(split, folds)
    save_table(tabela_alpha.round(5), "q3_grade_alpha")
    print("\n(b) CV conjunta (alpha x lambda):")
    print(tabela_alpha.round(5).to_string())
    print(f"    Melhor: α = {melhor.alpha:g}, λ = {melhor.lambda_min:.5f}, "
          f"RMSE CV = {melhor.cv_rmse_min:.2f}")
    if melhor.na_borda:
        print("    Atenção: λ na borda inferior da grade. A melhor penalização é ~0.")
    final = melhor.modelo
    rmse_final = rmse(split.y_te, final.predict(split.X_te_sc))
    print(f"    RMSE teste do modelo final: {rmse_final:.2f}")

    # (c) Zeros para o mesmo lambda
    alpha_cmp = melhor.alpha if 0 < melhor.alpha < 1 else 0.5
    lam_1se = q1["lasso"].lambda_1se
    zeros = pd.DataFrame({
        f"λ ótimo da CV ({melhor.lambda_min:.4g})": zeros_mesmo_lambda(melhor.lambda_min, alpha_cmp, split),
        f"λ 1-EP do LASSO ({lam_1se:.4g})": zeros_mesmo_lambda(lam_1se, alpha_cmp, split),
    })
    save_table(zeros, "q3_zeros_mesmo_lambda")
    print("\n(c) Coeficientes zerados (de 17) para o mesmo λ:")
    print(zeros.to_string())
    curva = curva_zeros(split)
    save_table(curva, "q3_curva_zeros")
    plot_curva_zeros(curva, {"1-EP": lam_1se})

    # (d) Top 3 do Elastic Net final
    coef_final = pd.Series(final.coef_, index=split.features)
    top3 = coef_final.reindex(coef_final.abs().sort_values(ascending=False).index).head(3)
    save_table(coef_final.sort_values(key=abs, ascending=False).round(2).rename("coef").to_frame(),
               "q3_coef_modelo_final")
    print("\n(d) Três maiores coeficientes (em módulo, preditores padronizados):")
    print(top3.round(1).to_string())

    # (e) Preditores correlacionados
    corr = df[["Apps", "Accept", "Enroll", "F_Undergrad", "P_Undergrad",
               "Top10perc", "Top25perc"]].corr()
    save_table(corr.round(3), "q3_correlacao_escala")
    plot_correlacao_grupo(corr)
    preds = df[split.features].corr().where(np.triu(np.ones((17, 17), bool), k=1)).stack()
    altos = preds[preds.abs() > 0.8].sort_values(ascending=False)
    print("\n(e) Pares de preditores com |r| > 0,8:")
    print(altos.round(3).to_string())

    # Cada método com o seu próprio λ pela regra de 1 erro padrão
    enet05 = fit_enet_cv(split, folds, l1_ratio=0.5)
    lam_enet_1se = enet05.lambda_1se
    lasso_1se, enet_1se = lasso_at(lam_1se, split), enet_at(lam_enet_1se, 0.5, split)
    print(f"\n    Modelos esparsos pela regra 1-EP: LASSO λ = {lam_1se:.2f} | "
          f"Elastic Net (α=0,5) λ = {lam_enet_1se:.4f}")
    print(f"    RMSE teste: LASSO 1-EP = {rmse(split.y_te, lasso_1se.predict(split.X_te_sc)):.2f} | "
          f"Elastic Net 1-EP = {rmse(split.y_te, enet_1se.predict(split.X_te_sc)):.2f}")

    grupo = ["Accept", "Enroll", "F_Undergrad", "P_Undergrad", "Top10perc", "Top25perc"]
    coefs_grupo = pd.DataFrame({
        "LASSO (1-EP)": lasso_1se.coef_,
        "Elastic Net α=0,5 (1-EP)": enet_1se.coef_,
        "OLS": LinearRegression().fit(split.X_tr_sc, split.y_tr).coef_,
    }, index=split.features).round(1)
    save_table(coefs_grupo, "q3_coef_lasso_vs_enet")
    print("\n    Coeficientes no grupo correlacionado:")
    print(coefs_grupo.loc[grupo].to_string())

    freq = frequencia_selecao(split, lam_1se, lam_enet_1se, 0.5)
    save_table(freq.round(3), "q3_frequencia_selecao_bootstrap")
    print("\n    Frequência de seleção em 200 reamostragens bootstrap:")
    print(freq.loc[grupo].round(2).to_string())

    return {"tabela_alpha": tabela_alpha, "melhor": melhor, "rmse_final": rmse_final,
            "zeros": zeros, "top3": top3, "freq": freq}
