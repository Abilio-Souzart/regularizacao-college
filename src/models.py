"""Ajuste dos modelos com validação cruzada.

Convenção de escala do lambda
-----------------------------
LASSO e Elastic Net do scikit-learn (e o glmnet do R) minimizam

    (1 / 2n) * RSS + lambda * [ (1 - alpha)/2 * ||b||^2 + alpha * ||b||_1 ]

onde `alpha` é chamado de `l1_ratio` no scikit-learn e `lambda` de `alpha`.

O `Ridge` do scikit-learn usa outra escala:  RSS + a * ||b||^2.
Com alpha = 0, multiplicando a função do Elastic Net por 2n, obtém-se
RSS + n * lambda * ||b||^2, logo  a = n * lambda.

Todas as funções abaixo reportam lambda na escala do Elastic Net, para que
Ridge, LASSO e Elastic Net possam ser comparados com "o mesmo lambda".
"""
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso, LassoCV, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold

from .config import LAMBDAS, N_FOLDS, SEED
from .data import Split


def make_folds(seed: int = SEED) -> KFold:
    """Os mesmos folds são usados por todos os modelos, para uma comparação justa."""
    return KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)


def ridge_sklearn_alpha(lam: float, n: int) -> float:
    """Converte lambda (escala Elastic Net) para o `alpha` do Ridge do scikit-learn."""
    return n * lam


def ridge_from_lambda(lam: float, split: Split) -> Ridge:
    return Ridge(alpha=ridge_sklearn_alpha(lam, split.n_train)).fit(split.X_tr_sc, split.y_tr)


@dataclass
class CVResult:
    nome: str
    modelo: object
    lambda_min: float
    lambdas: np.ndarray            # grade, em ordem decrescente
    cv_mse: np.ndarray             # MSE médio de validação por lambda
    cv_se: np.ndarray              # erro padrão do MSE entre folds
    alpha: float | None = None     # mistura L1/L2 (Elastic Net)
    extra: dict = field(default_factory=dict)

    @property
    def cv_rmse_min(self) -> float:
        return float(np.sqrt(self.cv_mse.min()))

    @property
    def lambda_1se(self) -> float:
        """Maior lambda cujo erro de CV fica a até 1 erro padrão do mínimo."""
        i = np.argmin(self.cv_mse)
        limite = self.cv_mse[i] + self.cv_se[i]
        return float(self.lambdas[self.cv_mse <= limite].max())

    @property
    def na_borda(self) -> bool:
        """True se o lambda escolhido é o menor (ou maior) valor da grade."""
        return bool(np.isclose(self.lambda_min, self.lambdas.min())
                    or np.isclose(self.lambda_min, self.lambdas.max()))


def _resumo_path(mse_path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """mse_path: (n_lambdas, n_folds) -> média e erro padrão."""
    return mse_path.mean(axis=1), mse_path.std(axis=1, ddof=1) / np.sqrt(mse_path.shape[1])


def fit_ols(split: Split) -> LinearRegression:
    # OLS é invariante à escala dos preditores; usa os dados originais.
    return LinearRegression().fit(split.X_tr, split.y_tr)


def fit_ridge_cv(split: Split, folds: KFold, lambdas=LAMBDAS) -> CVResult:
    lambdas = np.sort(lambdas)[::-1]
    grid = GridSearchCV(
        Ridge(),
        {"alpha": [ridge_sklearn_alpha(l, split.n_train) for l in lambdas]},
        cv=folds, scoring="neg_mean_squared_error",
    ).fit(split.X_tr_sc, split.y_tr)
    mse_path = -np.column_stack([grid.cv_results_[f"split{k}_test_score"]
                                 for k in range(folds.get_n_splits())])
    mse, se = _resumo_path(mse_path)
    lam = float(lambdas[np.argmin(mse)])
    return CVResult("Ridge", grid.best_estimator_, lam, lambdas, mse, se, alpha=0.0,
                    extra={"sklearn_alpha": grid.best_params_["alpha"]})


def fit_lasso_cv(split: Split, folds: KFold, lambdas=LAMBDAS) -> CVResult:
    cv = LassoCV(alphas=lambdas, cv=folds, max_iter=100_000).fit(split.X_tr_sc, split.y_tr)
    mse, se = _resumo_path(cv.mse_path_)
    return CVResult("LASSO", cv, float(cv.alpha_), cv.alphas_, mse, se, alpha=1.0)


def fit_enet_cv(split: Split, folds: KFold, l1_ratio, lambdas=LAMBDAS) -> CVResult:
    """Elastic Net com alpha fixo (float) ou escolhido por CV (lista)."""
    cv = ElasticNetCV(l1_ratio=l1_ratio, alphas=lambdas, cv=folds,
                      max_iter=100_000).fit(split.X_tr_sc, split.y_tr)
    mse_path = cv.mse_path_
    alphas = cv.alphas_
    if mse_path.ndim == 3:   # grade de l1_ratio: escolhe a linha do melhor alpha
        k = int(np.argmin(mse_path.mean(axis=2).min(axis=1)))
        mse_path, alphas = mse_path[k], alphas[k]
    mse, se = _resumo_path(mse_path)
    return CVResult("Elastic Net", cv, float(cv.alpha_), alphas, mse, se,
                    alpha=float(cv.l1_ratio_))


def lasso_at(lam: float, split: Split) -> Lasso:
    return Lasso(alpha=lam, max_iter=100_000).fit(split.X_tr_sc, split.y_tr)


def enet_at(lam: float, alpha: float, split: Split) -> ElasticNet:
    return ElasticNet(alpha=lam, l1_ratio=alpha, max_iter=100_000).fit(split.X_tr_sc, split.y_tr)
