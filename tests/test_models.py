"""Testes das conversões de escala e da preparação dos dados."""
import numpy as np
import pytest
from sklearn.linear_model import ElasticNet, Lasso

from src.data import load_college, make_split
from src.models import ridge_from_lambda


@pytest.fixture(scope="module")
def split():
    return make_split(load_college())


def test_private_codificada(split):
    df = load_college()
    assert "Private" not in df.columns
    assert set(df["Private_Yes"].unique()) <= {0, 1}
    assert len(split.features) == 17


def test_padronizacao_so_com_treino(split):
    assert np.allclose(split.X_tr_sc.mean(axis=0), 0, atol=1e-10)
    assert not np.allclose(split.X_te_sc.mean(axis=0), 0, atol=1e-3)


@pytest.mark.parametrize("lam", [0.1, 10.0, 300.0])
def test_ridge_equivale_elastic_net_com_alpha_zero(split, lam):
    """Ridge(alpha = n * λ) deve ser o mesmo modelo que o Elastic Net com α → 0."""
    ridge = ridge_from_lambda(lam, split)
    enet = ElasticNet(alpha=lam, l1_ratio=1e-8, max_iter=1_000_000, tol=1e-10).fit(
        split.X_tr_sc, split.y_tr)
    assert np.allclose(ridge.coef_, enet.coef_, rtol=1e-3, atol=1e-2)


def test_elastic_net_com_alpha_um_e_lasso(split):
    lam = 50.0
    a = ElasticNet(alpha=lam, l1_ratio=1.0, max_iter=100_000).fit(split.X_tr_sc, split.y_tr)
    b = Lasso(alpha=lam, max_iter=100_000).fit(split.X_tr_sc, split.y_tr)
    assert np.allclose(a.coef_, b.coef_)
