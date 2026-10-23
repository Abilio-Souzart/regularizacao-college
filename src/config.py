"""Caminhos e hiperparâmetros globais."""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "College.csv"
FIG_DIR = ROOT / "results" / "figures"
TABLE_DIR = ROOT / "results" / "tables"

TARGET = "Apps"
SEED = 42
TEST_SIZE = 0.30
N_FOLDS = 10

# Grade de lambda na escala do glmnet / ElasticNet do scikit-learn:
#   (1 / 2n) * RSS + lambda * [ (1 - alpha)/2 * ||b||^2 + alpha * ||b||_1 ]
# A grade vai bem perto de zero para que o CV possa escolher "quase OLS".
LAMBDAS = np.logspace(-4, 4, 200)
L1_RATIOS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]   # alpha do Elastic Net (0 = Ridge)
ENET_ALPHA_FIXO = 0.5

N_REPETICOES = 50   # divisões treino/teste na análise de robustez

PALETTE = {
    "OLS": "#E74C3C",
    "Ridge": "#3498DB",
    "LASSO": "#2ECC71",
    "LASSO (1-EP)": "#7FD8A4",
    "Elastic Net": "#9B59B6",
    "Elastic Net (1-EP)": "#C39BD3",
}


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
