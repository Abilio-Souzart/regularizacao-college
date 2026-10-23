"""Leitura, codificação e divisão treino/teste."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import DATA_PATH, SEED, TARGET, TEST_SIZE


def load_college() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.replace(".", "_") for c in df.columns]
    # Codificação explícita: não depende de como cada versão do pandas
    # classifica colunas de texto (object x str).
    df["Private_Yes"] = (df.pop("Private") == "Yes").astype(int)
    return df


@dataclass
class Split:
    features: list
    X_tr: np.ndarray       # sem padronizar (usado pelo OLS)
    X_te: np.ndarray
    X_tr_sc: np.ndarray    # padronizado com média/desvio do treino
    X_te_sc: np.ndarray
    y_tr: np.ndarray
    y_te: np.ndarray

    @property
    def n_train(self) -> int:
        return len(self.y_tr)


def make_split(df: pd.DataFrame, seed: int = SEED) -> Split:
    features = [c for c in df.columns if c != TARGET]
    X, y = df[features].to_numpy(float), df[TARGET].to_numpy(float)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=seed)
    scaler = StandardScaler().fit(X_tr)   # ajustado só no treino
    return Split(features, X_tr, X_te, scaler.transform(X_tr), scaler.transform(X_te), y_tr, y_te)
