"""Executa a análise completa de métodos de regularização.

Uso:
    python main.py                 # tudo
    python main.py --sem-robustez  # pula a análise de 50 divisões (mais lenta)
"""
import argparse
import warnings

import matplotlib

matplotlib.use("Agg")

from sklearn.exceptions import ConvergenceWarning  # noqa: E402

from src import q1_ridge_lasso, q2_elastic_net, q3_selecao, robustez  # noqa: E402
from src.config import FIG_DIR, TABLE_DIR, ensure_dirs  # noqa: E402
from src.data import load_college, make_split  # noqa: E402
from src.models import make_folds  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sem-robustez", action="store_true",
                        help="não roda a análise com várias divisões treino/teste")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    ensure_dirs()

    df = load_college()
    split, folds = make_split(df), make_folds()

    q1 = q1_ridge_lasso.run(split, folds)
    q2_elastic_net.run(split, folds, q1)
    q3_selecao.run(split, folds, df, q1)
    if not args.sem_robustez:
        robustez.run(df)

    print(f"\nTabelas: {TABLE_DIR}\nFiguras: {FIG_DIR}")


if __name__ == "__main__":
    main()
