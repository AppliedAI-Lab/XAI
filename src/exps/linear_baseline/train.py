"""
Method 1 — Linear / Lasso baseline on aggregation-pooled ECFP
(per substructure from BRICS + scaffold + FG).
"""
import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso, LassoCV, Ridge, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data.attn_dataset import FP_NBITS, get_substructure_features_fp
from src.data.fragments import load_faah_csv
from src.exps.dataset_utils import load_split_frames, start_training_log


def aggregate_substructure_features(features, pooling="sum_mean_max"):
    """Aggregate substructure embeddings into one ligand-level feature vector."""
    stacked = np.stack(features, axis=0)
    reducers = {
        "sum": stacked.sum(axis=0),
        "mean": stacked.mean(axis=0),
        "max": stacked.max(axis=0),
    }
    if pooling in reducers:
        return reducers[pooling]
    if pooling == "sum_mean":
        return np.concatenate([reducers["sum"], reducers["mean"]], axis=0)
    if pooling == "sum_mean_max":
        return np.concatenate([reducers["sum"], reducers["mean"], reducers["max"]], axis=0)
    raise ValueError("Unknown pooling: " + str(pooling))


def build_aggregated_dataset(df, fp_radius, fp_n_bits, pooling):
    """Each row: aggregation-pooled substructure ECFP; y = pic50."""
    pooled_rows = []
    y_list = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"ECFP {pooling} pool"):
        smi = row["smiles"]
        feats, _labels, _subs = get_substructure_features_fp(
            smi, fp_radius=fp_radius, fp_n_bits=fp_n_bits
        )
        if not feats:
            continue
        pooled_rows.append(aggregate_substructure_features(feats, pooling=pooling))
        y_list.append(float(row["pic50"]))
    if not pooled_rows:
        return None, None
    X = np.stack(pooled_rows, axis=0)
    y = np.asarray(y_list, dtype=np.float64)
    return X, y


def parse_int_list(value):
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_str_list(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def make_estimator_and_grid(regressor, seed, max_iter):
    if regressor == "ridge":
        return Ridge(solver="lsqr"), {"reg__alpha": np.logspace(0, 5, 11)}
    if regressor == "lasso":
        return Lasso(random_state=seed, max_iter=max_iter, tol=1e-3), {
            "reg__alpha": np.logspace(-2, 1, 8)
        }
    if regressor == "elasticnet":
        return ElasticNet(random_state=seed, max_iter=max_iter, tol=1e-3), {
            "reg__alpha": np.logspace(-2, 1, 8),
            "reg__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        }
    raise ValueError("Unknown regressor: " + str(regressor))


def run_grid_search(df_train, df_test, args, save_path, log_path):
    poolings = parse_str_list(args.grid_poolings)
    fp_radii = parse_int_list(args.grid_fp_radii)
    fp_bits = parse_int_list(args.grid_fp_bits)
    regressors = parse_str_list(args.grid_regressors)

    best = None
    results = []
    cv_splits = max(2, min(args.cv, len(df_train)))
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=args.seed)

    for pooling in poolings:
        for fp_radius in fp_radii:
            for fp_n_bits in fp_bits:
                print(
                    "\nGrid feature config:",
                    f"pooling={pooling}",
                    f"fp_radius={fp_radius}",
                    f"fp_bits={fp_n_bits}",
                )
                X_train, y_train = build_aggregated_dataset(df_train, fp_radius, fp_n_bits, pooling)
                if X_train is None:
                    print("Skipping config with no valid train molecules.")
                    continue
                if df_test is None:
                    from sklearn.model_selection import train_test_split

                    idx = np.arange(len(y_train))
                    tr, te = train_test_split(idx, test_size=args.test_size, random_state=args.seed)
                    X_fit, X_test = X_train[tr], X_train[te]
                    y_fit, y_test = y_train[tr], y_train[te]
                else:
                    X_fit, y_fit = X_train, y_train
                    X_test, y_test = build_aggregated_dataset(df_test, fp_radius, fp_n_bits, pooling)
                    if X_test is None:
                        print("Skipping config with no valid test molecules.")
                        continue

                for regressor in regressors:
                    estimator, param_grid = make_estimator_and_grid(
                        regressor, args.seed, args.max_iter
                    )
                    model = Pipeline([("scaler", StandardScaler()), ("reg", estimator)])
                    search = GridSearchCV(
                        model,
                        param_grid=param_grid,
                        scoring="r2",
                        cv=cv,
                        n_jobs=args.n_jobs,
                        refit=True,
                    )
                    search.fit(X_fit, y_fit)
                    pred_train = search.predict(X_fit)
                    pred_test = search.predict(X_test)
                    train_r2 = r2_score(y_fit, pred_train)
                    test_r2 = r2_score(y_test, pred_test)
                    row = {
                        "pooling": pooling,
                        "fp_radius": fp_radius,
                        "fp_n_bits": fp_n_bits,
                        "regressor": regressor,
                        "best_cv_r2": float(search.best_score_),
                        "train_r2": float(train_r2),
                        "test_r2": float(test_r2),
                        "best_params": search.best_params_,
                    }
                    results.append(row)
                    print(
                        "Grid result:",
                        row["regressor"],
                        "CV R2:",
                        round(row["best_cv_r2"], 6),
                        "Train R2:",
                        round(row["train_r2"], 6),
                        "Test R2:",
                        round(row["test_r2"], 6),
                        "Params:",
                        row["best_params"],
                    )
                    if best is None or row["best_cv_r2"] > best["row"]["best_cv_r2"]:
                        best = {
                            "row": row,
                            "model": search.best_estimator_,
                            "X_fit": X_fit,
                            "y_fit": y_fit,
                            "X_test": X_test,
                            "y_test": y_test,
                        }

    if best is None:
        print("No valid grid-search configuration.")
        sys.exit(1)

    checkpoint_dir = os.path.dirname(save_path) or "."
    results_path = os.path.join(checkpoint_dir, "grid_results.csv")
    results_df = pd.DataFrame(results)
    results_df["best_params"] = results_df["best_params"].astype(str)
    results_df.to_csv(results_path, index=False)

    best_model = best["model"]
    reg = best_model.named_steps["reg"]
    coef = getattr(reg, "coef_", np.array([]))
    if getattr(coef, "ndim", 1) > 1:
        coef = coef.ravel()
    best_row = best["row"]
    payload = {
        "model": best_model,
        "fp_radius": best_row["fp_radius"],
        "fp_n_bits": best_row["fp_n_bits"],
        "regressor": best_row["regressor"],
        "pooling": best_row["pooling"],
        "coef": coef,
        "dataset": args.dataset,
        "train_csv": args.train_csv,
        "test_csv": args.test_csv,
        "training_log": log_path,
        "grid_search": True,
        "grid_results": results_path,
        "best_cv_r2": best_row["best_cv_r2"],
        "train_r2": best_row["train_r2"],
        "test_r2": best_row["test_r2"],
        "best_params": best_row["best_params"],
    }
    joblib.dump(payload, save_path)
    print("\nBest grid config:", best_row)
    print("Grid results:", results_path)
    print("Saved:", save_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="FAAH", help="Dataset folder under data/datasets")
    p.add_argument("--csv", default=None)
    p.add_argument("--train_csv", default=None)
    p.add_argument("--test_csv", default=None)
    p.add_argument("--save_path", default=None)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--regressor", choices=["ridge", "lasso", "elasticnet"], default="ridge")
    p.add_argument(
        "--pooling",
        choices=["sum", "mean", "max", "sum_mean", "sum_mean_max"],
        default="sum_mean_max",
        help="Aggregation pooling over substructure embeddings",
    )
    p.add_argument("--fp_radius", type=int, default=2)
    p.add_argument("--fp_bits", type=int, default=FP_NBITS)
    p.add_argument("--grid_search", action="store_true", help="Tune Method 1 with grid search")
    p.add_argument("--grid_regressors", default="ridge,lasso,elasticnet")
    p.add_argument("--grid_poolings", default="sum_mean,sum_mean_max")
    p.add_argument("--grid_fp_radii", default="2,3")
    p.add_argument("--grid_fp_bits", default="2048,4096")
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--n_jobs", type=int, default=-1)
    p.add_argument("--max_iter", type=int, default=100000)
    args = p.parse_args()

    if args.grid_search:
        save_path = args.save_path or os.path.join(
            ROOT,
            "weights",
            "linear_baseline",
            "grid",
            args.dataset,
            f"linear_agg_ecfp_grid_{args.dataset}.joblib",
        )
    else:
        save_path = args.save_path or os.path.join(
            ROOT, "weights", "linear_baseline", args.regressor, args.dataset, f"linear_agg_ecfp_{args.regressor}_{args.dataset}_{args.pooling}.joblib"
        )
    checkpoint_dir = os.path.dirname(save_path) or "."
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_path = start_training_log(checkpoint_dir)

    if args.csv:
        df_train = load_faah_csv(args.csv)
        df_test = None
        print("Using random split from:", args.csv)
    else:
        try:
            df_train, df_test, train_path, test_path = load_split_frames(
                load_faah_csv,
                dataset=args.dataset,
                train_csv=args.train_csv,
                test_csv=args.test_csv,
            )
        except FileNotFoundError as exc:
            print(exc)
            sys.exit(1)
        print("Train CSV:", train_path)
        print("Test CSV:", test_path)

    if args.grid_search:
        run_grid_search(df_train, df_test, args, save_path, log_path)
        return

    X_train_all, y_train_all = build_aggregated_dataset(
        df_train, args.fp_radius, args.fp_bits, args.pooling
    )
    if X_train_all is None:
        print("No valid molecules after decomposition.")
        sys.exit(1)
    if df_test is None:
        from sklearn.model_selection import train_test_split

        idx = np.arange(len(y_train_all))
        tr, te = train_test_split(idx, test_size=args.test_size, random_state=args.seed)
        X_train, X_test = X_train_all[tr], X_train_all[te]
        y_train, y_test = y_train_all[tr], y_train_all[te]
    else:
        X_train, y_train = X_train_all, y_train_all
        X_test, y_test = build_aggregated_dataset(df_test, args.fp_radius, args.fp_bits, args.pooling)
        if X_test is None:
            print("No valid test molecules after decomposition.")
            sys.exit(1)

    if args.regressor == "ridge":
        reg = RidgeCV(alphas=np.logspace(-4, 4, 20))
    elif args.regressor == "lasso":
        reg = LassoCV(
            alphas=np.logspace(-2, 1, 8),
            cv=5,
            random_state=args.seed,
            max_iter=args.max_iter,
            tol=1e-3,
        )
    else:
        reg = ElasticNetCV(
            alphas=np.logspace(-2, 1, 8),
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
            cv=5,
            random_state=args.seed,
            max_iter=args.max_iter,
            tol=1e-3,
            n_jobs=args.n_jobs,
        )

    model = Pipeline([("scaler", StandardScaler()), ("reg", reg)])
    model.fit(X_train, y_train)
    pred_tr = model.predict(X_train)
    pred_te = model.predict(X_test)
    print("Train R2:", r2_score(y_train, pred_tr))
    print("Test R2:", r2_score(y_test, pred_te))

    coef = model.named_steps["reg"].coef_
    if coef.ndim > 1:
        coef = coef.ravel()

    payload = {
        "model": model,
        "fp_radius": args.fp_radius,
        "fp_n_bits": args.fp_bits,
        "regressor": args.regressor,
        "pooling": args.pooling,
        "coef": coef,
        "dataset": args.dataset,
        "train_csv": args.train_csv,
        "test_csv": args.test_csv,
        "training_log": log_path,
    }
    joblib.dump(payload, save_path)
    print("Saved:", save_path)


if __name__ == "__main__":
    main()
