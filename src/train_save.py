import json
import numpy as np
import pandas as pd
from pathlib import Path
import lightgbm as lgb
from joblib import dump
from sklearn.metrics import mean_absolute_error, mean_squared_error

def smape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)

def wape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sum(np.abs(y_true - y_pred)) / np.maximum(np.sum(np.abs(y_true)), eps) * 100.0)

def main():
    root = Path.cwd()
    processed = root / "data" / "processed"
    models_dir = root / "models"
    metrics_dir = root / "reports" / "metrics"
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    feat_path = processed / "features.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(f"Missing {feat_path}. Build features first.")

    print("Loading features...")
    df = pd.read_parquet(feat_path)
    df["date"] = pd.to_datetime(df["date"])

    # target
    y = df["y"].astype(np.float32)

    # features
    drop_cols = {"y", "date", "d", "wm_yr_wk"}
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # categorical
    cat_cols = ["item_id","dept_id","cat_id","store_id","state_id",
                "event_name_1","event_type_1","event_name_2","event_type_2"]
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].astype("category")

    # avoid overfitting on series id
    if "id" in X.columns:
        X = X.drop(columns=["id"])

    # time split (last 28 days)
    max_date = df["date"].max()
    valid_start = max_date - pd.Timedelta(days=27)

    train_idx = df["date"] < valid_start
    valid_idx = df["date"] >= valid_start

    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_valid, y_valid = X.loc[valid_idx], y.loc[valid_idx]

    print("Train:", X_train.shape, "Valid:", X_valid.shape)
    print("Valid range:", valid_start.date(), "→", max_date.date())

    model = lgb.LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=128,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        force_row_wise=True
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=True)]
    )

    pred = model.predict(X_valid)

    mae = float(mean_absolute_error(y_valid, pred))
    rmse = float(np.sqrt(mean_squared_error(y_valid, pred)))
    metrics = {
        "mae": mae,
        "rmse": rmse,
        "smape": smape(y_valid, pred),
        "wape": wape(y_valid, pred),
        "best_iteration": int(getattr(model, "best_iteration_", model.n_estimators)),
        "n_train_rows": int(X_train.shape[0]),
        "n_valid_rows": int(X_valid.shape[0]),
        "n_features": int(X_train.shape[1]),
        "valid_start": str(valid_start.date()),
        "valid_end": str(max_date.date())
    }

    # Save model bundle
    model_path = models_dir / "lgbm_model.joblib"
    dump({"model": model, "feature_cols": list(X.columns)}, model_path)

    # Save metrics
    metrics_path = metrics_dir / "train_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("✅ Saved model:", model_path)
    print("✅ Saved metrics:", metrics_path)

if __name__ == "__main__":
    main()
