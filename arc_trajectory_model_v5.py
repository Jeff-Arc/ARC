#!/usr/bin/env python3
"""
TSFresh trajectory death model for arc_v5.
Extracts time-series features from full cluster history,
trains XGBoost, compares AUC to static snapshot model baseline.
"""

import os
import sys
import pickle
import psycopg2
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

os.environ['PGHOST'] = '/var/run/postgresql'
os.environ['PGDATABASE'] = 'arc_v5'
os.environ['PGUSER'] = 'jeff'


def get_connection():
    return psycopg2.connect(
        host='/var/run/postgresql',
        dbname='arc_v5',
        user='jeff'
    )


def load_cluster_history(conn):
    """Load full cluster history — all periods per cluster."""
    query = """
    WITH windowed AS (
        SELECT
            corpus_id,
            persistent_cluster_id,
            period_start,
            is_new,
            COALESCE(size, 0)::float as size,
            COALESCE(cohesion, 0)::float as cohesion,
            COALESCE(drift_magnitude, 0)::float as drift_magnitude,
            COALESCE(elongation_ratio, 1.5)::float as elongation_ratio,
            COALESCE(mean_betweenness, 0)::float as mean_betweenness,
            COALESCE(n_attractors, 0)::float as n_attractors,
            COALESCE(mean_triangle_count, 0)::float as mean_triangle_count,
            COALESCE(boundary_pressure_rate, 0)::float as boundary_pressure_rate,
            COALESCE(convergence_score, 0)::float as convergence_score,
            -- Target: will this cluster die within the NEXT 4 periods?
            -- Compute on ALL rows (including is_new=true) so window sees future correctly
            COALESCE(MAX(is_dead::int) OVER (
                PARTITION BY corpus_id, persistent_cluster_id
                ORDER BY period_start
                ROWS BETWEEN 1 FOLLOWING AND 4 FOLLOWING
            ), 0) as dies_within_4
        FROM cluster_snapshot
        WHERE size >= 3
          AND cohesion IS NOT NULL
    )
    SELECT
        corpus_id || '_' || persistent_cluster_id as id,
        EXTRACT(EPOCH FROM period_start)::bigint as time,
        size, cohesion, drift_magnitude, elongation_ratio,
        mean_betweenness, n_attractors, mean_triangle_count,
        boundary_pressure_rate, convergence_score,
        dies_within_4
    FROM windowed
    WHERE is_new = false
    ORDER BY corpus_id, persistent_cluster_id, period_start
    """
    print("Loading cluster history...")
    df = pd.read_sql(query, conn)
    print(f"  Loaded {len(df)} rows, {df['id'].nunique()} unique clusters")
    label_counts = df.groupby('id')['dies_within_4'].max().value_counts()
    print(f"  Class balance (dies_within_4, max over history): {label_counts.to_dict()}")
    return df


def extract_tsfresh_features(df):
    """Extract TSFresh features from time series."""
    try:
        from tsfresh import extract_features
        from tsfresh.utilities.dataframe_functions import impute
        from tsfresh.feature_extraction import EfficientFCParameters, MinimalFCParameters
        tsfresh_available = True
    except ImportError:
        print("tsfresh not available, using manual feature extraction")
        tsfresh_available = False

    feature_cols = ['size', 'cohesion', 'drift_magnitude', 'elongation_ratio',
                    'mean_betweenness', 'n_attractors', 'mean_triangle_count',
                    'boundary_pressure_rate', 'convergence_score']

    if not tsfresh_available:
        return extract_manual_features(df)

    ts_df = df[['id', 'time'] + feature_cols].copy()

    print("Extracting TSFresh features (EfficientFCParameters, n_jobs=4)...")
    try:
        features = extract_features(
            ts_df,
            column_id='id',
            column_sort='time',
            default_fc_parameters=EfficientFCParameters(),
            n_jobs=4,
            show_warnings=False,
            disable_progressbar=False
        )
        impute(features)
        print(f"  Extracted {features.shape[1]} TSFresh features for {len(features)} clusters")
    except Exception as e:
        print(f"EfficientFCParameters failed ({e}), trying MinimalFCParameters...")
        try:
            features = extract_features(
                ts_df,
                column_id='id',
                column_sort='time',
                default_fc_parameters=MinimalFCParameters(),
                n_jobs=4,
                show_warnings=False
            )
            impute(features)
            print(f"  Extracted {features.shape[1]} minimal TSFresh features for {len(features)} clusters")
        except Exception as e2:
            print(f"MinimalFCParameters also failed ({e2}), falling back to manual")
            features = extract_manual_features(df)

    return features


def extract_manual_features(df):
    """Fallback: manual time series feature extraction."""
    print("Extracting manual time series features...")
    feature_cols = ['size', 'cohesion', 'drift_magnitude', 'elongation_ratio',
                    'mean_betweenness', 'n_attractors', 'boundary_pressure_rate',
                    'convergence_score']

    def ts_features(g):
        feats = {}
        for col in feature_cols:
            vals = g[col].values
            if len(vals) == 0:
                continue
            feats[f'{col}_mean'] = np.mean(vals)
            feats[f'{col}_std'] = np.std(vals) if len(vals) > 1 else 0.0
            feats[f'{col}_min'] = np.min(vals)
            feats[f'{col}_max'] = np.max(vals)
            feats[f'{col}_last'] = vals[-1]
            feats[f'{col}_first'] = vals[0]
            feats[f'{col}_range'] = np.max(vals) - np.min(vals)
            if len(vals) > 1:
                feats[f'{col}_trend'] = np.polyfit(range(len(vals)), vals, 1)[0]
                feats[f'{col}_last_diff'] = vals[-1] - vals[-2]
                feats[f'{col}_last3_mean'] = np.mean(vals[-3:]) if len(vals) >= 3 else vals[-1]
                feats[f'{col}_last3_trend'] = np.polyfit(range(len(vals[-3:])), vals[-3:], 1)[0] if len(vals) >= 3 else 0.0
            else:
                feats[f'{col}_trend'] = 0.0
                feats[f'{col}_last_diff'] = 0.0
                feats[f'{col}_last3_mean'] = vals[-1]
                feats[f'{col}_last3_trend'] = 0.0
        feats['n_periods'] = len(vals)
        return pd.Series(feats)

    features = df.groupby('id').apply(ts_features)
    features = features.fillna(0)
    print(f"  Extracted {features.shape[1]} manual features for {len(features)} clusters")
    return features


def train_model(features, labels):
    """Train XGBoost on extracted features."""
    # Align labels and features
    common_ids = features.index.intersection(labels.index)
    X = features.loc[common_ids]
    y = labels.loc[common_ids]

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)

    print(f"\nTraining XGBoost on {len(X)} clusters")
    print(f"  Positive (dies_within_4): {n_pos} ({100*n_pos/len(y):.1f}%)")
    print(f"  Negative (alive): {n_neg}")
    print(f"  scale_pos_weight: {scale_pos_weight:.1f}")
    print(f"  Features: {X.shape[1]}")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric='auc',
        n_jobs=4,
        random_state=42,
        verbosity=0
    )

    # 5-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_aucs = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    print(f"\n5-fold CV AUC: {cv_aucs.mean():.4f} ± {cv_aucs.std():.4f}")
    print(f"  Per fold: {[f'{a:.4f}' for a in cv_aucs]}")

    # Train final model on all data
    model.fit(X, y)

    # Feature importance (top 20)
    importance = pd.Series(model.feature_importances_, index=X.columns)
    top_features = importance.nlargest(20)
    print("\nTop 20 features by importance:")
    for feat, imp in top_features.items():
        print(f"  {feat}: {imp:.4f}")

    return model, cv_aucs.mean(), top_features


def main():
    conn = get_connection()

    # Load data
    df = load_cluster_history(conn)
    conn.close()

    # Get labels (one per cluster): MAX over all periods
    # "did this cluster ever have a forward 4-period death signal in its history?"
    # This identifies clusters that were heading toward death at any observed point
    labels = df.groupby('id')['dies_within_4'].max()

    # Extract features
    features = extract_tsfresh_features(df)

    # Train
    model, mean_auc, top_features = train_model(features, labels)

    # Save model
    os.makedirs('/home/jeff/arc/ml/models', exist_ok=True)
    model_path = '/home/jeff/arc/ml/models/death_trajectory_h4_v5.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'feature_names': features.columns.tolist(),
            'mean_cv_auc': mean_auc
        }, f)
    print(f"\nModel saved to {model_path}")

    # Save feature names for scoring
    feat_path = '/home/jeff/arc/ml/models/death_trajectory_h4_features.txt'
    with open(feat_path, 'w') as f:
        f.write('\n'.join(features.columns.tolist()))

    print(f"\n{'='*50}")
    print(f"TRAJECTORY MODEL RESULT")
    print(f"  Mean CV AUC: {mean_auc:.4f}")
    print(f"  Top feature: {top_features.index[0]} ({top_features.iloc[0]:.4f})")
    print(f"{'='*50}")

    return mean_auc


if __name__ == '__main__':
    auc = main()
    sys.exit(0 if auc > 0.5 else 1)
