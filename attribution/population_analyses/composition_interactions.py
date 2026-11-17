"""Does adding pairwise interactions between cluster shares close the gap between a composition-only model (Fig. 3,
AUROC .74) and SPECTRA (.82)? Same within-test 5x5 CV and L2 logistic as analyze_leiden.py; C fixed a priori."""
from pathlib import Path
import json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

REPO = Path(__file__).resolve().parents[2]
L = REPO / 'analyses/figure3_locked6_attribution_20261111/leiden'; OUT = REPO / 'analyses/population_analyses'
pool = pd.read_csv(L / 'donor_pool_fraction_private.csv', index_col=0)
don = pd.read_csv(REPO / 'runs/figure3_locked6_attribution_20261111/donors.csv', index_col=0)
y = don.loc[pool.index, 'label'].to_numpy()
X = np.log(pool.to_numpy() + 1e-3); X = X - X.mean(1, keepdims=True)
models = {'Composition, main effects (Fig. 3)': lambda: LogisticRegression(C=.1, class_weight='balanced', max_iter=4000),
          'Composition + all pairwise interactions': lambda: make_pipeline(PolynomialFeatures(2, interaction_only=True, include_bias=False), StandardScaler(),
                                                                          LogisticRegression(C=.01, class_weight='balanced', max_iter=8000)),
          'Composition, gradient-boosted trees (depth 3)': lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=.05, max_iter=200, class_weight='balanced', random_state=17)}
res = {}
for name, mk in models.items():
    a = []
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=17).split(X, y):
        m = mk().fit(X[tr], y[tr]); s = m.decision_function(X[te]) if hasattr(m, 'decision_function') else m.predict_proba(X[te])[:, 1]
        a.append(roc_auc_score(y[te], s))
    res[name] = dict(mean=float(np.mean(a)), sd=float(np.std(a)))
res['SPECTRA'] = dict(mean=float(roc_auc_score(y, don.loc[pool.index, 'probability'])))
json.dump(res, open(OUT / 'composition_interaction_models.json', 'w'), indent=1); print(json.dumps(res, indent=1))
