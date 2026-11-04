import numpy as np
import pandas as pd
import pytest

from src import models as M
from src.config import FEATURES_PATH
from src.split import dev_split, fit_frame


class Fixed:
    def __init__(self, value):
        self.value = value

    def predict(self, df):
        return np.full(len(df), self.value)


def test_seed_average_is_the_mean_of_models():
    frame = pd.DataFrame({"a": [1, 2]})
    assert M.SeedAverage([Fixed(0.2), Fixed(0.4)]).predict(frame) == pytest.approx([0.3, 0.3])


@pytest.mark.skipif(not FEATURES_PATH.exists(), reason="run python -m src.features first")
def test_catboost_refit_and_cv_write_no_files(tmp_path, monkeypatch):
    train, val = dev_split(pd.read_parquet(FEATURES_PATH))
    fit = fit_frame(train).sample(3000, random_state=0)
    monkeypatch.chdir(tmp_path)
    cv = M.cv_catboost(fit, params=M.DEFAULT_CATBOOST, max_iter=5)
    model = M.refit_catboost(fit, 5, params=M.DEFAULT_CATBOOST, seeds=(42, 1))
    p = model.predict(val.head(50))
    assert len(cv) == 3 and cv["iterations"].between(1, 5).all()
    assert len(model.fitted) == 2 and p.min() > 0 and p.max() < 1
    assert not (tmp_path / "catboost_info").exists()


@pytest.mark.skipif(not FEATURES_PATH.exists(), reason="run python -m src.features first")
def test_lightgbm_refit_gives_probabilities():
    train, val = dev_split(pd.read_parquet(FEATURES_PATH))
    fit = fit_frame(train).sample(3000, random_state=0)
    model = M.refit_lightgbm(fit, 5, seeds=(42, 1))
    p = model.predict(val.head(50))
    assert len(model.fitted) == 2 and p.min() > 0 and p.max() < 1


@pytest.mark.skipif(not FEATURES_PATH.exists(), reason="run python -m src.features first")
def test_tuning_resumes_from_saved_trials(tmp_path):
    train, _ = dev_split(pd.read_parquet(FEATURES_PATH))
    fit = fit_frame(train).sample(3000, random_state=0)
    storage = f"sqlite:///{(tmp_path / 'study.db').as_posix()}"
    assert len(M.tune_lightgbm(fit, n_trials=1, max_iter=3, storage=storage).trials) == 1
    assert len(M.tune_lightgbm(fit, n_trials=2, max_iter=3, storage=storage).trials) == 2
    assert len(M.tune_lightgbm(fit, n_trials=2, max_iter=3, storage=storage).trials) == 2


@pytest.mark.skipif(not FEATURES_PATH.exists(), reason="run python -m src.features first")
def test_lightgbm_cv_uses_three_time_folds():
    train, _ = dev_split(pd.read_parquet(FEATURES_PATH))
    fit = fit_frame(train).sample(3000, random_state=0)
    cv = M.cv_lightgbm(fit, max_iter=5)
    assert len(cv) == 3 and cv["iterations"].between(1, 5).all()
