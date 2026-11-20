"""[3] 모델 담당의 시계열·연속학습 후보 3종 (칼만필터, 시변계수, 최근가중).

지금까지 데이터에서 본 구조: 다음 달 변화 ≈ 달 전체 공통 움직임(달 길이 등) + 칸별 고유 움직임.
같은 달 안의 관계는 달마다 바뀌었다 (docs/0.7_탐색기록.md). 그래서 '매달 새 자료로 상태를 고쳐 가는' 방법을 후보로 둔다.

1. 칼만상태공간: 칸(고객층×업종)마다 '하루 평균 지출(로그)'을 숨은 상태로 보고 칼만필터로 매달 갱신.
   다음 달 합계 예측 = 다음 달 하루 평균 예측 × 다음 달 일수 → 이번 달보다 크면 증가.
   달 길이 효과가 식 안에 들어가면서도 칸마다 다른 답이 나온다.
   - 모형 '수준': 하루 평균이 조금씩 흔들리며 이어짐 / '수준추세': 기울기도 추적 /
     '공통요인': 48칸 전체의 공통 흐름과 칸별 차이를 따로 추적해 더함
   - 신호대잡음비: 클수록 새 달에 빨리 따라감(빨리 잊음), 작을수록 천천히
   - 보정 '부호': 예측 변화의 부호 그대로 / '로지스틱': 학습 기간 정답으로 예측 변화 → 증가확률을 맞춤
2. 시변계수회귀: 변수 → 다음 달 증감률의 계수가 매달 조금씩 변한다고 보고 칼만필터로 계수를 추적 (변화속도 = 계수가 한 달에 변하는 정도)
3. 최근가중로지스틱: 로지스틱 회귀에 최근 달일수록 큰 가중치 (망각계수^몇 달 전). 연속학습의 가장 단순한 형태

모든 후보는 기준월 t의 문제에 t월까지의 금액만 쓴다 (forward 필터). 정답(다음 달 금액)은 학습 문제의 것만 쓴다.
행 → 문제(기준월·칸) 대응은 problem_lookup으로 찾으므로, 본 프로세스에서 돈다 (model_agent.LOOKUP_MODELS).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from agents import problem_lookup

KALMAN, TVC, RECENCY = "칼만상태공간", "시변계수회귀", "최근가중로지스틱"
NAMES = (KALMAN, TVC, RECENCY)
CELL = ["sex", "age_band", "cat_l"]
_PANEL: dict = {}        # 정답표 금액 (칸 × 월) — data_agent.build_truth가 등록


def set_panel(truth: pd.DataFrame):
    """정답표(거래를 빼지 않은 칸×월 금액)를 등록. 모델은 문제의 기준월까지만 잘라 쓴다."""
    am = truth.pivot_table(index=CELL, columns="crym", values="am", aggfunc="sum").sort_index(axis=1)
    months = list(am.columns)
    days = {m: pd.Timestamp(m // 100, m % 100, 1).days_in_month for m in months}
    _PANEL.update({"am": am, "months": months, "days": days,
                   "rate": truth.set_index(CELL + ["crym"])["rate_next"]})


def _mindex(m: int) -> int:
    return (m // 100) * 12 + m % 100


def _keys(data_name, X) -> pd.DataFrame:
    lk = problem_lookup.get(data_name)
    return lk["keys"].iloc[problem_lookup.find(data_name, X)].reset_index(drop=True)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


# ---------------------------------------------------------------- 1. 칼만 상태공간
def _filter(z: np.ndarray, snr: float, trend: bool) -> np.ndarray:
    """1차원 칼만필터. 반환: 각 시점 t까지 보고 낸 t+1 예측값 (forward만, 미래 값 안 씀)."""
    n = len(z)
    out = np.full(n, np.nan)
    if trend:
        F, H = np.array([[1.0, 1.0], [0.0, 1.0]]), np.array([1.0, 0.0])
        Q, R = np.diag([snr, snr * 0.1]), 1.0
        x, P = np.array([z[0], 0.0]), np.eye(2) * 10.0
    else:
        F, H, Q, R = np.array([[1.0]]), np.array([1.0]), np.array([[snr]]), 1.0
        x, P = np.array([z[0]]), np.eye(1) * 10.0
    for t in range(n):
        if t > 0:                                       # 예측 단계
            x, P = F @ x, F @ P @ F.T + Q
        S = H @ P @ H + R                               # 갱신 단계 (t월 관측)
        K = P @ H / S
        x = x + K * (z[t] - H @ x)
        P = P - np.outer(K, H) @ P
        out[t] = H @ (F @ x)                            # t월까지 보고 낸 t+1월 예측
    return out


def kalman_delta(모형: str, snr: float) -> pd.Series:
    """칸×기준월마다 '예측한 다음 달 로그 금액 − 이번 달 로그 금액'. 다음 달 일수가 곱해져 달 길이 효과 포함."""
    am, months, days = _PANEL["am"], _PANEL["months"], _PANEL["days"]
    logd = np.log(np.array([days[m] for m in months], dtype=float))
    Z = np.log(am.to_numpy(float) + 1.0) - logd                 # 로그 하루 평균
    if 모형 == "공통요인":
        common = np.log(am.to_numpy(float).sum(axis=0) + 1.0) - logd
        c_hat = _filter(common, snr, trend=True)
        pred = np.vstack([_filter(z - common, snr, trend=False) for z in Z]) + c_hat
    else:
        pred = np.vstack([_filter(z, snr, trend=(모형 == "수준추세")) for z in Z])
    nxt = np.append(logd[1:], np.log(pd.Timestamp(months[-1] // 100, months[-1] % 100, 1).days_in_month))  # 마지막 달은 쓰이지 않음
    delta = pred + nxt - (Z + logd)                             # log(예측 다음 달 합계) − log(이번 달 합계)
    return pd.DataFrame(delta, index=am.index, columns=pd.Index(months, name="crym")).stack().rename("delta")


class KalmanStateSpaceClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, 모형="수준", 신호대잡음비=1.0, 보정="부호", data_name=""):
        self.모형, self.신호대잡음비, self.보정, self.data_name = 모형, 신호대잡음비, 보정, data_name

    def _delta(self, X) -> np.ndarray:
        if not hasattr(self, "_table"):
            self._table = kalman_delta(self.모형, float(self.신호대잡음비))
        k = _keys(self.data_name, X)
        return self._table.reindex(pd.MultiIndex.from_frame(k[CELL + ["crym"]])).fillna(0.0).to_numpy()

    def fit(self, X, y):
        d, y = self._delta(X), np.asarray(y).astype(int)
        self.scale_ = float(np.std(d)) or 1.0
        if self.보정 == "로지스틱" and len(set(y)) == 2:
            self.cal_ = LogisticRegression(C=1.0, class_weight="balanced").fit((d / self.scale_)[:, None], y)
        else:
            self.cal_ = None
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        d = self._delta(X) / self.scale_
        p = self.cal_.predict_proba(d[:, None])[:, 1] if self.cal_ is not None else _sigmoid(2.0 * d)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ---------------------------------------------------------------- 2. 시변계수 회귀
class TimeVaryingRegressionClassifier(BaseEstimator, ClassifierMixin):
    """계수 β가 매달 무작위로 조금씩 움직인다고 보고(β_m = β_(m-1) + 잡음) 칼만필터로 추적한다.
    학습 기간의 달을 차례로 넣으며 β를 고치고, 마지막 학습 달 이후의 달은 β를 그대로 쓰되 불확실성만 키운다."""

    def __init__(self, 변화속도=0.1, data_name=""):
        self.변화속도, self.data_name = 변화속도, data_name

    def _prep(self, X):
        Xs = (np.nan_to_num(np.asarray(X, dtype=float)) - self.mu_) / self.sd_
        return np.column_stack([np.ones(len(Xs)), Xs])

    def fit(self, X, y):
        X = np.nan_to_num(np.asarray(X, dtype=float))
        k = _keys(self.data_name, X)
        rate = _PANEL["rate"].reindex(pd.MultiIndex.from_frame(k[CELL + ["crym"]])).to_numpy(float)
        t = np.log1p(np.clip(np.nan_to_num(rate), -0.9, 3.0))
        self.mu_, self.sd_ = X.mean(axis=0), X.std(axis=0) + 1e-9
        self.tsd_ = float(t.std()) or 1.0
        A, t = self._prep(X), t / self.tsd_
        p = A.shape[1]
        beta, P = np.zeros(p), np.eye(p)                       # 처음 β ~ N(0, I): 릿지와 같은 역할
        q = float(self.변화속도)
        months = k["crym"].to_numpy()
        for i, m in enumerate(sorted(set(months))):
            if i > 0:
                P = P + q * np.eye(p)                           # 한 달 지남: 계수가 변할 수 있음
            sel = months == m
            Am, tm = A[sel], t[sel]
            S = Am @ P @ Am.T + np.eye(sel.sum())               # 관측 잡음 1 (표준화된 목표)
            K = P @ Am.T @ np.linalg.solve(S, np.eye(sel.sum()))
            beta = beta + K @ (tm - Am @ beta)
            P = P - K @ Am @ P
        self.beta_ = beta
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X):
        return self._prep(X) @ self.beta_

    def predict_proba(self, X):
        p = _sigmoid(2.0 * self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ---------------------------------------------------------------- 3. 최근가중 로지스틱 (연속학습)
class RecencyWeightedLogistic(BaseEstimator, ClassifierMixin):
    def __init__(self, C=0.1, 망각계수=0.8, data_name=""):
        self.C, self.망각계수, self.data_name = C, 망각계수, data_name

    def fit(self, X, y):
        X = np.nan_to_num(np.asarray(X, dtype=float))
        mi = np.array([_mindex(int(m)) for m in _keys(self.data_name, X)["crym"]])
        w = float(self.망각계수) ** (mi.max() - mi)               # 가장 최근 달 1, 한 달 전 λ, 두 달 전 λ² ...
        self.mu_, self.sd_ = X.mean(axis=0), X.std(axis=0) + 1e-9
        self.clf_ = LogisticRegression(C=float(self.C), class_weight="balanced", max_iter=5000).fit(
            (X - self.mu_) / self.sd_, np.asarray(y).astype(int), sample_weight=w / w.mean())
        self.classes_ = self.clf_.classes_
        return self

    def predict_proba(self, X):
        return self.clf_.predict_proba((np.nan_to_num(np.asarray(X, dtype=float)) - self.mu_) / self.sd_)

    def predict(self, X):
        return self.clf_.predict(((np.nan_to_num(np.asarray(X, dtype=float)) - self.mu_) / self.sd_))


def build(name: str, p: dict, recipe: dict):
    cls = {KALMAN: KalmanStateSpaceClassifier, TVC: TimeVaryingRegressionClassifier, RECENCY: RecencyWeightedLogistic}[name]
    return cls(**p, data_name=recipe["이름"])
