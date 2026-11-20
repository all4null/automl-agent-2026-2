"""[3] 모델 담당의 후보 '중분류합산': 중분류 단위로 예측해서 대분류로 더한다 (계획서 4-2 ③).

왜: 대분류 방향을 바로 맞히면 학습 칸이 한 달 48개뿐이다. 중분류로 내려가면 칸이 몇 배로 늘고,
    업종 안에서 어느 중분류가 움직였는지를 모델이 직접 본다.

어떻게 (문제 하나 = 고객층 × 대분류 × 기준월 t):
  1. 그 대분류의 중분류마다 '다음 달 금액 변화 ÷ 이번 달 대분류 금액'을 회귀 모델로 예측한다.
     (이렇게 나누면 중분류 값들을 더한 것이 정확히 대분류의 다음 달 증감률이 된다)
  2. 예측값을 더해 0보다 크면 증가, 작으면 감소. 더한 값을 0~1 사이 '증가확률'로 바꿔 다른 모델과 같은 모양으로 낸다.
  - 1~5월 월평균 결제건수가 기준(원본변수 > 최소_월평균건수, 기본 100건) 미만인 중분류는
    대분류 안의 '(기타)' 한 칸으로 합친다 (계획서 4-2 ③의 '대분류로 되돌림'). 그래서 중분류 칸을 더하면 늘 대분류 금액과 같다.
  - 정답표·문제·채점은 그대로다. 바뀌는 것은 '무엇을 학습해서 방향을 내는가'뿐이다.

다른 후보 모델과 같은 자리(검증·채택 규칙·테스트)에 끼우기 위해, 이 모델은 대분류 변수표의 행을 받아
그 행이 어느 문제인지 찾은 뒤(agents/problem_lookup.py) 그 문제의 중분류 칸들로 학습·예측한다.
실데이터 결과: 검증 F1 0.41~0.51로 로지스틱(0.55)보다 낮았다. 같은 달 안의 순위 신호가 달마다 뒤집혀서,
칸을 늘려도 신호가 늘지 않았다 (docs/0.7_탐색기록.md).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler

from agents import problem_lookup
from common import SafeError, slog

NAME = "중분류합산"
ETC = "(기타)"
_REG: dict = {}          # 레시피 이름 → 중분류 자료 (같은 프로세스 안에서만 씀)

UNIT_COLS = ["중분류_비중", "중분류_비중변화", "중분류_전월기여", "중분류_전전월기여", "중분류_증감률",
             "중분류_3개월변동성", "중분류_최근방향", "중분류_건수", "중분류_기타여부"]

COMPLEXITY = {   # 복잡도 → 회귀 모델 설정 (비전공자가 숫자 대신 '낮음/중간/높음'으로 고르도록)
    "릿지": {"낮음": {"alpha": 100.0}, "중간": {"alpha": 10.0}, "높음": {"alpha": 1.0}},
    "LightGBM": {"낮음": {"num_leaves": 4, "n_estimators": 150, "min_child_samples": 40},
                 "중간": {"num_leaves": 7, "n_estimators": 200, "min_child_samples": 20},
                 "높음": {"num_leaves": 15, "n_estimators": 300, "min_child_samples": 10}},
    "랜덤포레스트": {"낮음": {"max_depth": 4, "min_samples_leaf": 20},
                 "중간": {"max_depth": 8, "min_samples_leaf": 10},
                 "높음": {"max_depth": None, "min_samples_leaf": 5}},
}


# ---------------------------------------------------------------- 자료 만들기 (data_agent.build가 부름)
def build_table(mid: pd.DataFrame, keys: list[str], unit: list[str], months: list[int], stat: list[int],
                min_ct: float) -> pd.DataFrame:
    """고객층 × 대분류 × 중분류 × 월 합계(mid) → 중분류 칸 변수와 목표값.

    mid: 정제 원본 전체(거래를 빼지 않음)의 합계. 칸을 더하면 정답표 금액과 1원 단위로 같다.
    """
    ok = (mid[mid["crym"].isin(stat)].groupby(unit + ["cat_m"])["ct"].sum() / max(len(stat), 1)) >= min_ct
    ok = ok[ok].reset_index()[unit + ["cat_m"]].assign(_ok=True)
    m = mid.merge(ok, on=unit + ["cat_m"], how="left")
    m["cat_m"] = np.where(m["_ok"].fillna(False).astype(bool), m["cat_m"], ETC)
    m = m.groupby(keys + ["cat_m"], as_index=False)[["am", "ct"]].sum()

    # 모든 월로 펼쳐 0 채움 (전월대비 계산용). 키는 집계 후라 유일
    cells = m[unit + ["cat_m"]].drop_duplicates()
    grid = cells.merge(pd.DataFrame({"crym": months}), how="cross")
    t = grid.merge(m, on=keys + ["cat_m"], how="left", validate="one_to_one").fillna({"am": 0, "ct": 0})
    t = t.sort_values(unit + ["cat_m", "crym"]).reset_index(drop=True)
    g = t.groupby(unit + ["cat_m"], sort=False)
    am = t["am"].astype(float)
    am1, am2, nxt = g["am"].shift(1).astype(float), g["am"].shift(2).astype(float), g["am"].shift(-1).astype(float)
    par = t.groupby(keys)["am"].transform("sum").astype(float)                  # 대분류 금액 (= 정답표 금액)
    pg = t.assign(_p=par).groupby(unit + ["cat_m"], sort=False)["_p"]
    par1, par2 = pg.shift(1).astype(float), pg.shift(2).astype(float)

    def div(a, b):
        return np.where(b > 0, a / np.where(b > 0, b, 1), 0.0)

    share, share1 = div(am, par), div(am1.fillna(0), par1.fillna(0))
    three = pd.concat([am2, am1, am], axis=1)
    mean, std = three.mean(axis=1), three.std(axis=1, ddof=0)
    t["중분류_비중"] = share
    t["중분류_비중변화"] = share - share1
    t["중분류_전월기여"] = div((am - am1).fillna(0), par1.fillna(0))
    t["중분류_전전월기여"] = div((am1 - am2).fillna(0), par2.fillna(0))
    t["중분류_증감률"] = np.clip(div((am - am1).fillna(0), am1.fillna(0)), -1, 3)
    t["중분류_3개월변동성"] = div(std.fillna(0), mean.fillna(0))
    t["중분류_최근방향"] = (am > am1).astype(float)
    t["중분류_건수"] = slog(t["ct"].astype(float))                            # 건수는 항상 로그
    t["중분류_기타여부"] = (t["cat_m"] == ETC).astype(float)
    t["목표"] = div((nxt - am).fillna(0), par)                                  # 다음 달 변화 ÷ 이번 달 대분류 금액
    return t


def register(name: str, table: pd.DataFrame, keys: list[str]) -> str | None:
    """레시피의 문제 대응표(problem_lookup)에 중분류 표를 붙여 둔다. 등록 못 하면 이유를 돌려준다."""
    if not problem_lookup.has(name):
        _REG.pop(name, None)
        return "문제 대응표가 없음"
    lk = problem_lookup.get(name)
    pk = lk["keys"][keys].assign(_pid=np.arange(len(lk["keys"])))
    t = table.merge(pk, on=keys, how="inner")
    codes = {c: i for i, c in enumerate(sorted(table["cat_m"].unique()))}
    _REG[name] = {"P": lk["X"], "pid": t["_pid"].to_numpy(), "M": t[UNIT_COLS].to_numpy(float),
                  "code": t["cat_m"].map(codes).to_numpy(float), "target": t["목표"].to_numpy(float),
                  "n_units": int(len(t)), "n_cells": int(table[["sex", "age_band", "cat_l", "cat_m"]].drop_duplicates().shape[0])}
    return None


def info(name: str) -> dict | None:
    r = _REG.get(name)
    return None if r is None else {"중분류_칸": r["n_cells"], "학습_행(전체)": r["n_units"], "대분류_문제": len(r["P"])}


# ---------------------------------------------------------------- 모델
class MidSumClassifier(BaseEstimator, ClassifierMixin):
    """중분류 변화를 회귀로 예측해 더하는 분류기. 다른 후보와 같이 fit(X, y) / predict_proba(X)를 쓴다."""

    def __init__(self, 회귀모델="릿지", 복잡도="중간", 극단값_상한=True, data_name="", seed=0):
        self.회귀모델, self.복잡도, self.극단값_상한, self.data_name, self.seed = 회귀모델, 복잡도, 극단값_상한, data_name, seed

    def _data(self):
        d = _REG.get(self.data_name)
        if d is None:
            raise SafeError(f"중분류합산: 레시피 {self.data_name}의 중분류 자료가 없습니다.")
        return d

    def _pids(self, X):
        return problem_lookup.find(self.data_name, X)

    def _features(self, d, sel):
        Z = np.hstack([d["P"][d["pid"][sel]], d["M"][sel]])
        if self.회귀모델 != "릿지":
            Z = np.hstack([Z, d["code"][sel][:, None]])     # 나무 모델은 중분류 번호도 씀
        return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

    def fit(self, X, y=None):
        d = self._data()
        pids = self._pids(X)
        sel = np.isin(d["pid"], pids)
        Z, t = self._features(d, sel), d["target"][sel].copy()
        if self.극단값_상한 and len(t) > 10:
            self.lo_, self.hi_ = np.percentile(t, [1, 99])
            t = np.clip(t, self.lo_, self.hi_)
        s = pd.Series(d["target"][sel]).groupby(d["pid"][sel]).sum()
        self.scale_ = float(s.std()) or 1.0                       # 대분류 증감률의 퍼짐 → 확률로 바꿀 때 씀
        p = COMPLEXITY[self.회귀모델][self.복잡도]
        if self.회귀모델 == "릿지":
            from sklearn.linear_model import Ridge
            self.sc_ = StandardScaler().fit(Z)
            self.reg_ = Ridge(**p).fit(self.sc_.transform(Z), t)
        elif self.회귀모델 == "LightGBM":
            from lightgbm import LGBMRegressor
            self.reg_ = LGBMRegressor(**p, learning_rate=0.05, random_state=self.seed, n_jobs=1, verbose=-1).fit(Z, t)
        else:
            from sklearn.ensemble import RandomForestRegressor
            self.reg_ = RandomForestRegressor(**p, n_estimators=300, random_state=self.seed, n_jobs=1).fit(Z, t)
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X):
        """문제마다 '중분류 예측을 더한 다음 달 증감률'."""
        d = self._data()
        pids = self._pids(X)
        sel = np.isin(d["pid"], pids)
        Z = self._features(d, sel)
        pr = self.reg_.predict(self.sc_.transform(Z) if self.회귀모델 == "릿지" else Z)
        s = pd.Series(pr).groupby(d["pid"][sel]).sum()
        return s.reindex(pids).fillna(0.0).to_numpy()

    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(-2.0 * self.decision_function(X) / self.scale_))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
