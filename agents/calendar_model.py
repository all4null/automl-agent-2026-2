"""[3] 모델 담당의 후보 '달력방향': 다음 달이 이번 달보다 긴지(일수 또는 영업일수)만 보고 방향을 정한다.

왜: 정답이 '월 합계' 기준이라, 다음 달이 길면 대부분의 업종이 늘고 짧거나 같으면 대부분 준다.
    검증까지의 6개월(기준월 3~8월)에서 '다음 달이 더 길다' ↔ '그 달 문제의 과반이 증가'가 6번 모두 맞았다.
    반면 같은 달 안에서 어느 고객층·업종이 반대로 갈지는 변수로 거의 맞히지 못했다 (docs/0.7_탐색기록.md).

어떻게: 학습 기간의 문제를 '다음 달이 더 김 / 아님' 두 무리로 나눠 무리마다 증가 비율을 센다.
       새 문제는 같은 무리의 증가 비율을 증가확률로 낸다 (0.5 이상이면 증가). 규칙을 손으로 정하지 않고 학습 기간에서 배운다.
       같은 달의 문제는 모두 같은 답이 나온다 (그래서 월별 Macro-F1은 낮게 나오고, 3개월 합친 점수로 봐야 한다).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from agents import problem_lookup
from common import month_add

NAME = "달력방향"


def month_length(m: int, 기준: str, holidays: frozenset) -> int:
    first = pd.Timestamp(m // 100, m % 100, 1)
    days = pd.date_range(first, periods=first.days_in_month)
    if 기준 == "일수":
        return len(days)
    return sum(d.weekday() < 5 and d.date() not in holidays for d in days)        # 영업일수


class CalendarDirectionClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, 기준="일수", 공휴일=(), data_name=""):
        self.기준, self.공휴일, self.data_name = 기준, 공휴일, data_name

    def _longer(self, X) -> np.ndarray:
        hol = frozenset(pd.Timestamp(d).date() for d in self.공휴일)
        crym = problem_lookup.get(self.data_name)["keys"]["crym"].to_numpy()[problem_lookup.find(self.data_name, X)]
        return np.array([month_length(month_add(int(m), 1), self.기준, hol) > month_length(int(m), self.기준, hol) for m in crym])

    def fit(self, X, y):
        g, y = self._longer(X), np.asarray(y, dtype=float)
        self.overall_ = float(y.mean())
        # 무리마다 증가 비율 (문제가 없는 무리는 전체 비율). 1건씩 앞뒤로 더해 극단값을 막음
        self.rate_ = {v: float((y[g == v].sum() + self.overall_) / ((g == v).sum() + 1)) if (g == v).any() else self.overall_
                      for v in (True, False)}
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = np.array([self.rate_[bool(v)] for v in self._longer(X)])
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
