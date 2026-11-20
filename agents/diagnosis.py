"""[5] 검토 담당의 진단 도구: 모델을 돌리기 전에 데이터 구조와 평가 신뢰도를 본다.

사람이 0.7 탐색 때 손으로 했던 진단을 코드로 옮긴 것이다 (docs/0.7_탐색기록.md).
  1. 달 효과: 그 달 문제 전체가 같이 움직이는 정도, '다음 달이 더 길다' ↔ '과반 증가'가 맞은 달 수
  2. 같은 달 안 신호의 안정성: '지난달 증감률 ↔ 다음 달 증감률' 순위상관이 달마다 뒤집히는지
  3. 정보 상한: 달 / 달×업종 / 달×고객층의 방향을 미리 안다면 낼 수 있는 점수
  4. 단위 일치도: 같은 달·같은 업종 안에서 고객층들이 같은 방향인 정도 (높으면 칸을 잘게 나눠도 새 정보가 적음)
  5. 평가 신뢰도: validation 선택지마다 달 수, 첫 회차 학습 달 수, 달 효과를 감안한 '실질 표본 수'

모두 기준월이 validation 마지막 달 이하인 문제만 쓴다 (test 기간 정보 없음). 화면·LLM에는 집계값만 나간다.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import macro_f1, month_add, month_range


def _dim(m: int) -> int:
    return pd.Timestamp(m // 100, m % 100, 1).days_in_month


def _base(state) -> pd.DataFrame:
    """진단에 쓰는 문제: 첫 기준월 ~ validation 마지막 달 (test 문제 제외)."""
    from agents.data_agent import periods
    P = periods(state.config)
    t = state.truth
    t = t[t["문제"] & (t["crym"] >= P["first"]) & (t["crym"] <= P["dev_end"])].copy()
    t["prev_rate"] = np.where(t["am_prev"] > 0, t["am"] / t["am_prev"].where(t["am_prev"] > 0) - 1, 0.0)
    return t


def month_effect(state) -> dict:
    t = _base(state)
    rows = []
    for m, g in t.groupby("crym"):
        rows.append({"기준월": int(m), "증가비율": round(float(g["label"].mean()), 3),
                     "다음달_일수차이": _dim(month_add(int(m), 1)) - _dim(int(m))})
    y = t["label"].to_numpy(float)
    eta2 = float(t.groupby("crym")["label"].transform("mean").var(ddof=0) / max(y.var(), 1e-9))
    agree = sum((r["다음달_일수차이"] > 0) == (r["증가비율"] > 0.5) for r in rows)
    return {"달별": rows, "달효과_비율": round(eta2, 3), "일수규칙_맞은달": f"{agree}/{len(rows)}",
            "설명": "달효과_비율 = 정답의 흔들림 중 '그 달 전체가 같이 움직인' 몫 (0~1). 클수록 달 단위 패턴이 점수를 좌우함"}


def within_month_signal(state) -> dict:
    t = _base(state)
    rc = {int(m): round(float(spearmanr(g["prev_rate"], g["rate_next"])[0]), 2) for m, g in t.groupby("crym") if len(g) > 5}
    v = [x for x in rc.values() if x == x]
    flips = sum(1 for a, b in zip(v, v[1:]) if (a > 0) != (b > 0))
    return {"달별_순위상관": rc, "부호가_바뀐_횟수": flips, "평균_절댓값": round(float(np.mean(np.abs(v))) if v else 0, 2),
            "설명": "같은 달 안에서 '지난달 많이 오른 칸이 다음 달도 오르나'. 부호가 자주 바뀌면 칸별 패턴을 외울수록 틀림"}


def ceilings(state, months: list[int] | None = None) -> dict:
    t = _base(state)
    if months:
        t = t[t["crym"].isin(months)]
    y = t["label"].to_numpy().astype(int)
    out = {}
    for name, keys in [("달", ["crym"]), ("달×업종", ["crym", "cat_l"]), ("달×고객층", ["crym", "sex", "age_band"])]:
        p = (t.groupby(keys)["label"].transform("mean") >= 0.5).astype(int).to_numpy()
        out[name] = round(macro_f1(y, p), 3)
    return {"상한_F1": out, "대상_기준월": sorted(int(m) for m in t["crym"].unique()),
            "설명": "그 단위의 방향을 정답처럼 미리 안다면 낼 수 있는 Macro-F1. 어느 정보가 점수를 좌우하는지 보여 줌"}


def unit_agreement(state) -> dict:
    t = _base(state)
    g = t.groupby(["crym", "cat_l"])["label"].mean()
    return {"달×업종_안_고객층_일치도": round(float(np.maximum(g, 1 - g).mean()), 3),
            "설명": "같은 달·같은 업종에서 고객층들이 같은 방향인 비율의 평균. 0.85 이상이면 칸을 잘게 나눠도 새 정보가 적음"}


def validation_options(state, eta2: float) -> list[dict]:
    cfg = state.config
    first, last = int(cfg["기간"]["첫_기준월"]), int(cfg["기간"]["검증월"][-1])
    out = []
    for n in (cfg.get("진단") or {}).get("검증월_수_선택지", [3, 4, 5]):
        months = month_range(month_add(last, -(int(n) - 1)), last)
        train_first = len(month_range(first, month_add(months[0], -1))) if months[0] > first else 0
        if train_first < 1:
            continue
        n_items = 48 * len(months)
        n_eff = n_items / (1 + (48 - 1) * max(eta2, 0))          # 같은 달 문제끼리 닮은 정도를 감안한 실질 표본 수
        out.append({"검증월_수": int(n), "검증월": months, "첫회차_학습_달수": train_first,
                    "문제수": n_items, "실질_표본수(달효과 감안)": round(n_eff, 1)})
    return out


def category_month_effect(state) -> dict:
    """업종마다 '다음 달이 더 긴 달 / 아닌 달'의 증가 비율 (검토 담당의 도구)."""
    t = _base(state)
    longer = t["crym"].map(lambda m: _dim(month_add(int(m), 1)) > _dim(int(m)))
    g = t.assign(더긴달=longer).groupby(["cat_l", "더긴달"])["label"].mean().unstack()
    return {c: {"더 긴 달 증가비율": round(float(r.get(True, float("nan"))), 2),
                "아닌 달 증가비율": round(float(r.get(False, float("nan"))), 2)} for c, r in g.iterrows()}


def run_all(state) -> dict:
    me = month_effect(state)
    d = {"달_효과": me, "같은달_신호": within_month_signal(state), "정보_상한": ceilings(state),
         "단위_일치도": unit_agreement(state)}
    d["평가_선택지"] = validation_options(state, me["달효과_비율"])
    return d


def rule_decision(cfg, d: dict) -> dict:
    """LLM이 없을 때의 판단 (0.7 탐색에서 배운 것을 규칙으로)."""
    th = cfg.get("진단") or {}
    eta2 = d["달_효과"]["달효과_비율"]
    strong = eta2 >= float(th.get("달효과_기준", 0.2))
    opts = d["평가_선택지"]
    n = max(o["검증월_수"] for o in opts) if strong else min(o["검증월_수"] for o in opts)
    agree_n, total = (int(x) for x in d["달_효과"]["일수규칙_맞은달"].split("/"))
    on = ["일수보정", "칼만예측"] + (["달력차이"] if agree_n >= math.ceil(total * 0.8) else [])
    off = []
    if d["같은달_신호"]["부호가_바뀐_횟수"] >= 2 or d["단위_일치도"]["달×업종_안_고객층_일치도"] >= 0.85:
        off.append("중분류합산")              # 칸을 잘게 나누는 후보: 새 정보가 적고 뒤집히는 패턴을 외움
    why = (f"달효과 비율 {eta2:.2f} {'≥' if strong else '<'} 기준 → validation {n}개월, 부트스트랩 {'달' if strong else '문제'} 단위. "
           f"일수 규칙 {d['달_효과']['일수규칙_맞은달']} 맞음. 같은 달 신호 부호 바뀜 {d['같은달_신호']['부호가_바뀐_횟수']}회.")
    return {"검증월_수": n, "부트스트랩_단위": "달" if strong else "문제", "켤_변수": on, "끌_후보모델": off,
            "테스트_방식": "매달재학습", "주의점": [], "이유": why}
