"""[3] 모델 담당: 후보(모델 × 하이퍼파라미터)마다 롤링 검증을 돌려 1등을 고른다.

검증 순서 (후보 하나마다):
  1회차: 기준월 3~5월로 학습 → 6월 48문제 채점
  2회차: 3~6월로 처음부터 다시 학습 → 7월 48문제 채점
  3회차: 3~7월로 처음부터 다시 학습 → 8월 48문제 채점
  → 세 회차 예측을 합친 144문제의 Macro-F1 = 검증 점수
극단값 상한·표준화 기준은 회차마다 그 회차의 학습 문제로만 정한다.
테스트 문제(봉인 칸)는 이 파일에서 열지 않는다.
"""
from __future__ import annotations

import importlib.util
import itertools
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import adoption_test, is_on, macro_f1
from agents.data_agent import periods


# ---------------------------------------------------------------- 모델 목록
def _lr(p, seed):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(**p, random_state=seed)


def _lda(p, seed):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    return LinearDiscriminantAnalysis(**p)


def _sgd(p, seed):
    from sklearn.linear_model import SGDClassifier
    return SGDClassifier(**p, random_state=seed)


def _nb(p, seed):
    from sklearn.naive_bayes import GaussianNB
    return GaussianNB(**p)


def _knn(p, seed):
    from sklearn.neighbors import KNeighborsClassifier
    return KNeighborsClassifier(**p)


def _svm(p, seed):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.svm import SVC
    return CalibratedClassifierCV(SVC(**p, random_state=seed), ensemble=False, cv=3)   # 확률은 보정해서 출력


def _tree(p, seed):
    from sklearn.tree import DecisionTreeClassifier
    return DecisionTreeClassifier(**p, random_state=seed)


def _rf(p, seed):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(**p, random_state=seed, n_jobs=1)


def _et(p, seed):
    from sklearn.ensemble import ExtraTreesClassifier
    return ExtraTreesClassifier(**p, random_state=seed, n_jobs=1)


def _lgbm(p, seed):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(**p, random_state=seed, n_jobs=1)


def _hgb(p, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(**p, random_state=seed)


def _mlp(p, seed):
    from sklearn.neural_network import MLPClassifier
    p = dict(p)
    h = p.pop("hidden_layer_sizes", 8)
    return MLPClassifier(hidden_layer_sizes=(int(h),), **p, random_state=seed)


def _xgb(p, seed):
    from xgboost import XGBClassifier
    return XGBClassifier(**p, random_state=seed, n_jobs=1)


def _cat(p, seed):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(**p, random_seed=seed, thread_count=1)


MODEL_BUILDERS = {"로지스틱회귀": _lr, "로지스틱_L1": _lr, "선형판별분석": _lda, "SGD": _sgd, "나이브베이즈": _nb,
                  "최근접이웃": _knn, "서포트벡터머신": _svm, "결정나무": _tree, "랜덤포레스트": _rf,
                  "엑스트라트리": _et, "LightGBM": _lgbm, "히스트그래디언트부스팅": _hgb, "신경망": _mlp,
                  "XGBoost": _xgb, "CatBoost": _cat}
_PACKAGE = {"XGBoost": "xgboost", "CatBoost": "catboost", "LightGBM": "lightgbm"}


def optional_missing(name: str) -> bool:
    pkg = _PACKAGE.get(name)
    return bool(pkg) and importlib.util.find_spec(pkg) is None


class QuantileClip(BaseEstimator, TransformerMixin):
    """학습 문제의 상·하위 1%를 넘는 값을 그 값으로 자른다 (0/1 표시 변수는 건드리지 않음)."""

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.cols_ = [j for j in range(X.shape[1]) if len(np.unique(X[:, j])) > 2]
        self.lo_ = np.percentile(X[:, self.cols_], 1, axis=0) if self.cols_ else []
        self.hi_ = np.percentile(X[:, self.cols_], 99, axis=0) if self.cols_ else []
        return self

    def transform(self, X):
        X = np.array(X, dtype=float, copy=True)
        if self.cols_:
            X[:, self.cols_] = np.clip(X[:, self.cols_], self.lo_, self.hi_)
        return X


class ScaleExcept(BaseEstimator, TransformerMixin):
    """표준화(평균 0, 표준편차 1). 단, 고객층·업종 원-핫 표시(skip 칸)는 0/1 그대로 둔다 (노트북 baseline과 같음)."""

    def __init__(self, skip=()):
        self.skip = skip

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.cols_ = [j for j in range(X.shape[1]) if j not in set(self.skip)]
        self.sc_ = StandardScaler().fit(X[:, self.cols_]) if self.cols_ else None
        return self

    def transform(self, X):
        X = np.array(X, dtype=float, copy=True)
        if self.cols_:
            X[:, self.cols_] = self.sc_.transform(X[:, self.cols_])
        return X


def onehot_idx(ds: dict) -> tuple:
    """변수표에서 고객층·업종 원-핫 표시 칸의 위치 (표준화하지 않음)."""
    return tuple(i for i, c in enumerate(ds["cols"]) if ds["groups"][c] == "고객층_업종_구분")


def make_pipeline(cfg, model: str, params: dict, recipe: dict, seed: int, skip: tuple = ()) -> Pipeline:
    m = cfg["후보모델"][model]
    p = {**(m.get("고정설정") or {}), **params}
    steps = []
    if recipe["전처리"]["극단값_상한"]:
        steps.append(("clip", QuantileClip()))
    if is_on(m.get("표준화_필요")):
        steps.append(("scale", ScaleExcept(skip)))
    steps.append(("model", MODEL_BUILDERS[model](p, seed)))
    return Pipeline(steps)


def proba1(pipe, X) -> np.ndarray:
    return pipe.predict_proba(X)[:, list(pipe.classes_).index(1)] if 1 in pipe.classes_ else np.zeros(len(X))


# ---------------------------------------------------------------- 롤링 검증
def folds(cfg, meta: pd.DataFrame, recipe: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    """(학습 행, 채점 행) 3쌍. ±2% 미세변동 제외는 학습 행에만 적용 (채점은 전부)."""
    P = periods(cfg)
    out = []
    for vm in P["valid"]:
        tr = (meta["crym"] >= P["first"]) & (meta["crym"] < vm)
        if recipe["전처리"]["미세변동_제외"]:
            tr &= meta["rate_next"].abs() > 0.02
        out.append((np.where(tr)[0], np.where(meta["crym"] == vm)[0]))
    return out


def _eval_candidate(cfg, model, params, recipe, X, y, fold_list, seed, skip=()):
    warnings.filterwarnings("ignore")
    proba = []
    for tr, va in fold_list:
        if len(set(y[tr])) < 2:
            proba.append(np.full(len(va), float(y[tr].mean())))
            continue
        pipe = make_pipeline(cfg, model, params, recipe, seed, skip).fit(X[tr], y[tr])
        proba.append(proba1(pipe, X[va]))
    return np.concatenate(proba)


def _grid(space: dict) -> list[dict]:
    keys = list(space)
    return [dict(zip(keys, v)) for v in itertools.product(*[space[k] for k in keys])] or [{}]


def _rules(meta, fold_list, y) -> dict:
    maj, last = [], []
    for tr, va in fold_list:
        maj.append(np.full(len(va), int(y[tr].mean() >= 0.5)))
        last.append(meta["last_dir"].to_numpy()[va])
    return {"다수방향": np.concatenate(maj), "직전방향유지": np.concatenate(last)}


def evaluate(state, ds: dict, recipe: dict) -> dict:
    """레시피 하나에 대해 모든 후보를 검증하고, 채택 규칙으로 모델을 고른다."""
    cfg, log = state.config, state.log
    seed = cfg["난수시드"]
    X = ds["dev"]["X"].to_numpy(dtype=float)
    skip = onehot_idx(ds)
    meta = ds["dev"]["meta"]
    y = meta["label"].to_numpy().astype(int)
    fl = folds(cfg, meta, recipe)
    va_idx = np.concatenate([va for _, va in fl])
    yv, mv = y[va_idx], meta["crym"].to_numpy()[va_idx]

    used = [m for m, v in cfg["후보모델"].items() if is_on(v.get("사용")) and not optional_missing(m)]
    method = cfg["탐색"]["방법"]
    if method == "optuna" and importlib.util.find_spec("optuna") is None:
        method = "grid"
    t0 = time.time()
    if method == "grid":
        if state.search_plan:      # 모델 담당이 세운 탐색 계획 (모든 레시피에 같은 계획)
            tasks = [(m, p) for m, grid in state.search_plan.items() for p in grid]
        else:
            tasks = [(m, p) for m in used for p in _grid(cfg["후보모델"][m]["좁은범위"])]
        probs = Parallel(n_jobs=cfg["탐색"]["병렬_작업수"])(
            delayed(_eval_candidate)(cfg, m, p, recipe, X, y, fl, seed, skip) for m, p in tasks)
    else:
        tasks, probs = _optuna(cfg, used, recipe, X, y, fl, yv, seed, skip)
    state.n_validations += len(tasks) * len(fl)

    rows = []
    for (m, p), pr in zip(tasks, probs):
        pred = (pr >= 0.5).astype(int)
        f = macro_f1(yv, pred)
        mf = {int(k): round(macro_f1(yv[mv == k], pred[mv == k]), 4) for k in sorted(set(mv))}
        rows.append({"레시피": recipe["이름"], "모델": m, "설정": p, "검증F1": round(f, 4), "월별F1": mf, "_proba": pr})
    for r in rows:
        state.trials.append({k: v for k, v in r.items() if k != "_proba"})

    rule = _rules(meta, fl, y)
    rule_f1 = {k: round(macro_f1(yv, v), 4) for k, v in rule.items()}     # 규칙 예측도 같은 144문제 순서

    # 모델별 1등 설정
    best = {}
    for r in rows:
        if r["모델"] not in best or r["검증F1"] > best[r["모델"]]["검증F1"]:
            best[r["모델"]] = r
    base_name = cfg["기준모델"]
    chosen = best[base_name]
    decisions = []
    for r in sorted(best.values(), key=lambda r: -r["검증F1"]):
        if r["모델"] == base_name:
            break
        t = adoption_test(yv, mv, (chosen["_proba"] >= 0.5), (r["_proba"] >= 0.5), cfg["판단규칙"], seed)
        decisions.append((r["모델"], t))
        if t["채택"]:
            chosen = r
            break

    # 앙상블 (선택): 상위 모델 확률 평균을 마지막 후보로 1회만 비교
    ens = cfg.get("앙상블", {})
    if is_on(ens.get("사용")):
        top = sorted(best.values(), key=lambda r: -r["검증F1"])[: int(ens.get("모델_수", 3))]
        pr = np.mean([r["_proba"] for r in top], axis=0)
        t = adoption_test(yv, mv, chosen["_proba"] >= 0.5, pr >= 0.5, cfg["판단규칙"], seed)
        decisions.append(("앙상블", t))
        if t["채택"]:
            chosen = {"모델": "앙상블", "설정": {"구성": [(r["모델"], r["설정"]) for r in top]}, "_proba": pr,
                      "검증F1": round(macro_f1(yv, pr >= 0.5), 4),
                      "월별F1": {int(k): round(macro_f1(yv[mv == k], (pr >= 0.5)[mv == k]), 4) for k in sorted(set(mv))}}

    table = pd.DataFrame([{"모델": r["모델"], "설정": str(r["설정"]), "검증F1": r["검증F1"],
                           **{f"{k}월": v for k, v in r["월별F1"].items()}} for r in best.values()]).sort_values("검증F1", ascending=False)
    result = {"레시피": recipe["이름"], "모델": chosen["모델"], "설정": chosen["설정"], "검증F1": chosen["검증F1"],
              "월별F1": chosen["월별F1"], "proba": chosen["_proba"], "pred": (chosen["_proba"] >= 0.5).astype(int),
              "y": yv, "months": mv, "모델별": table, "규칙F1": rule_f1, "기준모델F1": best[base_name]["검증F1"],
              "후보수": len(tasks), "소요초": round(time.time() - t0, 1), "탐색방법": method,
              "best_by_model": {m: {"설정": r["설정"], "검증F1": r["검증F1"], "proba": r["_proba"]} for m, r in best.items()}}
    state.results[recipe["이름"]] = result

    cand = ", ".join(f"{r['모델']} {r['검증F1']:.3f}" for r in sorted(best.values(), key=lambda r: -r["검증F1"])[:5])
    why = "; ".join(f"{m}: {'채택' if t['채택'] else '기각'} ({t['이유']})" for m, t in decisions) or "기준모델이 검증 1등"
    state.decide("③ 모델", f"모델 선택 (레시피 {recipe['이름']})", cand, f"{chosen['모델']} {chosen['설정']}", "탐색+채택규칙", why)
    log(f"[3] 레시피 {recipe['이름']}: 후보 {len(tasks)}개 × 검증 {len(fl)}회차 ({result['소요초']}초) → "
        f"{chosen['모델']} 검증 F1 {chosen['검증F1']:.3f} (기준모델 {best[base_name]['검증F1']:.3f}, "
        f"규칙 {max(rule_f1.values()):.3f})")
    return result


# ---------------------------------------------------------------- 모델 담당의 생각 (LLM): 탐색 계획
MAX_PER_MODEL, MAX_TOTAL = 12, 100


def _space_info(cfg, used) -> list[dict]:
    out = []
    for m in used:
        v = cfg["후보모델"][m]
        out.append({"모델": m, "설명": v.get("설명", ""), "표준화": v.get("표준화_필요"),
                    "고정설정": v.get("고정설정") or {}, "기본_탐색값": v.get("좁은범위") or {},
                    "허용_범위(최소,최대)": v.get("넓은범위") or {}})
    return out


def _plan_from(cfg, used, ans) -> tuple[dict | None, str]:
    """LLM 답 → 탐색 계획 {모델: [설정, ...]}. 허용 범위 밖·조합 과다면 무효."""
    if not isinstance(ans, dict) or not isinstance(ans.get("모델"), list):
        return None, "모델 목록 없음"
    models = [m for m in ans["모델"] if m in used]
    if cfg["기준모델"] not in models:
        return None, f"기준모델({cfg['기준모델']})이 빠짐"
    ranges = ans.get("범위") or {}
    plan, total = {}, 0
    for m in models:
        narrow, wide = cfg["후보모델"][m].get("좁은범위") or {}, cfg["후보모델"][m].get("넓은범위") or {}
        space = dict(narrow)
        for k, vals in (ranges.get(m) or {}).items():
            if k not in narrow and k not in wide:
                return None, f"{m}의 '{k}'는 탐색할 수 없는 설정"
            if not isinstance(vals, list) or not vals:
                return None, f"{m}의 '{k}' 값 목록이 비었음"
            fixed = []
            for v in vals:
                lo_hi = wide.get(k)
                if isinstance(v, (int, float)) and not isinstance(v, bool) and isinstance(lo_hi, list) and len(lo_hi) == 2 \
                        and all(isinstance(x, (int, float)) for x in lo_hi):
                    if not (lo_hi[0] <= v <= lo_hi[1]):
                        return None, f"{m}의 {k}={v}가 허용 범위 {lo_hi} 밖"
                    fixed.append(int(v) if all(isinstance(x, int) for x in lo_hi) else float(v))
                elif v in (narrow.get(k) or []):
                    fixed.append(v)
                else:
                    return None, f"{m}의 {k}={v}는 허용되지 않는 값"
            uniq = list(dict.fromkeys(fixed))
            space[k] = sorted(uniq) if all(isinstance(x, (int, float)) for x in uniq) else uniq
        grid = _grid(space)
        if len(grid) > MAX_PER_MODEL:
            return None, f"{m}의 조합이 {len(grid)}개로 {MAX_PER_MODEL}개 초과"
        plan[m], total = grid, total + len(grid)
    if total > MAX_TOTAL:
        return None, f"전체 조합 {total}개로 {MAX_TOTAL}개 초과"
    return plan, ""


def plan(state, ds: dict, recipe: dict) -> dict:
    """모델 담당: ① 탐색 계획 → 탐색 실행 → ② 1차 결과를 보고 범위 조정 → 최종 계획을 모든 레시피에 사용."""
    import llm
    from agents.reviewer import adoption_rule_text
    cfg = state.config
    used = [m for m, v in cfg["후보모델"].items() if is_on(v.get("사용")) and not optional_missing(m)]
    meta = ds["dev"]["meta"]
    groups = pd.Series(ds["groups"]).value_counts().to_dict()
    summary = {"학습_문제": {"3~5월": int((meta["crym"] < periods(cfg)["valid"][0]).sum()), "3~7월(최대)": int((meta["crym"] < periods(cfg)["dev_end"]).sum())},
               "검증_문제": int(meta["crym"].isin(periods(cfg)["valid"]).sum()), "증가_비율": round(float(meta["label"].mean()), 3),
               "변수_수": len(ds["cols"]), "변수_묶음별_개수": groups}
    rule = adoption_rule_text(cfg)
    default_plan = {m: _grid(cfg["후보모델"][m]["좁은범위"]) for m in used}

    def check(ans):
        p, why = _plan_from(cfg, used, ans)
        return p is not None, why

    data = {"데이터_요약": summary, "후보_모델": _space_info(cfg, used), "기준모델": cfg["기준모델"],
            "제한": f"모델당 조합 {MAX_PER_MODEL}개 이하, 전체 {MAX_TOTAL}개 이하. '범위'에 안 적은 모델은 기본_탐색값을 그대로 씀",
            "채택_규칙": rule + " 모델도 이 규칙으로 기준모델을 이겨야만 선택된다."}
    tools = {"변수_목록": {"인자": "", "설명": "지금 레시피의 변수 이름과 묶음",
                         "함수": lambda a: {c: ds["groups"][c] for c in ds["cols"]}}}
    ans, who = llm.ask(state, "모델 탐색 계획", "model_plan.md", data, {"모델": used, "범위": {}, "이유": "모든 후보 모델의 기본 탐색값 (규칙)"},
                       check, tools)
    state.search_plan = _plan_from(cfg, used, ans)[0] or default_plan
    n = sum(len(g) for g in state.search_plan.values())
    state.decide("③ 모델", "탐색 계획 (1차)", f"후보 모델 {len(used)}종", f"{len(state.search_plan)}종 · 조합 {n}개", who, ans.get("이유", ""))
    state.log(f"[3] 모델 담당 탐색 계획: {len(state.search_plan)}종 · 조합 {n}개 ({who})")
    res = evaluate(state, ds, recipe)

    if who.startswith("LLM"):       # 1차 결과를 보고 범위 조정 (1회)
        data2 = {**data, "지금_계획": {m: (ans.get("범위") or {}).get(m, "기본_탐색값") for m in state.search_plan},
                 "1차_결과_모델별_1등": res["모델별"].to_dict("records"), "선택된_모델": f"{res['모델']} {res['설정']}",
                 "규칙_검증F1": res["규칙F1"]}
        ans2, who2 = llm.ask(state, "모델 탐색 조정", "model_refine.md", data2, {"유지": True, "이유": "조정 없음"},
                             lambda a: (True, "") if isinstance(a, dict) and a.get("유지") is True else check(a))
        if not ans2.get("유지"):
            new = _plan_from(cfg, used, ans2)[0]
            if new and new != state.search_plan:
                state.search_plan = new
                n = sum(len(g) for g in new.values())
                state.decide("③ 모델", "탐색 계획 조정 (2차)", "1차 결과", f"{len(new)}종 · 조합 {n}개", who2, ans2.get("이유", ""))
                state.log(f"[3] 모델 담당 범위 조정: {len(new)}종 · 조합 {n}개 → 레시피 {recipe['이름']} 다시 검증")
                res = evaluate(state, ds, recipe)
        else:
            state.decide("③ 모델", "탐색 계획 조정 (2차)", "1차 결과", "유지", who2, ans2.get("이유", ""))
    return res


# ---------------------------------------------------------------- Optuna (선택)
def _optuna(cfg, used, recipe, X, y, fl, yv, seed, skip=()):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    tasks, probs = [], []
    for m in used:
        wide = cfg["후보모델"][m]["넓은범위"] or {}

        def objective(trial, m=m, wide=wide):
            p = {}
            for k, (lo, hi) in ((k, v) for k, v in wide.items() if isinstance(v, list) and len(v) == 2):
                if isinstance(lo, int) and isinstance(hi, int):
                    p[k] = trial.suggest_int(k, lo, hi, log=hi / max(lo, 1) >= 100)
                elif isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
                    p[k] = trial.suggest_float(k, float(lo), float(hi), log=lo > 0 and hi / lo >= 100)
            for k, v in (cfg["후보모델"][m]["좁은범위"] or {}).items():
                if k not in p:
                    p[k] = trial.suggest_categorical(k, v)
            pr = _eval_candidate(cfg, m, p, recipe, X, y, fl, seed, skip)
            trial.set_user_attr("proba", pr.tolist())
            return macro_f1(yv, pr >= 0.5)

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=cfg["탐색"]["optuna_모델당_시도"], timeout=cfg["탐색"]["모델당_시간상한_분"] * 60)
        for t in study.trials:
            if t.value is not None:
                tasks.append((m, t.params))
                probs.append(np.array(t.user_attrs["proba"]))
    return tasks, probs
