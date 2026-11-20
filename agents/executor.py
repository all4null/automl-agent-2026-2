"""[4] 실행 담당: 테스트를 단 한 번 연다.

1. 최종 레시피의 선택 모델로 3~8월(학습+검증 기간) 재학습 → 9~11월 테스트 144문제 채점
2. 처음 레시피도 같은 방식으로 채점해 나란히 보고 (고르는 데 쓰지 않음. 검증 과적합 점검용)
3. 월별 갱신 실험 (계획서 5-3): 고정 / 확장 재학습 / SGD 순차 갱신. 점수는 항상 갱신 전에 기록
4. 예측표·오류 분석·변수 영향
"""
from __future__ import annotations

import copy
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support

from agents import bottomup
from agents.data_agent import periods
from agents.model_agent import LOOKUP_MODELS, make_pipeline, onehot_idx, proba1
from common import SafeError, bootstrap_f1, cfg_path, macro_f1, month_add, write_csv


def _train_rows(meta, recipe, extra_mask=None):
    tr = np.ones(len(meta), bool) if extra_mask is None else extra_mask.copy()
    if recipe["전처리"]["미세변동_제외"]:
        tr &= meta["rate_next"].abs().to_numpy() > 0.02
    return tr


def _fit(cfg, res, recipe, X, y, seed, skip=()):
    """선택된 모델(또는 앙상블 구성 모델들)을 학습해 돌려준다."""
    if res["모델"] == "앙상블":
        return [make_pipeline(cfg, m, p, recipe, seed, skip).fit(X, y) for m, p in res["설정"]["구성"]]
    return [make_pipeline(cfg, res["모델"], res["설정"], recipe, seed, skip).fit(X, y)]


def _proba(pipes, X):
    return np.mean([proba1(p, X) for p in pipes], axis=0)


def run(state):
    if state.test is not None:
        raise SafeError("테스트는 이미 열렸습니다. 실행 1회에 테스트는 한 번만 엽니다.")
    cfg, log = state.config, state.log
    seed, P = cfg["난수시드"], periods(cfg)
    n_boot = cfg["평가"]["부트스트랩_횟수"]
    final, first = state.current, state.first_recipe
    state.exec_plan = _plan_monthly(state)          # 실행 담당: 테스트를 열기 전에 월별 갱신 실험 계획을 정함
    log(f"[4] 테스트 개봉: 최종 레시피 {final}" + (f", 비교용 처음 레시피 {first}" if first != final else "")
        + f" (채점 방식: {(cfg.get('평가') or {}).get('테스트_방식', '고정')})")

    scored, models = {}, {}
    for name in dict.fromkeys([final, first]):
        ds, res, recipe = state.datasets[name], state.results[name], state.recipes[name]
        Xd, md = ds["dev"]["X"].to_numpy(float), ds["dev"]["meta"]
        Xt, mt = ds["sealed"]["X"].to_numpy(float), ds["sealed"]["meta"]       # 봉인 해제 (여기서만)
        yd = md["label"].to_numpy().astype(int)
        tr = _train_rows(md, recipe)
        pipes = _fit(cfg, res, recipe, Xd[tr], yd[tr], seed, onehot_idx(ds))
        models[name] = pipes
        if (cfg.get("평가") or {}).get("테스트_방식") == "매달재학습":
            # test 달마다 그때까지 확정된 정답(3월 ~ 지난달)으로 다시 학습해 그 달만 예측 (실제 운영과 같은 방식)
            Xa = np.vstack([Xd, Xt])
            ma = pd.concat([md, mt], ignore_index=True)
            ya = ma["label"].to_numpy().astype(int)
            ca = ma["crym"].to_numpy()
            p = np.zeros(len(mt))
            for m in P["test"]:
                trm = _train_rows(ma, recipe, ca < m)
                k = mt["crym"].to_numpy() == m
                p[k] = _proba(_fit(cfg, res, recipe, Xa[trm], ya[trm], seed, onehot_idx(ds)), Xt[k])
            scored[name] = p
        else:
            scored[name] = _proba(pipes, Xt)

    ds, res, recipe = state.datasets[final], state.results[final], state.recipes[final]
    mt, md = ds["sealed"]["meta"], ds["dev"]["meta"]
    yt = mt["label"].to_numpy().astype(int)
    months = mt["crym"].to_numpy()
    preds = {f"최종 ({final}: {res['모델']})": (scored[final] >= 0.5).astype(int)}
    if first != final:
        preds[f"처음 ({first}: {state.results[first]['모델']})"] = (scored[first] >= 0.5).astype(int)
    preds["규칙1 다수방향"] = np.full(len(yt), int(md["label"].mean() >= 0.5))
    preds["규칙2 직전방향유지"] = mt["last_dir"].to_numpy()

    rows = []
    for k, p in preds.items():
        b = bootstrap_f1(yt, p, n_boot, seed)
        rows.append({"모델": k, "macro_f1": round(macro_f1(yt, p), 4),
                     "95%구간_하": round(float(np.percentile(b, 2.5)), 4), "95%구간_상": round(float(np.percentile(b, 97.5)), 4),
                     "balanced_acc": round(balanced_accuracy_score(yt, p), 4), "정답률": round(float((yt == p).mean()), 4),
                     **{f"{m}월": round(macro_f1(yt[months == m], p[months == m]), 4) for m in P["test"]}, "n": len(yt)})
    fkey = next(iter(preds))
    fm = rows[0]
    rule_rows = [r for r in rows if r["모델"].startswith("규칙")]
    win_months = sum(fm[f"{m}월"] > max(r[f"{m}월"] for r in rule_rows) for m in P["test"])
    pf = preds[fkey]
    cm = confusion_matrix(yt, pf, labels=[0, 1])
    pr, rc, f1s, sup = precision_recall_fscore_support(yt, pf, labels=[0, 1], zero_division=0)
    first_f1 = rows[1]["macro_f1"] if first != final else fm["macro_f1"]

    # 예측표 (계획서 1-4)
    pt = pd.DataFrame({
        "성별": mt["sex"].map({str(k): v for k, v in cfg["성별"].items()}), "연령구간": mt["age_band"], "업종대분류": mt["cat_l"],
        "기준월": mt["crym"], "예측대상월": mt["crym"].map(lambda m: month_add(m, 1)),
        "증가확률": scored[final].round(4), "예측방향": pf, "실제방향": yt, "맞음": (pf == yt).astype(int),
        "모델버전": f"{final}:{res['모델']}:{res['설정']}@{P['first']}~{P['dev_end']}"})
    state.predictions = pt

    state.test = {
        "최종": {"레시피": final, "모델": res["모델"], "설정": str(res["설정"]), "macro_f1": fm["macro_f1"],
                 "검증F1": res["검증F1"], "검증_테스트_차": round(res["검증F1"] - fm["macro_f1"], 4)},
        "처음": {"레시피": first, "macro_f1": first_f1},
        "표": rows, "성공월수": int(win_months), "성공": bool(win_months >= 2),
        "과적합경고": bool(first != final and fm["macro_f1"] < first_f1),
        "혼동행렬": {"실제감소_예측감소": int(cm[0, 0]), "실제감소_예측증가": int(cm[0, 1]),
                     "실제증가_예측감소": int(cm[1, 0]), "실제증가_예측증가": int(cm[1, 1])},
        "클래스별": {"감소": {"precision": round(pr[0], 4), "recall": round(rc[0], 4)},
                     "증가": {"precision": round(pr[1], 4), "recall": round(rc[1], 4)}},
        "오류분석": _errors(state, pt), "사례": _cases(state, pt),
        "변수영향": _importance(state, models[final], ds, res, recipe, yt), "_preds": preds,
    }
    _test_log(state)
    log(f"[4] 테스트 Macro-F1 {fm['macro_f1']:.3f} (95% {fm['95%구간_하']:.3f}~{fm['95%구간_상']:.3f}), "
        f"규칙 {max(r['macro_f1'] for r in rule_rows):.3f}, 두 규칙을 이긴 달 {win_months}/{len(P['test'])} → "
        f"{'성공' if win_months >= 2 else '실패'}" + (", 처음 레시피보다 낮음: 검증 과적합 의심" if state.test["과적합경고"] else ""))
    _monthly(state)
    state.test_interpretation = _interpret(state)   # 실행 담당: 테스트 해석 (다시 고르는 데 쓰지 않음)
    return state


def _test_log(state):
    path = cfg_path(state.config, "결과폴더") / "test_log.csv"
    t = state.test["최종"]
    row = pd.DataFrame([{"시각": datetime.now().isoformat(timespec="seconds"), "실행": state.run_dir.name,
                         "레시피": t["레시피"], "모델": t["모델"], "테스트F1": t["macro_f1"]}])
    old = pd.read_csv(path, encoding="utf-8-sig") if path.exists() else pd.DataFrame()
    write_csv(pd.concat([old, row], ignore_index=True), path)


def _errors(state, pt) -> dict:
    """고객층·업종별 정답률 (셀당 3문제라 참고용), 명절·쿠폰이 있는 예측대상월의 정답률."""
    from agents.data_agent import _calendar
    seg = (pt["성별"] + "_" + pt["연령구간"])
    by_seg = pt.groupby(seg)["맞음"].mean().round(3).to_dict()
    by_cat = pt.groupby("업종대분류")["맞음"].mean().round(3).to_dict()
    grid = pt.assign(고객층=seg).pivot_table(index="고객층", columns="업종대분류", values="맞음", aggfunc="mean").round(2)
    cal = _calendar(state.config, sorted(pt["기준월"].unique()))
    pt2 = pt.assign(명절=pt["기준월"].map(cal["명절"]), 쿠폰=pt["기준월"].map(cal["소비쿠폰"]))
    ev = {f"{k}_{'있음' if v else '없음'}": round(float(g["맞음"].mean()), 3)
          for k in ["명절", "쿠폰"] for v, g in pt2.groupby(k)}
    return {"고객층별": by_seg, "업종별": by_cat, "고객층x업종": grid.reset_index().to_dict("records"), "행사월": ev}


def _cases(state, pt) -> dict:
    """성공 3건·실패 3건: 확신이 가장 컸던 사례 (OT 최종보고서 항목). 고객층·업종·월 단위 집계값만."""
    t = state.truth.set_index(["crym", "sex", "age_band", "cat_l"])
    d = pt.assign(확신=(pt["증가확률"] - 0.5).abs())
    sex_code = {v: k for k, v in state.config["성별"].items()}

    def info(r):
        k = (r["기준월"], sex_code[r["성별"]], r["연령구간"], r["업종대분류"])
        row = t.loc[k]
        prev = (row["am"] / row["am_prev"] - 1) * 100 if row["am_prev"] else None
        return {"고객층": f"{r['성별']}_{r['연령구간']}", "업종": r["업종대분류"], "기준월": int(r["기준월"]),
                "증가확률": float(r["증가확률"]), "예측": "증가" if r["예측방향"] else "감소", "실제": "증가" if r["실제방향"] else "감소",
                "이번달_전월대비%": round(prev, 1) if prev is not None else None,
                "다음달_실제변화%": round(float(row["rate_next"]) * 100, 1)}

    ok = d[d["맞음"] == 1].nlargest(3, "확신")
    bad = d[d["맞음"] == 0].nlargest(3, "확신")
    return {"성공": [info(r) for _, r in ok.iterrows()], "실패": [info(r) for _, r in bad.iterrows()]}


def _importance(state, pipes, ds, res, recipe, yt) -> pd.DataFrame:
    """변수 영향: 순열 중요도(최종 모델, 테스트), 로지스틱 계수, LightGBM 중요도 (계획서 1-4)."""
    cfg, seed = state.config, state.config["난수시드"]
    cols = ds["cols"]
    Xt = ds["sealed"]["X"].to_numpy(float)
    Xd, md = ds["dev"]["X"].to_numpy(float), ds["dev"]["meta"]
    yd = md["label"].to_numpy().astype(int)
    out = pd.DataFrame({"변수": cols, "묶음": [ds["groups"][c] for c in cols]})
    if res["모델"] in LOOKUP_MODELS:    # 중분류합산·달력방향은 행 값을 섞으면 문제를 찾을 수 없어 순열 중요도를 못 냄
        out["순열중요도(테스트)"] = np.nan
    else:
        pi = permutation_importance(pipes[0], Xt, yt, scoring="f1_macro", n_repeats=10, random_state=seed, n_jobs=1)
        out["순열중요도(테스트)"] = pi.importances_mean.round(4)
    bbm = res["best_by_model"]
    if "로지스틱회귀" in bbm:
        lr = make_pipeline(cfg, "로지스틱회귀", bbm["로지스틱회귀"]["설정"], recipe, seed, onehot_idx(ds)).fit(Xd, yd)
        out["로지스틱계수(표준화)"] = lr[-1].coef_[0].round(4)
    if "LightGBM" in bbm:
        lg = make_pipeline(cfg, "LightGBM", bbm["LightGBM"]["설정"], recipe, seed, onehot_idx(ds)).fit(Xd, yd)
        imp = lg[-1].booster_.feature_importance(importance_type="gain")
        out["LightGBM중요도(gain)"] = (imp / max(imp.sum(), 1e-9)).round(4)
    return out.sort_values("순열중요도(테스트)", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- 월별 갱신 실험 (계획서 5-3)
def _monthly(state):
    cfg, log = state.config, state.log
    seed, P = cfg["난수시드"], periods(cfg)
    final = state.current
    ds, res, recipe = state.datasets[final], state.results[final], state.recipes[final]
    X = np.vstack([ds["dev"]["X"].to_numpy(float), ds["sealed"]["X"].to_numpy(float)])
    meta = pd.concat([ds["dev"]["meta"], ds["sealed"]["meta"]], ignore_index=True)
    y = meta["label"].to_numpy().astype(int)
    crym = meta["crym"].to_numpy()
    dev = crym <= P["dev_end"]
    rows, forget, drift = [], [], []

    # 고정: 3~8월로 한 번 학습 (테스트 예측 그대로)
    p_fixed = state.test["_preds"][next(iter(state.test["_preds"]))]
    for m in P["test"]:
        k = crym[~dev] == m
        rows.append({"방식": "고정", "모델": res["모델"], "월": int(m), "macro_f1": round(macro_f1(y[~dev][k], p_fixed[k]), 4)})

    # 확장 재학습: 매달 누적 기간으로 처음부터 다시 학습
    bbm = res["best_by_model"]
    plan = state.exec_plan or {"확장재학습_모델": cfg["월별갱신"]["확장재학습_모델"], "순차갱신": True}
    names = [res["모델"]] + [m for m in plan["확장재학습_모델"] if m in bbm and m != res["모델"]]
    for name in names:
        for m in P["test"]:
            tr = _train_rows(meta, recipe, crym < m)
            if name == res["모델"]:
                pipes = _fit(cfg, res, recipe, X[tr], y[tr], seed, onehot_idx(ds))
            else:
                pipes = [make_pipeline(cfg, name, bbm[name]["설정"], recipe, seed, onehot_idx(ds)).fit(X[tr], y[tr])]
            k = crym == m
            rows.append({"방식": "확장재학습", "모델": name, "월": int(m),
                         "macro_f1": round(macro_f1(y[k], _proba(pipes, X[k]) >= 0.5), 4)})

    # 순차 갱신: SGD가 새 달 데이터만으로 조금씩 고침. 갱신 전 점수 기록 + 지식 손실 점검
    if "SGD" in bbm and plan.get("순차갱신", True):
        th = cfg["월별갱신"]["지식손실_기준"]
        tr0 = _train_rows(meta, recipe, dev)
        pipe = make_pipeline(cfg, "SGD", bbm["SGD"]["설정"], recipe, seed, onehot_idx(ds)).fit(X[tr0], y[tr0])
        pre, clf = pipe[:-1], pipe[-1]
        rng = np.random.default_rng(seed)
        for m in P["test"]:
            k = crym == m
            rows.append({"방식": "순차갱신", "모델": "SGD", "월": int(m),
                         "macro_f1": round(macro_f1(y[k], proba1(pipe, X[k]) >= 0.5), 4)})   # 갱신 전에 기록
            before = macro_f1(y[dev], proba1(pipe, X[dev]) >= 0.5)
            old = copy.deepcopy(clf)
            Xm = pre.transform(X[k])
            clf.partial_fit(Xm, y[k])
            after = macro_f1(y[dev], proba1(pipe, X[dev]) >= 0.5)
            rec = {"갱신월": int(m), "갱신전_과거F1": round(before, 4), "갱신후_과거F1": round(after, 4),
                   "지식손실": bool(before - after >= th)}
            if rec["지식손실"]:       # 대응 ①: 학습률 낮추기, ②: 과거 표본 섞기 (계획서 5-3)
                a = copy.deepcopy(old); a.set_params(eta0=a.eta0 * 0.1); a.partial_fit(Xm, y[k])
                rec["대응_eta0낮춤_과거F1"] = round(macro_f1(y[dev], _clf_pred(pre, a, X[dev])), 4)
                b = copy.deepcopy(old)
                past = rng.choice(np.where(dev)[0], size=min(k.sum(), dev.sum()), replace=False)
                b.partial_fit(np.vstack([Xm, pre.transform(X[past])]), np.concatenate([y[k], y[past]]))
                rec["대응_과거섞기_과거F1"] = round(macro_f1(y[dev], _clf_pred(pre, b, X[dev])), 4)
            forget.append(rec)

    # 분포 변화: 매월 증가 비율과 주요 변수 평균을 학습 구간과 비교
    cols = ds["cols"]
    base = {"증가비율": float(y[dev].mean())}
    for c in ["최근월_금액", "전월대비_증감률"]:
        if c in cols:
            base[c] = float(X[dev, cols.index(c)].mean())
    for m in P["test"]:
        k = crym == m
        r = {"월": int(m), "증가비율": round(float(y[k].mean()), 3), "학습구간_증가비율": round(base["증가비율"], 3)}
        for c in ["최근월_금액", "전월대비_증감률"]:
            if c in cols:
                r[f"{c}_평균"] = round(float(X[k, cols.index(c)].mean()), 4)
                r[f"{c}_학습구간평균"] = round(base[c], 4)
        drift.append(r)
    state.monthly = {"표": rows, "지식손실": forget, "분포변화": drift}
    t = pd.DataFrame(rows).pivot_table(index=["방식", "모델"], values="macro_f1", aggfunc="mean").round(3)
    log("[4] 월별 갱신 실험 (테스트 3개월 평균 F1): " + ", ".join(f"{a}/{b} {v:.3f}" for (a, b), v in t["macro_f1"].items()))


def _clf_pred(pre, clf, X):
    p = clf.predict_proba(pre.transform(X))[:, list(clf.classes_).index(1)]
    return (p >= 0.5).astype(int)


# ---------------------------------------------------------------- 실행 담당의 생각 (LLM)
def _plan_monthly(state) -> dict:
    """테스트를 열기 전에, 월별 갱신 실험에서 어떤 모델을 비교할지 정한다 (검증 결과만 봄)."""
    import llm
    cfg = state.config
    res = state.results[state.current]
    bbm = res["best_by_model"]
    data = {"최종_레시피": state.current, "최종_모델": f"{res['모델']} {res['설정']}",
            "모델별_검증F1": {m: v["검증F1"] for m, v in bbm.items()},
            "실험_설명": "테스트 3개월 동안 매달 '예측 → 점수 기록 → 정답 공개 → 갱신'. "
                         "확장재학습 = 누적 데이터로 매달 처음부터 다시 학습, 순차갱신 = SGD가 새 달 데이터만 추가 학습",
            "설정_기본값": {"확장재학습_모델": cfg["월별갱신"]["확장재학습_모델"], "순차갱신": "SGD" in bbm},
            "제한": "확장재학습_모델은 모델별_검증F1에 있는 모델 중 최대 4개 (최종 모델은 항상 포함)"}

    def ok(a):
        good = (isinstance(a, dict) and isinstance(a.get("확장재학습_모델"), list) and len(a["확장재학습_모델"]) <= 4
                and all(m in bbm for m in a["확장재학습_모델"]) and isinstance(a.get("순차갱신"), bool))
        return good, "" if good else "모델 이름이 목록 밖이거나 형식 오류"

    default = {"확장재학습_모델": [m for m in cfg["월별갱신"]["확장재학습_모델"] if m in bbm], "순차갱신": "SGD" in bbm,
               "이유": "설정 기본값 (규칙)"}
    ans, who = llm.ask(state, "실행 계획", "exec_plan.md", data, default, ok)
    state.decide("④ 실행", "월별 갱신 실험 계획 (테스트 전)", ", ".join(bbm),
                 f"확장재학습 {ans['확장재학습_모델']}, 순차갱신 {'O' if ans['순차갱신'] else 'X'}", who, ans.get("이유", ""))
    return ans


def _interpret(state) -> dict | None:
    """테스트를 연 뒤 결과를 해석한다. 모든 선택이 끝난 뒤라 테스트 점수를 봐도 결과에 영향이 없다."""
    import llm
    t, res = state.test, state.results[state.current]
    months = [k for k in t["표"][0] if k.endswith("월") and k[:-1].isdigit()]
    final_row = t["표"][0]
    rules = [r for r in t["표"] if r["모델"].startswith("규칙")]
    # 월별 비교는 코드가 계산해서 준다 (LLM이 표를 잘못 읽지 않게)
    compare = {m: {"최종": final_row[m], **{r["모델"]: r[m] for r in rules},
                   "두_규칙_모두_이김": all(final_row[m] > r[m] for r in rules)} for m in months}
    wins = {r["모델"]: sum(final_row[m] > r[m] for m in months) for r in rules}
    tf, vf = t["최종"]["macro_f1"], t["최종"]["검증F1"]
    mon = pd.DataFrame(state.monthly["표"]).groupby(["방식", "모델"])["macro_f1"].mean().round(3)
    fixed = mon.xs("고정", level="방식").iloc[0] if "고정" in mon.index.get_level_values(0) else None
    worst_cat = min(t["오류분석"]["업종별"].items(), key=lambda kv: kv[1])
    cm = t["혼동행렬"]
    facts = [  # 해석이 이 사실과 모순되지 않게, 코드가 문장으로 만들어 준다
        f"최종 모델은 규칙1(다수방향)을 {wins.get('규칙1 다수방향', 0)}/{len(months)}개월, 규칙2(직전방향유지)를 "
        f"{wins.get('규칙2 직전방향유지', 0)}/{len(months)}개월 이겼다. 두 규칙을 모두 이긴 달은 {t['성공월수']}개월이고, "
        f"성공 판정은 '{'성공' if t['성공'] else '실패'}'이다.",
        f"테스트 Macro-F1은 {tf:.3f} (95% 구간 {final_row['95%구간_하']:.3f}~{final_row['95%구간_상']:.3f}), 검증 Macro-F1은 {vf:.3f}이다. "
        f"테스트가 검증보다 {abs(tf - vf):.3f} {'높다' if tf > vf else '낮다' if tf < vf else '같다'}.",
        (f"처음 레시피({t['처음']['레시피']})의 테스트 점수는 {t['처음']['macro_f1']:.3f}로 최종과 같은 레시피다."
         if t["처음"]["레시피"] == t["최종"]["레시피"] else
         f"처음 레시피({t['처음']['레시피']}) 테스트 {t['처음']['macro_f1']:.3f}, 최종 레시피({t['최종']['레시피']}) 테스트 {tf:.3f}. "
         f"과적합 경고는 '{'있음' if t['과적합경고'] else '없음'}'."),
        f"혼동행렬: 실제 증가인데 감소로 예측 {cm['실제증가_예측감소']}건, 실제 감소인데 증가로 예측 {cm['실제감소_예측증가']}건.",
        "월별 갱신(테스트 3개월 평균 F1): " + ", ".join(f"{a}/{b} {v:.3f}" for (a, b), v in mon.items())
        + (f" (고정 모델 {fixed:.3f}과 비교)" if fixed is not None else ""),
        f"업종별 정답률이 가장 낮은 곳은 {worst_cat[0]} ({worst_cat[1]:.3f})이다 (셀이 작아 참고용).",
    ]
    data = {"확인된_사실(코드가_계산)": facts, "월별_규칙_비교(코드가_계산)": compare, "규칙별_이긴_달수(코드가_계산)": wins,
            "테스트_표": t["표"], "최종": t["최종"], "처음": t["처음"], "성공판정": t["성공"], "성공월수": t["성공월수"],
            "과적합경고": t["과적합경고"], "혼동행렬": t["혼동행렬"], "클래스별": t["클래스별"],
            "검증_월별F1": res["월별F1"], "월별갱신": state.monthly["표"], "지식손실": state.monthly["지식손실"],
            "분포변화": state.monthly["분포변화"], "업종별_정답률": t["오류분석"]["업종별"], "고객층별_정답률": t["오류분석"]["고객층별"]}

    def ok(a):
        good = isinstance(a, dict) and isinstance(a.get("해석"), str) and a["해석"].strip()
        return bool(good), "" if good else "해석 없음"

    ans, who = llm.ask(state, "테스트 해석", "test_interpret.md", data, None, ok)
    if not ans:
        return None
    state.decide("④ 실행", "테스트 해석 (테스트 후, 선택에 쓰지 않음)", "-", "보고서에 기록", who, ans["해석"][:200])
    return {**ans, "결정자": who, "모델": llm.model_name(state, "실행")}
