"""[5] 검토 담당.

check()  : 검증 점수만 보고 수정안을 하나씩 시험한다 (테스트 점수는 이 시점에 존재하지 않음).
           수정안 → 변수표 → 사전 점검(🟢만 통과) → 검증 → 채택 규칙을 넘으면 채택. 최대 '수정_최대횟수'.
report() : 노트에 적힌 내용을 결과 파일과 대시보드로 정리한다 (새로 고르는 것은 없음).
"""
from __future__ import annotations


import pandas as pd

import llm
from agents import data_agent, model_agent
from agents.planner import apply_change, change_label, describe, menu_changes
from common import adoption_test, write_csv, write_json


# ---------------------------------------------------------------- 처음 레시피 A·B 비교
def choose_initial(state):
    """A와 B를 검증하고, B가 채택 규칙을 넘을 때만 B로 시작한다."""
    ra, rb = state.results["A"], state.results["B"]
    t = adoption_test(ra["y"], ra["months"], ra["pred"], rb["pred"], state.config["판단규칙"], state.config["난수시드"])
    state.current = "B" if t["채택"] else "A"
    state.decide("① 계획", "처음 레시피 선택", f"A {ra['검증F1']:.3f} / B {rb['검증F1']:.3f}", state.current, "채택규칙", t["이유"])
    state.log(f"[5] 처음 레시피: {state.current} 로 시작 (B {'채택' if t['채택'] else '기각'}: {t['이유']})")


# ---------------------------------------------------------------- 수정 루프
def check(state):
    cfg, log = state.config, state.log
    rule = cfg["판단규칙"]
    seed = cfg["난수시드"]
    max_try, max_pc = rule["수정_최대횟수"], cfg["사전점검"]["최대횟수"]
    tried, n_try, n_pc = set(), 0, 0
    bundle_queue = _bundle_candidates(state)

    while n_try < max_try and n_pc < max_pc:
        cur = state.recipes[state.current]
        menu = [c for c in menu_changes(cfg, cur) if _key(c) not in tried]
        menu += [c for c in bundle_queue if _key(c) not in tried and len(cur["묶음"]) < cfg["소분류처리"]["묶음_최대"]]
        menu.sort(key=_priority)
        if not menu:
            log("[5] 더 시험할 수정안이 없습니다")
            break
        change, who, why = _pick(state, menu)
        tried.add(_key(change))
        name = f"R{len(state.revisions) + 1}"
        new = apply_change(cur, change, name)
        ds_new = data_agent.build(state, new)
        pc = data_agent.precheck(state, ds_new, state.datasets[state.current], change)
        n_pc += 1
        rec = {"번호": name, "기준레시피": state.current, "수정안": change_label(change), "결정자": who, "제안이유": why,
               "사전점검": pc["신호등"], "사전점검_이유": " / ".join(pc["이유"]), "결과": pc["판정"], "precheck": pc}
        state.prechecks.append({"번호": name, **{k: v for k, v in pc.items() if k not in ("분포",)}})
        if pc["신호등"] != "🟢":
            state.revisions.append(rec)
            state.decide("⑤ 검토", f"수정 시도 {name}: {change_label(change)}", "-", pc["판정"], who, " / ".join(pc["이유"]))
            log(f"[5] {name} {change_label(change)} → 사전 점검 {pc['신호등']} {pc['판정']}: {pc['이유'][-1] if pc['이유'] else ''}")
            continue
        state.recipes[name] = new
        state.datasets[name] = ds_new
        res = model_agent.evaluate(state, ds_new, new)
        n_try += 1
        base = state.results[state.current]
        t = adoption_test(base["y"], base["months"], base["pred"], res["pred"], rule, seed)
        rec.update({"결과": "채택" if t["채택"] else "기각", "기존F1": t["기존F1"], "새F1": t["새F1"], "채택판단": t["이유"]})
        state.revisions.append(rec)
        state.decide("⑤ 검토", f"수정 시도 {name}: {change_label(change)}", f"{t['기존F1']:.3f} → {t['새F1']:.3f}",
                     "채택" if t["채택"] else "기각", f"{who} → 채택규칙", t["이유"])
        log(f"[5] {name} {change_label(change)}: 검증 {t['기존F1']:.3f} → {t['새F1']:.3f} {'채택' if t['채택'] else '기각'} ({t['이유']})")
        if t["채택"]:
            state.current = name
            if change["종류"] != "묶음":     # 방금 채택한 수정을 바로 되돌리는 시도는 하지 않음
                old = cur["지역_단위"] if change["종류"] == "지역_단위" else cur[change["종류"]][change["항목"]]
                tried.add(f"{change['종류']}|{change['항목']}|{old}")
    log(f"[5] 수정 종료: 검증까지 간 시도 {n_try}회, 사전 점검 {n_pc}회 → 최종 레시피 {state.current} ({describe(state.recipes[state.current])})")
    return state


_ORDER = [("원본변수", "중분류_요약"), ("원본변수", "중분류_분해"), ("전처리", "미세변동_제외"), ("묶음", None),
          ("원본변수", "소분류_요약"), ("원본변수", "지역_요약"), ("원본변수", "지역_분해"), ("지역_단위", None),
          ("변수", None), ("원본변수", "원본행수"), ("전처리", None)]


def _priority(c: dict) -> int:
    """규칙 모드의 시험 순서. 계획서 4-2(③ 중분류 → ④ 미세변동) 다음 묶음·지역.
    기본 변수(① 객단가·업종비중·달력)는 처음 레시피 B에서 이미 한꺼번에 시험했으므로 뒤로 보낸다."""
    for i, (kind, item) in enumerate(_ORDER):
        if c["종류"] == kind and (item is None or c.get("항목") == item):
            return i
    return len(_ORDER)


def _key(c: dict) -> str:
    if c["종류"] == "묶음":
        b = c["묶음"]
        return f"묶음|{b['처리']}|{'|'.join(sorted(b['소분류']))}"
    return f"{c['종류']}|{c['항목']}|{c['값']}"


def _bundle_candidates(state) -> list[dict]:
    """규칙 모드의 소분류 묶음 후보: ① 변동 주도 소분류 → 분리, ② 힌트 목록 → 제외, 분리."""
    cfg = state.config
    st = data_agent.subcat_stats(state)
    have = set(st["소분류"])
    out = []
    th = cfg["소분류처리"]["변동주도_기준"]
    drv = st[(st["대분류변화기여도"] >= th) & (st["금액비중"] >= cfg["사전점검"]["최소변화비중"])]["소분류"].tolist()
    if drv:
        out.append({"종류": "묶음", "묶음": {"이름": "변동주도", "소분류": drv, "처리": "분리",
                                          "이유": f"1~5월 같은 대분류 안 월 변화의 {th:.0%} 이상을 만든 소분류"}})
    for name, items in (cfg["소분류처리"].get("힌트") or {}).items():
        subs = [s for s in items if s in have]
        if subs:
            for how in ("분리", "제외"):
                out.append({"종류": "묶음", "묶음": {"이름": name, "소분류": subs, "처리": how, "이유": f"설정의 힌트 목록 '{name}'"}})
    return out


def _pick(state, menu: list[dict]):
    """다음 수정안 고르기. 규칙 모드는 목록 순서(계획서 4-2), LLM 모드는 LLM이 메뉴에서 고르거나 묶음을 만든다."""
    default = {"번호": 0, "이유": "계획서 4-2 순서대로 아직 안 해 본 첫 수정안 (규칙)"}
    st = data_agent.subcat_stats(state)
    have = set(st["소분류"])
    cur = state.results[state.current]
    data = {"지금_레시피": describe(state.recipes[state.current]), "지금_검증F1": cur["검증F1"], "월별F1": cur["월별F1"],
            "지난_시도": [{k: r.get(k) for k in ["수정안", "사전점검", "사전점검_이유", "결과", "기존F1", "새F1"]} for r in state.revisions],
            "남은_수정안": [{"번호": i, "수정안": change_label(c)} for i, c in enumerate(menu)],
            "소분류_통계": st.sort_values("대분류변화기여도", ascending=False).head(60).to_dict("records"),
            "묶음_최대": state.config["소분류처리"]["묶음_최대"]}

    def ok(ans):
        if not isinstance(ans, dict):
            return False, "형식 오류"
        if "묶음" in ans:
            b = ans["묶음"]
            if not isinstance(b, dict) or b.get("처리") not in ("제외", "분리"):
                return False, "묶음 처리 방법이 제외/분리가 아님"
            subs = [s for s in b.get("소분류", []) if s in have]
            if not subs:
                return False, "목록에 있는 소분류가 하나도 없음"
            b["소분류"] = subs
            b["이름"] = str(b.get("이름") or "LLM묶음")[:20]
            return True, ""
        return (isinstance(ans.get("번호"), int) and 0 <= ans["번호"] < len(menu)), "번호가 목록 밖"

    ans, who = llm.ask(state, "다음 수정안", "reviewer.md", data, default, ok)
    if "묶음" in ans:
        b = dict(ans["묶음"]); b["이유"] = ans.get("이유", "")
        return {"종류": "묶음", "묶음": b}, who, ans.get("이유", "")
    c = menu[ans["번호"]]
    why = ans.get("이유") or (c["묶음"].get("이유") if c["종류"] == "묶음" else "")
    return c, who, why


# ---------------------------------------------------------------- 보고
def report(state):
    """결과 파일·대시보드 작성. 집계값만 쓴다."""
    import dashboard
    d, cfg = state.run_dir, state.config
    write_csv(pd.DataFrame(state.decisions), d / "decisions.csv")
    write_csv(pd.DataFrame([{**{k: v for k, v in t.items() if k != "월별F1"}, "설정": str(t["설정"]),
                             **{f"{m}월": f for m, f in t["월별F1"].items()}} for t in state.trials]), d / "trials.csv")
    vs = pd.concat([r["모델별"].assign(레시피=name) for name, r in state.results.items()], ignore_index=True)
    write_csv(vs, d / "valid_scores.csv")
    write_csv(pd.DataFrame([{k: v for k, v in r.items() if k != "precheck"} for r in state.revisions]), d / "revisions.csv")
    if state.test:
        write_csv(pd.DataFrame(state.test["표"]), d / "test_scores.csv")
        write_csv(state.predictions, d / "predictions.csv")
        write_csv(pd.DataFrame(state.monthly["표"]), d / "monthly_f1.csv")
        write_csv(state.test["변수영향"], d / "feature_importance.csv")
    insights = _insights(state)
    write_csv(insights["업종월별"], d / "insight_category_monthly.csv")
    write_csv(insights["행사전후"], d / "insight_events.csv")
    recipes = {k: {**v, "요약": describe(v)} for k, v in state.recipes.items()}
    trace = {"실행": d.name, "시작": state.started_at, "LLM모드": llm.mode(state), "탐색방법": cfg["탐색"]["방법"],
             "레시피": recipes, "처음레시피": state.first_recipe, "최종레시피": state.current,
             "검증횟수": state.n_validations, "비교한_후보수": len(state.trials),
             "수정시도": [{k: v for k, v in r.items() if k != "precheck"} for r in state.revisions],
             "결정기록": state.decisions, "원본점검": state.quality,
             "테스트": {k: v for k, v in (state.test or {}).items() if k not in ("표", "변수영향", "_preds")}}
    write_json(trace, d / "agent_trace.json")
    llm_out = {}
    for c in state.llm_calls:
        llm_out.setdefault(c["목적"], []).append(c)
    write_json(llm_out, d / "llm_proposals.json")
    _append_all_runs(state)
    dashboard.build(state, insights)
    dashboard.build_index(d.parent.parent)
    state.log(f"[5] 보고서: {d / 'report.html'}")
    return state


def _insights(state) -> dict:
    """업종별 인사이트 (계획서 1-4): 업종별 월 지출 추이·증가율, 명절·소비쿠폰 전후 비교."""
    cfg = state.config
    t = state.truth
    cm = t.groupby(["cat_l", "crym"])["am"].sum().unstack("crym")
    growth = cm.pct_change(axis=1) * 100
    tab = cm.stack().rename("금액").to_frame().join(growth.stack().rename("전월대비%")).reset_index()
    from common import month_add
    ms = sorted(t["crym"].unique())
    cal = data_agent._calendar(cfg, [month_add(ms[0], -1)] + ms)
    # 기준월 t의 다음 달이 명절/쿠폰이면 t+1이 행사월 → 행사월 = t+1
    event = {month_add(m, 1): row for m, row in cal.iterrows()}
    fest = [m for m, r in event.items() if r["명절"] > 0 and m in cm.columns]
    coup = [m for m, r in event.items() if r["소비쿠폰"] > 0 and m in cm.columns]
    rows = []
    for c in cm.index:
        g = growth.loc[c]
        rows.append({"대분류": c,
                     "명절월_전월대비%평균": round(float(g[fest].mean()), 2) if fest else None,
                     "그외월_전월대비%평균": round(float(g.drop(fest, errors="ignore").mean()), 2),
                     "쿠폰기간_월평균금액": float(cm.loc[c, coup].mean()) if coup else None,
                     "쿠폰전_월평균금액": float(cm.loc[c, [m for m in cm.columns if coup and m < min(coup)]].mean()) if coup else None})
    ev = pd.DataFrame(rows)
    if coup:
        ev["쿠폰기간_증감%"] = ((ev["쿠폰기간_월평균금액"] / ev["쿠폰전_월평균금액"] - 1) * 100).round(2)
    return {"업종월별": tab, "행사전후": ev, "명절월": fest, "쿠폰월": coup}


def _append_all_runs(state):
    root = state.run_dir.parent.parent
    path = root / "all_runs.csv"
    row = {"실행": state.run_dir.name, "LLM모드": llm.mode(state), "탐색": state.config["탐색"]["방법"],
           "최종레시피": describe(state.recipes[state.current]), "모델": state.results[state.current]["모델"],
           "검증F1": state.results[state.current]["검증F1"],
           "테스트F1": state.test["최종"]["macro_f1"] if state.test else None,
           "성공판정": state.test["성공"] if state.test else None, "검증횟수": state.n_validations}
    old = pd.read_csv(path, encoding="utf-8-sig") if path.exists() else pd.DataFrame()
    write_csv(pd.concat([old, pd.DataFrame([row])], ignore_index=True), path)
