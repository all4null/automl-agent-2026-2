"""[5] 검토 담당.

check()  : 검증 점수만 보고 수정안을 하나씩 시험한다 (테스트 점수는 이 시점에 존재하지 않음).
           수정안 → 변수표 → 사전 점검(🟢만 통과) → 검증 → 채택 규칙을 넘으면 채택. 최대 '수정_최대횟수'.
report() : 노트에 적힌 내용을 결과 파일과 대시보드로 정리한다 (새로 고르는 것은 없음).
"""
from __future__ import annotations


import pandas as pd

import llm
from agents import data_agent, model_agent
from agents.planner import DESC as DESC_VARS, apply_change, change_desc, change_label, describe, menu_changes
from common import adoption_test, write_csv, write_json


# ---------------------------------------------------------------- 진단·평가 설계 (모델을 돌리기 전)
def diagnose(state):
    """데이터 구조를 진단하고 이번 실행의 평가 방식·개선안을 정한다. 모델 점수를 보기 전에 한 번만 정한다."""
    import math
    from agents import diagnosis
    cfg = state.config
    if not llm.is_on((cfg.get("진단") or {}).get("사용", "예")):
        return state
    d = diagnosis.run_all(state)
    default = diagnosis.rule_decision(cfg, d)
    var_menu = [k for k, v in cfg["변수"].items() if llm.is_on(v.get("시험")) and v.get("처음값") != "항상"]
    models = [m for m, v in cfg["후보모델"].items() if llm.is_on(v.get("사용"))]
    opts = [o["검증월_수"] for o in d["평가_선택지"]]
    data = {"진단": d, "변수_메뉴": {k: DESC_VARS.get(k, "") for k in var_menu}, "후보모델_목록": models,
            "기준모델": cfg["기준모델"], "규칙의_기본_판단": default}
    tools = {"업종별_달효과": {"인자": "", "설명": "업종마다 '다음 달이 더 김' 여부에 따른 증가 비율 (기준월 ~ validation 마지막 달)",
                            "함수": lambda a: diagnosis.category_month_effect(state)}}

    def ok(a):
        if not isinstance(a, dict):
            return False, "형식 오류"
        if a.get("검증월_수") not in opts:
            return False, f"검증월_수는 {opts} 중 하나"
        if a.get("부트스트랩_단위") not in ("달", "문제") or a.get("테스트_방식") not in ("고정", "매달재학습"):
            return False, "부트스트랩_단위 또는 테스트_방식 값 오류"
        on, off = a.get("켤_변수", []), a.get("끌_후보모델", [])
        if not (isinstance(on, list) and len(on) <= 4 and all(k in var_menu for k in on)):
            return False, "켤_변수가 메뉴 밖이거나 4개 초과"
        if not (isinstance(off, list) and len(off) <= 5 and all(m in models and m != cfg["기준모델"] for m in off)):
            return False, "끌_후보모델이 목록 밖이거나 기준모델 포함"
        return True, ""

    ans, who = llm.ask(state, "진단·평가 설계", "diagnose.md", data, default, ok, tools)
    # 반영 (이번 실행의 설정에만. config.yaml 파일은 바꾸지 않음)
    n = int(ans["검증월_수"])
    chosen = next(o for o in d["평가_선택지"] if o["검증월_수"] == n)
    cfg["기간"]["검증월"] = chosen["검증월"]
    cfg["판단규칙"]["부트스트랩_단위"] = ans["부트스트랩_단위"]
    cfg["판단규칙"]["최소상승월수"] = n // 2 + 1                       # 달 수가 바뀌면 '과반 달에서 상승'으로 맞춤
    for k in ans.get("켤_변수", []):
        cfg["변수"][k]["처음값"] = "켜기"
    for m in ans.get("끌_후보모델", []):
        cfg["후보모델"][m]["사용"] = "아니오"
    cfg["평가"]["테스트_방식"] = ans["테스트_방식"]
    state.diagnosis = {"진단": d, "결정": ans, "결정자": who}
    state.decide("⑤ 검토", "진단·평가 설계", f"달효과 {d['달_효과']['달효과_비율']}, 같은 달 신호 부호 바뀜 "
                 f"{d['같은달_신호']['부호가_바뀐_횟수']}회, 상한(달/달×업종) {d['정보_상한']['상한_F1']['달']}/"
                 f"{d['정보_상한']['상한_F1']['달×업종']}",
                 f"validation {n}개월({chosen['검증월'][0] % 100}~{chosen['검증월'][-1] % 100}월), 부트스트랩 {ans['부트스트랩_단위']} 단위, "
                 f"test {ans['테스트_방식']}, 켬 {ans.get('켤_변수', [])}, 끔 {ans.get('끌_후보모델', [])}", who, ans.get("이유", ""))
    state.log(f"[5] 검토 담당 진단: 달효과 {d['달_효과']['달효과_비율']:.2f}, 일수 규칙 {d['달_효과']['일수규칙_맞은달']}, "
              f"같은 달 신호 부호 바뀜 {d['같은달_신호']['부호가_바뀐_횟수']}회 → validation {n}개월 · 부트스트랩 "
              f"{ans['부트스트랩_단위']} 단위 · test {ans['테스트_방식']} ({who})")
    return state


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
        change, who, why = _pick(state, menu, max_try - n_try)
        tried.add(_key(change))
        name = f"R{len(state.revisions) + 1}"
        new = apply_change(cur, change, name)
        new = data_agent.decide_details(state, new, change)                         # 데이터 담당: 분해 변수 개수
        ds_new = data_agent.build(state, new)
        pc = data_agent.precheck(state, ds_new, state.datasets[state.current], change)
        pc = data_agent.judge_precheck(state, pc, change_label(change), max_try - n_try)   # 데이터 담당: 🟢를 진행할지
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
    hints = state.hints or (cfg["소분류처리"].get("힌트") or {})    # 설정 힌트 + 원본 담당이 제안한 후보
    for name, items in hints.items():
        subs = [s for s in items if s in have]
        if subs:
            for how in ("분리", "제외"):
                out.append({"종류": "묶음", "묶음": {"이름": name, "소분류": subs, "처리": how, "이유": f"설정의 힌트 목록 '{name}'"}})
    return out


def _preview(state, menu, num):
    """검토 담당의 도구: 수정안 하나를 사전 점검만 해 본다."""
    try:
        c = menu[int(num)]
    except (TypeError, ValueError, IndexError):
        return {"오류": "번호가 목록 밖"}
    cur = state.recipes[state.current]
    ds = data_agent.build(state, apply_change(cur, c, "미리보기"))
    pc = data_agent.precheck(state, ds, state.datasets[state.current], c)
    top = sorted(pc["신호"].items(), key=lambda kv: -abs(kv[1] - 0.5))[:5]
    return {"수정안": change_label(c), "신호등": pc["신호등"], "이유": pc["이유"], "신호_상위5(AUC)": dict(top)}


def review_plan(state, A: dict, ans: dict, who: str, menu: list, names: list) -> dict:
    """검토 담당이 계획 담당의 레시피 B 제안을 검토한다. 승인하거나, 메뉴 안에서 고친다."""
    from agents.planner import _signal_preview, DESC
    data = {"레시피A": describe(A), "계획_담당_제안": {"켤_항목": ans["켤_항목"], "이유": ans.get("이유", ""), "결정자": who},
            "메뉴": [{"항목": c["항목"], "설명": DESC.get(c["항목"], "")} for c in menu],
            "학습_문제_수": "3~5월 144개 (검증 3회차 중 가장 적을 때)"}
    tools = {"변수_신호": {"인자": "항목", "설명": "A에 그 항목을 켰을 때의 사전 점검 결과와 새 변수 신호",
                         "함수": lambda a: _signal_preview(state, A, menu, a.get("항목"))}}

    def ok(r):
        if not isinstance(r, dict) or not isinstance(r.get("승인"), bool):
            return False, "승인 값이 true/false가 아님"
        if r["승인"]:
            return True, ""
        items = r.get("고친_켤_항목")
        good = isinstance(items, list) and 1 <= len(items) <= 4 and all(i in names for i in items)
        return good, "" if good else "고친 항목이 메뉴 밖이거나 개수가 1~4개가 아님"

    r, who2 = llm.ask(state, "계획 검토", "plan_review.md", data, {"승인": True, "이유": "검토 생략 (규칙)"}, ok, tools)
    if r["승인"]:
        state.decide("① 계획", "계획 검토", ", ".join(ans["켤_항목"]), "승인", who2, r.get("이유", ""))
        return ans
    state.decide("① 계획", "계획 검토", ", ".join(ans["켤_항목"]), "수정: " + ", ".join(r["고친_켤_항목"]), who2, r.get("이유", ""))
    state.log(f"[5] 검토 담당이 계획을 고침: {ans['켤_항목']} → {r['고친_켤_항목']}")
    return {"켤_항목": r["고친_켤_항목"], "이유": f"{ans.get('이유', '')} / 검토 수정: {r.get('이유', '')}"}


def adoption_rule_text(cfg) -> str:
    r = cfg["판단규칙"]
    n = len(cfg["기간"]["검증월"])
    unit = "달을 통째로" if r.get("부트스트랩_단위") == "달" else "문제를"
    return (f"새 레시피는 지금 레시피와 같은 검증 {48 * n}문제로 비교해 ① {unit} 다시 뽑아 채점한 {r['부트스트랩_횟수']}번 중 "
            f"{r['부트스트랩_승률']:.0%} 이상에서 이기고 ② 검증 Macro-F1이 {r['최소상승폭']} 이상 오르고 "
            f"③ 검증 {n}개월 중 {r['최소상승월수']}개월 이상에서 월별 {'정답률' if r.get('월별_비교_지표') == '정답률' else 'Macro-F1'}이 올라야 "
            f"채택된다. 하나라도 못 넘으면 기각.")


def _pick(state, menu: list[dict], remaining: int):
    """다음 수정안 고르기. 규칙 모드는 목록 순서(계획서 4-2), LLM 모드는 LLM이 메뉴에서 고르거나 묶음을 만든다."""
    default = {"번호": 0, "이유": "계획서 4-2 순서대로 아직 안 해 본 첫 수정안 (규칙)"}
    cfg = state.config
    st = data_agent.subcat_stats(state)
    have = set(st["소분류"])
    cur = state.results[state.current]
    data = {"지금_레시피": describe(state.recipes[state.current]), "지금_검증F1": cur["검증F1"],
            "지금_검증_월별F1": cur["월별F1"], "비교용_규칙_검증F1": cur["규칙F1"],
            "채택_규칙": adoption_rule_text(cfg),
            "남은_시도_횟수": f"{remaining}회 (사전 점검에서 🟡 건너뜀·🔴 탈락이면 횟수를 쓰지 않음)",
            "사전점검_의미": "모델 학습 전에 3~5월 학습 데이터만 보고 거름. 🔴 데이터가 망가짐, 🟡 바뀐 게 거의 없거나 정답과 관련 신호가 약함",
            "지난_시도": [{k: r.get(k) for k in ["수정안", "제안이유", "사전점검", "사전점검_이유", "결과", "기존F1", "새F1", "채택판단"]}
                         for r in state.revisions],
            "남은_수정안": [{"번호": i, "수정안": change_label(c), "설명": change_desc(c)} for i, c in enumerate(menu)],
            "소분류_통계_설명": "1~5월 기준. 금액비중=전체 지출 중 비율, 월변동계수=월별 금액의 흔들림, "
                                "대분류변화기여도=같은 대분류 안 월 변화량 중 이 소분류 몫(대분류마다 합 1), 가맹점1위시도비중=한 시도에 몰린 정도",
            "소분류_통계_대분류변화기여도_상위15": st.sort_values("대분류변화기여도", ascending=False).head(15).to_dict("records"),
            "원본_담당의_주의점": (state.data_review or {}).get("데이터_주의점", []),
            "묶음_최대": cfg["소분류처리"]["묶음_최대"], "지금_묶음_수": len(state.recipes[state.current]["묶음"])}
    tools = {
        "소분류_통계": {"인자": "대분류", "설명": "그 대분류의 모든 소분류 통계 (1~5월)",
                       "함수": lambda a: st[st["소분류"].str.startswith(str(a.get("대분류", "")) + " > ")].to_dict("records")},
        "수정안_미리보기": {"인자": "번호", "설명": "남은_수정안 하나의 사전 점검 결과(🟢🟡🔴, 이유, 새 변수 신호). 검증 시도 횟수를 쓰지 않음",
                         "함수": lambda a: _preview(state, menu, a.get("번호"))},
    }

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
        ok_num = isinstance(ans.get("번호"), int) and 0 <= ans["번호"] < len(menu)
        return ok_num, "" if ok_num else "번호가 목록 밖"

    ans, who = llm.ask(state, "다음 수정안", "reviewer.md", data, default, ok, tools)
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
    _explain(state, insights)
    recipes = {k: {**v, "요약": describe(v)} for k, v in state.recipes.items()}
    trace = {"실행": d.name, "시작": state.started_at, "LLM모드": llm.mode(state), "탐색방법": cfg["탐색"]["방법"],
             "레시피": recipes, "처음레시피": state.first_recipe, "최종레시피": state.current,
             "검증횟수": state.n_validations, "비교한_후보수": len(state.trials),
             "수정시도": [{k: v for k, v in r.items() if k != "precheck"} for r in state.revisions],
             "결정기록": state.decisions, "원본점검": state.quality, "LLM_해석": state.explanation,
             "진단": state.diagnosis,
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


def _explain(state, insights: dict):
    """LLM이 결과 해석을 쓴다 (LLM 모드 + 맡길_일 > 결과_설명 이 켜진 경우만). 테스트 점수는 보내지 않는다."""
    final = state.results[state.current]
    coef = None
    if state.test is not None and "로지스틱계수(표준화)" in state.test["변수영향"]:
        fi = state.test["변수영향"]           # 로지스틱 계수는 3~8월 학습 데이터로 구한 값 (테스트 사용 안 함)
        coef = fi.reindex(fi["로지스틱계수(표준화)"].abs().sort_values(ascending=False).index).head(10)[
            ["변수", "로지스틱계수(표준화)"]].to_dict("records")
    growth = insights["업종월별"].pivot_table(index="cat_l", columns="crym", values="전월대비%").round(1)
    tried = {r["수정안"] for r in state.revisions}
    untried = [f"{change_label(c)}: {change_desc(c)}" for c in menu_changes(state.config, state.recipes[state.current])
               if change_label(c) not in tried]
    data = {"과제": "고객층 6 × 업종 대분류 8의 다음 달 지출 증가/감소 예측. 검증은 6·7·8월 144문제",
            "채택_규칙": adoption_rule_text(state.config),
            "모델_선택_규칙": f"기준모델({state.config['기준모델']})을 위 채택 규칙으로 이긴 모델만 선택됨. "
                              "그래서 검증 점수 1등 모델이 선택되지 않을 수 있음",
            "아직_시험하지_않은_수정안": untried[:10],
            "레시피": {k: {"내용": describe(v), "검증F1": state.results[k]["검증F1"] if k in state.results else None}
                       for k, v in state.recipes.items()},
            "처음레시피": state.first_recipe, "최종레시피": state.current,
            "처음레시피_검증F1": state.results[state.first_recipe]["검증F1"],
            "최종레시피_검증F1": final["검증F1"],
            "최종모델": f"{final['모델']} {final['설정']}", "최종_검증_월별F1": final["월별F1"],
            "규칙_검증F1": final["규칙F1"],
            "모델별_검증F1_상위5": final["모델별"].head(5)[["모델", "검증F1"]].to_dict("records"),
            "수정시도": [{k: r.get(k) for k in ["수정안", "결정자", "제안이유", "사전점검", "사전점검_이유", "결과", "기존F1", "새F1", "채택판단"]}
                         for r in state.revisions],
            "로지스틱_계수_상위10": coef,
            "업종별_전월대비%": {c: {str(m): v for m, v in row.items() if v == v} for c, row in growth.iterrows()},
            "명절·쿠폰_전후": insights["행사전후"].round(2).to_dict("records")}

    def ok(ans):
        good = isinstance(ans, dict) and isinstance(ans.get("요약"), str) and ans["요약"].strip()
        return bool(good), "" if good else "요약 없음"

    ans, who = llm.ask(state, "결과 설명", "explainer.md", data, None, ok)
    state.explanation = {**ans, "결정자": who, "모델": llm.model_name(state)} if ans else None


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
