"""[1] 계획 담당: 설정을 검사하고, 처음 레시피 2개(A, B)를 세운다.

레시피 = 에이전트가 정하는 전처리·변수 선택 한 세트.
  변수      : 기본 변수 켜기/끄기
  원본변수  : 중·소분류, 지역 칼럼으로 만드는 변수 켜기/끄기, 지역 단위
  전처리    : 로그 변환, 극단값 상한, 고객층 표현, 미세변동 제외
  묶음      : 소분류 묶음과 처리(제외/분리)
A = 설정 파일의 처음값 그대로. B = A에서 '시험: 예'인 변수를 켠 것 (LLM 모드면 LLM이 고름).
"""
from __future__ import annotations

import copy

import llm
from common import SafeError, check_config

TOGGLES_O = ["중분류_요약", "중분류_분해", "소분류_요약", "지역_요약", "지역_분해", "원본행수"]


def run(state):
    cfg, log = state.config, state.log
    err, warn = check_config(cfg)
    if err:
        raise SafeError("설정 오류:\n  - " + "\n  - ".join(err))
    for w in warn:
        log(f"[1] 주의: {w}")

    A = initial_recipe(cfg)
    menu = [c for c in menu_changes(cfg, A) if c["종류"] in ("변수", "원본변수")]
    names = [c["항목"] for c in menu]
    basic = [c["항목"] for c in menu if c["종류"] == "변수"]
    default = {"켤_항목": basic or names[:1],
               "이유": "계획서 4-2 ①처럼 기본 변수 중 시험 대상(객단가·업종비중·달력 등)을 켠 버전과 비교. 원본 칼럼 변수는 수정 단계에서 하나씩 시험 (규칙)"}

    def ok(ans):
        items = ans.get("켤_항목") if isinstance(ans, dict) else None
        if not isinstance(items, list) or not items:
            return False, "켤_항목 없음"
        bad = [i for i in items if i not in names]
        return (not bad), (f"메뉴 밖 항목: {bad}" if bad else "")

    from agents import data_agent, reviewer
    state.recipes = {"A": A}
    state.datasets["A"] = data_agent.build(state, A)
    data = {"레시피A": describe(A), "레시피A_기본변수": [k for k, v in A["변수"].items() if v],
            "메뉴": [{"항목": c["항목"], "설명": DESC.get(c["항목"], "")} for c in menu],
            "업종_대분류": state.quality["업종"]["대분류"], "학습_문제_수": "3~5월 144개 (검증 3회차 중 가장 적을 때)",
            "원본_담당의_주의점": (state.data_review or {}).get("데이터_주의점", [])}
    tools = {
        "변수_신호": {"인자": "항목 (메뉴의 항목 이름)", "설명": "A에 그 항목을 켰을 때의 사전 점검 결과(🟢🟡🔴)와 새 변수의 신호(3~5월 학습 데이터 기준)",
                     "함수": lambda a: _signal_preview(state, A, menu, a.get("항목"))},
        "업종별_추이": {"인자": "", "설명": "1~5월 업종 대분류별 월 지출 전월대비(%)",
                      "함수": lambda a: _cat_trend(state)},
    }
    ans, who = llm.ask(state, "처음 레시피 B", "planner.md", data, default, ok, tools)
    ans = reviewer.review_plan(state, A, ans, who, menu, names)   # 검토 담당이 계획을 검토 (승인 또는 수정)
    B = copy.deepcopy(A)
    B["이름"], B["설명"] = "B", "A + " + ", ".join(ans["켤_항목"])
    for c in menu:
        if c["항목"] in ans["켤_항목"]:
            B = apply_change(B, c, "B")
    B["설명"] = "A + " + ", ".join(ans["켤_항목"])
    state.recipes = {"A": A, "B": B}
    state.datasets["B"] = data_agent.build(state, B)
    state.first_recipe = "A"
    state.decide("① 계획", "처음 레시피 2개", f"A: {describe(A)} / B: {describe(B)}", "A와 B를 둘 다 검증", who, ans["이유"])
    log(f"[1] 처음 레시피 A: {describe(A)}")
    log(f"[1] 처음 레시피 B: {B['설명']} (결정: {who})")
    return state


def _signal_preview(state, A, menu, item):
    """계획 담당의 도구: A에 항목 하나를 켜 보고 사전 점검만 해 본다 (검증은 하지 않음)."""
    from agents import data_agent
    c = next((c for c in menu if c["항목"] == item), None)
    if c is None:
        return {"오류": f"메뉴에 없는 항목: {item}"}
    ds = data_agent.build(state, apply_change(A, c, "미리보기"))
    pc = data_agent.precheck(state, ds, state.datasets["A"], c)
    top = sorted(pc["신호"].items(), key=lambda kv: -abs(kv[1] - 0.5))[:5]
    return {"신호등": pc["신호등"], "이유": pc["이유"], "새변수_수": len(pc["추가변수"]),
            "신호_상위5(AUC)": dict(top)}


def _cat_trend(state):
    from agents.data_agent import periods
    t = state.truth
    t = t[t["crym"] <= periods(state.config)["stat_end"]]
    g = t.groupby(["cat_l", "crym"])["am"].sum().unstack("crym").pct_change(axis=1) * 100
    return {c: {str(m): round(v, 1) for m, v in row.items() if v == v} for c, row in g.iterrows()}


# ---------------------------------------------------------------- 메뉴 항목 설명 (LLM·사람이 읽는 용도)
DESC = {
    "객단가": "결제 1건당 평균 금액(금액÷건수). 같은 지출이라도 건수와 단가 중 무엇이 변했는지 구분",
    "업종비중": "그 고객층의 전체 지출 중 이 업종이 차지하는 비율",
    "달력변수": "다음 달의 영업일 수·명절 포함 여부·소비쿠폰 기간 여부 (미리 알 수 있는 일정)",
    "달력차이": "다음 달이 이번 달보다 일수·영업일수·휴일수·토일수·소비쿠폰 일수가 얼마나 많은지 (월 합계라 달이 길면 늘기 쉬움)",
    "중분류_요약": "업종 안 중분류들의 쏠림 정도(집중도), 1위 중분류 비중, 지난달보다 늘어난 중분류의 비율",
    "중분류_분해": "업종 안 금액 상위 몇 개 중분류 각각의 비중과 전월 대비 변화",
    "소분류_요약": "중분류_요약과 같은 지표를 더 잘게 나눈 소분류 기준으로",
    "원본행수": "원본 거래 조합의 수 (지출이 얼마나 여러 곳에 퍼져 있는지)",
    "지역_요약": "다른 시도에서 결제한 비중, 가맹점·고객 지역의 쏠림 정도",
    "지역_분해": "전체 금액 상위 몇 개 지역(코드) 각각의 비중과 전월 대비 변화",
    "지역_단위": "지역을 시도 단위로 볼지, 수도권만 시군구로 더 잘게 볼지",
    "미세변동_제외": "다음 달 변화가 ±2% 이내인 애매한 사례를 학습에서만 뺌 (채점은 전부)",
    "극단값_상한": "학습 데이터의 상·하위 1% 극단값을 잘라 냄",
    "고객층_표현": "성별·연령구간을 따로 표시할지, 고객층 6개 중 하나로 합쳐 표시할지",
    "금액_로그변환": "금액 크기 차이를 log로 줄임",
}


def change_desc(c: dict) -> str:
    """수정안 하나에 대한 쉬운 설명."""
    if c["종류"] == "묶음":
        b = c["묶음"]
        how = "그 거래를 빼고 변수를 계산" if b["처리"] == "제외" else "빼고 계산한 뒤 묶음 금액·전월대비를 별도 변수로 추가"
        names = ", ".join(s.split(" > ")[-1] for s in b["소분류"][:4]) + (" 등" if len(b["소분류"]) > 4 else "")
        return f"소분류 {len(b['소분류'])}개({names})를 {b['처리']}: {how}. 근거: {b.get('이유', '')}"
    return DESC.get(c["항목"], "")


# ---------------------------------------------------------------- 레시피 도구
def initial_recipe(cfg) -> dict:
    o = cfg["원본변수"]
    return {
        "이름": "A", "설명": "설정 파일의 처음값",
        "변수": {k: v["처음값"] in ("항상", "켜기") for k, v in cfg["변수"].items()},
        "원본변수": {k: o[k]["처음값"] in ("켜기", "항상") for k in TOGGLES_O},
        "지역_단위": o["지역_단위"]["처음값"], "상위지역_수": o["상위지역_수"], "상위중분류_수": o["상위중분류_수"],
        "전처리": {"금액_로그변환": cfg["전처리"]["금액_로그변환"]["처음값"] == "켜기",
                   "극단값_상한": cfg["전처리"]["극단값_상한"]["처음값"] == "켜기",
                   "고객층_표현": cfg["전처리"]["고객층_표현"]["처음값"],
                   "미세변동_제외": cfg["전처리"]["미세변동_제외"]["처음값"] == "켜기"},
        "묶음": [],
    }


def menu_changes(cfg, r: dict) -> list[dict]:
    """지금 레시피 r에서 시험해 볼 수 있는 수정안 목록 (계획서 4-2 순서). 묶음은 reviewer가 따로 만든다."""
    out = []
    for k, v in cfg["변수"].items():
        if v["시험"] == "예":
            out.append({"종류": "변수", "항목": k, "값": not r["변수"][k]})
    for k in ["중분류_요약", "중분류_분해", "소분류_요약", "원본행수"]:
        if cfg["원본변수"][k]["시험"] == "예":
            out.append({"종류": "원본변수", "항목": k, "값": not r["원본변수"][k]})
    for k in ["미세변동_제외", "극단값_상한", "고객층_표현", "금액_로그변환"]:
        v = cfg["전처리"][k]
        if v["시험"] == "예":
            cur = r["전처리"][k]
            if isinstance(cur, bool):
                out.append({"종류": "전처리", "항목": k, "값": not cur})
            else:
                for alt in v["선택지"]:
                    if alt != cur:
                        out.append({"종류": "전처리", "항목": k, "값": alt})
    for k in ["지역_요약", "지역_분해"]:
        if cfg["원본변수"][k]["시험"] == "예":
            out.append({"종류": "원본변수", "항목": k, "값": not r["원본변수"][k]})
    lv = cfg["원본변수"]["지역_단위"]
    if lv["시험"] == "예" and (r["원본변수"]["지역_요약"] or r["원본변수"]["지역_분해"]):
        for alt in lv["선택지"]:
            if alt != r["지역_단위"]:
                out.append({"종류": "지역_단위", "항목": "지역_단위", "값": alt})
    return out


def apply_change(r: dict, c: dict, name: str) -> dict:
    n = copy.deepcopy(r)
    n["이름"] = name
    if c["종류"] in ("변수", "원본변수", "전처리"):
        n[c["종류"]][c["항목"]] = c["값"]
    elif c["종류"] == "지역_단위":
        n["지역_단위"] = c["값"]
    elif c["종류"] == "묶음":
        n["묶음"] = n["묶음"] + [c["묶음"]]
    n["설명"] = f"{r['이름']} + {change_label(c)}"
    return n


def change_label(c: dict) -> str:
    if c["종류"] == "묶음":
        b = c["묶음"]
        return f"묶음 '{b['이름']}' {b['처리']} (소분류 {len(b['소분류'])}개)"
    v = c["값"]
    if isinstance(v, bool):
        return f"{c['항목']} {'켜기' if v else '끄기'}"
    return f"{c['항목']} → {v}"


def describe(r: dict) -> str:
    on = [k for k, v in r["변수"].items() if v]
    o = [k for k, v in r["원본변수"].items() if v]
    p = r["전처리"]
    parts = [f"변수 {len(on)}종"]
    extra = [k for k in ["객단가", "업종비중", "달력변수"] if r["변수"].get(k)]
    if extra:
        parts.append("+" + "·".join(extra))
    if o:
        parts.append("원본 " + "·".join(o) + (f"(지역 {r['지역_단위']})" if any(x.startswith("지역") for x in o) else ""))
    parts.append(f"로그{'O' if p['금액_로그변환'] else 'X'}·상한{'O' if p['극단값_상한'] else 'X'}·고객층{p['고객층_표현']}"
                 + ("·±2%제외" if p["미세변동_제외"] else ""))
    if r["묶음"]:
        parts.append("묶음 " + ", ".join(f"{b['이름']}({b['처리']})" for b in r["묶음"]))
    return " / ".join(parts)
