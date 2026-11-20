"""에이전트의 '생각' 엔진: LLM이 도구를 써 가며 판단하고, 코드가 그 결정을 검사한다.

동작 (결정 1번):
  1. 에이전트 지시문 + 자료 + 쓸 수 있는 도구 목록을 LLM에게 보낸다.
  2. LLM은 매 턴 둘 중 하나로 답한다.
       {"도구": "이름", "인자": {...}, "생각": "왜 필요한지"}   → 코드가 집계값을 계산해 돌려주고 다시 묻는다 (최대 N회)
       답 형식대로 결정                                         → 코드가 검사(validate)해서 통과하면 반영
  3. 검사에 실패하거나 LLM 연결이 안 되면 규칙 모드의 기본 결정을 쓴다.

- 모드: 규칙 / ollama / openai (config.yaml > LLM). 에이전트마다 모델을 다르게 쓸 수 있다 (에이전트별_모델).
- 맡길 일은 config.yaml > LLM > 맡길_일 에서 하나씩 켜고 끈다.
- 도구는 집계값만 돌려준다. 원본 행·지역 이름·테스트 점수(테스트 해석 전)는 LLM에게 가지 않는다.
- 대화 전체(프롬프트·도구 호출·도구 결과·원문 응답)를 runs/<실행>/llm/ 에 저장한다.
- 에이전트별 context: 에이전트마다 '자기 대화'만 이어서 기억한다 (다른 에이전트의 대화·도구 결과는 섞지 않음).
- 정보 차단벽: test를 열기 전의 결정에는 validation 마지막 달보다 뒤의 월 정보가 가지 않는다.
  보내기 직전에 코드가 자료·도구 결과를 검사해서 지우고, 지운 개수를 기록한다.
- --replay 로 이전 실행의 결정을 그대로 다시 쓸 수 있다 (도구는 다시 부르지 않음).
"""
from __future__ import annotations

import json
import os
import re
from copy import deepcopy
import urllib.request
from datetime import datetime
from pathlib import Path

from common import is_on

PROMPTS = Path(__file__).resolve().parent / "prompts"

# 목적 → (에이전트, 맡길_일 항목)
PURPOSE = {
    "원본 해석": ("원본", "원본_해석"),
    "진단·평가 설계": ("검토", "진단"),
    "처음 레시피 B": ("계획", "처음_레시피"),
    "계획 검토": ("검토", "계획_검토"),
    "모델 탐색 계획": ("모델", "모델_탐색_계획"),
    "모델 탐색 조정": ("모델", "모델_탐색_계획"),
    "변수 세부 설정": ("데이터", "데이터_판단"),
    "사전 점검 판단": ("데이터", "데이터_판단"),
    "다음 수정안": ("검토", "수정안_고르기"),
    "실행 계획": ("실행", "실행_판단"),
    "테스트 해석": ("실행", "실행_판단"),
    "결과 설명": ("검토", "결과_설명"),
}


AFTER_TEST = {"테스트 해석", "결과 설명"}       # test를 연 뒤의 결정만 test 기간 정보를 볼 수 있음
_MONTH = re.compile(r"(?<!\d)(20\d{2})(0[1-9]|1[0-2])(?!\d)")


def _limit(state, 목적: str) -> int | None:
    """이 결정이 볼 수 있는 마지막 기준월 (validation 마지막 달). None이면 제한 없음."""
    if 목적 in AFTER_TEST:
        return None
    from agents.data_agent import periods
    return periods(state.config)["dev_end"]


def guard(obj, last: int | None, cut: list):
    """자료에서 last보다 뒤의 월 정보를 지운다 (키·값·목록 항목·문장 속 월 표기). 지운 것은 cut에 센다."""
    if last is None:
        return obj

    def late(x) -> bool:
        s = str(x).replace("월", "")
        return bool(re.fullmatch(r"20\d{2}(0[1-9]|1[0-2])", s)) and int(s) > last

    def rep(mt):
        if int(mt.group(0)) > last:
            cut.append(mt.group(0))
            return "[차단된 달]"
        return mt.group(0)

    def walk(x):
        if isinstance(x, dict):
            if any(k in ("crym", "기준월", "월") and late(v) for k, v in x.items()):
                return None                              # 이 항목 전체가 늦은 달의 것
            out = {}
            for k, v in x.items():
                if late(k):
                    cut.append(str(k))
                    continue
                w = walk(v)
                if w is None and v is not None:
                    cut.append("항목")
                    continue
                out[k] = w
            return out
        if isinstance(x, (list, tuple)):
            kept = []
            for v in x:
                if late(v):
                    cut.append(str(v))
                    continue
                w = walk(v)
                if w is None and v is not None:
                    cut.append("항목")
                    continue
                kept.append(w)
            return kept
        if isinstance(x, str):
            return _MONTH.sub(rep, x)
        if isinstance(x, (int, float)) and not isinstance(x, bool) and late(int(x)) and float(x).is_integer():
            cut.append(str(x))
            return "[차단된 달]"
        return x
    return walk(deepcopy(obj))


def mode(state) -> str:
    m = state.config["LLM"]["모드"]
    if m == "openai" and not os.environ.get("OPENAI_API_KEY"):
        return "규칙"
    return m


def model_name(state, agent: str = "") -> str:
    m = mode(state)
    if m not in ("ollama", "openai"):
        return "-"
    per = state.config["LLM"].get("에이전트별_모델") or {}
    if m == "openai" and per.get(agent):
        return per[agent]
    return state.config["LLM"][m]["모델"]


def ask(state, 목적: str, prompt_file: str, data: dict, default, validate, tools: dict | None = None):
    """에이전트에게 결정을 묻는다. (결정, 결정자) 반환. tools = {이름: {"설명", "인자", "함수"}}"""
    agent, role = PURPOSE.get(목적, ("검토", ""))
    m = mode(state)
    role_on = is_on((state.config["LLM"].get("맡길_일") or {}).get(role, "예"))
    if not role_on:
        return default, "규칙(LLM 끔)"
    if m == "규칙" and not state.replay:
        return default, "규칙"
    call = {"번호": len(state.llm_calls) + 1, "시각": datetime.now().isoformat(timespec="seconds"),
            "에이전트": agent, "목적": 목적, "모드": m, "모델": model_name(state, agent), "프롬프트_파일": prompt_file}

    # --replay: 이전 실행의 결정을 순서대로 재사용 (LLM·도구를 다시 부르지 않음)
    if state.replay is not None:
        prev = state.replay.get(목적, [])
        used = sum(1 for c in state.llm_calls if c["목적"] == 목적)
        if used < len(prev):
            ans = prev[used].get("답")
            ok, why = validate(ans) if ans is not None else (False, "이전 답 없음")
            call.update({"모드": "replay", "대화": prev[used].get("대화", []), "도구사용": prev[used].get("도구사용", []),
                         "답": ans, "채택": ok, "비고": why})
            _record(state, call)
            return (ans, "LLM(replay)") if ok else (default, "규칙(replay 답 무효)")
        if m == "규칙":
            return default, "규칙"

    cut: list = []
    last = _limit(state, 목적)
    data = guard(data, last, cut)
    system = (PROMPTS / "system.md").read_text(encoding="utf-8") + (
        f"\n\n## 너의 역할\n너는 '{agent} 담당'이다. 다른 담당의 대화는 볼 수 없고, 아래 자료와 너의 이전 결정만 안다.")
    user = (PROMPTS / prompt_file).read_text(encoding="utf-8") + "\n\n## 자료 (JSON)\n" + json.dumps(data, ensure_ascii=False, indent=1, default=str)
    max_tools = int(state.config["LLM"].get("도구_호출_최대", 3)) if tools else 0
    if tools:
        user += f"\n\n## 쓸 수 있는 도구 (이번 결정에서 최대 {max_tools}회)\n" + "\n".join(
            f"- {name}({t.get('인자', '')}): {t['설명']}" for name, t in tools.items())
        user += ('\n\n도구로 집계를 더 보고 싶으면 {"도구": "이름", "인자": {...}, "생각": "왜 필요한지"} 로 답한다. '
                 "결정할 준비가 되면 위의 '답 형식'대로 답한다.")
    # 에이전트별 context: 같은 담당의 이전 결정만 이어 붙인다 (다른 담당 것은 없음)
    keep = is_on(state.config["LLM"].get("에이전트별_대화유지", "예"))
    past = []
    for h in (state.llm_sessions.get(agent, []) if keep else [])[-6:]:
        past += [{"role": "user", "content": f"(너의 이전 결정) {h['목적']}"},
                 {"role": "assistant", "content": json.dumps(guard(h["답"], last, cut), ensure_ascii=False, default=str)}]
    messages = [{"role": "system", "content": system}] + past + [{"role": "user", "content": user}]
    used_tools, ans, ok, why = [], None, False, ""
    try:
        for turn in range(max_tools + 2):
            raw = _chat(state.config, m, model_name(state, agent), messages)
            messages.append({"role": "assistant", "content": raw})
            obj = json.loads(_strip_fence(raw))
            if isinstance(obj, dict) and "도구" in obj and tools:
                left = max_tools - len(used_tools)
                if left <= 0:
                    messages.append({"role": "user", "content": "도구 호출 횟수를 다 썼다. 지금 '답 형식'대로 결정하라."})
                    continue
                name, args = obj.get("도구"), obj.get("인자") or {}
                result = _run_tool(tools, name, args, last, cut)
                used_tools.append({"도구": name, "인자": args, "생각": obj.get("생각", ""), "결과_요약": result[:300]})
                if state.config["LLM"].get("기록", {}).get("화면_표시", "예") in ("예", True):
                    state.log(f"      [{agent} 담당 생각] 도구 {name}({json.dumps(args, ensure_ascii=False)[:60]}) — {str(obj.get('생각', ''))[:80]}")
                messages.append({"role": "user", "content": f"## 도구 결과: {name}\n{result}\n\n도구 호출이 {left - 1}회 남았다. "
                                                            "더 볼 것이 없으면 '답 형식'대로 결정하라."})
                continue
            ans = obj
            ok, why = validate(ans)
            break
        else:
            why = "결정을 내리지 않음"
    except Exception as e:      # 네트워크·형식 오류 등: 규칙으로 대체
        ok, why = False, f"{type(e).__name__}"
    call.update({"대화": messages, "도구사용": used_tools, "답": ans, "채택": ok, "비고": why,
                 "차단": len(cut), "이전결정_수": len(past) // 2})
    if cut:
        state.log(f"    [정보 차단벽] {agent} 담당에게 가는 자료에서 {last % 100}월 이후 월 정보 {len(cut)}건을 지움")
    _record(state, call)
    if ok:
        state.llm_sessions.setdefault(agent, []).append({"목적": 목적, "답": ans})
        return ans, f"LLM({m})"
    state.log(f"    [{agent} 담당] LLM 답을 쓰지 못해 규칙으로 대체 ({why})")
    return default, "규칙(LLM 실패)"


def _run_tool(tools: dict, name, args, last=None, cut=None) -> str:
    """도구 실행. 결과는 집계값 JSON 문자열 (너무 길면 자름). 정보 차단벽을 거친다."""
    if name not in tools:
        return json.dumps({"오류": f"'{name}' 도구는 없다. 가능한 도구: {list(tools)}"}, ensure_ascii=False)
    try:
        out = tools[name]["함수"](args if isinstance(args, dict) else {})
    except Exception as e:
        out = {"오류": f"{type(e).__name__}: 인자를 확인하라"}
    s = json.dumps(guard(out, last, cut if cut is not None else []), ensure_ascii=False, default=str)
    return s if len(s) <= 6000 else s[:6000] + " …(이하 생략)"


def _strip_fence(s: str) -> str:
    """```json ... ``` 로 감싼 답도 읽을 수 있게."""
    m = re.search(r"```(?:json)?\s*(.*?)```", s, re.S)
    return m.group(1) if m else s


def _record(state, call: dict):
    """호출 기록: 노트 + (선택) 화면 한 줄 + (선택) 대화 전체 파일."""
    state.llm_calls.append(call)
    rec = state.config["LLM"].get("기록") or {}
    if is_on(rec.get("화면_표시", "예")):
        a = call["답"]
        why = a.get("이유", "") if isinstance(a, dict) else ""
        brief = {k: v for k, v in a.items() if k != "이유"} if isinstance(a, dict) else a
        if isinstance(brief, dict) and isinstance(brief.get("묶음"), dict):
            b = brief["묶음"]
            brief = {"묶음": f"{b.get('이름')} {b.get('처리')} (소분류 {len(b.get('소분류', []))}개)"}
        tools = f" · 도구 {len(call.get('도구사용', []))}회" if call.get("도구사용") else ""
        state.log(f"    [LLM {call['번호']} · {call['에이전트']} 담당 · {call['모델']}{tools}] {call['목적']} → "
                  f"{json.dumps(brief, ensure_ascii=False)[:160]}{' | 이유: ' + str(why)[:120] if why else ''} "
                  f"({'채택' if call['채택'] else '무효: ' + str(call['비고'])})")
    if is_on(rec.get("프롬프트_저장", "예")):
        d = state.run_dir / "llm"
        d.mkdir(parents=True, exist_ok=True)
        name = f"{call['번호']:02d}_{call['에이전트']}_{call['목적'].replace(' ', '')}.md"
        parts = [f"# LLM 호출 {call['번호']}: {call['에이전트']} 담당 — {call['목적']}\n",
                 f"- 시각: {call['시각']} / 모드: {call['모드']} / 모델: {call['모델']}",
                 f"- 도구 사용: {len(call.get('도구사용', []))}회 / 검사 결과: {'채택' if call['채택'] else '무효'} {call['비고']}\n"]
        title = {"system": "시스템 프롬프트", "user": "사용자", "assistant": "LLM 응답"}
        for i, msg in enumerate(call.get("대화", [])):
            head = title.get(msg["role"], msg["role"])
            if msg["role"] == "user" and str(msg["content"]).startswith("(너의 이전 결정)"):
                head = "이 담당의 이전 결정 (다른 담당 것은 없음)"
            elif msg["role"] == "user" and i > 1 and not str(msg["content"]).startswith("#"):
                head = "도구 결과 / 추가 지시"
            elif msg["role"] == "assistant" and i + 1 < len(call.get("대화", [])) and str(call["대화"][i - 1]["content"]).startswith("(너의 이전 결정)"):
                head = "이전에 낸 답"
            body = msg["content"] if msg["role"] != "assistant" else f"```\n{msg['content']}\n```"
            parts.append(f"## {i}. {head}\n\n{body}\n")
        (d / name).write_text("\n".join(parts), encoding="utf-8")
        call["파일"] = f"llm/{name}"


def _chat(cfg, m: str, model: str, messages: list[dict]) -> str:
    c = cfg["LLM"][m]
    body = {"model": model, "temperature": 0, "response_format": {"type": "json_object"}, "messages": messages}
    headers = {"Content-Type": "application/json"}
    if m == "openai":
        headers["Authorization"] = f"Bearer {os.environ['OPENAI_API_KEY']}"
    req = urllib.request.Request(c["주소"].rstrip("/") + "/chat/completions", data=json.dumps(body).encode("utf-8"),
                                 headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=cfg["LLM"]["시간초과_초"]) as r:
        out = json.loads(r.read().decode("utf-8"))
    return out["choices"][0]["message"]["content"]
