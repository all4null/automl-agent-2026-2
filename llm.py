"""LLM 호출 (선택). 모드: 규칙 / ollama / openai  (config.yaml > LLM).

- ollama와 openai 모두 OpenAI 호환 주소(/chat/completions)로 같은 코드를 쓴다. 추가 패키지 없이 파이썬 기본 기능만 사용.
- LLM에게는 업종 이름·집계 통계·검증 점수·지역 코드만 보낸다. 원본 행·지역 이름·테스트 점수는 보내지 않는다.
- 오류·시간초과·형식이 틀린 답이면 규칙 모드의 기본값을 쓰고 '규칙(LLM 실패)'로 기록한다.
- 모든 호출은 state.llm_calls에 남고 llm_proposals.json으로 저장된다. --replay 로 같은 답을 다시 쓸 수 있다.
"""
from __future__ import annotations

import json
import os
import urllib.request
from datetime import datetime
from pathlib import Path

PROMPTS = Path(__file__).resolve().parent / "prompts"


def mode(state) -> str:
    m = state.config["LLM"]["모드"]
    if m == "openai" and not os.environ.get("OPENAI_API_KEY"):
        return "규칙"
    return m


def ask(state, 목적: str, prompt_file: str, data: dict, default, validate):
    """LLM에게 묻고, validate(답)이 통과하면 그 답을, 아니면 default를 돌려준다. (답, 결정자) 반환."""
    m = mode(state)
    call = {"시각": datetime.now().isoformat(timespec="seconds"), "목적": 목적, "모드": m}
    if m == "규칙" and not state.replay:
        return default, "규칙"

    # --replay: 이전 실행의 답을 순서대로 재사용 (LLM을 다시 부르지 않음)
    if state.replay is not None:
        prev = state.replay.get(목적, [])
        used = sum(1 for c in state.llm_calls if c["목적"] == 목적)
        if used < len(prev):
            ans = prev[used]["답"]
            ok, why = validate(ans) if ans is not None else (False, "이전 답 없음")
            call.update({"모드": "replay", "질문요약": prev[used].get("질문요약", ""), "답": ans, "채택": ok, "비고": why})
            state.llm_calls.append(call)
            return (ans, "LLM(replay)") if ok else (default, "규칙(replay 답 무효)")
        if m == "규칙":
            return default, "규칙"

    system = (PROMPTS / "system.md").read_text(encoding="utf-8")
    user = (PROMPTS / prompt_file).read_text(encoding="utf-8") + "\n\n## 자료 (JSON)\n" + json.dumps(data, ensure_ascii=False)
    call["질문요약"] = f"{prompt_file} + 자료 {len(json.dumps(data, ensure_ascii=False))}자"
    try:
        ans = _chat(state.config, m, system, user)
        ok, why = validate(ans)
    except Exception as e:      # 네트워크·형식 오류 등: 규칙으로 대체
        ans, ok, why = None, False, f"{type(e).__name__}"
    call.update({"답": ans, "채택": ok, "비고": why})
    state.llm_calls.append(call)
    if ok:
        return ans, f"LLM({m})"
    state.log(f"    LLM 답을 쓰지 못해 규칙으로 대체 ({why})")
    return default, "규칙(LLM 실패)"


def _chat(cfg, m: str, system: str, user: str):
    c = cfg["LLM"][m]
    body = {"model": c["모델"], "temperature": 0, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    headers = {"Content-Type": "application/json"}
    if m == "openai":
        headers["Authorization"] = f"Bearer {os.environ['OPENAI_API_KEY']}"
    req = urllib.request.Request(c["주소"].rstrip("/") + "/chat/completions", data=json.dumps(body).encode("utf-8"),
                                 headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=cfg["LLM"]["시간초과_초"]) as r:
        out = json.loads(r.read().decode("utf-8"))
    return json.loads(out["choices"][0]["message"]["content"])
