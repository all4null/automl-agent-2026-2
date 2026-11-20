"""AutoML-Agent 실행 버튼. 이 파일만 읽어도 전체 흐름이 보인다.

실행 (AutoML-AGENT 폴더에서, 프로젝트의 가상환경 사용):
    ..\\.venv\\Scripts\\python run.py --check                 설정 요약과 오류만 확인 (학습 안 함)
    ..\\.venv\\Scripts\\python run.py                         config.yaml의 원본으로 전체 실행
    ..\\.venv\\Scripts\\python run.py --raw <가짜원본.csv>     원본 경로를 바꿔 실행 (가짜데이터 시험용)
    ..\\.venv\\Scripts\\python run.py --replay <실행폴더>      이전 실행의 LLM 답을 그대로 다시 써서 재현
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import RunLog, SafeError, cfg_path, check_config, is_on, load_config  # noqa: E402
from state import State  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="AutoML-Agent")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--check", action="store_true", help="설정 요약과 오류만 확인")
    ap.add_argument("--raw", help="원본 CSV 경로 (config 대신)")
    ap.add_argument("--replay", help="이전 실행 폴더 (그 실행의 LLM 답을 재사용)")
    a = ap.parse_args()

    try:
        cfg = load_config(a.config)
        if a.raw and is_fake(a.raw):
            # 가짜데이터 시험은 실데이터 결과·보안 폴더와 섞이지 않게 따로 둔다
            cfg["경로"]["결과폴더"] = cfg["경로"].get("가짜_결과폴더", "../결과/테스트_agent")
            cfg["경로"]["보안폴더"] = cfg["경로"].get("가짜_보안폴더", "../데이터/가짜데이터/secure/agent")
            print(f"[가짜데이터 실행] 결과: {cfg['경로']['결과폴더']}, 행 단위 파일: {cfg['경로']['보안폴더']}")
        err, warn = check_config(cfg)
        if a.check:
            print_summary(cfg, err, warn, a.raw)
            sys.exit(1 if err else 0)
        if err:
            raise SafeError("설정 오류:\n  - " + "\n  - ".join(err) + "\n  (python run.py --check 로 확인하세요)")
    except SafeError as e:
        print(f"[중단] {e}", file=sys.stderr)
        sys.exit(1)

    run_dir = cfg_path(cfg, "결과폴더") / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    state = State(config=cfg, run_dir=run_dir, started_at=datetime.now().isoformat(timespec="seconds"), log=RunLog())
    if a.replay:
        p = Path(a.replay) / "llm_proposals.json"
        state.replay = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    try:
        run_all(state, a.raw)
    except SafeError as e:
        state.log(f"[중단] {e}")
        sys.exit(1)
    except Exception as e:
        # 오류 메시지에 원본 내용이 섞일 수 있으므로 화면에는 종류만, 상세는 보안 폴더에
        sec = cfg_path(cfg, "보안폴더")
        sec.mkdir(parents=True, exist_ok=True)
        err_file = sec / f"error_{run_dir.name}.log"
        err_file.write_text(traceback.format_exc(), encoding="utf-8")
        state.log(f"[오류] {type(e).__name__} 발생. 상세는 보안 폴더의 {err_file.name} 참고 "
                  f"(원본 내용이 섞여 있을 수 있으니 공유 전 확인)")
        sys.exit(2)
    finally:
        state.log.save(run_dir / "run_log.txt")


def is_fake(raw: str) -> bool:
    fake_dir = (Path(__file__).resolve().parent.parent / "데이터" / "가짜데이터").resolve()
    return fake_dir in Path(raw).resolve().parents


def run_all(state, raw=None):
    """전체 흐름: [0] 원본 → [1] 계획 → [2] 데이터 → [3] 검증 → [5] 수정(최대 N회) → [4] 테스트 1회 → [5] 보고."""
    from agents import data_agent, executor, model_agent, planner, raw_agent, reviewer

    raw_agent.run(state, raw)                         # [0] 원본 점검·정제 (원본을 읽는 유일한 단계)
    planner.run(state)                                # [1] 설정 검사, 처음 레시피 A·B
    data_agent.build_truth(state)                     # [2] 정답표 (고정)
    for name in ["A", "B"]:                           # [2]+[3] 처음 레시피 둘 다 검증
        state.datasets[name] = data_agent.build(state, state.recipes[name])
        model_agent.evaluate(state, state.datasets[name], state.recipes[name])
    reviewer.choose_initial(state)                    # [5] B가 채택 규칙을 넘을 때만 B로 시작
    reviewer.check(state)                             # [5] 수정안 → 사전 점검 → 검증 → 채택/기각
    executor.run(state)                               # [4] 테스트 단 1회 + 월별 갱신 실험
    reviewer.report(state)                            # [5] 결과 파일·대시보드
    return state


def print_summary(cfg, err, warn, raw=None):
    from agents.model_agent import optional_missing
    from agents.planner import initial_recipe, menu_changes
    p = cfg["기간"]
    raw_path = Path(raw) if raw else cfg_path(cfg, "원본파일")
    if not raw and not raw_path.exists():
        alt = raw_path.parent / cfg["경로"].get("원본_다른이름", "")
        raw_path = alt if alt.exists() else raw_path
    always = [k for k, v in cfg["변수"].items() if v["처음값"] == "항상"]
    trial_v = [k for k, v in cfg["변수"].items() if v["시험"] == "예"]
    trial_o = [k for k, v in cfg["원본변수"].items() if isinstance(v, dict) and v.get("시험") == "예"]
    trial_p = [k for k, v in cfg["전처리"].items() if v["시험"] == "예"]
    used = [m for m, v in cfg["후보모델"].items() if is_on(v.get("사용")) and not optional_missing(m)]
    n_grid = sum(len(list(itertools.product(*(cfg["후보모델"][m]["좁은범위"] or {"_": [0]}).values()))) for m in used)
    r = cfg["판단규칙"]
    try:
        n_menu = len(menu_changes(cfg, initial_recipe(cfg)))
    except Exception:
        n_menu = "?"
    print("[설정 요약]")
    print(f" 원본: {raw_path.name} ({'있음' if raw_path.exists() else '없음'})")
    print(f" 기간: 검증 {p['검증월']} (롤링 3회차) / 테스트 {p['테스트월']} (실행당 1회)")
    print(f" 변수: 항상 {len(always)}종 + 에이전트가 시험 {len(trial_v)}종 ({', '.join(trial_v)})")
    print(f" 원본 칼럼 변수: 시험 {', '.join(trial_o)}")
    print(f" 소분류 묶음: 에이전트가 직접 구성 (최대 {cfg['소분류처리']['묶음_최대']}개, 처리=제외/분리), "
          f"힌트 {', '.join(cfg['소분류처리'].get('힌트', {}))}")
    print(f" 전처리: 시험 {', '.join(trial_p)}  → 처음 레시피 기준 수정안 메뉴 {n_menu}개 + 묶음 후보")
    print(f" 후보 모델: {len(used)}개 ({', '.join(used)}) + 비교용 규칙 2개, 기준모델 {cfg['기준모델']}")
    print(f" 탐색: {cfg['탐색']['방법']}, 레시피 1개당 {n_grid}개 조합 × 검증 {len(p['검증월'])}회차")
    print(f" 판단: 짝 비교 부트스트랩 {r['부트스트랩_승률']:.0%} + 상승폭 {r['최소상승폭']} + {len(p['검증월'])}개월 중 "
          f"{r['최소상승월수']}개월 이상 상승 → 채택. 수정 최대 {r['수정_최대횟수']}회")
    s = cfg["사전점검"]
    print(f" 사전 점검: 잔여 {s['최소잔여비율']:.0%} / 변화 {s['최소변화비중']:.1%} / 신호 {s['최소신호']} / 상관 {s['최대상관']} "
          f"(최대 {s['최대횟수']}회)")
    import llm
    print(f" LLM: {cfg['LLM']['모드']}" + ("" if cfg["LLM"]["모드"] == "규칙" else f" (모델 {cfg['LLM'][cfg['LLM']['모드']]['모델']})"))
    for w in warn:
        print(f" ⚠ {w}")
    if err:
        for e in err:
            print(f" ✘ {e}")
    else:
        print(" ✔ 설정에 오류가 없습니다.")


if __name__ == "__main__":
    main()
