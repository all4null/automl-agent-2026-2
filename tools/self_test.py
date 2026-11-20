"""가짜데이터로 에이전트의 안전장치를 자동 점검한다 (계획서 Verification).

    ..\\.venv\\Scripts\\python tools\\self_test.py

확인 항목
  1. 원본 행을 지우지 않았나 (검산: 행 수·금액 1원 단위 일치)
  2. 대분류 하나의 거래를 통째로 빼도 문제 432개·정답이 그대로인가 (키 격자 고정, 거래없음 표시)
  3. 나쁜 묶음이 사전 점검에서 걸러지나 (남는 금액이 거의 없는 '제외' → 🔴, 아주 작은 묶음 → 🟡)
  4. 테스트를 두 번 열면 멈추나
"""
import copy
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import RunLog, SafeError, cfg_path, load_config  # noqa: E402
from state import State  # noqa: E402

FAKE = Path(__file__).resolve().parents[2] / "데이터" / "가짜데이터" / "fake_raw.csv"


def main():
    from agents import data_agent, executor, model_agent, planner, raw_agent
    cfg = load_config("config.yaml")
    cfg["경로"]["결과폴더"], cfg["경로"]["보안폴더"] = cfg["경로"]["가짜_결과폴더"], cfg["경로"]["가짜_보안폴더"]
    run_dir = cfg_path(cfg, "결과폴더") / "self_test" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    s = State(config=cfg, run_dir=run_dir, log=RunLog())
    results = []

    def check(name, ok, detail=""):
        results.append(ok)
        print(f"{'✔' if ok else '✘'} {name} {detail}")

    raw_agent.run(s, str(FAKE))
    q = s.quality
    check("1. 원본 행을 지우지 않음", q["검산"]["통과"] and q["검산"]["원본_행수"] == q["행수"]["원본"],
          f"(원본 {q['검산']['원본_행수']:,}행 = 정제 {q['행수']['원본']:,}행, 중복 초과 {q.get('중복', {}).get('완전중복_초과행수', '?')}행 유지)")

    data_agent.build_truth(s)        # run.py와 같은 순서: 정답표 → 계획
    planner.run(s)
    A = s.recipes["A"]
    dA = s.datasets["A"]
    st = data_agent.subcat_stats(s)
    cat = s.quality["업종"]["대분류"][-1]
    whole = [x for x in st["소분류"] if x.startswith(cat + " > ")]
    R = copy.deepcopy(A); R["이름"] = "T1"
    R["묶음"] = [{"이름": "통째로", "소분류": whole, "처리": "제외", "이유": "시험"}]
    dR = data_agent.build(s, R)
    same = (dR["dev"]["meta"][["crym", "sex", "age_band", "cat_l", "label"]].equals(dA["dev"]["meta"][["crym", "sex", "age_band", "cat_l", "label"]])
            and len(dR["dev"]["X"]) + len(dR["sealed"]["X"]) == 432)
    check("2. 대분류 통째로 빼도 같은 문제", same and "거래없음" in dR["cols"],
          f"(문제 {len(dR['dev']['X']) + len(dR['sealed']['X'])}개, 거래없음 표시 {'있음' if '거래없음' in dR['cols'] else '없음'})")

    pc = data_agent.precheck(s, dR, dA, {"종류": "묶음", "묶음": R["묶음"][0]})
    check("3a. 남는 금액이 없는 '제외' 묶음 → 🔴", pc["신호등"] == "🔴", f"({' / '.join(pc['이유'])})")
    small = st.nsmallest(1, "금액비중")["소분류"].tolist()
    s.config["사전점검"]["최소변화비중"] = float(st["금액비중"].min()) * 2      # 가장 작은 소분류가 기준 미만이 되게
    T = copy.deepcopy(A); T["이름"] = "T2"; T["묶음"] = [{"이름": "작은묶음", "소분류": small, "처리": "분리", "이유": "시험"}]
    pc2 = data_agent.precheck(s, data_agent.build(s, T), dA, {"종류": "묶음", "묶음": T["묶음"][0]})
    check("3b. 아주 작은 묶음 → 🟡", pc2["신호등"] == "🟡", f"({' / '.join(pc2['이유'])})")

    s.datasets["A"] = dA
    model_agent.evaluate(s, dA, A)
    s.current = "A"
    executor.run(s)
    try:
        executor.run(s)
        check("4. 테스트 두 번 열기 차단", False)
    except SafeError as e:
        check("4. 테스트 두 번 열기 차단", True, f"({e})")
    print(f"\n{sum(results)}/{len(results)} 통과")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
