"""노트북 baseline(테스트 Macro-F1 0.5197) 재현 점검 — 에이전트 코드가 노트북과 같은 계산을 하는지 확인한다.

노트북과 같은 조건으로 에이전트의 부품만 돌린다:
  - 검증: 8월 한 달 (롤링 아님), 학습 3~7월 → C 선택 → 3~8월로 재학습 → 9~11월 테스트
  - 모델: 로지스틱 회귀 (C ∈ {0.01, 0.1, 1, 10}, class_weight=balanced), 동점이면 작은 C
  - 변수: 기본 + 객단가 + 업종비중 + 달력, 극단값 상한 없음, 원-핫 표시는 표준화하지 않음
확인 항목
  1. 정답표 금액이 결과/01_집계/panel_monthly.csv 와 1원 단위로 같은가
  2. 테스트 Macro-F1이 결과/02_baseline/test_scores.csv 의 로지스틱 점수와 같은가

    .venv\\Scripts\\python tools\\reproduce_baseline.py                 실데이터 (원본을 읽음: 사용자 요청 시에만 실행)
    .venv\\Scripts\\python tools\\reproduce_baseline.py --raw <가짜.csv> 가짜데이터로 동작만 확인 (비교 대상 없음)
화면과 결과에는 집계값(점수·불일치 칸 수)만 출력한다.
"""
import argparse
import copy
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RunLog, cfg_path, load_config  # noqa: E402
from state import State  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw")
    a = ap.parse_args()
    from agents import data_agent, executor, model_agent, planner, raw_agent
    from run import is_fake

    cfg = load_config("config.yaml")
    fake = bool(a.raw and is_fake(a.raw))
    if fake:
        cfg["경로"]["결과폴더"], cfg["경로"]["보안폴더"] = cfg["경로"]["가짜_결과폴더"], cfg["경로"]["가짜_보안폴더"]
    cfg["기간"]["검증월"] = [202508]
    cfg["판단규칙"]["최소상승월수"] = 1
    for m, v in cfg["후보모델"].items():
        v["사용"] = "예" if m == "로지스틱회귀" else "아니오"
    run_dir = cfg_path(cfg, "결과폴더") / "reproduce" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    s = State(config=cfg, run_dir=run_dir, log=RunLog())

    raw_agent.run(s, a.raw)
    data_agent.build_truth(s)

    # 1) 정답표 금액 대조
    ref = PROJECT / "결과" / "01_집계" / "panel_monthly.csv"
    if not fake and ref.exists():
        p = pd.read_csv(ref, encoding="utf-8-sig", dtype={"sex": str})
        p = p.rename(columns={"category_l": "cat_l", "sl_am": "am_ref", "sl_ct": "ct_ref"})
        j = s.truth.merge(p[["crym", "sex", "age_band", "cat_l", "am_ref", "ct_ref"]],
                          on=["crym", "sex", "age_band", "cat_l"], how="outer", indicator=True)
        bad = int(((j["_merge"] != "both") | (j["am"] != j["am_ref"]) | (j["ct"] != j["ct_ref"])).sum())
        print(f"[대조1] 정답표 {len(s.truth)}칸 vs panel_monthly {len(p)}칸 → 금액·건수 불일치 {bad}칸 "
              f"({'✔ 일치' if bad == 0 else '✘ 다름'})")

    # 2) 노트북과 같은 레시피로 검증·테스트
    r = planner.initial_recipe(cfg)
    r["이름"], r["설명"] = "BASE", "노트북 baseline 조건"
    for k in ["객단가", "업종비중", "달력변수"]:
        r["변수"][k] = True
    r["전처리"]["극단값_상한"] = False
    s.recipes = {"BASE": r}
    s.first_recipe = s.current = "BASE"
    s.datasets["BASE"] = data_agent.build(s, r)
    res = model_agent.evaluate(s, s.datasets["BASE"], r)
    executor.run(s)
    got = s.test["최종"]["macro_f1"]
    print(f"[대조2] 선택 C = {res['설정'].get('C')}, 8월 검증 F1 {res['검증F1']:.4f}, 테스트 F1 {got:.4f}")
    ref2 = PROJECT / "결과" / "02_baseline" / "test_scores.csv"
    if not fake and ref2.exists():
        t = pd.read_csv(ref2, encoding="utf-8-sig")
        exp = float(t.loc[t.iloc[:, 0].astype(str).str.startswith("로지스틱"), "macro_f1"].iloc[0])
        print(f"         노트북 테스트 F1 {exp:.4f} → {'✔ 재현' if abs(exp - got) < 5e-4 else '✘ 다름 (차이 %.4f)' % (got - exp)}")
    s.log.save(run_dir / "run_log.txt")


if __name__ == "__main__":
    main()
