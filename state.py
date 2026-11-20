"""작업 노트 양식.

에이전트들은 이 노트 하나를 차례로 넘겨받아 '자기 칸'만 채운다.
어느 칸이 비어 있으면, 그 칸을 채우는 단계까지 아직 오지 않은 것이다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class State:
    # ---- run.py 가 채움 ----
    config: dict                                    # config.yaml 내용
    run_dir: Path                                   # 이번 실행의 결과 폴더 (결과/03_agent/runs/실행시각)
    started_at: str = ""
    replay: dict | None = None                      # --replay 로 불러온 이전 LLM 답 (없으면 None)

    # ---- [0] raw_agent 가 채움 ----
    clean_path: Path | None = None                  # 정제 원본 parquet 경로 (보안 폴더)
    quality: dict = field(default_factory=dict)     # 원본 점검 결과 요약 (집계값만)
    data_review: dict | None = None                 # 원본 담당의 해석: 데이터 주의점, LLM이 제안한 소분류 후보 목록
    hints: dict = field(default_factory=dict)       # 소분류 묶음 후보 목록 (config 힌트 + 원본 담당 제안)
    search_plan: dict | None = None                 # [3] 모델 담당의 탐색 계획: 모델 → 하이퍼파라미터 범위 (모든 레시피에 같은 계획)

    # ---- [1] planner 가 채움 ----
    recipes: dict[str, dict] = field(default_factory=dict)   # 레시피 이름 → 레시피 (전처리·변수 선택 묶음)
    first_recipe: str = ""                          # 처음 레시피 이름 (테스트에서 최종과 비교용)

    # ---- [2] data_agent 가 채움 ----
    truth: pd.DataFrame | None = None               # 정답표: 고객층×대분류×월 금액과 다음 달 증가/감소 (고정)
    problem_sig: tuple | None = None                # 문제 지문 (기준월·고객층·대분류·실제방향). 레시피마다 같아야 함
    datasets: dict[str, dict] = field(default_factory=dict)  # 레시피 이름 → 변수표 (검증용 dev / 봉인된 test)
    subcat_stats: pd.DataFrame | None = None        # 1~5월 소분류 통계 (묶음 만들기 참고용)
    prechecks: list[dict] = field(default_factory=list)      # 사전 점검 기록 (신호등)

    # ---- [3] model_agent 가 채움 ----
    trials: list[dict] = field(default_factory=list)          # 검증한 모든 후보와 점수
    results: dict[str, dict] = field(default_factory=dict)    # 레시피 이름 → 그 레시피의 선택 모델·검증 예측
    n_validations: int = 0                          # 검증을 몇 번 돌렸는지 (후보 × 회차)

    # ---- [5] reviewer.check 가 채움 ----
    current: str = ""                               # 지금 채택된 레시피 이름
    revisions: list[dict] = field(default_factory=list)       # 수정 시도 기록 (채택/기각과 이유)

    # ---- [4] executor 가 채움 ----
    test: dict | None = None                        # 테스트 결과 (한 번만 채워짐)
    monthly: dict = field(default_factory=dict)     # 월별 갱신 실험 결과
    predictions: pd.DataFrame | None = None         # 예측표 (계획서 1-4 형식)
    exec_plan: dict | None = None                   # 실행 담당의 월별 갱신 실험 계획 (테스트 전에 정함)
    test_interpretation: dict | None = None         # 실행 담당의 테스트 해석 (테스트 후, 고르는 데 쓰지 않음)

    # ---- 모두 ----
    decisions: list[dict] = field(default_factory=list)       # 결정 기록: 단계 | 항목 | 후보 | 선택 | 결정자 | 이유
    llm_calls: list[dict] = field(default_factory=list)       # LLM 호출 기록 (AI 활용 내역)
    explanation: dict | None = None                 # [5] reviewer.report: LLM이 쓴 결과 해석 (LLM 모드일 때만)
    log: Any = None                                 # common.RunLog

    def decide(self, 단계: str, 항목: str, 후보: str, 선택: str, 결정자: str, 이유: str):
        """결정 기록에 한 줄 추가 (대시보드 '에이전트 결정 과정' 표가 됨)."""
        self.decisions.append({"단계": 단계, "결정 항목": 항목, "후보(점수)": 후보,
                               "선택": 선택, "결정자": 결정자, "이유": 이유})
