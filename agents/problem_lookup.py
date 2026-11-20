"""변수표의 행 → 그 행이 어느 문제(기준월·고객층·대분류)인지 찾기.

대부분의 후보 모델은 변수 값만 보면 되지만, 일부 후보(중분류합산, 달력방향)는 '이 행이 몇 월 문제인지'나
'이 문제 아래 중분류 칸들'이 필요하다. 다른 모델과 같은 자리(검증·채택 규칙·테스트)에 끼우기 위해,
레시피마다 변수표 행 내용 → 문제 번호 대응을 등록해 두고 모델이 행 내용으로 문제를 찾는다.
같은 프로세스 안에서만 쓰므로, 이 모델들은 병렬 작업자 대신 본 프로세스에서 돌린다 (model_agent.evaluate).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import SafeError

_REG: dict = {}      # 레시피 이름 → {"rows": 행내용→번호, "keys": 문제 키 표, "X": 변수표}


def register(name: str, X_all: np.ndarray, keys: pd.DataFrame) -> str | None:
    """X_all: 검증용+봉인 행을 이어 붙인 변수표, keys: 같은 순서의 문제 키. 등록 못 하면 이유를 돌려준다."""
    rows = {r.tobytes(): i for i, r in enumerate(np.asarray(X_all, dtype=float))}
    if len(rows) != len(X_all):
        _REG.pop(name, None)
        return "변수표에 내용이 똑같은 행이 있어 문제를 구분할 수 없음"
    _REG[name] = {"rows": rows, "keys": keys.reset_index(drop=True), "X": np.asarray(X_all, dtype=float)}
    return None


def has(name: str) -> bool:
    return name in _REG


def get(name: str) -> dict:
    if name not in _REG:
        raise SafeError(f"레시피 {name}의 문제 대응표가 없습니다.")
    return _REG[name]


def find(name: str, X) -> np.ndarray:
    """변수표 행들 → 문제 번호들."""
    rows = get(name)["rows"]
    try:
        return np.array([rows[np.asarray(r, dtype=float).tobytes()] for r in np.asarray(X, dtype=float)], dtype=int)
    except KeyError:
        raise SafeError("변수표의 원래 행만 받을 수 있습니다 (값을 바꾼 행은 문제를 찾을 수 없음).") from None
