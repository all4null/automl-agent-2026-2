"""공통 도구: 설정 읽기·검사, 로그, 점수 계산처럼 여러 에이전트가 함께 쓰는 작은 함수들."""
from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent          # 저장소 최상위 폴더
YES = {"예", "켜기", "항상", True}              # 설정 파일에서 '참'으로 보는 값


class SafeError(Exception):
    """화면에 그대로 보여 줘도 되는 오류 (원본 데이터 내용이 들어 있지 않음)."""


# ---------------------------------------------------------------- .env (API 키)
def load_dotenv(path: Path = ROOT / ".env"):
    """저장소 최상위 .env 의 '이름=값' 줄을 환경변수로 읽는다. 이미 설정된 환경변수는 덮어쓰지 않는다.
    값(키)은 화면이나 로그에 절대 출력하지 않는다. .env 는 .gitignore로 GitHub에 올라가지 않는다."""
    import os
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        val = val.strip().strip('"').strip("'")
        if key and val and key not in os.environ:
            os.environ[key] = val


# ---------------------------------------------------------------- 설정
def load_config(path: str | Path) -> dict:
    load_dotenv()
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise SafeError(f"설정 파일이 없습니다: {path}")
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        mark = getattr(e, "problem_mark", None)
        where = f" ({mark.line + 1}번째 줄 근처)" if mark else ""
        raise SafeError(f"설정 파일 형식 오류{where}. 들여쓰기와 '이름: 값' 사이 띄어쓰기를 확인하세요.")
    cfg["_path"] = str(path)
    return cfg


def cfg_path(cfg: dict, key: str) -> Path:
    """설정의 경로 값을 저장소 최상위 폴더 기준 절대경로로."""
    p = Path(cfg["경로"][key])
    return p if p.is_absolute() else (ROOT / p).resolve()


def is_on(v) -> bool:
    return v in YES


def check_config(cfg: dict) -> tuple[list[str], list[str]]:
    """설정을 검사해 (오류 목록, 경고 목록)을 돌려준다. 오류는 비전공자도 고칠 수 있게 쓴다."""
    err, warn = [], []

    def need(section, key=None):
        if section not in cfg:
            err.append(f"'{section}' 항목이 없습니다.")
            return False
        if key and key not in cfg[section]:
            err.append(f"{section} > '{key}' 항목이 없습니다.")
            return False
        return True

    for s in ["경로", "기간", "성별", "연령구간", "변수", "원본변수", "전처리", "후보모델", "탐색", "판단규칙", "사전점검", "LLM"]:
        need(s)
    if err:
        return err, warn

    # 기간
    p = cfg["기간"]
    try:
        vm, tm, first = [int(x) for x in p["검증월"]], [int(x) for x in p["테스트월"]], int(p["첫_기준월"])
        if sorted(vm) != vm or sorted(tm) != tm:
            err.append("기간 > 검증월·테스트월은 작은 달부터 순서대로 적어야 합니다.")
        if vm and tm and max(vm) >= min(tm):
            err.append("기간 > 검증월은 모두 테스트월보다 앞이어야 합니다.")
        if vm and first >= min(vm):
            err.append("기간 > 첫_기준월은 첫 검증월보다 앞이어야 합니다.")
        if tm and month_add(max(tm), 1) > int(p["전체_끝월"]):
            err.append("기간 > 마지막 테스트월의 다음 달이 전체_끝월 안에 있어야 정답을 만들 수 있습니다.")
    except (KeyError, TypeError, ValueError):
        err.append("기간 > 검증월·테스트월·첫_기준월은 202506 같은 숫자로 적어야 합니다.")

    # 켜기/끄기 형식
    for name, v in cfg["변수"].items():
        if v.get("처음값") not in {"항상", "끄기", "켜기"}:
            err.append(f"변수 > {name} > 처음값은 '항상' 또는 '끄기'여야 합니다. (지금 값: {v.get('처음값')})")
        if v.get("시험") not in {"예", "아니오"}:
            err.append(f"변수 > {name} > 시험은 '예' 또는 '아니오'여야 합니다.")
    for name, v in cfg["원본변수"].items():
        if isinstance(v, dict) and "선택지" not in v and v.get("처음값") not in {"켜기", "끄기", "항상"}:
            err.append(f"원본변수 > {name} > 처음값은 '켜기' 또는 '끄기'여야 합니다.")
    for name, v in cfg["전처리"].items():
        if v.get("처음값") not in v.get("선택지", []):
            err.append(f"전처리 > {name} > 처음값({v.get('처음값')})이 선택지 {v.get('선택지')} 안에 없습니다.")

    # 모델
    from agents.model_agent import MODEL_BUILDERS, optional_missing
    used = [m for m, v in cfg["후보모델"].items() if is_on(v.get("사용"))]
    for m, v in cfg["후보모델"].items():
        if m not in MODEL_BUILDERS:
            err.append(f"후보모델 > '{m}' 은(는) 없는 모델 이름입니다. 가능한 이름: {', '.join(MODEL_BUILDERS)}")
        elif is_on(v.get("사용")) and optional_missing(m):
            warn.append(f"후보모델 > {m}: 패키지가 설치되어 있지 않아 건너뜁니다.")
    base = cfg.get("기준모델")
    if base not in used:
        err.append(f"기준모델({base})은 후보모델에서 '사용: 예'여야 합니다.")
    if not used:
        err.append("후보모델 중 '사용: 예'인 모델이 하나도 없습니다.")

    # 탐색·판단 규칙
    if cfg["탐색"].get("방법") not in {"grid", "optuna"}:
        err.append(f"탐색 > 방법은 grid 또는 optuna 여야 합니다. (지금 값: {cfg['탐색'].get('방법')})")
    if cfg["탐색"].get("방법") == "optuna":
        try:
            import optuna  # noqa: F401
        except ImportError:
            warn.append("탐색 > optuna가 설치되어 있지 않아 grid로 대신 실행합니다.")
    r = cfg["판단규칙"]
    for k, lo, hi in [("부트스트랩_승률", 0.5, 1.0), ("최소상승폭", 0.0, 1.0)]:
        if not (isinstance(r.get(k), (int, float)) and lo <= r[k] <= hi):
            err.append(f"판단규칙 > {k} 은(는) {lo}~{hi} 사이 숫자여야 합니다. (지금 값: {r.get(k)})")
    nv = len(cfg["기간"].get("검증월", []))
    if not (isinstance(r.get("최소상승월수"), int) and 1 <= r["최소상승월수"] <= max(nv, 1)):
        err.append(f"판단규칙 > 최소상승월수 는 1~{nv} 사이 정수여야 합니다.")

    # LLM
    mode = cfg["LLM"].get("모드")
    if mode not in {"규칙", "ollama", "openai"}:
        err.append(f"LLM > 모드는 규칙 / ollama / openai 중 하나여야 합니다. (지금 값: {mode})")
    import os
    if mode == "openai" and not os.environ.get("OPENAI_API_KEY"):
        warn.append("LLM > openai 모드인데 OPENAI_API_KEY가 없습니다 (저장소 최상위의 .env 확인). 규칙 모드로 대신 실행합니다.")
    return err, warn


# ---------------------------------------------------------------- 로그
class RunLog:
    """화면에 출력하면서 기록도 남기는 로그. 원본 값은 절대 넣지 않는다."""

    def __init__(self):
        self.lines: list[str] = []

    def __call__(self, msg: str):
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        self.lines.append(line)

    def save(self, path: Path):
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- 날짜·문자열
def month_add(yyyymm: int, k: int) -> int:
    y, m = divmod(int(yyyymm), 100)
    t = y * 12 + (m - 1) + k
    return (t // 12) * 100 + t % 12 + 1


def month_range(start: int, end: int) -> list[int]:
    out, m = [], int(start)
    while m <= int(end):
        out.append(m)
        m = month_add(m, 1)
    return out


def norm_text(s):
    """전각→반각(NFKC), 연속 공백 1개로, 앞뒤 공백 제거. 빈 값은 None (계획서 3-3)."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    s = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(s))).strip()
    return s or None


def slog(x):
    """sign(x)·ln(1+|x|) (계획서 3-5)."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log1p(np.abs(x))


def write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    return str(o)


# ---------------------------------------------------------------- 점수
def _f1_from_counts(tp, fp, fn, tn):
    """증가(1)·감소(0) 두 클래스 F1의 평균 = Macro-F1. 배열도 받는다."""
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_pos = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        f1_neg = np.where(2 * tn + fn + fp > 0, 2 * tn / (2 * tn + fn + fp), 0.0)
    return (f1_pos + f1_neg) / 2


def macro_f1(y, p) -> float:
    y, p = np.asarray(y).astype(int), np.asarray(p).astype(int)
    tp = np.sum((y == 1) & (p == 1)); fp = np.sum((y == 0) & (p == 1))
    fn = np.sum((y == 1) & (p == 0)); tn = np.sum((y == 0) & (p == 0))
    return float(_f1_from_counts(tp, fp, fn, tn))


def bootstrap_f1(y, p, n: int, seed: int) -> np.ndarray:
    """문제를 무작위로 다시 뽑아 n번 채점한 Macro-F1 배열."""
    y, p = np.asarray(y).astype(int), np.asarray(p).astype(int)
    idx = np.random.default_rng(seed).integers(0, len(y), size=(n, len(y)))
    Y, P = y[idx], p[idx]
    tp = ((Y == 1) & (P == 1)).sum(1); fp = ((Y == 0) & (P == 1)).sum(1)
    fn = ((Y == 1) & (P == 0)).sum(1); tn = ((Y == 0) & (P == 0)).sum(1)
    return _f1_from_counts(tp, fp, fn, tn)


def paired_win_rate(y, p_old, p_new, n: int, seed: int, blocks=None) -> float:
    """같은 문제 묶음을 다시 뽑아 채점했을 때 새 예측이 기존 예측보다 '높은' 비율 (짝 비교).

    blocks(예: 기준월)를 주면 문제가 아니라 '달'을 통째로 다시 뽑는다 (달 단위 블록 부트스트랩).
    같은 달 문제끼리 함께 움직일 때, 문제 단위로 뽑으면 근거를 실제보다 크게 센다.
    """
    y, a, b = (np.asarray(v).astype(int) for v in (y, p_old, p_new))
    rng = np.random.default_rng(seed)
    if blocks is not None:
        groups = [np.where(np.asarray(blocks) == g)[0] for g in sorted(set(np.asarray(blocks).tolist()))]
        pick = rng.integers(0, len(groups), size=(n, len(groups)))        # 매번 달을 '달 수만큼' 중복 허용으로 다시 뽑음
        if len({len(g) for g in groups}) > 1:                              # 달마다 문제 수가 다르면 한 번씩 채점
            sets = [np.concatenate([groups[j] for j in row]) for row in pick]
            return float(np.mean([macro_f1(y[s], b[s]) > macro_f1(y[s], a[s]) for s in sets]))
        idx = np.array([np.concatenate([groups[j] for j in row]) for row in pick])
    else:
        idx = rng.integers(0, len(y), size=(n, len(y)))
    Y, A, B = y[idx], a[idx], b[idx]

    def f1(P):
        tp = ((Y == 1) & (P == 1)).sum(1); fp = ((Y == 0) & (P == 1)).sum(1)
        fn = ((Y == 1) & (P == 0)).sum(1); tn = ((Y == 0) & (P == 0)).sum(1)
        return _f1_from_counts(tp, fp, fn, tn)

    return float(np.mean(f1(B) > f1(A)))


def adoption_test(y, months, p_old, p_new, rule: dict, seed: int) -> dict:
    """채택 규칙 (config 판단규칙): ① 짝 비교 부트스트랩 승률 ② 최소상승폭 ③ 최소상승월수. 모두 넘어야 채택."""
    y, months = np.asarray(y), np.asarray(months)
    f_old, f_new = macro_f1(y, p_old), macro_f1(y, p_new)
    by_month = rule.get("부트스트랩_단위", "문제") == "달"
    win = paired_win_rate(y, p_old, p_new, rule["부트스트랩_횟수"], seed, months if by_month else None)
    up_months = 0
    per_month = {}
    acc = rule.get("월별_비교_지표", "macro_f1") == "정답률"     # ③을 월별 정답률로 볼지, 월별 Macro-F1로 볼지

    def month_score(t, p):
        return float((t == p).mean()) if acc else macro_f1(t, p)
    for m in sorted(set(months.tolist())):
        k = months == m
        a, b = month_score(y[k], np.asarray(p_old)[k].astype(int)), month_score(y[k], np.asarray(p_new)[k].astype(int))
        per_month[int(m)] = (round(a, 4), round(b, 4))
        up_months += int(b > a)
    ok1 = win >= rule["부트스트랩_승률"]
    ok2 = (f_new - f_old) >= rule["최소상승폭"]
    ok3 = up_months >= rule["최소상승월수"]
    reasons = []
    reasons.append(f"{'달 단위 ' if by_month else ''}부트스트랩 승률 {win:.0%} ({'통과' if ok1 else '미달'}, 기준 {rule['부트스트랩_승률']:.0%})")
    reasons.append(f"상승폭 {f_new - f_old:+.3f} ({'통과' if ok2 else '미달'}, 기준 +{rule['최소상승폭']})")
    reasons.append(f"{'정답률' if acc else 'F1'} 상승 월 {up_months}/{len(per_month)} ({'통과' if ok3 else '미달'}, 기준 {rule['최소상승월수']})")
    return {"채택": bool(ok1 and ok2 and ok3), "기존F1": round(f_old, 4), "새F1": round(f_new, 4),
            "승률": round(win, 3), "상승월수": up_months, "월별": per_month, "이유": " / ".join(reasons)}
