"""결과 폴더에 지역 이름이 새지 않았는지 검사한다 (계획서 Verification 4).

보안 폴더의 지역 대응표(region_codes.csv)에 있는 이름이 결과 폴더의 파일에 나오는지 센다.
이름 자체는 화면에 출력하지 않고 '발견 n건'과 파일 이름만 출력한다.

    ..\\.venv\\Scripts\\python tools\\check_results.py                 실데이터 결과 (결과/03_agent)
    ..\\.venv\\Scripts\\python tools\\check_results.py --fake          가짜데이터 결과 (결과/테스트_agent)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import cfg_path, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true")
    a = ap.parse_args()
    cfg = load_config("config.yaml")
    res = cfg_path(cfg, "가짜_결과폴더" if a.fake else "결과폴더")
    sec = cfg_path(cfg, "가짜_보안폴더" if a.fake else "보안폴더")
    names = set()
    for f in sec.glob("clean_*/region_codes.csv"):
        d = pd.read_csv(f, encoding="utf-8-sig", dtype=str)
        names |= {n for n in pd.concat([d["시도"], d["시군구"]]).dropna() if len(n) >= 2}
    if not names:
        print("지역 대응표가 없습니다 (아직 실행 전이거나 보안폴더 경로 확인 필요)")
        return
    files = [p for p in res.rglob("*") if p.suffix in {".csv", ".json", ".html", ".txt"}]
    hits = {}
    for p in files:
        text = p.read_text(encoding="utf-8-sig", errors="ignore")
        n = sum(text.count(nm) for nm in names)
        if n:
            hits[p.relative_to(res)] = n
    print(f"검사: 결과 파일 {len(files)}개 × 지역 이름 {len(names)}개")
    if hits:
        print(f"✘ 지역 이름 발견 {sum(hits.values())}건:")
        for p, n in hits.items():
            print(f"   {p}: {n}건")
        sys.exit(1)
    print("✔ 발견 0건")


if __name__ == "__main__":
    main()
