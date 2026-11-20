"""명세서(카드거래 set1) 형식의 가짜 원본 CSV 생성기 — 실제 데이터 없이 에이전트 전체를 시험하기 위함.

모든 값은 난수로 만든 가상값이다. 지역 이름도 '가상구1' 같은 가짜 이름이다.
실제 데이터의 특징을 흉내 낸다:
  - 비수도권은 시군구 칸에 시도 이름이 들어감 (명세서 익명처리: 상위범주화)
  - 그래서 '키가 같은 행'과 '12칼럼이 모두 같은 행(익명처리형 중복)'이 생김
  - 건수 1건 + 천원 단위 금액 행이 많음
  - 세금 같은 비소비성 소분류가 특정 달에 크게 튐
  - 일부러 넣은 이상값: 범주 밖 성별·연령, 대분류 결측, 쉼표 붙은 금액, 전각 공백

실행 (AutoML-AGENT 폴더에서):
    ..\\.venv\\Scripts\\python tools\\make_fake_raw.py --rows 300000
    ..\\.venv\\Scripts\\python tools\\make_fake_raw.py --rows 300000 --encoding cp949 --out ../데이터/가짜데이터/fake_raw_cp949.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

CATS = {   # 대분류 → 중분류 → 소분류 (일부는 config 힌트와 같은 이름, 나머지는 가상)
    "교육/교양/서비스": {"서비스": ["국세/지방세", "공과금", "통신비(시청료)", "기타서비스"], "학원": ["입시학원", "외국어학원"]},
    "숙박/여행": {"렌터카": ["렌터카"], "여행/교통": ["항공사", "철도", "여행사"], "숙박": ["호텔", "콘도"]},
    "식생활": {"일반음식점": ["한식", "양식", "중식"], "주점": ["주점"], "식료품": ["정육", "제과"]},
    "유통": {"유통업": ["연쇄점", "편의점", "면세점"], "대형마트": ["대형마트"]},
    "의료/보험/위생": {"보험": ["보험"], "의료기관": ["의원", "치과", "성형외과"], "약국": ["약국"]},
    "의생활": {"의복": ["정장", "캐주얼"], "신발": ["신발"]},
    "자동차/주유소": {"주유소": ["주유소", "LPG충전소"], "자동차판매": ["자동차판매"]},
    "주생활": {"가구": ["가구"], "가전": ["가전제품"], "인테리어": ["인테리어"]},
}
SUBS = [(l, m, s) for l, ms in CATS.items() for m, ss in ms.items() for s in ss]

CAPITAL = {"서울": [f"가상구{i}" for i in range(1, 7)],
           "경기": [f"가상시{i}" for i in range(1, 6)],
           "인천": [f"가상구{i}" for i in range(1, 4)]}          # '가상구1'이 서울·인천 둘 다에 있음 (쌍 코드 시험용)
NON_CAPITAL = [f"가상도{c}" for c in "ABCDE"]                    # 비수도권: 시군구 = 시도 이름
REGIONS = [(s, g) for s, gs in CAPITAL.items() for g in gs] + [(s, s) for s in NON_CAPITAL]
MONTHS = [f"2025{m:02d}" for m in range(1, 13)]
SEXES, AGES = ["1", "2"], ["20", "30", "40", "50", "60", "70"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=300_000)
    ap.add_argument("--out", default=str(ROOT.parent / "데이터" / "가짜데이터" / "fake_raw.csv"))
    ap.add_argument("--encoding", default="utf-8", choices=["utf-8", "cp949"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    n = a.rows

    # 고객층 × 대분류 × 월의 흐름 (예측할 만한 구조: 업종·고객층마다 다른 추세와 계절성)
    seg_cat = [(s, g, l) for s in SEXES for g in AGES for l in CATS]
    flow = {}
    for key in seg_cat:
        walk = np.cumsum(rng.normal(0, 0.06, 12))
        season = 0.08 * np.sin(np.arange(12) / 12 * 2 * np.pi + rng.uniform(0, 6.28))
        flow[key] = np.exp(walk + season)
    tax_spike = {m: (2.5 if m in {"202503", "202505", "202507", "202509", "202512"} else 1.0) for m in MONTHS}

    month = rng.choice(MONTHS, n)
    sex = rng.choice(SEXES, n)
    age = rng.choice(AGES, n, p=[0.12, 0.18, 0.22, 0.22, 0.16, 0.10])
    sub_i = rng.integers(0, len(SUBS), n)
    cust_i = rng.integers(0, len(REGIONS), n)
    same = rng.random(n) < 0.7                                 # 70%는 사는 지역에서 결제
    merch_i = np.where(same, cust_i, rng.integers(0, len(REGIONS), n))

    l = np.array([SUBS[i][0] for i in sub_i]); m = np.array([SUBS[i][1] for i in sub_i]); s = np.array([SUBS[i][2] for i in sub_i])
    mult = np.array([flow[(sx, ag, ll)][int(mm[-2:]) - 1] for sx, ag, ll, mm in zip(sex, age, l, month)])
    mult *= np.where(s == "국세/지방세", [tax_spike[mm] for mm in month], 1.0)
    ct = np.maximum(1, rng.poisson(np.where(rng.random(n) < 0.25, 0.0, 6.0)))   # 약 25%는 1건
    am = np.round(ct * 30000 * mult * rng.lognormal(0, 0.5, n)).astype(np.int64)
    one = ct == 1
    am[one] = (np.round(am[one] / 1000) * 1000).astype(np.int64)            # 1건은 천원 단위 (상단라운딩 흉내)

    df = pd.DataFrame({
        "crym": month, "ma_fem_dc": sex, "age_g": age,
        "mgpo_nm": [REGIONS[i][0] for i in cust_i], "ccd_nm": [REGIONS[i][1] for i in cust_i],
        "mgpo_nm_mc": [REGIONS[i][0] for i in merch_i], "ccd_nm_mc": [REGIONS[i][1] for i in merch_i],
        "tco_bt_hlv_c_nm": l, "tco_bt_mcls_c_nm": m, "tco_btc_nm": s,
        "sl_ct": ct.astype(str), "sl_am": am.astype(str),
    })

    # 익명처리형 중복: 비수도권·1건 행 일부를 그대로 한 번 더 (실제 원본처럼 약 2%)
    noncap = df["mgpo_nm"].isin(NON_CAPITAL).to_numpy() & one
    dup = df[noncap].sample(frac=min(1.0, 0.02 * n / max(noncap.sum(), 1)), random_state=a.seed)
    df = pd.concat([df, dup], ignore_index=True)

    # 일부러 넣는 이상값 (raw_agent 점검 시험용)
    k = len(df)
    pick = lambda c: rng.choice(k, c, replace=False)
    df.loc[pick(30), "ma_fem_dc"] = "9"                        # 범주 밖 성별 → 제외
    df.loc[pick(20), "age_g"] = "15"                           # 범주 밖 연령 → 제외
    df.loc[pick(10), "tco_bt_hlv_c_nm"] = ""                   # 대분류 결측 → 제외
    i = pick(50); df.loc[i, "sl_am"] = df.loc[i, "sl_am"].astype(int).map("{:,}".format)   # 쉼표 붙은 금액
    i = pick(50); df.loc[i, "tco_bt_hlv_c_nm"] = df.loc[i, "tco_bt_hlv_c_nm"] + "　"   # 전각 공백
    df = df.sample(frac=1, random_state=a.seed).reset_index(drop=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding=a.encoding)
    print(f"가짜 원본 {len(df):,}행 → {out} ({a.encoding}, 익명처리형 중복 {len(dup):,}행 포함)")


if __name__ == "__main__":
    main()
