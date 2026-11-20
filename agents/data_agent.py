"""[2] 데이터 담당: 정답표(고정)와 레시피별 변수표를 만들고, 모델에 넣기 전에 사전 점검을 한다.

정답표: 원본 거래를 하나도 빼지 않고 고객층×대분류×월로 합산 → 다음 달 증가(1)/감소(0). 레시피와 무관하게 고정.
변수표: 레시피(소분류 묶음 처리, 변수 선택)에 따라 만든 입력 변수. 정답표의 키 격자에 붙인다.
정제 원본은 행 단위지만, 이 파일 밖으로는 고객층×대분류×월(또는 그 아래 업종·지역 코드) 합계만 나간다.
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from agents import bottomup, problem_lookup, timeseries_models
from common import SafeError, is_on, month_add, month_range, slog, write_csv

UNIT = ["sex", "age_band", "cat_l"]          # 고객층(성별·연령구간) × 대분류 = 시계열 하나
KEYS = ["crym"] + UNIT                       # 문제 하나의 키


# ---------------------------------------------------------------- 기간 도우미
def periods(cfg) -> dict:
    p = cfg["기간"]
    valid, test = [int(m) for m in p["검증월"]], [int(m) for m in p["테스트월"]]
    return {"all": month_range(p["전체_시작월"], p["전체_끝월"]), "first": int(p["첫_기준월"]),
            "valid": valid, "test": test, "dev_end": valid[-1],
            "stat_end": month_add(valid[0], -1)}       # 통계(1~5월)는 첫 검증월 전까지만 사용


def _con(state):
    con = duckdb.connect()
    path = str(state.clean_path).replace("\\", "/").replace("'", "''")
    con.execute(f"CREATE VIEW clean AS SELECT * FROM read_parquet('{path}') WHERE reason IS NULL")
    return con


def _bands(cfg) -> list[str]:
    seen = []
    for v in cfg["연령구간"].values():
        if v not in seen:
            seen.append(v)
    return seen


def _agg(con, cfg, extra: list[str], view: str = "clean", where: str = "") -> pd.DataFrame:
    """정제 원본 → 월 × 성별 × 연령구간 × 대분류 (+ extra) 합계. 같은 키는 항상 합산 (중복 행도 지우지 않음)."""
    cols = ", ".join(["crym", "sex", "age_g", "cat_l"] + extra)
    df = con.execute(f"SELECT {cols}, CAST(sum(am) AS BIGINT) am, CAST(sum(ct) AS BIGINT) ct, count(*) n "
                     f"FROM {view} {where} GROUP BY ALL").df()
    df["age_band"] = df["age_g"].map({str(k): v for k, v in cfg["연령구간"].items()})
    return df.groupby(KEYS + extra, as_index=False)[["am", "ct", "n"]].sum()


def _grid(cfg, cats: list[str], months: list[int]) -> pd.DataFrame:
    idx = pd.MultiIndex.from_product([months, sorted(cfg["성별"]), _bands(cfg), cats], names=KEYS)
    return idx.to_frame(index=False)


def _on_grid(df: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    """키 격자에 LEFT JOIN: 거래가 하나도 없는 칸도 행을 남기고 0으로 채운다."""
    out = grid.merge(df, on=KEYS, how="left", validate="one_to_one")
    for c in ["am", "ct", "n"]:
        out[c] = out[c].fillna(0)
    return out.sort_values(UNIT + ["crym"]).reset_index(drop=True)


# ---------------------------------------------------------------- 정답표 (고정)
def build_truth(state):
    cfg, log = state.config, state.log
    P = periods(cfg)
    con = _con(state)
    cats = list(state.quality["업종"]["대분류"])
    panel = _on_grid(_agg(con, cfg, []), _grid(cfg, cats, P["all"]))
    con.close()
    g = panel.groupby(UNIT, sort=False)
    panel["am_prev"] = g["am"].shift(1)
    panel["am_next"] = g["am"].shift(-1)
    panel["label"] = np.where(panel["am_next"] > panel["am"], 1.0,
                              np.where(panel["am_next"] < panel["am"], 0.0, np.nan))
    panel["rate_next"] = np.where(panel["am"] > 0, (panel["am_next"] - panel["am"]) / panel["am"].where(panel["am"] > 0), 0.0)
    panel["last_dir"] = (panel["am"] > panel["am_prev"]).astype(int)       # 직전방향유지 규칙용
    base_months = month_range(P["first"], P["test"][-1])
    panel["문제"] = panel["crym"].isin(base_months) & panel["label"].notna()
    keep_same = int((panel["crym"].isin(base_months) & panel["label"].isna()).sum())
    panel["sex_nm"] = panel["sex"].map({str(k): v for k, v in cfg["성별"].items()})
    state.truth = panel
    timeseries_models.set_panel(panel)          # 칼만필터 등 시계열 후보가 쓰는 칸×월 금액 (기준월까지만 잘라 씀)
    state.quality["유지_건수"] = keep_same                                      # 계획서 5-1: 금액이 같아 빠진 문제 수
    write_csv(panel.drop(columns=["am_prev"]), state.run_dir / "panel_truth.csv")
    n = panel.groupby(panel["crym"].map(lambda m: "검증" if m in P["valid"] else "테스트" if m in P["test"]
                                        else "학습" if m < P["valid"][0] else "기타"))["문제"].sum()
    log(f"[2] 정답표: 문제 {int(panel['문제'].sum())}개 (학습 {int(n.get('학습', 0))} / 검증 {int(n.get('검증', 0))} / "
        f"테스트 {int(n.get('테스트', 0))}), 금액이 같아 빠진 '유지' {keep_same}개")
    return state


def problems(state) -> pd.DataFrame:
    t = state.truth
    return t.loc[t["문제"], KEYS + ["label", "rate_next", "last_dir", "am"]].reset_index(drop=True)


# ---------------------------------------------------------------- 변수표 (레시피별)
def build(state, recipe: dict) -> dict:
    cfg = state.config
    P = periods(cfg)
    log1p_on = recipe["전처리"]["금액_로그변환"]
    L = slog if log1p_on else (lambda x: np.asarray(x, dtype=float))
    con = _con(state)
    cats = list(state.quality["업종"]["대분류"])
    grid = _grid(cfg, cats, P["all"])

    # 소분류 묶음: 제외·분리 모두 본 계산에서 뺀다 (분리는 따로 변수로 더함)
    bx = pd.DataFrame([dict(zip(["cat_l", "cat_m", "cat_s"], s.split(" > ")), bname=b["이름"], 처리=b["처리"])
                       for b in recipe.get("묶음", []) for s in b["소분류"]],
                      columns=["cat_l", "cat_m", "cat_s", "bname", "처리"]).astype(str)
    con.register("bx", bx)
    con.execute("""CREATE VIEW base AS SELECT c.* FROM clean c
                   ANTI JOIN bx ON c.cat_l = bx.cat_l AND c.cat_m = bx.cat_m AND c.cat_s = bx.cat_s""")
    con.execute("""CREATE VIEW bundled AS SELECT c.*, bx.bname FROM clean c
                   JOIN bx ON c.cat_l = bx.cat_l AND c.cat_m = bx.cat_m AND c.cat_s = bx.cat_s""")

    main = _on_grid(_agg(con, cfg, [], "base"), grid)
    g = main.groupby(UNIT, sort=False)
    am, ct, n = main["am"].astype(float), main["ct"].astype(float), main["n"].astype(float)
    am1, am2 = g["am"].shift(1).astype(float), g["am"].shift(2).astype(float)
    F, groups = {}, {}

    def put(name, values, group):
        F[name] = np.asarray(values, dtype=float)
        groups[name] = group

    v = {k: (val["처음값"] == "항상" or bool(recipe["변수"].get(k))) for k, val in cfg["변수"].items()}
    if v.get("최근월_금액"):   put("최근월_금액", L(am), "최근월_금액")
    if v.get("최근월_건수"):   put("최근월_건수", L(ct), "최근월_건수")
    if v.get("전월대비_차이"): put("전월대비_차이", L(am - am1), "전월대비_차이")
    if v.get("전월대비_증감률"):
        put("전월대비_증감률", np.where(am1 > 0, (am - am1) / am1.where(am1 > 0), 0.0), "전월대비_증감률")
        put("전월0원여부", (am1.fillna(0) <= 0).astype(float), "전월대비_증감률")
    if v.get("최근3개월_평균") or v.get("최근3개월_변동성"):
        three = pd.concat([am2, am1, am], axis=1)
        mean, std = three.mean(axis=1), three.std(axis=1, ddof=0)
        if v.get("최근3개월_평균"):   put("최근3개월_평균", L(mean), "최근3개월_평균")
        if v.get("최근3개월_변동성"): put("최근3개월_변동성", np.where(mean > 0, std / mean.where(mean > 0), 0.0), "최근3개월_변동성")
    if v.get("최근_증감방향"): put("최근_증감방향", (am > am1).astype(float), "최근_증감방향")
    if v.get("고객층_업종_구분"):
        sex_nm = main["sex"].map({str(k): v for k, v in cfg["성별"].items()})
        if recipe["전처리"]["고객층_표현"] == "합침":
            for seg in sorted(set(sex_nm + "_" + main["age_band"])):
                put(f"고객층_{seg}", ((sex_nm + "_" + main["age_band"]) == seg).astype(float), "고객층_업종_구분")
        else:
            for s in sorted(set(sex_nm)):
                put(f"성별_{s}", (sex_nm == s).astype(float), "고객층_업종_구분")
            for b in _bands(cfg):
                put(f"연령_{b}", (main["age_band"] == b).astype(float), "고객층_업종_구분")
        for c in cats:
            put(f"업종_{c}", (main["cat_l"] == c).astype(float), "고객층_업종_구분")
    if v.get("객단가"):
        put("객단가", L(np.where(ct > 0, am / ct.where(ct > 0), 0.0)), "객단가")
    if v.get("업종비중"):
        tot = main.groupby(["crym", "sex", "age_band"])["am"].transform("sum").astype(float)
        put("업종비중", np.where(tot > 0, am / tot.where(tot > 0), 0.0), "업종비중")
    if v.get("달력변수"):
        cal = _calendar(cfg, P["all"])
        for col in cal.columns:
            put(f"다음달_{col}", main["crym"].map(cal[col]), "달력변수")
    if v.get("달력차이"):
        cd = _calendar_diff(cfg, P["all"])
        for col in cd.columns:
            put(f"달력차이_{col}", main["crym"].map(cd[col]), "달력차이")
    if v.get("일수보정"):
        # 월 합계는 달 길이에 따라 커지므로, 하루 평균으로 바꿔 '진짜 변화'와 '날짜 수 효과'를 나눈다
        def dim(m):
            return float(pd.Timestamp(m // 100, m % 100, 1).days_in_month)
        d0 = main["crym"].map(dim).to_numpy()
        d1 = main["crym"].map(lambda m: dim(month_add(int(m), -1))).to_numpy()
        dn = main["crym"].map(lambda m: dim(month_add(int(m), 1))).to_numpy()
        put("하루평균_금액", L(am / d0), "일수보정")
        put("하루평균_전월대비", np.where(am1 > 0, (am / d0) / (am1 / d1).where(am1 > 0) - 1, 0.0), "일수보정")
        put("다음달_일수차이", dn - d0, "일수보정")
    if v.get("칼만예측"):
        # 칼만 상태공간 모델이 기준월까지의 금액만으로 낸 '다음 달 예상 변화(로그)'를 변수로 (스태킹)
        idx = pd.MultiIndex.from_frame(main[["sex", "age_band", "cat_l", "crym"]])
        for mdl in ["수준", "공통요인"]:
            put(f"칼만예측_{mdl}", timeseries_models.kalman_delta(mdl, 1.0).reindex(idx).fillna(0).to_numpy(), "칼만예측")
    if (n == 0).any():
        put("거래없음", (n == 0).astype(float), "거래없음")

    # ---- 원본 칼럼 변수 ----
    o = recipe["원본변수"]
    min_ct = cfg["원본변수"]["최소_월평균건수"]
    stat = [m for m in P["all"] if m <= P["stat_end"]]
    if o.get("중분류_요약") or o.get("중분류_분해"):
        mid = _sub_panel(_agg(con, cfg, ["cat_m"], "base"), "cat_m", main)
        if o.get("중분류_요약"):
            for k, val in _summary(mid, "cat_m", main).items():
                put(f"중분류_{k}", val, "중분류_요약")
        if o.get("중분류_분해"):
            for k, val in _rank_decomp(mid, "cat_m", main, recipe["상위중분류_수"], stat, min_ct).items():
                put(f"중분류_{k}", val, "중분류_분해")
    if o.get("소분류_요약"):
        sub = _sub_panel(_agg(con, cfg, ["cat_m", "cat_s"], "base").assign(
            sub=lambda d: d["cat_m"] + " > " + d["cat_s"]).drop(columns=["cat_m", "cat_s"]), "sub", main)
        for k, val in _summary(sub, "sub", main).items():
            put(f"소분류_{k}", val, "소분류_요약")
    if o.get("지역_요약") or o.get("지역_분해"):
        lv = "sido" if recipe["지역_단위"] == "시도" else "reg"
        for side, label in [("merch", "가맹점"), ("cust", "고객")]:
            reg = _sub_panel(_agg(con, cfg, [f"{side}_{lv}"], "base"), f"{side}_{lv}", main)
            if o.get("지역_요약"):
                for k, val in _summary(reg, f"{side}_{lv}", main, inc=False).items():
                    put(f"{label}지역_{k}", val, "지역_요약")
            if o.get("지역_분해"):
                for k, val in _global_decomp(reg, f"{side}_{lv}", main, recipe["상위지역_수"], stat, min_ct).items():
                    put(f"{label}_{k}", val, "지역_분해")
        if o.get("지역_요약"):
            cross = _agg(con, cfg, [], "base", "WHERE cross_sido")
            cross = main[KEYS].merge(cross, on=KEYS, how="left")["am"].fillna(0).to_numpy()
            put("타시도_결제비중", np.where(am > 0, cross / am.where(am > 0), 0.0), "지역_요약")
    if o.get("원본행수"):
        put("원본행수", L(n), "원본행수")

    # ---- 분리 묶음: 따로 변수 2개씩 ----
    bundle_share = {}
    tot_stat = float(main.loc[main["crym"].isin(stat), "am"].sum()) or 1.0
    if len(bx):
        ball = con.execute("SELECT crym, bname, CAST(sum(am) AS BIGINT) am FROM bundled GROUP BY ALL").df()
        for b in recipe["묶음"]:
            sb = ball[(ball["bname"] == b["이름"]) & ball["crym"].isin(stat)]["am"].sum()
            bundle_share[b["이름"]] = float(sb) / (tot_stat + float(sb))
            if b["처리"] != "분리":
                continue
            bp = _on_grid(_agg(con, cfg, [], "bundled", f"WHERE bname = '{_sq(b['이름'])}'"), grid)
            bam = bp["am"].astype(float)
            bam1 = bp.groupby(UNIT, sort=False)["am"].shift(1).astype(float)
            put(f"묶음_{b['이름']}_금액", L(bam), "묶음")
            put(f"묶음_{b['이름']}_증감률", np.where(bam1 > 0, (bam - bam1) / bam1.where(bam1 > 0), 0.0), "묶음")
    con.close()

    # ---- 정답표의 문제 칸에 붙이기 (키 격자 고정) ----
    feats = pd.concat([main[KEYS], pd.DataFrame(F)], axis=1)
    prob = problems(state)
    X = prob[KEYS].merge(feats, on=KEYS, how="left", validate="one_to_one")
    _same_problems(state, X, prob)
    cols = [c for c in F]
    X = X[KEYS + cols].copy()
    X[cols] = X[cols].replace([np.inf, -np.inf], np.nan)
    dev = prob["crym"] <= P["dev_end"]
    const = [c for c in cols if X.loc[dev, c].nunique(dropna=False) <= 1]
    cols = [c for c in cols if c not in const]

    meta = prob[KEYS + ["label", "rate_next", "last_dir"]]
    test = prob["crym"].isin(P["test"])
    out = {
        "이름": recipe["이름"], "cols": cols, "groups": {c: groups[c] for c in cols}, "상수라뺀변수": const,
        "dev": {"X": X.loc[dev, cols].reset_index(drop=True), "meta": meta[dev].reset_index(drop=True)},
        "sealed": {"X": X.loc[test, cols].reset_index(drop=True), "meta": meta[test].reset_index(drop=True)},
        "panel": main[KEYS + ["am", "ct", "n"]], "묶음비중": bundle_share,
    }
    X_all = np.vstack([out["dev"]["X"].to_numpy(float), out["sealed"]["X"].to_numpy(float)])
    keys = pd.concat([out["dev"]["meta"][KEYS], out["sealed"]["meta"][KEYS]], ignore_index=True)
    why = problem_lookup.register(out["이름"], X_all, keys)        # 행 → 문제 대응 (달력방향·중분류합산 후보가 씀)
    if why:
        out["문제대응"] = why
    if is_on(cfg["후보모델"].get(bottomup.NAME, {}).get("사용")):
        _register_mid(state, out, stat, min_ct)
    return out


def _register_mid(state, ds: dict, stat: list[int], min_ct: float):
    """중분류합산 후보가 쓸 중분류 표를 이 레시피의 변수표와 묶어 둔다 (중분류 표는 원본 전체로 한 번만 만듦)."""
    cfg = state.config
    if state.mid_table is None:
        con = _con(state)
        mid = _agg(con, cfg, ["cat_m"], "clean")
        con.close()
        state.mid_table = bottomup.build_table(mid, KEYS, UNIT, periods(cfg)["all"], stat, min_ct)
    why = bottomup.register(ds["이름"], state.mid_table, KEYS)
    ds["중분류합산"] = bottomup.info(ds["이름"]) or {"사용불가": why}


def _sq(s: str) -> str:
    return s.replace("'", "''")


def _same_problems(state, X: pd.DataFrame, prob: pd.DataFrame):
    """모든 레시피가 같은 문제(기준월·고객층·대분류·실제방향)를 푸는지 확인. 행 수만 보지 않는다."""
    a = X[KEYS].astype(str).agg("|".join, axis=1).tolist()
    b = prob[KEYS].astype(str).agg("|".join, axis=1).tolist()
    if a != b or len(set(a)) != len(a):
        raise SafeError("레시피의 문제 칸이 정답표와 다릅니다 (같은 문제로 비교할 수 없음).")
    sig = tuple(zip(b, prob["label"].tolist()))
    if state.problem_sig is None:
        state.problem_sig = sig
    elif state.problem_sig != sig:
        raise SafeError("정답(실제방향)이 처음 레시피와 다릅니다. 정답표는 고정이어야 합니다.")


def _calendar(cfg, months) -> pd.DataFrame:
    """기준월 t에 대해 t+1월의 영업일 수·명절 포함·소비쿠폰 기간 여부 (미리 알 수 있는 일정이라 누수 아님)."""
    hol = {pd.Timestamp(d).date() for d in cfg["달력"]["공휴일"]}
    fest = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in cfg["달력"]["명절"].values()]
    coup = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in cfg["달력"]["소비쿠폰기간"].values()]
    rows = {}
    for m in months:
        nm = month_add(m, 1)
        days = pd.date_range(pd.Timestamp(nm // 100, nm % 100, 1), periods=pd.Timestamp(nm // 100, nm % 100, 1).days_in_month)
        rows[m] = {"영업일수": sum(d.weekday() < 5 and d.date() not in hol for d in days),
                   "명절": float(any(a <= d <= b for d in days for a, b in fest)),
                   "소비쿠폰": float(any(not (b < days[0] or a > days[-1]) for a, b in coup))}
    return pd.DataFrame.from_dict(rows, orient="index")


def _calendar_diff(cfg, months) -> pd.DataFrame:
    """기준월 t → t+1월이 t월보다 며칠 긴지 (일수·영업일수·휴일수·토일수·소비쿠폰 일수의 차이).

    정답이 '월 합계'라 다음 달이 길면 늘기 쉽다. 미리 알 수 있는 달력이라 누수가 아니다.
    """
    hol = {pd.Timestamp(d).date() for d in cfg["달력"]["공휴일"]}
    coup = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in cfg["달력"]["소비쿠폰기간"].values()]

    def one(m):
        first = pd.Timestamp(m // 100, m % 100, 1)
        d = pd.date_range(first, periods=first.days_in_month)
        off = sum(x.weekday() >= 5 or x.date() in hol for x in d)
        return {"일수": len(d), "영업일수": len(d) - off, "휴일수": off, "토일수": sum(x.weekday() >= 5 for x in d),
                "쿠폰일수": sum(any(a <= x <= b for a, b in coup) for x in d)}
    rows = {}
    for m in months:
        a, b = one(m), one(month_add(m, 1))
        rows[m] = {k: float(b[k] - a[k]) for k in a}
    return pd.DataFrame.from_dict(rows, orient="index")


def _sub_panel(df: pd.DataFrame, col: str, main: pd.DataFrame) -> pd.DataFrame:
    """하위 칸(중분류·지역 등)을 모든 월에 펼쳐 0 채움 (전월대비 계산을 위해). 키는 집계 후라 유일하다."""
    wide = df.pivot_table(index=UNIT + [col], columns="crym", values=["am", "ct"], aggfunc="sum", fill_value=0)
    months = sorted(main["crym"].unique())
    wide = wide.reindex(columns=pd.MultiIndex.from_product([["am", "ct"], months]), fill_value=0)
    return wide


def _summary(wide: pd.DataFrame, col: str, main: pd.DataFrame, inc: bool = True) -> dict:
    """집중도(HHI), 1위 비중, 전월보다 늘어난 하위 칸 비율 — 고객층×대분류×월마다."""
    am = wide["am"]
    tot = am.groupby(level=UNIT).transform("sum")
    share = (am / tot.where(tot > 0)).fillna(0)
    hhi = (share ** 2).groupby(level=UNIT).sum()
    top1 = share.groupby(level=UNIT).max()
    out = {"집중도": hhi, "1위비중": top1}
    if inc:
        prev = am.shift(1, axis=1)
        up = ((am > prev) & (prev > 0)).groupby(level=UNIT).sum()
        had = (prev > 0).groupby(level=UNIT).sum()
        out["증가비율"] = (up / had.where(had > 0)).fillna(0)
    return {k: _to_main(v, main) for k, v in out.items()}


def _to_main(unit_by_month: pd.DataFrame, main: pd.DataFrame) -> np.ndarray:
    s = unit_by_month.stack()
    s.index = s.index.set_names(UNIT + ["crym"])
    return main.merge(s.rename("v").reset_index(), on=UNIT + ["crym"], how="left")["v"].fillna(0).to_numpy()


def _rank_decomp(wide, col, main, top_n, stat, min_ct) -> dict:
    """업종 안 금액 상위 N개 하위 칸(1~5월 기준)의 비중·전월대비. 칸 이름 대신 '상위1, 상위2…'로 부른다."""
    am, ctw = wide["am"], wide["ct"]
    rank_basis = am[[m for m in am.columns if m in stat]].sum(axis=1).groupby(level=["cat_l", col]).sum()
    ranks = rank_basis.groupby(level="cat_l").rank(ascending=False, method="first")
    ok = ctw[[m for m in ctw.columns if m in stat]].mean(axis=1) >= min_ct        # 월평균 건수 기준 (계획서 4-2 ③)
    tot = am.groupby(level=UNIT).transform("sum")
    share = (am / tot.where(tot > 0)).fillna(0).where(ok, 0)
    prev = am.shift(1, axis=1)
    rate = ((am - prev) / prev.where(prev > 0)).fillna(0).where(ok, 0)
    r = pd.Series([ranks.get((l, c), 99) for l, c in zip(am.index.get_level_values("cat_l"), am.index.get_level_values(col))], index=am.index)
    out = {}
    for k in range(1, top_n + 1):
        sel = r == k
        out[f"상위{k}_비중"] = _to_main(share[sel].groupby(level=UNIT).sum(), main)
        out[f"상위{k}_전월대비"] = _to_main(rate[sel].groupby(level=UNIT).sum(), main)
    return out


def _global_decomp(wide, col, main, top_n, stat, min_ct) -> dict:
    """전체 금액 상위 N개 지역(1~5월 기준)의 비중·전월대비. 이름은 지역 코드 (예: R053_비중)."""
    am, ctw = wide["am"], wide["ct"]
    top = am[[m for m in am.columns if m in stat]].sum(axis=1).groupby(level=col).sum().nlargest(top_n).index
    ok = ctw[[m for m in ctw.columns if m in stat]].mean(axis=1) >= min_ct
    tot = am.groupby(level=UNIT).transform("sum")
    share = (am / tot.where(tot > 0)).fillna(0).where(ok, 0)
    prev = am.shift(1, axis=1)
    rate = ((am - prev) / prev.where(prev > 0)).fillna(0).where(ok, 0)
    codes = am.index.get_level_values(col)
    out = {}
    for c in top:
        sel = codes == c
        out[f"{c}_비중"] = _to_main(share[sel].groupby(level=UNIT).sum(), main)
        out[f"{c}_전월대비"] = _to_main(rate[sel].groupby(level=UNIT).sum(), main)
    return out


# ---------------------------------------------------------------- 소분류 통계 (묶음 만들기 참고용, 1~5월만)
def subcat_stats(state) -> pd.DataFrame:
    if state.subcat_stats is not None:
        return state.subcat_stats
    cfg = state.config
    P = periods(cfg)
    con = _con(state)
    d = con.execute(f"""SELECT cat_l, cat_m, cat_s, crym, CAST(sum(am) AS BIGINT) am, CAST(sum(ct) AS BIGINT) ct
                        FROM clean WHERE crym <= {P['stat_end']} GROUP BY ALL""").df()
    top = con.execute(f"""SELECT cat_l, cat_m, cat_s, max(share) top FROM (
                            SELECT cat_l, cat_m, cat_s, merch_sido, sum(am) / sum(sum(am)) OVER (PARTITION BY cat_l, cat_m, cat_s) share
                            FROM clean WHERE crym <= {P["stat_end"]} GROUP BY cat_l, cat_m, cat_s, merch_sido) GROUP BY ALL""").df()
    con.close()
    w = d.pivot_table(index=["cat_l", "cat_m", "cat_s"], columns="crym", values="am", aggfunc="sum", fill_value=0)
    lw = w.groupby(level="cat_l").sum()
    dsub = w.diff(axis=1).abs().iloc[:, 1:].sum(axis=1)
    dcat = lw.diff(axis=1).abs().iloc[:, 1:].sum(axis=1)
    out = pd.DataFrame({
        "금액비중": w.sum(axis=1) / w.values.sum(),
        "월변동계수": w.std(axis=1, ddof=0) / w.mean(axis=1).where(w.mean(axis=1) > 0),
        # 같은 대분류 안 소분류들의 월 변화량(절댓값) 합 중 이 소분류의 몫. 대분류마다 합이 1
        "대분류변화기여도": dsub / dsub.groupby(level="cat_l").transform("sum").where(lambda s: s > 0),
        "월평균건수": d.groupby(["cat_l", "cat_m", "cat_s"])["ct"].sum() / len(w.columns),
    }).fillna(0)
    out = out.join(top.set_index(["cat_l", "cat_m", "cat_s"])["top"].rename("가맹점1위시도비중"))
    out = out.reset_index()
    out.insert(0, "소분류", out["cat_l"] + " > " + out["cat_m"] + " > " + out["cat_s"])
    state.subcat_stats = out.drop(columns=["cat_l", "cat_m", "cat_s"]).round(4)
    write_csv(state.subcat_stats, state.run_dir / "subcat_stats.csv")
    return state.subcat_stats


# ---------------------------------------------------------------- 에이전트 도구 (집계값만 돌려줌, 1~5월만)
def sub_monthly(state, name: str) -> dict:
    """소분류 하나의 1~5월 월별 금액과 그 대분류 안 비중."""
    parts = name.split(" > ")
    if len(parts) != 3:
        return {"오류": "소분류는 '대 > 중 > 소' 형식"}
    P = periods(state.config)
    con = _con(state)
    con.register("q", pd.DataFrame([parts], columns=["l", "m", "s"]))
    d = con.execute(f"""SELECT crym, CAST(sum(am) FILTER (WHERE cat_m = q.m AND cat_s = q.s) AS BIGINT) sub,
                               CAST(sum(am) AS BIGINT) cat
                        FROM clean, q WHERE cat_l = q.l AND crym <= {P['stat_end']} GROUP BY crym ORDER BY crym""").df()
    con.close()
    return {str(r.crym): {"금액": int(r.sub or 0), "대분류_안_비중": round(float((r.sub or 0) / r.cat), 4) if r.cat else 0}
            for r in d.itertuples()}


def region_shares(state, level: str = "시도", side: str = "가맹점") -> list:
    """1~5월 지역(코드)별 금액 비중과 월평균 건수 상위 15개."""
    P = periods(state.config)
    col = ("merch" if side == "가맹점" else "cust") + ("_sido" if level == "시도" else "_reg")
    con = _con(state)
    d = con.execute(f"""SELECT {col} code, sum(am) am, sum(ct) / count(DISTINCT crym) ct FROM clean
                        WHERE crym <= {P['stat_end']} GROUP BY 1 ORDER BY am DESC""").df()
    con.close()
    d["비중"] = (d["am"] / d["am"].sum()).round(4)
    return [{"코드": r.code, "금액비중": r.비중, "월평균건수": int(r.ct)} for r in d.head(15).itertuples()]


def mid_shares(state) -> dict:
    """1~5월 대분류별 중분류 금액 비중."""
    P = periods(state.config)
    con = _con(state)
    d = con.execute(f"SELECT cat_l, cat_m, sum(am) am FROM clean WHERE crym <= {P['stat_end']} GROUP BY ALL").df()
    con.close()
    d["비중"] = d["am"] / d.groupby("cat_l")["am"].transform("sum")
    return {l: {r.cat_m: round(float(r.비중), 3) for r in g.sort_values("비중", ascending=False).itertuples()}
            for l, g in d.groupby("cat_l")}


# ---------------------------------------------------------------- 데이터 담당의 생각 (LLM)
def decide_details(state, recipe: dict, change: dict) -> dict:
    """분해 변수를 켤 때, 상위 몇 개 지역·중분류를 따로 볼지 정한다 (설정값 범위 안)."""
    import llm
    o = recipe["원본변수"]
    if not (o.get("지역_분해") or o.get("중분류_분해")) or change.get("종류") not in ("원본변수", "지역_단위"):
        return recipe
    data = {"수정안": change.get("항목"), "지역_단위": recipe["지역_단위"],
            "지금_설정": {"상위지역_수": recipe["상위지역_수"], "상위중분류_수": recipe["상위중분류_수"]},
            "허용_범위": {"상위지역_수": [1, 8], "상위중분류_수": [1, 5]},
            "최소_월평균건수": state.config["원본변수"]["최소_월평균건수"],
            "참고": "분해 변수는 개수마다 비중·전월대비 2개씩 늘어남. 학습 문제가 144~240개라 너무 많으면 과적합"}
    tools = {
        "지역별_금액비중": {"인자": "단위(시도|세부), 쪽(가맹점|고객)", "설명": "1~5월 지역 코드별 금액 비중·월평균 건수 상위 15",
                          "함수": lambda a: region_shares(state, a.get("단위", recipe["지역_단위"]), a.get("쪽", "가맹점"))},
        "중분류_금액비중": {"인자": "", "설명": "1~5월 대분류별 중분류 금액 비중", "함수": lambda a: mid_shares(state)},
    }

    def ok(ans):
        try:
            r, m = int(ans.get("상위지역_수")), int(ans.get("상위중분류_수"))
        except (TypeError, ValueError, AttributeError):
            return False, "숫자가 아님"
        good = 1 <= r <= 8 and 1 <= m <= 5
        return good, "" if good else "허용 범위 밖"

    default = {"상위지역_수": recipe["상위지역_수"], "상위중분류_수": recipe["상위중분류_수"], "이유": "설정값 그대로 (규칙)"}
    ans, who = llm.ask(state, "변수 세부 설정", "data_details.md", data, default, ok, tools)
    recipe["상위지역_수"], recipe["상위중분류_수"] = int(ans["상위지역_수"]), int(ans["상위중분류_수"])
    state.decide("② 데이터", f"분해 변수 개수 ({recipe['이름']})", "지역 1~8 / 중분류 1~5",
                 f"지역 {recipe['상위지역_수']} / 중분류 {recipe['상위중분류_수']}", who, ans.get("이유", ""))
    return recipe


def judge_precheck(state, pc: dict, change_label: str, remaining: int) -> dict:
    """사전 점검을 통과(🟢)한 수정안을 실제로 검증까지 진행할지 판단한다. 🟡·🔴는 뒤집지 못한다."""
    import llm
    if pc["신호등"] != "🟢":
        return pc
    data = {"수정안": change_label, "사전점검": {k: pc[k] for k in ["신호등", "이유", "추가변수", "제거변수", "바뀐변수", "바뀐값비율", "신호"]},
            "남은_검증_시도": remaining,
            "참고": "진행하면 검증 시도 1회를 쓴다. 신호가 약하거나 기존 변수와 같은 정보로 보이면 아끼는 편이 낫다"}

    def ok(ans):
        good = isinstance(ans, dict) and isinstance(ans.get("진행"), bool)
        return good, "" if good else "진행 값이 true/false가 아님"

    ans, who = llm.ask(state, "사전 점검 판단", "precheck_judge.md", data, {"진행": True, "이유": "🟢 통과 (규칙)"}, ok)
    if not ans["진행"]:
        pc = {**pc, "신호등": "🟡", "판정": "건너뜀", "이유": pc["이유"] + [f"데이터 담당 판단으로 건너뜀: {ans.get('이유', '')}"]}
    pc["데이터담당"] = {"진행": ans["진행"], "이유": ans.get("이유", ""), "결정자": who}
    return pc


# ---------------------------------------------------------------- 사전 점검
def precheck(state, new: dict, old: dict, change: dict) -> dict:
    """모델에 넣기 전에 '할 만한가' 판단. 첫 회차 학습 구간(3~5월) 정답만 보고, 검증월 정답은 보지 않는다."""
    from sklearn.metrics import roc_auc_score
    cfg = state.config
    th = cfg["사전점검"]
    P = periods(cfg)
    reasons, light = [], "🟢"
    Xn, Xo, meta = new["dev"]["X"], old["dev"]["X"], new["dev"]["meta"]
    tr = (meta["crym"] < P["valid"][0]).to_numpy()
    y = meta["label"].to_numpy()[tr]

    # ① 건강
    bad = int(Xn.isna().sum().sum())
    if bad:
        light = "🔴"; reasons.append(f"빈 값·무한값 {bad}개 발생")
    if new["상수라뺀변수"]:
        reasons.append(f"값이 전부 같아 뺀 변수 {len(new['상수라뺀변수'])}개: {', '.join(new['상수라뺀변수'][:5])}")
    truth = state.truth.set_index(KEYS)["am"]
    pn = new["panel"].set_index(KEYS)["am"]
    rem = (pn / truth.where(truth > 0)).dropna()
    low = int((rem < th["최소잔여비율"]).sum())
    if low:
        msg = f"묶음을 빼고 남은 금액이 {th['최소잔여비율']:.0%} 미만인 칸 {low}개"
        if change.get("종류") == "묶음" and change["묶음"]["처리"] == "제외":
            light = "🔴"; reasons.append(msg)
        else:   # '분리'는 뺀 금액이 별도 변수로 남으므로 참고로만 표시
            reasons.append(msg + " (분리라 정보는 별도 변수로 남음)")

    # ② 변화
    common = [c for c in new["cols"] if c in old["cols"]]
    added = [c for c in new["cols"] if c not in old["cols"]]
    removed = [c for c in old["cols"] if c not in new["cols"]]
    diff = (Xn[common].to_numpy() - Xo[common].to_numpy()) if common else np.zeros((len(Xn), 0))
    changed_cols = [c for c, d in zip(common, np.abs(diff).max(axis=0) if diff.size else []) if d > 1e-9]
    frac = float((np.abs(diff) > 1e-9).mean()) if diff.size else 0.0
    if change.get("종류") == "묶음":
        share = new["묶음비중"].get(change["묶음"]["이름"], 0.0)
        if share < th["최소변화비중"] and light == "🟢":
            light = "🟡"; reasons.append(f"묶음 금액이 전체의 {share:.2%}라 변화가 거의 없음 (기준 {th['최소변화비중']:.1%})")
    train_side = change.get("종류") == "전처리" and change.get("항목") in ("미세변동_제외", "극단값_상한")
    if train_side:
        # 변수 값이 아니라 '학습 방식'을 바꾸는 수정안: 학습 문제가 실제로 달라지는지만 본다 (3~5월 학습 문제 기준)
        if change["항목"] == "미세변동_제외" and change["값"]:
            gone = float((meta["rate_next"].abs().to_numpy()[tr] <= 0.02).mean())
            if gone == 0 and light == "🟢":
                light = "🟡"; reasons.append("±2% 이내로 변한 학습 문제가 없어 빠지는 문제가 없음")
            elif gone > 0.5:
                light = "🔴"; reasons.append(f"학습 문제의 {gone:.0%}가 빠져 학습할 문제가 너무 적음")
            else:
                reasons.append(f"학습 문제의 {gone:.0%}가 빠짐 (검증·테스트는 그대로)")
        else:
            reasons.append("학습 방식만 바뀌는 수정안 (변수 값 그대로)")
    elif not added and not removed and frac < 0.01 and light == "🟢":
        light = "🟡"; reasons.append(f"바뀐 변수 값이 {frac:.1%}뿐이라 해 봐야 점수가 같음")

    # ③ 신호 (새로 생기거나 바뀐 변수만)
    # 같은 달이면 모든 문제가 같은 값인 '월 단위 변수'(달력 등)는 3개월치로 신호를 판단할 수 없어 점검에서 뺀다
    months_tr = meta["crym"].to_numpy()[tr]
    month_level = [c for c in added + changed_cols
                   if pd.Series(Xn[c].to_numpy()[tr]).groupby(months_tr).nunique().max() <= 1]
    if month_level:
        reasons.append(f"월 단위 변수 {', '.join(month_level)}는 신호 점검 생략 (검증으로 판단)")
    look = [c for c in added + changed_cols if c not in month_level]
    aucs = {}
    for c in look:
        x = Xn[c].to_numpy()[tr]
        if np.nanstd(x) > 0 and len(set(y)) == 2:
            aucs[c] = float(roc_auc_score(y, x))
    signal = max((abs(a - 0.5) for a in aucs.values()), default=None)
    if look and light == "🟢":
        if signal is None or signal < th["최소신호"]:
            light = "🟡"; reasons.append(f"새·바뀐 변수의 신호(AUC−0.5)가 {signal or 0:.3f}로 기준 {th['최소신호']} 미만")
        elif added and not changed_cols:
            base_cols = [c for c in new["cols"] if c not in added]
            if base_cols:
                corr = pd.concat([Xn.loc[tr, added], Xn.loc[tr, base_cols]], axis=1).corr().loc[added, base_cols].abs()
                if (corr.max(axis=1) > th["최대상관"]).all():
                    light = "🟡"; reasons.append(f"새 변수가 기존 변수와 {th['최대상관']} 넘게 겹침")
    if light == "🟢":
        reasons.append("통과" + (f" (신호 {signal:.3f})" if signal is not None
                                 else " (신호를 볼 새 변수가 없어 점검 생략)" if not removed else " (변수 제거라 신호 점검 생략)"))

    # 대시보드용: 대분류별 금액 전/후, 가장 신호가 큰 변수의 분포 전/후
    po = old["panel"].groupby("cat_l")["am"].sum()
    pn2 = new["panel"].groupby("cat_l")["am"].sum()
    top_col = max(aucs, key=lambda c: abs(aucs[c] - 0.5)) if aucs else None
    hist = None
    if top_col:
        hist = {"변수": top_col, "후": Xn.loc[tr, top_col].tolist(),
                "전": Xo.loc[tr, top_col].tolist() if top_col in Xo else None}
    result = {"신호등": light, "판정": {"🟢": "통과", "🟡": "건너뜀", "🔴": "탈락"}[light], "이유": reasons,
              "추가변수": added, "제거변수": removed, "바뀐변수": changed_cols, "바뀐값비율": round(frac, 4),
              "신호": {k: round(v, 3) for k, v in aucs.items()},
              "대분류금액_전후": {c: [float(po.get(c, 0)), float(pn2.get(c, 0))] for c in po.index}, "분포": hist}
    return result
