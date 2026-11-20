"""[0] 원본 담당: 원본 CSV → 점검·정제 → 정제 원본(parquet).

원본 CSV를 읽는 코드는 이 파일에만 있다.
- 원본 행은 줄이지도 합치지도 않고 그대로 옮긴다 (중복처럼 보이는 행도 지우지 않음).
- 지역 이름은 이 단계에서 코드(S01, C001, R001)로 바뀌고, 이후 단계에는 지역 이름이 나오지 않는다.
- 화면과 결과 폴더에는 집계값만 쓴다. 행 단위 파일·지역 대응표는 보안 폴더에만 쓴다.
"""
from __future__ import annotations

import codecs
import json
import time
from pathlib import Path

import duckdb
import pandas as pd

from common import SafeError, cfg_path, month_range, norm_text, write_csv, write_json

RAW_VERSION = "raw-v1"          # 정제 규칙이 바뀌면 올려서 정제 원본을 다시 만들게 함
SPEC_COLS = ["crym", "ma_fem_dc", "age_g", "mgpo_nm", "ccd_nm", "mgpo_nm_mc", "ccd_nm_mc",
             "tco_bt_hlv_c_nm", "tco_bt_mcls_c_nm", "tco_btc_nm", "sl_ct", "sl_am"]   # 명세서 1-2 순서
CAPITAL = ("서울", "경기", "인천")   # 명세서: 수도권만 시군구가 살아 있음
NULL = "§없음§"                      # 빈 값을 조인 키로 쓰기 위한 표시


def run(state, raw_override: str | None = None):
    cfg, log = state.config, state.log
    raw = _resolve_raw(cfg, raw_override)
    secure = cfg_path(cfg, "보안폴더") / f"clean_{raw.stem}"
    report_dir = cfg_path(cfg, "결과폴더") / "원본점검" / raw.stem
    secure.mkdir(parents=True, exist_ok=True)
    clean = secure / "clean.parquet"
    meta_path = secure / "meta.json"

    fp = {"raw": str(raw), "size": raw.stat().st_size, "mtime": raw.stat().st_mtime, "version": RAW_VERSION,
          "months": [cfg["기간"]["전체_시작월"], cfg["기간"]["전체_끝월"]],
          "sex": sorted(cfg["성별"]), "age": sorted(cfg["연령구간"])}
    if clean.exists() and meta_path.exists() and (report_dir / "quality_report.json").exists():
        if json.loads(meta_path.read_text(encoding="utf-8")) == fp:
            state.clean_path = clean
            state.quality = json.loads((report_dir / "quality_report.json").read_text(encoding="utf-8"))
            log(f"[0] 원본이 바뀌지 않아 정제 원본을 재사용합니다 ({state.quality['행수']['원본']:,}행)")
            return state

    t0 = time.time()
    info = _probe(raw)
    log(f"[0] 원본 판별: 인코딩 {info['encoding']}, 구분자 {info['delimiter_name']}, "
        f"헤더 {'있음' if info['header'] else '없음'}, {info['size'] / 1e9:.2f}GB")
    src = _ensure_utf8(raw, secure, info["encoding"], log)

    con = duckdb.connect()
    tmp = secure / "duckdb_tmp"
    tmp.mkdir(exist_ok=True)
    con.execute(f"SET temp_directory='{_q(tmp)}'")
    con.execute(f"SET memory_limit='{cfg['DuckDB']['메모리_상한']}'")
    con.execute("SET preserve_insertion_order=false")
    names = "" if info["header"] else ", names=[" + ", ".join(f"'{c}'" for c in SPEC_COLS) + "]"
    delim = info["delimiter"].replace("\t", "\\t")
    con.execute(f"CREATE VIEW raw AS SELECT * FROM read_csv('{_q(src)}', delim=E'{delim}', "
                f"header={str(info['header']).lower()}, all_varchar=true, quote='\"', normalize_names=true{names})")
    con.execute(PARSE_SQL)

    # 1) 키 값 목록 (원본 1회 스캔) → 정규화·코드화 대응표
    log("[0] 키 값 목록 수집 (원본 1회 스캔)")
    maps = _build_maps(con, cfg)
    for name, df in maps.items():
        con.register(name, df)
    # 지역 대응표는 보안 폴더에만 저장 (사람이 직접 열어 확인하는 용도)
    write_csv(maps["m_region_names"], secure / "region_codes.csv")
    con.unregister("m_region_names")

    # 2) 정제 원본 저장 (원본 1회 스캔)
    log("[0] 정제 원본(parquet) 저장 중 (원본 1회 스캔)")
    con.execute(f"COPY ({CLEAN_SQL}) TO '{_q(clean)}' (FORMAT parquet, COMPRESSION zstd)")
    con.execute(f"CREATE VIEW clean AS SELECT * FROM read_parquet('{_q(clean)}')")

    # 3) 점검 (정제 원본으로 빠르게) + 검산 (원본 1회 스캔)
    log("[0] 점검·검산")
    q = _quality(con, cfg, info, maps, secure, report_dir, log)
    if cfg["DuckDB"].get("중복진단") in ("예", True):
        log("[0] 적재 중복 원인 진단")
        q["중복"] = _duplicates(con, report_dir)
    q["소요초"] = round(time.time() - t0, 1)
    write_json(q, report_dir / "quality_report.json")
    meta_path.write_text(json.dumps(fp, ensure_ascii=False), encoding="utf-8")
    con.close()

    state.clean_path, state.quality = clean, q
    log(f"[0] 완료: 원본 {q['행수']['원본']:,}행 → 정제 원본 {q['행수']['사용']:,}행 "
        f"(제외 {q['행수']['제외']:,}행), 검산 {'통과' if q['검산']['통과'] else '실패'} ({q['소요초']}초)")
    if not q["검산"]["통과"]:
        raise SafeError("검산 실패: 정제 원본 합계가 원본 합계와 다릅니다. 결과/03_agent/원본점검 의 quality_report.json을 확인하세요.")
    return state


# ---------------------------------------------------------------- 원본 담당의 생각 (LLM)
def review(state):
    """품질 점검표·업종 목록(집계값)을 읽고 데이터 주의점과 소분류 묶음 후보를 제안한다. 원본 행은 보지 않는다."""
    import llm
    from agents import data_agent
    q = state.quality
    st = data_agent.subcat_stats(state)
    have = set(st["소분류"])
    hints = {k: [s for s in v if s in have] for k, v in (state.config["소분류처리"].get("힌트") or {}).items()}
    data = {"원본_요약": {k: q.get(k) for k in ["행수", "제외내역", "점검", "월별", "업종", "지역", "유지_건수"]},
            "중복_진단_요약": q.get("중복"),
            "설정의_힌트_목록": hints,
            "소분류_목록_금액상위30": st.sort_values("금액비중", ascending=False).head(30).to_dict("records"),
            "소분류_통계_설명": "1~5월 기준. 금액비중=전체 지출 중 비율, 월변동계수=월별 금액의 흔들림, "
                                "대분류변화기여도=같은 대분류 안 월 변화량 중 이 소분류 몫, 가맹점1위시도비중=한 시도에 몰린 정도"}
    tools = {
        "소분류_목록": {"인자": "대분류", "설명": "그 대분류에 속한 모든 소분류와 통계 (1~5월)",
                       "함수": lambda a: st[st["소분류"].str.startswith(str(a.get("대분류", "")) + " > ")].to_dict("records")},
        "소분류_월별": {"인자": "소분류 ('대 > 중 > 소')", "설명": "그 소분류의 1~5월 월별 금액과 대분류 안 비중",
                       "함수": lambda a: data_agent.sub_monthly(state, str(a.get("소분류", "")))},
    }

    def ok(ans):
        if not isinstance(ans, dict):
            return False, "형식 오류"
        for k in ["비소비성_후보", "본사집계의심_후보"]:
            v = ans.get(k, [])
            if not isinstance(v, list):
                return False, f"{k}가 목록이 아님"
            ans[k] = [s for s in v if s in have][:20]
        if not isinstance(ans.get("데이터_주의점", []), list):
            return False, "데이터_주의점이 목록이 아님"
        return True, ""

    default = {"데이터_주의점": [], "비소비성_후보": [], "본사집계의심_후보": [], "이유": "규칙 모드: 설정의 힌트 목록만 사용"}
    ans, who = llm.ask(state, "원본 해석", "raw.md", data, default, ok, tools)
    state.data_review = {**ans, "결정자": who}
    state.hints = dict(hints)
    for key, name in [("비소비성_후보", "LLM_비소비성"), ("본사집계의심_후보", "LLM_본사집계의심")]:
        if ans.get(key):
            state.hints[name] = ans[key]
    state.decide("⓪ 원본", "데이터 주의점·소분류 후보", f"소분류 {len(have)}개",
                 f"비소비성 {len(ans.get('비소비성_후보', []))}개, 본사집계의심 {len(ans.get('본사집계의심_후보', []))}개 제안",
                 who, ans.get("이유", ""))
    return state


# ---------------------------------------------------------------- 파일 판별
def _resolve_raw(cfg, override):
    if override:
        p = Path(override)
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        if not p.exists():
            raise SafeError(f"원본 파일이 없습니다: {p.name}")
        return p
    p = cfg_path(cfg, "원본파일")
    if p.exists():
        return p
    alt = p.parent / cfg["경로"].get("원본_다른이름", "")
    if alt.exists():
        return alt
    raise SafeError(f"원본 파일이 없습니다: {p.name} (명세서 파일명 {alt.name}도 없음). config.yaml 경로 > 원본파일을 확인하세요.")


def _probe(path: Path) -> dict:
    """인코딩·구분자·헤더 판별. 화면에는 판별 결과만 남긴다 (첫 줄 내용은 출력하지 않음)."""
    with open(path, "rb") as f:
        head = f.read(1 << 20)
    enc = None
    for cand in ("utf-8", "cp949"):
        try:
            codecs.getincrementaldecoder(cand)().decode(head, final=False)
            enc = cand
            break
        except UnicodeDecodeError:
            continue
    if enc is None:
        raise SafeError("인코딩 판별 실패 (utf-8, cp949 모두 아님)")
    text = codecs.getincrementaldecoder(enc)().decode(head, final=False).lstrip("﻿")
    first = text.split("\n", 1)[0].rstrip("\r")
    delim = next((d for d in [",", "\t", "|", ";"] if first.count(d) == len(SPEC_COLS) - 1), None)
    if delim is None:
        raise SafeError(f"구분자 판별 실패: 첫 줄의 칸 수가 명세({len(SPEC_COLS)}개)와 다릅니다.")
    tokens = [t.strip().strip('"').strip().lower() for t in first.split(delim)]
    if set(tokens) == set(SPEC_COLS):
        header = True
    elif tokens[0].isdigit() and len(tokens[0]) == 6:
        header = False
    else:
        raise SafeError("첫 줄이 명세 칼럼명과 다르고 데이터 행처럼 보이지도 않습니다. 헤더를 확인하세요.")
    names = {",": "쉼표", "\t": "탭", "|": "세로줄", ";": "세미콜론"}
    return {"encoding": enc, "delimiter": delim, "delimiter_name": names[delim], "header": header,
            "size": path.stat().st_size, "file": path.name}


def _ensure_utf8(src: Path, secure: Path, enc: str, log) -> Path:
    """DuckDB는 utf-8 기준이라, cp949면 보안 폴더에 utf-8 사본을 1회 만든다."""
    if enc == "utf-8":
        return src
    dst = secure / "raw.utf8.csv"
    done = secure / "raw.utf8.done"
    if dst.exists() and done.exists() and done.read_text() == f"{src.stat().st_size}:{src.stat().st_mtime}":
        log("[0] utf-8 사본 재사용")
        return dst
    log(f"[0] {enc} → utf-8 변환 (보안 폴더에 저장)")
    dec = codecs.getincrementaldecoder(enc)(errors="strict")
    with open(src, "rb") as fi, open(dst, "w", encoding="utf-8", newline="") as fo:
        while chunk := fi.read(1 << 24):
            fo.write(dec.decode(chunk))
        fo.write(dec.decode(b"", final=True))
    done.write_text(f"{src.stat().st_size}:{src.stat().st_mtime}")
    return dst


def _q(p) -> str:
    return str(p).replace("\\", "/").replace("'", "''")


# ---------------------------------------------------------------- SQL
PARSE_SQL = r"""
CREATE VIEW parsed AS
WITH c AS (
    SELECT *,
        nullif(regexp_replace(coalesce(sl_ct, ''), '[,\s원건]', '', 'g'), '') AS ct_clean,
        nullif(regexp_replace(coalesce(sl_am, ''), '[,\s원]',   '', 'g'), '') AS am_clean
    FROM raw
), d AS (
    SELECT *, TRY_CAST(ct_clean AS DECIMAL(38,3)) AS ct_dec, TRY_CAST(am_clean AS DECIMAL(38,3)) AS am_dec FROM c
)
SELECT *,
    CAST(round(ct_dec) AS BIGINT) AS ct,
    CAST(round(am_dec) AS BIGINT) AS am,
    (ct_clean IS NOT NULL AND ct_dec IS NULL) AS ct_castfail,
    (am_clean IS NOT NULL AND am_dec IS NULL) AS am_castfail,
    (ct_dec <> round(ct_dec) OR am_dec <> round(am_dec)) AS fraction
FROM d
"""

CLEAN_SQL = f"""
SELECT
    mc.crym, ms.sex, ma.age_g,
    cr.sido AS cust_sido, cr.reg AS cust_reg, cr.cap AS cust_cap,
    mr.sido AS merch_sido, mr.reg AS merch_reg, mr.cap AS merch_cap,
    (cr.sido IS DISTINCT FROM mr.sido) AS cross_sido,
    ca.cat_l, ca.cat_m, ca.cat_s,
    p.ct, p.am,
    CASE WHEN mc.crym IS NULL THEN 'crym_invalid'
         WHEN ms.sex IS NULL THEN 'sex_invalid'
         WHEN ma.age_g IS NULL THEN 'age_invalid'
         WHEN ca.cat_l IS NULL THEN 'cat_l_missing'
         WHEN p.ct IS NULL OR p.am IS NULL THEN 'value_missing' END AS reason,
    p.ct_castfail, p.am_castfail, coalesce(p.fraction, false) AS fraction
FROM parsed p
LEFT JOIN m_crym mc ON coalesce(p.crym, '{NULL}') = mc.k
LEFT JOIN m_sex  ms ON coalesce(p.ma_fem_dc, '{NULL}') = ms.k
LEFT JOIN m_age  ma ON coalesce(p.age_g, '{NULL}') = ma.k
LEFT JOIN m_region cr ON coalesce(p.mgpo_nm, '{NULL}') = cr.k1 AND coalesce(p.ccd_nm, '{NULL}') = cr.k2 AND cr.side = 'C'
LEFT JOIN m_region mr ON coalesce(p.mgpo_nm_mc, '{NULL}') = mr.k1 AND coalesce(p.ccd_nm_mc, '{NULL}') = mr.k2 AND mr.side = 'R'
LEFT JOIN m_cat ca ON coalesce(p.tco_bt_hlv_c_nm, '{NULL}') = ca.k1 AND coalesce(p.tco_bt_mcls_c_nm, '{NULL}') = ca.k2
                  AND coalesce(p.tco_btc_nm, '{NULL}') = ca.k3
"""


# ---------------------------------------------------------------- 대응표
def _build_maps(con, cfg) -> dict[str, pd.DataFrame]:
    """키 칼럼의 서로 다른 값만 모아(작은 표) 파이썬에서 정규화·코드화한다."""
    sets = "(crym), (ma_fem_dc), (age_g), (mgpo_nm, ccd_nm), (mgpo_nm_mc, ccd_nm_mc), (tco_bt_hlv_c_nm, tco_bt_mcls_c_nm, tco_btc_nm)"
    cols = ["crym", "ma_fem_dc", "age_g", "mgpo_nm", "ccd_nm", "mgpo_nm_mc", "ccd_nm_mc",
            "tco_bt_hlv_c_nm", "tco_bt_mcls_c_nm", "tco_btc_nm"]
    sel = ", ".join(cols) + ", " + ", ".join(f"grouping({c}) AS g_{c}" for c in cols)
    g = con.execute(f"SELECT {sel} FROM raw GROUP BY GROUPING SETS ({sets})").df()

    def part(keys):
        k = g[(g[[f"g_{c}" for c in keys]] == 0).all(1) & (g[[f"g_{c}" for c in cols if c not in keys]] == 1).all(1)]
        return k[keys].astype(object).where(k[keys].notna(), None)

    months = {m for m in month_range(cfg["기간"]["전체_시작월"], cfg["기간"]["전체_끝월"])}

    def crym(v):
        s = norm_text(v)
        s = s.replace("-", "").replace(".", "").replace("/", "") if s else s
        return int(s) if s and s.isdigit() and len(s) == 6 and int(s) in months else None

    def code(v, allowed):
        s = norm_text(v)
        if s and s.endswith(".0") and s[:-2].isdigit():
            s = s[:-2]
        return s if s in allowed else None

    kk = lambda v: NULL if v is None else v
    c = part(["crym"]); m_crym = pd.DataFrame({"k": c["crym"].map(kk), "crym": c["crym"].map(crym).astype("Int64")})
    c = part(["ma_fem_dc"]); m_sex = pd.DataFrame({"k": c["ma_fem_dc"].map(kk), "sex": c["ma_fem_dc"].map(lambda v: code(v, set(cfg["성별"])))})
    c = part(["age_g"]); m_age = pd.DataFrame({"k": c["age_g"].map(kk), "age_g": c["age_g"].map(lambda v: code(v, set(cfg["연령구간"])))})

    # 지역: (시도, 시군구) 쌍마다 코드. 고객·가맹점이 같은 번호 체계를 씀 (C053 = R053 같은 곳)
    cust = part(["mgpo_nm", "ccd_nm"]).rename(columns={"mgpo_nm": "a", "ccd_nm": "b"}).assign(side="C")
    merch = part(["mgpo_nm_mc", "ccd_nm_mc"]).rename(columns={"mgpo_nm_mc": "a", "ccd_nm_mc": "b"}).assign(side="R")
    reg = pd.concat([cust, merch], ignore_index=True)
    reg["na"], reg["nb"] = reg["a"].map(norm_text), reg["b"].map(norm_text)
    sidos = sorted({x for x in reg["na"] if x})
    sido_code = {s: f"S{i:02d}" for i, s in enumerate(sidos, 1)}
    pairs = sorted({(a, b if b else a) for a, b in zip(reg["na"], reg["nb"]) if a})
    pair_code = {p: i for i, p in enumerate(pairs, 1)}
    reg["sido"] = reg["na"].map(lambda s: sido_code.get(s, "S00"))
    reg["num"] = [pair_code.get((a, b if b else a), 0) if a else 0 for a, b in zip(reg["na"], reg["nb"])]
    reg["reg"] = reg["side"] + reg["num"].map(lambda i: f"{i:03d}")
    reg["cap"] = reg["na"].map(lambda s: bool(s) and s.startswith(CAPITAL))
    m_region = pd.DataFrame({"k1": reg["a"].map(kk), "k2": reg["b"].map(kk), "side": reg["side"],
                             "sido": reg["sido"], "reg": reg["reg"], "cap": reg["cap"]})
    names = pd.DataFrame([{"코드번호": f"{i:03d}", "시도코드": sido_code[a], "시도": a, "시군구": b} for (a, b), i in pair_code.items()])

    # 업종: (대, 중, 소) 조합으로 식별 (명세서: 소분류는 재분류되어 있음)
    c = part(["tco_bt_hlv_c_nm", "tco_bt_mcls_c_nm", "tco_btc_nm"])
    m_cat = pd.DataFrame({"k1": c["tco_bt_hlv_c_nm"].map(kk), "k2": c["tco_bt_mcls_c_nm"].map(kk), "k3": c["tco_btc_nm"].map(kk),
                          "cat_l": c["tco_bt_hlv_c_nm"].map(norm_text),
                          "cat_m": c["tco_bt_mcls_c_nm"].map(lambda v: norm_text(v) or "미상"),
                          "cat_s": c["tco_btc_nm"].map(lambda v: norm_text(v) or "미상")})
    for df, key in [(m_crym, ["k"]), (m_sex, ["k"]), (m_age, ["k"]), (m_region, ["k1", "k2", "side"]), (m_cat, ["k1", "k2", "k3"])]:
        assert not df.duplicated(key).any(), "대응표 키가 중복되면 원본 행이 늘어남"
    return {"m_crym": m_crym, "m_sex": m_sex, "m_age": m_age, "m_region": m_region, "m_cat": m_cat,
            "m_region_names": names}


# ---------------------------------------------------------------- 점검·검산
def _quality(con, cfg, info, maps, secure, report_dir, log) -> dict:
    one = lambda sql: con.execute(sql).fetchone()
    n_all, n_ok = one("SELECT count(*), count(*) FILTER (WHERE reason IS NULL) FROM clean")
    excl = con.execute("SELECT reason, count(*) n, coalesce(sum(am),0) am FROM clean WHERE reason IS NOT NULL GROUP BY 1").df()
    q = {"파일": {k: info[k] for k in ["file", "encoding", "delimiter_name", "header", "size"]},
         "행수": {"원본": int(n_all), "사용": int(n_ok), "제외": int(n_all - n_ok)},
         "제외내역": {r.reason: {"행수": int(r.n), "금액": int(r.am)} for r in excl.itertuples()}}

    dist = lambda col: {str(k if k is not None and k == k else "범주밖·결측"): int(v)
                        for k, v in con.execute(f"SELECT {col}, count(*) FROM clean GROUP BY 1 ORDER BY 1").fetchall()}
    q["범주값"] = {"기준년월": dist("crym"), "성별": dist("sex"), "연령대": dist("age_g")}
    cats = [r[0] for r in con.execute("SELECT DISTINCT cat_l FROM clean WHERE reason IS NULL ORDER BY 1").fetchall()]
    q["업종"] = {"대분류_K": len(cats), "대분류": cats,
                 "중분류_수": one("SELECT count(DISTINCT (cat_l, cat_m)) FROM clean WHERE reason IS NULL")[0],
                 "소분류_수": one("SELECT count(DISTINCT (cat_l, cat_m, cat_s)) FROM clean WHERE reason IS NULL")[0]}
    q["지역"] = {"시도_수": one("SELECT count(DISTINCT cust_sido) FROM clean")[0],
                 "고객지역_수": one("SELECT count(DISTINCT cust_reg) FROM clean")[0],
                 "가맹점지역_수": one("SELECT count(DISTINCT merch_reg) FROM clean")[0]}
    f = one("""SELECT count(*) FILTER (WHERE am < 0), coalesce(sum(am) FILTER (WHERE am < 0), 0),
                      count(*) FILTER (WHERE ct = 0 AND am <> 0), count(*) FILTER (WHERE ct = 1 AND am % 1000 = 0),
                      count(*) FILTER (WHERE ct_castfail), count(*) FILTER (WHERE am_castfail), count(*) FILTER (WHERE fraction)
               FROM clean WHERE reason IS NULL""")
    q["점검"] = {"음수금액_행수": int(f[0]), "음수금액_합계": int(f[1]), "건수0_금액있음_행수": int(f[2]),
                 "건수1_천원단위_행수": int(f[3]), "건수_변환실패": int(f[4]), "금액_변환실패": int(f[5]), "소수점값": int(f[6])}
    if f[0] or f[2]:
        log(f"[0] 경고: 음수 금액 {f[0]:,}행, 건수0·금액있음 {f[2]:,}행 (합산에는 포함)")
    if f[2]:
        con.execute(f"COPY (SELECT * FROM clean WHERE reason IS NULL AND ct = 0 AND am <> 0) TO '{_q(secure / 'ct0_am_nonzero.csv')}' (HEADER)")

    # 월별 수집 완결성 (계획서 3-1)
    months = month_range(cfg["기간"]["전체_시작월"], cfg["기간"]["전체_끝월"])
    ms = con.execute("SELECT crym, count(*) n, sum(am) am FROM clean WHERE reason IS NULL GROUP BY 1").df()
    ms = ms.set_index("crym").reindex(months)
    ms["금액_전월대비%"] = (ms["am"].astype(float).pct_change() * 100).round(2)
    ms["행수_전월대비%"] = (ms["n"].astype(float).pct_change() * 100).round(2)
    ms["경고(±30%)"] = ((ms["금액_전월대비%"].abs() >= 30) | (ms["행수_전월대비%"].abs() >= 30)).astype(int)
    ms = ms.reset_index().rename(columns={"index": "crym", "n": "행수", "am": "금액"})
    write_csv(ms, report_dir / "monthly_summary.csv")
    q["월별"] = {"누락월": [int(m) for m, n in zip(ms["crym"], ms["행수"]) if pd.isna(n)],
                 "급변월": [int(m) for m, a in zip(ms["crym"], ms["경고(±30%)"]) if a == 1]}

    # 검산 (계획서 3-4): 원본을 다시 읽은 월별 행수·금액 = 정제 원본(제외 포함) 월별 행수·금액
    r = con.execute(f"SELECT coalesce(crym, '{NULL}') k, count(*) n, coalesce(sum(am), 0) am FROM parsed GROUP BY 1").df()
    r = r.merge(maps["m_crym"], on="k", how="left").groupby("crym", dropna=False)[["n", "am"]].sum()
    c = con.execute("SELECT crym, count(*) n, coalesce(sum(am), 0) am FROM clean GROUP BY 1").df().set_index("crym")
    r.index = r.index.astype("Int64"); c.index = c.index.astype("Int64")
    j = r.join(c, lsuffix="_원본", rsuffix="_정제", how="outer").fillna(0)
    ok = bool((j["n_원본"] == j["n_정제"]).all() and (j["am_원본"] == j["am_정제"]).all())
    excl_am = int(excl["am"].sum()) if len(excl) else 0
    ok_am = one("SELECT coalesce(sum(am), 0) FROM clean WHERE reason IS NULL")[0]
    q["검산"] = {"통과": ok and int(j["n_원본"].sum()) == n_all,
                 "원본_행수": int(j["n_원본"].sum()), "정제_행수": int(n_all),
                 "원본_금액": int(j["am_원본"].sum()), "사용_금액": int(ok_am), "제외_금액": excl_am}
    return q


def _duplicates(con, report_dir) -> dict:
    """12칼럼이 모두 같은 행·키만 같은 행이 어디에 몰려 있는지 (수도권 여부 × 건수 구간 × 천원단위)."""
    keys = "crym, sex, age_g, cust_reg, merch_reg, cat_l, cat_m, cat_s"
    d = con.execute(f"""
        SELECT CASE WHEN cust_cap THEN '수도권' ELSE '비수도권' END AS 고객지역,
               CASE WHEN merch_cap THEN '수도권' ELSE '비수도권' END AS 가맹점지역,
               CASE WHEN ct <= 1 THEN '1건' WHEN ct <= 5 THEN '2~5건' ELSE '6건 이상' END AS 건수구간,
               CASE WHEN am % 1000 = 0 THEN '천원단위' ELSE '그 외' END AS 금액단위,
               count(*) AS 전체_행수,
               count(*) FILTER (WHERE rn > 1) AS 완전중복_초과행수,
               count(*) FILTER (WHERE nk > 1) AS 키중복_행수
        FROM (SELECT *, row_number() OVER (PARTITION BY {keys}, ct, am) rn, count(*) OVER (PARTITION BY {keys}) nk FROM clean)
        GROUP BY ALL ORDER BY ALL""").df()
    tot, dup = d["전체_행수"].sum(), d["완전중복_초과행수"].sum()
    d["전체_중_비율%"] = (d["전체_행수"] / tot * 100).round(2)
    d["완전중복_중_비율%"] = (d["완전중복_초과행수"] / max(dup, 1) * 100).round(2)
    write_csv(d, report_dir / "duplicate_diagnosis.csv")
    non_cap = ((d["고객지역"] == "비수도권") | (d["가맹점지역"] == "비수도권"))
    s = lambda mask, col: float(d.loc[mask, col].sum() / max(d[col].sum(), 1) * 100)
    groups = con.execute(f"SELECT count(*) FROM (SELECT 1 FROM clean GROUP BY {keys}, ct, am HAVING count(*) > 1)").fetchone()[0]
    return {"완전중복_묶음수": int(groups), "완전중복_초과행수": int(dup),
            "완전중복_중_비수도권포함%": round(s(non_cap, "완전중복_초과행수"), 1),
            "전체_중_비수도권포함%": round(s(non_cap, "전체_행수"), 1),
            "완전중복_중_1건%": round(s(d["건수구간"] == "1건", "완전중복_초과행수"), 1),
            "전체_중_1건%": round(s(d["건수구간"] == "1건", "전체_행수"), 1),
            "키중복_행수": int(d["키중복_행수"].sum()),
            "처리": "지우지 않고 합산 (익명처리로 생긴 정상 행일 가능성)"}
