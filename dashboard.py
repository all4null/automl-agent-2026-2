"""대시보드: 작업 노트 → report.html (실행 1회), index.html (전체 실행 목록).

계산은 하지 않고, 노트에 적힌 집계값을 표와 그림으로 옮기기만 한다.
그림은 PNG를 HTML 안에 넣어서 파일 하나로 열린다 (인터넷 불필요). 더블클릭하면 브라우저로 열림.
"""
from __future__ import annotations

import base64
import html
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from matplotlib import font_manager  # noqa: E402

_FONTS = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = next((f for f in ["Malgun Gothic", "AppleGothic", "NanumGothic"] if f in _FONTS), "DejaVu Sans")
plt.rcParams["axes.unicode_minus"] = False
C1, C2, C3, GREY = "#2f6fdf", "#e07b39", "#2a9d6f", "#9aa3ad"

CSS = """
:root{--bg:#f7f8fa;--card:#fff;--ink:#1d2433;--sub:#5b6474;--line:#e3e6eb;--ok:#2a9d6f;--warn:#d9822b;--bad:#d64545;--acc:#2f6fdf}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 "Malgun Gothic","Apple SD Gothic Neo",sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 64px}h1{font-size:24px;margin:0 0 4px}h2{font-size:19px;margin:36px 0 12px;padding-top:8px;border-top:2px solid var(--line)}
h3{font-size:16px;margin:20px 0 8px}.sub{color:var(--sub);font-size:13px}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:12px 0}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.kpi b{display:block;font-size:22px}.kpi span{color:var(--sub);font-size:13px}.ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}
.tbl{overflow-x:auto;margin:8px 0}table{border-collapse:collapse;font-size:13px;min-width:100%}th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
th{background:#eef1f5;position:sticky;top:0}td.num{text-align:right;font-variant-numeric:tabular-nums}img{max-width:100%;height:auto}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:12px}.pill{display:inline-block;padding:1px 8px;border-radius:99px;background:#eef1f5;font-size:12px}
details{margin:6px 0}summary{cursor:pointer;color:var(--acc)}nav a{margin-right:12px;font-size:13px;color:var(--acc);text-decoration:none}
"""


def _img(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return f'<img alt="" src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def _table(df: pd.DataFrame, max_rows: int = 200) -> str:
    if df is None or len(df) == 0:
        return '<p class="sub">(없음)</p>'
    df = df.head(max_rows)
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = []
    for _, r in df.iterrows():
        tds = []
        for v in r:
            num = isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                s = ""
            elif isinstance(v, (float, np.floating)):
                s = f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.4g}"
            else:
                s = f"{v:,}" if num else str(v)
            tds.append(f'<td class="{"num" if num else ""}">{html.escape(s)}</td>')
        body.append("<tr>" + "".join(tds) + "</tr>")
    return f'<div class="tbl"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _kv(d: dict) -> pd.DataFrame:
    return pd.DataFrame([{"항목": k, "값": v if not isinstance(v, (dict, list)) else str(v)} for k, v in d.items()])


# ---------------------------------------------------------------- report.html
def build(state, insights: dict):
    from agents.planner import describe
    import llm
    cfg, q, t = state.config, state.quality, state.test
    final = state.results[state.current]
    S = []
    S.append(f"<h1>AutoML-Agent 실행 보고서</h1><p class='sub'>실행 {state.run_dir.name} · LLM 모드 {llm.mode(state)} · "
             f"탐색 {final['탐색방법']} · 원본 {html.escape(q['파일']['file'])}</p>")
    S.append("<nav>" + "".join(f'<a href="#s{i}">{n}</a>' for i, n in enumerate(
        ["요약", "원본 점검", "설정", "결정 과정", "성능", "오류 분석", "업종 인사이트", "Mini Agent", "AI 활용 내역"], 1)) + "</nav>")

    # 1. 요약
    S.append('<h2 id="s1">1. 요약</h2>')
    if t:
        verdict = f"<b class='{'ok' if t['성공'] else 'bad'}'>{'성공' if t['성공'] else '실패'}</b>"
        k = [(verdict, f"성공 기준: 테스트 3개월 중 2개월 이상 두 규칙보다 높음 ({t['성공월수']}/3)"),
             (f"<b>{t['최종']['macro_f1']:.3f}</b>", "테스트 Macro-F1 (144문제, 1회)"),
             (f"<b>{final['검증F1']:.3f}</b>", "검증 Macro-F1 (144문제)"),
             (f"<b class='{'warn' if t['최종']['검증_테스트_차'] > 0.05 else ''}'>{t['최종']['검증_테스트_차']:+.3f}</b>", "검증 − 테스트 점수 차"),
             (f"<b>{len(state.trials):,}</b>", f"비교한 후보 수 (검증 {state.n_validations:,}회 학습)"),
             ("<b>1회</b>", "이번 실행에서 테스트를 연 횟수")]
        S.append('<div class="kpis">' + "".join(f'<div class="kpi">{a}<span>{b}</span></div>' for a, b in k) + "</div>")
        if t["과적합경고"]:
            S.append(f"<div class='card warn'>⚠ 최종 레시피의 테스트 점수({t['최종']['macro_f1']:.3f})가 처음 레시피({t['처음']['macro_f1']:.3f})보다 "
                     "낮습니다. 수정 단계에서 검증 데이터에 과적합됐을 수 있습니다.</div>")
    S.append(f"<div class='card'><b>최종 레시피 {state.current}</b>: {html.escape(describe(state.recipes[state.current]))}<br>"
             f"<b>모델</b>: {html.escape(final['모델'])} {html.escape(str(final['설정']))}</div>")
    ex = state.explanation
    if ex:
        lists = "".join(f"<h3>{html.escape(t)}</h3><ul>" + "".join(f"<li>{html.escape(str(x))}</li>" for x in ex.get(k) or []) + "</ul>"
                        for k, t in [("결정_해석", "에이전트 결정 해석"), ("업종_인사이트", "업종 인사이트"), ("다음에_해볼_것", "다음에 해볼 것")]
                        if ex.get(k))
        S.append(f"<div class='card'><b>LLM이 쓴 해석</b> <span class='pill'>{html.escape(ex['결정자'])} · {html.escape(ex['모델'])}</span> "
                 f"<span class='sub'>검증되지 않은 해석입니다. 테스트 점수는 보지 않고 썼습니다.</span>"
                 f"<p>{html.escape(ex['요약'])}</p>{lists}</div>")

    # 2. 원본 점검
    S.append('<h2 id="s2">2. 원본 점검</h2>')
    S.append(_table(_kv({"원본 행수": q["행수"]["원본"], "사용 행수": q["행수"]["사용"], "제외 행수": q["행수"]["제외"],
                         "제외 내역": q["제외내역"], "검산": "통과" if q["검산"]["통과"] else "실패",
                         "대분류 K": q["업종"]["대분류_K"], "중분류 수": q["업종"]["중분류_수"], "소분류 수": q["업종"]["소분류_수"],
                         "고객·가맹점 지역 수": f"{q['지역']['고객지역_수']} / {q['지역']['가맹점지역_수']} (코드로만 표시)",
                         "금액이 같아 빠진 '유지' 문제": q.get("유지_건수"), "월별 급변(±30%)": q["월별"]["급변월"] or "없음"})))
    S.append("<h3>점검 항목</h3>" + _table(_kv(q["점검"])))
    dr = state.data_review
    if dr:
        S.append(f"<div class='card'><b>원본 담당의 판단</b> <span class='pill'>{html.escape(dr['결정자'])}</span><ul>"
                 + "".join(f"<li>{html.escape(str(x))}</li>" for x in dr.get("데이터_주의점") or ["(주의점 없음)"]) + "</ul>"
                 + f"<p class='sub'>소분류 묶음 후보 제안: 비소비성 {len(dr.get('비소비성_후보', []))}개, "
                   f"본사집계의심 {len(dr.get('본사집계의심_후보', []))}개. {html.escape(dr.get('이유', ''))}</p></div>")
    if "중복" in q:
        d = q["중복"]
        S.append(f"<h3>적재 중복 진단</h3><p>12칼럼이 모두 같은 행: {d['완전중복_묶음수']:,}묶음, 초과 {d['완전중복_초과행수']:,}행. "
                 f"이 중 비수도권이 끼어 있는 비율 <b>{d['완전중복_중_비수도권포함%']}%</b> (전체 행에서는 {d['전체_중_비수도권포함%']}%), "
                 f"1건 결제 비율 <b>{d['완전중복_중_1건%']}%</b> (전체 {d['전체_중_1건%']}%). 처리: {d['처리']}.</p>")
        p = state.run_dir.parent.parent / "원본점검" / Path(q["파일"]["file"]).stem / "duplicate_diagnosis.csv"
        if p.exists():
            S.append("<details><summary>구간별 표</summary>" + _table(pd.read_csv(p, encoding="utf-8-sig")) + "</details>")

    # 3. 설정
    S.append('<h2 id="s3">3. 설정 요약</h2>')
    used = [m for m, v in cfg["후보모델"].items() if v.get("사용") in ("예", True)]
    S.append(_table(_kv({"검증월 (롤링)": cfg["기간"]["검증월"], "테스트월": cfg["기간"]["테스트월"],
                         "후보 모델": ", ".join(used), "기준 모델": cfg["기준모델"],
                         "채택 규칙": f"짝 비교 부트스트랩 {cfg['판단규칙']['부트스트랩_승률']:.0%} + 상승폭 {cfg['판단규칙']['최소상승폭']} + "
                                     f"{len(cfg['기간']['검증월'])}개월 중 {cfg['판단규칙']['최소상승월수']}개월 상승",
                         "수정 최대 횟수": cfg["판단규칙"]["수정_최대횟수"], "LLM 모드": llm.mode(state)})))

    # 4. 결정 과정
    S.append('<h2 id="s4">4. 에이전트 결정 과정</h2>')
    S.append(_table(pd.DataFrame(state.decisions)))
    S.append("<h3>레시피</h3>" + _table(pd.DataFrame([{"이름": k, "설명": v.get("설명", ""), "내용": describe(v),
                                                         "검증F1": state.results[k]["검증F1"] if k in state.results else None}
                                                        for k, v in state.recipes.items()])))
    if state.revisions:
        S.append("<h3>수정 시도 (사전 점검 신호등)</h3>")
        S.append(_table(pd.DataFrame([{k: r.get(k) for k in ["번호", "수정안", "결정자", "사전점검", "사전점검_이유", "결과", "기존F1", "새F1", "채택판단"]}
                                      for r in state.revisions])))
        for r in state.revisions:
            S.append(_precheck_block(r))

    # 5. 성능
    S.append('<h2 id="s5">5. 성능</h2>')
    S.append("<h3>검증: 모델별 1등 설정 (최종 레시피)</h3>")
    S.append(_bar(final["모델별"].set_index("모델")["검증F1"], f"검증 Macro-F1 (규칙 최고 {max(final['규칙F1'].values()):.3f})",
                  ref=max(final["규칙F1"].values()), mark=final["모델"]))
    S.append(_table(final["모델별"]))
    if t:
        S.append("<h3>테스트 (1회)</h3>" + _table(pd.DataFrame(t["표"])))
        ti = state.test_interpretation
        if ti:
            S.append(f"<div class='card'><b>실행 담당의 테스트 해석</b> <span class='pill'>{html.escape(ti['결정자'])} · {html.escape(ti['모델'])}</span>"
                     f"<p>{html.escape(ti['해석'])}</p><ul>" + "".join(f"<li>{html.escape(str(x))}</li>" for x in ti.get("주의점") or [])
                     + "</ul><p class='sub'>모든 선택이 끝난 뒤 작성된 해석이며, 결과를 바꾸지 않습니다.</p></div>")
        cm = t["혼동행렬"]
        S.append('<div class="grid2">' + _confusion(cm) + _monthly_fig(state) + "</div>")
        S.append("<h3>클래스별</h3>" + _table(pd.DataFrame(t["클래스별"]).T.reset_index().rename(columns={"index": "클래스"})))
        S.append("<h3>월별 갱신 실험</h3>" + _table(pd.DataFrame(state.monthly["표"]).pivot_table(
            index=["방식", "모델"], columns="월", values="macro_f1").reset_index()))
        if state.monthly["지식손실"]:
            S.append("<h3>지식 손실 점검 (SGD 순차 갱신)</h3>" + _table(pd.DataFrame(state.monthly["지식손실"])))
        S.append("<h3>분포 변화</h3>" + _table(pd.DataFrame(state.monthly["분포변화"])))

    # 6. 오류 분석
    S.append('<h2 id="s6">6. 오류 분석</h2>')
    if t:
        e = t["오류분석"]
        S.append(_heat(pd.DataFrame(e["고객층x업종"]).set_index("고객층"), "고객층 × 업종 정답률 (셀당 3문제라 참고용)"))
        S.append(_table(_kv({"명절·쿠폰 예측대상월 정답률": e["행사월"]})))
        S.append("<h3>성공 사례 3건</h3>" + _table(pd.DataFrame(t["사례"]["성공"])))
        S.append("<h3>실패 사례 3건</h3>" + _table(pd.DataFrame(t["사례"]["실패"])))

    # 7. 업종 인사이트
    S.append('<h2 id="s7">7. 업종별 인사이트</h2>')
    S.append(_trend(insights["업종월별"], insights))
    S.append("<h3>명절·소비쿠폰 전후 비교</h3>" + _table(insights["행사전후"]))
    if t:
        fi = t["변수영향"]
        S.append(_bar(fi.set_index("변수")["순열중요도(테스트)"].head(15)[::-1], "변수 영향: 순열 중요도 상위 15 (값을 섞으면 점수가 얼마나 떨어지나)", horizontal=True))
        S.append("<details><summary>변수 영향 전체 표 (로지스틱 계수·LightGBM 중요도 포함)</summary>" + _table(fi) + "</details>")

    # 8. Mini Agent
    S.append('<h2 id="s8">8. Mini AutoML-Agent 구성</h2>')
    S.append(_table(pd.DataFrame([
        ["[0] 원본 (raw_agent)", "원본 CSV 점검·정제 (코드)", "품질 점검 해석, 소분류 묶음 후보 제안", "원본 행은 안 봄, 중복 삭제 금지"],
        ["[1] 계획 (planner)", "처음 레시피 A", "레시피 B에 켤 변수 (도구: 변수 신호 미리보기)", "메뉴 밖 불가, 검토 담당이 검토"],
        ["[2] 데이터 (data_agent)", "정답표(고정)·변수표·사전 점검 (코드)", "분해 변수 개수, 🟢 수정안을 진행할지", "🟡🔴 못 뒤집음, 정답 고정"],
        ["[3] 모델 (model_agent)", "후보 학습·롤링 검증 (탐색 도구)", "탐색할 모델 계열·범위, 1차 결과 보고 범위 조정", "숫자는 탐색이 찾음, 허용 범위 안, 채택 규칙"],
        ["[4] 실행 (executor)", "테스트 1회·월별 갱신·예측표 (코드)", "테스트 전 실험 계획, 테스트 후 해석", "테스트 본 뒤 다시 고르기 금지"],
        ["[5] 검토 (reviewer)", "채택 규칙 판정·보고서 (코드)", "계획 검토, 다음 수정안·묶음, 결과 해석", "채택은 규칙이 판정"]],
        columns=["에이전트", "코드가 하는 일", "LLM이 생각하는 일", "넘을 수 없는 선"])))
    pdir = Path(__file__).resolve().parent / "prompts"
    for f in sorted(pdir.glob("*.md")):
        S.append(f"<details><summary>프롬프트: {f.name}</summary><pre style='white-space:pre-wrap'>{html.escape(f.read_text(encoding='utf-8'))}</pre></details>")

    # 9. AI 활용 내역
    S.append('<h2 id="s9">9. AI 활용 내역</h2>')
    if state.llm_calls:
        S.append(_table(pd.DataFrame([{"번호": c["번호"], "시각": c["시각"], "에이전트": c.get("에이전트"), "목적": c["목적"],
                                       "모드": c["모드"], "모델": c.get("모델"), "도구": len(c.get("도구사용", [])),
                                       "채택": c.get("채택"), "답": str(c.get("답"))[:300], "비고": c.get("비고", ""),
                                       "기록 파일": c.get("파일", "")} for c in state.llm_calls])))
        title = {"system": "시스템 프롬프트", "user": "사용자 (지시문 + 자료)", "assistant": "LLM 응답"}
        for c in state.llm_calls:
            turns = []
            for i, msg in enumerate(c.get("대화", [])):
                head = "도구 결과 / 추가 지시" if (msg["role"] == "user" and i > 1) else title.get(msg["role"], msg["role"])
                turns.append(f"<h3>{i}. {head}</h3><pre style='white-space:pre-wrap'>{html.escape(str(msg['content']))}</pre>")
            S.append(f"<details class='card'><summary>LLM 호출 {c['번호']} · {html.escape(c.get('에이전트', ''))} 담당 · "
                     f"{html.escape(c['목적'])} · 도구 {len(c.get('도구사용', []))}회 — {'채택' if c.get('채택') else '무효'}</summary>"
                     + "".join(turns) + "</details>")
    else:
        S.append("<p>이번 실행은 LLM 없이 규칙 모드로 판단했습니다. 결정 기록의 '결정자' 칸에서 규칙·탐색·LLM을 구분할 수 있습니다.</p>")

    doc = (f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>AutoML-Agent 보고서 {state.run_dir.name}</title><style>{CSS}</style></head><body><main>{''.join(S)}</main></body></html>")
    (state.run_dir / "report.html").write_text(doc, encoding="utf-8")


def _precheck_block(r: dict) -> str:
    pc = r.get("precheck") or {}
    figs = []
    ba = pc.get("대분류금액_전후") or {}
    if ba and any(abs(a - b) > 0 for a, b in ba.values()):
        cats = list(ba)
        fig, ax = plt.subplots(figsize=(6, 2.8))
        x = np.arange(len(cats))
        ax.bar(x - 0.2, [ba[c][0] / 1e8 for c in cats], 0.4, label="전", color=GREY)
        ax.bar(x + 0.2, [ba[c][1] / 1e8 for c in cats], 0.4, label="후", color=C1)
        ax.set_xticks(x, cats, rotation=30, ha="right", fontsize=8); ax.set_ylabel("금액(억)"); ax.legend(fontsize=8)
        ax.set_title("대분류별 변수 계산 금액 (전/후)", fontsize=10)
        figs.append(_img(fig))
    h = pc.get("분포")
    if h:
        fig, ax = plt.subplots(figsize=(5, 2.8))
        if h.get("전") is not None:
            ax.hist(h["전"], bins=20, alpha=0.5, color=GREY, label="전")
        ax.hist(h["후"], bins=20, alpha=0.6, color=C1, label="후")
        ax.set_title(f"{h['변수']} 분포 (3~5월 학습 문제)", fontsize=10); ax.legend(fontsize=8)
        figs.append(_img(fig))
    sig = pc.get("신호") or {}
    body = f"<b>{r['번호']} {html.escape(r['수정안'])}</b> {r['사전점검']} {html.escape(r['결과'])}<br><span class='sub'>{html.escape(r['사전점검_이유'])}</span>"
    if sig:
        body += _table(pd.DataFrame([{"변수": k, "AUC": v, "신호(|AUC−0.5|)": round(abs(v - 0.5), 3)} for k, v in sig.items()])
                       .sort_values("신호(|AUC−0.5|)", ascending=False).head(10))
    return f"<details class='card'><summary>{r['번호']} {html.escape(r['수정안'])} — {r['사전점검']} {html.escape(r['결과'])}</summary>{body}<div class='grid2'>{''.join(figs)}</div></details>"


def _bar(s: pd.Series, title: str, ref=None, mark=None, horizontal=False) -> str:
    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.32 * len(s)) if horizontal else 3.2))
    colors = [C2 if i == mark else C1 for i in s.index]
    if horizontal:
        ax.barh(range(len(s)), s.values, color=colors); ax.set_yticks(range(len(s)), s.index, fontsize=8)
    else:
        ax.bar(range(len(s)), s.values, color=colors); ax.set_xticks(range(len(s)), s.index, rotation=35, ha="right", fontsize=8)
        if ref is not None:
            ax.axhline(ref, color=GREY, ls="--", lw=1)
        ax.set_ylim(max(0, min(s.min(), ref or 1) - 0.1), min(1, s.max() + 0.08))
    ax.set_title(title, fontsize=10)
    return _img(fig)


def _confusion(cm: dict) -> str:
    m = np.array([[cm["실제감소_예측감소"], cm["실제감소_예측증가"]], [cm["실제증가_예측감소"], cm["실제증가_예측증가"]]])
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(m, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, m[i, j], ha="center", va="center", color="white" if m[i, j] > m.max() / 2 else "black")
    ax.set_xticks([0, 1], ["예측 감소", "예측 증가"]); ax.set_yticks([0, 1], ["실제 감소", "실제 증가"])
    ax.set_title("혼동행렬 (테스트)", fontsize=10)
    return _img(fig)


def _monthly_fig(state) -> str:
    df = pd.DataFrame(state.monthly["표"])
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    for (a, b), g in df.groupby(["방식", "모델"]):
        ax.plot(g["월"].astype(str), g["macro_f1"], marker="o", label=f"{a}/{b}")
    for r in state.test["표"]:
        if r["모델"].startswith("규칙"):
            ax.plot([str(m) for m in state.config["기간"]["테스트월"]], [r[f"{m}월"] for m in state.config["기간"]["테스트월"]],
                    ls=":", color=GREY, lw=1)
    ax.set_ylim(0, 1); ax.legend(fontsize=7); ax.set_title("월별 Macro-F1 (기준월, 점선=규칙)", fontsize=10)
    return _img(fig)


def _heat(df: pd.DataFrame, title: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.imshow(df.values.astype(float), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(df.shape[1]), df.columns, rotation=30, ha="right", fontsize=8); ax.set_yticks(range(df.shape[0]), df.index, fontsize=8)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            ax.text(j, i, f"{df.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title(title, fontsize=10)
    return _img(fig)


def _trend(tab: pd.DataFrame, ins: dict) -> str:
    w = tab.pivot_table(index="crym", columns="cat_l", values="금액")
    idx = w / w.iloc[0] * 100
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for c in idx.columns:
        ax.plot(idx.index.astype(str), idx[c], marker=".", label=c)
    for m in ins["명절월"]:
        ax.axvline(str(m), color=C2, alpha=0.3, lw=6)
    ax.set_title("업종별 월 지출 지수 (1월=100, 주황 띠=명절월)", fontsize=10)
    ax.legend(fontsize=7, ncol=4); ax.tick_params(axis="x", labelsize=8)
    return _img(fig)


# ---------------------------------------------------------------- index.html
def build_index(root: Path):
    p = root / "all_runs.csv"
    if not p.exists():
        return
    df = pd.read_csv(p, encoding="utf-8-sig").iloc[::-1]
    rows = []
    for _, r in df.iterrows():
        link = f"runs/{r['실행']}/report.html"
        rows.append("<tr>" + f"<td><a href='{link}'>{html.escape(str(r['실행']))}</a></td>" +
                    "".join(f"<td>{html.escape(str(r[c]))}</td>" for c in df.columns if c != "실행") + "</tr>")
    head = "".join(f"<th>{html.escape(c)}</th>" for c in df.columns)
    doc = (f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>AutoML-Agent 실행 목록</title><style>{CSS}</style></head><body><main><h1>AutoML-Agent 실행 목록</h1>"
           f"<p class='sub'>실행 이름을 누르면 그 실행의 보고서가 열립니다. 최신 실행이 위.</p>"
           f"<div class='tbl'><table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div></main></body></html>")
    (root / "index.html").write_text(doc, encoding="utf-8")
