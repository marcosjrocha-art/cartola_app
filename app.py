import streamlit as st
import pandas as pd
from pathlib import Path
import subprocess
import textwrap

st.set_page_config(
    page_title="Cartola FC — Pro Lab 2026",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_TEAMS = Path("data/teams_2026.csv")
DATA_SUMMARY = Path("data/teams_2026_summary.csv")

# =========
# Helpers
# =========
POS_NAME = {
    1: "GOL",
    2: "LAT",
    3: "ZAG",
    4: "MEI",
    5: "ATA",
    6: "TEC",
}

def fmt_money(x: float) -> str:
    try:
        return f"{float(x):.2f}"
    except Exception:
        return "0.00"

def fmt_score(x: float) -> str:
    try:
        return f"{float(x):.2f}"
    except Exception:
        return "0.00"

def pill(label: str) -> str:
    return f"<span style='padding:4px 10px;border-radius:999px;background:rgba(15,23,42,0.06);font-size:12px'>{label}</span>"

def card(title: str, value: str, subtitle: str = "") -> str:
    sub = f"<div style='color:rgba(15,23,42,0.65);font-size:12px;margin-top:2px'>{subtitle}</div>" if subtitle else ""
    return f"""
    <div style="border:1px solid rgba(15,23,42,0.10);border-radius:16px;padding:14px 14px 12px 14px;background:white">
        <div style="font-size:12px;color:rgba(15,23,42,0.70);font-weight:600;letter-spacing:.2px">{title}</div>
        <div style="font-size:26px;font-weight:800;line-height:1.1;margin-top:6px;color:rgba(15,23,42,0.95)">{value}</div>
        {sub}
    </div>
    """

# =========
# Global CSS (clean & elegant)
# =========
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 3rem;}
      h1, h2, h3 {letter-spacing: -0.5px;}
      .stDataFrame {border-radius: 14px; overflow: hidden;}
      div[data-testid="stMetric"] {border:1px solid rgba(15,23,42,0.10); border-radius: 16px; padding: 10px;}
      .muted {color: rgba(15,23,42,0.65); font-size: 13px;}
      .section {margin-top: 6px; padding: 14px; border: 1px solid rgba(15,23,42,0.10); border-radius: 16px; background: white;}
      .kpiRow {display:flex; gap: 12px; flex-wrap: wrap;}
    </style>
    """,
    unsafe_allow_html=True
)

# ======================
# SIDEBAR
# ======================
st.sidebar.title("⚙️ Configurações")

n_times = st.sidebar.slider("Quantidade de times", 1, 10, 3)
budget = st.sidebar.slider("Orçamento total (C$)", 80.0, 200.0, 120.0)
formation = st.sidebar.selectbox(
    "Formação",
    ["3-4-3", "3-5-2", "4-3-3", "4-4-2", "5-3-2", "5-4-1"],
    index=2
)
allow_repeat = st.sidebar.checkbox("Permitir repetir atletas entre times", value=False)

with st.sidebar.expander("⚙️ Avançado", expanded=False):
    history_root = st.text_input("History root", value="cartola/data/01_raw")
    years = st.text_input("Years (separados por espaço)", value="2023 2024 2025")
    min_games = st.number_input("Mínimo de jogos no histórico", min_value=1, max_value=50, value=1, step=1)
    shrink_k = st.number_input("Suavização (shrink-k)", min_value=1, max_value=50, value=5, step=1)
    require_complete = st.checkbox("Falhar se não montar time completo", value=False)

show_logs = st.sidebar.checkbox("Mostrar logs detalhados", value=False)

if st.sidebar.button("🚀 Gerar times", use_container_width=True):
    cmd = [
        "python",
        "scripts/team_generator_2026.py",
        "--history-root", str(history_root),
        "--years", *years.split(),
        "--n-times", str(int(n_times)),
        "--budget", str(float(budget)),
        "--formation", str(formation),
        "--min-games", str(int(min_games)),
        "--shrink-k", str(int(shrink_k)),
        "--out", str(DATA_TEAMS),
    ]
    if allow_repeat:
        cmd.append("--allow-repeat")
    if require_complete:
        cmd.append("--require-complete")

    with st.spinner("Gerando times (histórico + mercado 2026)..."):
        result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        st.error("Erro ao gerar os times. Veja os logs abaixo (ou no terminal).")
        if result.stdout:
            st.code(result.stdout)
        if result.stderr:
            st.code(result.stderr)
    else:
        st.success("Times gerados com sucesso!")
        if show_logs:
            if result.stdout:
                st.code(result.stdout)
            if result.stderr:
                st.code(result.stderr)

# ======================
# HEADER
# ======================
st.markdown("## ⚽ Cartola FC — Pro Lab 2026")
st.markdown(
    "<div class='muted'>Geração inteligente: histórico (2023–2025) + mercado 2026 • com <b>Capitão</b>, <b>Técnico</b>, <b>Reservas</b> e <b>Reserva de Luxo</b>.</div>",
    unsafe_allow_html=True
)

# ======================
# LOAD DATA
# ======================
if not DATA_TEAMS.exists():
    st.info("Nenhum time gerado ainda. Use o botão **🚀 Gerar times** na barra lateral.")
    st.stop()

try:
    df = pd.read_csv(DATA_TEAMS)
except Exception as e:
    st.error(f"Falha ao ler {DATA_TEAMS}: {e}")
    st.stop()

df_sum = None
if DATA_SUMMARY.exists():
    try:
        df_sum = pd.read_csv(DATA_SUMMARY)
    except Exception:
        df_sum = None

# Normalizações de colunas esperadas
for col in ["team_id", "slot", "posicao_id", "atleta_id", "preco_num", "score_pred"]:
    if col not in df.columns:
        st.error(f"CSV não tem a coluna obrigatória: {col}. Gere os times novamente.")
        st.stop()

df["posicao"] = df["posicao_id"].map(POS_NAME).fillna(df["posicao_id"].astype(str))
df["preco_num"] = pd.to_numeric(df["preco_num"], errors="coerce").fillna(0.0)
df["score_pred"] = pd.to_numeric(df["score_pred"], errors="coerce").fillna(0.0)

# ======================
# KPIs (summary first)
# ======================
st.markdown("### 📊 Resumo")

teams_n = int(df["team_id"].nunique())
unique_players = int(df["atleta_id"].nunique())

if df_sum is not None and not df_sum.empty:
    score_mean = float(df_sum["score_total_pred"].mean())
    budget_used_mean = float(df_sum["budget_used"].mean())
    budget_left_mean = float(df_sum["budget_left"].mean())
else:
    # fallback: calcula do detalhado
    # score_total_pred ~ starters + tecnico
    tmp = df[df["slot"].isin(["starter", "tecnico"])].groupby("team_id")["score_pred"].sum()
    score_mean = float(tmp.mean()) if len(tmp) else 0.0
    budget_used_mean = float(df.groupby("team_id")["preco_num"].sum().mean()) if teams_n else 0.0
    budget_left_mean = max(0.0, float(budget) - budget_used_mean)

k1, k2, k3, k4 = st.columns(4)
k1.markdown(card("Times gerados", str(teams_n), "geração atual"), unsafe_allow_html=True)
k2.markdown(card("Score médio previsto", fmt_score(score_mean), "titulares + técnico"), unsafe_allow_html=True)
k3.markdown(card("Preço médio usado", fmt_money(budget_used_mean), "C$"), unsafe_allow_html=True)
k4.markdown(card("Atletas únicos", str(unique_players), "no conjunto"), unsafe_allow_html=True)

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# ======================
# TIMES
# ======================
st.markdown("### 🧩 Times gerados")

team_ids = sorted(df["team_id"].unique().tolist())

# filtro superior (bonito e útil)
top_bar = st.columns([2, 2, 3, 3])
with top_bar[0]:
    team_sel = st.selectbox("Selecionar time", options=["Todos"] + [f"Time {t}" for t in team_ids], index=0)
with top_bar[1]:
    show_only_best = st.checkbox("Ordenar por score (desc)", value=True)
with top_bar[2]:
    show_tables = st.checkbox("Mostrar tabelas completas", value=True)
with top_bar[3]:
    compact = st.checkbox("Modo compacto", value=False)

def render_team(team_id: int, team: pd.DataFrame):
    starters = team[team["slot"] == "starter"].copy()
    tec = team[team["slot"] == "tecnico"].copy()
    reserves = team[team["slot"] == "reserve"].copy()

    # capitão
    cap = starters[starters.get("is_captain", 0).astype(int) == 1]
    cap_name = "—"
    cap_pos = "—"
    cap_score = 0.0
    if len(cap):
        cap_name = str(cap.iloc[0].get("apelido") or cap.iloc[0].get("nome") or "—")
        cap_pos = str(cap.iloc[0].get("posicao") or "—")
        cap_score = float(cap.iloc[0].get("score_pred", 0.0))

    # técnico
    tec_name = "—"
    tec_score = 0.0
    if len(tec):
        tec_name = str(tec.iloc[0].get("apelido") or tec.iloc[0].get("nome") or "—")
        tec_score = float(tec.iloc[0].get("score_pred", 0.0))

    # reserva de luxo
    luxo = reserves[reserves.get("is_luxury_reserve", 0).astype(int) == 1]
    luxo_name = "—"
    luxo_score = 0.0
    if len(luxo):
        luxo_name = str(luxo.iloc[0].get("apelido") or luxo.iloc[0].get("nome") or "—")
        luxo_score = float(luxo.iloc[0].get("score_pred", 0.0))

    # KPIs por time
    budget_used = float(team["preco_num"].sum())
    score_total = float(starters["score_pred"].sum()) + tec_score
    budget_left = max(0.0, float(budget) - budget_used)

    # header
    st.markdown(
        f"""
        <div class="section">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:10px;flex-wrap:wrap">
            <div>
              <div style="font-size:18px;font-weight:800;color:rgba(15,23,42,0.95)">🟦 Time {team_id}</div>
              <div class="muted">Formação: <b>{formation}</b> &nbsp;•&nbsp; {pill(f"Capitão: {cap_name} ({cap_pos})")} &nbsp; {pill(f"Técnico: {tec_name}")} &nbsp; {pill(f"Luxo: {luxo_name}")}</div>
            </div>
            <div style="display:flex;gap:10px;flex-wrap:wrap">
              <div style="text-align:right">
                <div class="muted">Score previsto (XI+TEC)</div>
                <div style="font-size:22px;font-weight:900">{fmt_score(score_total)}</div>
              </div>
              <div style="text-align:right">
                <div class="muted">Preço usado</div>
                <div style="font-size:22px;font-weight:900">C$ {fmt_money(budget_used)}</div>
              </div>
              <div style="text-align:right">
                <div class="muted">Saldo</div>
                <div style="font-size:22px;font-weight:900">C$ {fmt_money(budget_left)}</div>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Conteúdo
    cA, cB = st.columns([2.2, 1.3])

    with cA:
        st.markdown("#### Titulares")
        view = starters.copy()
        view["C"] = view.get("is_captain", 0).astype(int).apply(lambda x: "👑" if x == 1 else "")
        view["pos"] = view["posicao"]
        view["preço"] = view["preco_num"].apply(fmt_money)
        view["score"] = view["score_pred"].apply(fmt_score)

        cols = ["C", "apelido", "pos", "preço", "score"]
        for col in cols:
            if col not in view.columns:
                view[col] = ""

        view = view[cols]
        if show_only_best:
            # ordena por score (string -> converter)
            starters2 = starters.sort_values(["score_pred", "preco_num"], ascending=[False, True]).copy()
            view = starters2.copy()
            view["C"] = view.get("is_captain", 0).astype(int).apply(lambda x: "👑" if x == 1 else "")
            view["pos"] = view["posicao"]
            view["preço"] = view["preco_num"].apply(fmt_money)
            view["score"] = view["score_pred"].apply(fmt_score)
            view = view[cols]

        st.dataframe(view, use_container_width=True, hide_index=True)

    with cB:
        st.markdown("#### Técnico & Banco")
        # mini-cards
        st.markdown(
            f"""
            <div class="section" style="padding:12px">
              <div style="display:flex;flex-direction:column;gap:10px">
                <div>
                  <div class="muted">👑 Capitão (maior score previsto)</div>
                  <div style="font-size:16px;font-weight:900">{cap_name}</div>
                  <div class="muted">{cap_pos} • score {fmt_score(cap_score)}</div>
                </div>
                <div>
                  <div class="muted">🧑‍🏫 Técnico</div>
                  <div style="font-size:16px;font-weight:900">{tec_name}</div>
                  <div class="muted">score {fmt_score(tec_score)}</div>
                </div>
                <div>
                  <div class="muted">💎 Reserva de luxo</div>
                  <div style="font-size:16px;font-weight:900">{luxo_name}</div>
                  <div class="muted">score {fmt_score(luxo_score)}</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if reserves is None or reserves.empty:
            st.info("Reservas não geradas (orçamento/mercado).")
        else:
            r = reserves.copy()
            r["💎"] = r.get("is_luxury_reserve", 0).astype(int).apply(lambda x: "💎" if x == 1 else "")
            r["setor"] = r.get("reserva_setor", "").fillna("")
            r["pos"] = r["posicao"]
            r["preço"] = r["preco_num"].apply(fmt_money)
            r["score"] = r["score_pred"].apply(fmt_score)

            cols = ["💎", "setor", "apelido", "pos", "preço", "score"]
            for col in cols:
                if col not in r.columns:
                    r[col] = ""

            # luxo primeiro
            r = r.sort_values(["is_luxury_reserve", "score_pred"], ascending=[False, False])
            st.dataframe(r[cols], use_container_width=True, hide_index=True)

    if show_tables:
        with st.expander("📋 Ver dados brutos do time (debug)", expanded=False):
            st.dataframe(team.sort_values(["slot", "posicao_id", "score_pred"], ascending=[True, True, False]), use_container_width=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# render
if team_sel == "Todos":
    for tid in team_ids:
        render_team(tid, df[df["team_id"] == tid].copy())
else:
    tid = int(team_sel.replace("Time ", "").strip())
    render_team(tid, df[df["team_id"] == tid].copy())

# ======================
# DOWNLOADS
# ======================
st.markdown("### ⬇️ Downloads")

d1, d2 = st.columns([1, 1])
with d1:
    st.download_button(
        "Baixar CSV detalhado (times_2026.csv)",
        df.to_csv(index=False),
        file_name="teams_2026.csv",
        mime="text/csv",
        use_container_width=True
    )
with d2:
    if df_sum is not None and not df_sum.empty:
        st.download_button(
            "Baixar resumo (teams_2026_summary.csv)",
            df_sum.to_csv(index=False),
            file_name="teams_2026_summary.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("Resumo não encontrado. Gere os times novamente para criar o _summary.csv.")
