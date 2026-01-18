import streamlit as st
import pandas as pd
from pathlib import Path
import subprocess

st.set_page_config(
    page_title="Cartola FC — Pro Lab 2026",
    layout="wide"
)

DATA_TEAMS = Path("data/teams_2026.csv")

# ======================
# SIDEBAR
# ======================
st.sidebar.title("⚙️ Configurações")

n_times = st.sidebar.slider("Quantidade de times", 1, 10, 3)
budget = st.sidebar.slider("Cartoletas disponíveis (titulares + técnico)", 80.0, 250.0, 120.0)
formation = st.sidebar.selectbox(
    "Formação",
    ["3-4-3", "3-5-2", "4-3-3", "4-4-2", "4-5-1", "5-3-2", "5-4-1"]
)

allow_repeat = st.sidebar.checkbox("Permitir repetir atletas entre times")
max_repeat = 1
if allow_repeat:
    max_repeat = st.sidebar.slider("Limite de repetição (entre times)", 1, 5, 2)

show_logs = st.sidebar.checkbox("Mostrar logs detalhados do gerador", value=False)

if st.sidebar.button("🚀 Gerar times"):
    cmd = [
        "python",
        "scripts/team_generator_2026.py",
        "--n-times", str(n_times),
        "--budget", str(budget),
        "--formation", formation,
        "--out", str(DATA_TEAMS),
    ]

    # Banco não consome cartoletas, mas mantemos compatibilidade com o script (ele ignora)
    cmd += ["--bench-budget-ratio", "0.0"]

    if allow_repeat:
        cmd.append("--allow-repeat")
        cmd += ["--max-repeat", str(max_repeat)]

    with st.spinner("Gerando times..."):
        result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        st.error("Erro ao gerar os times.")
        if show_logs:
            st.subheader("📜 Logs do gerador")
            st.code((result.stdout or "") + "\n" + (result.stderr or ""))
        else:
            st.info("Ative 'Mostrar logs detalhados do gerador' para ver o erro completo.")
    else:
        st.success("Times gerados com sucesso!")
        if show_logs and ((result.stdout or "").strip() or (result.stderr or "").strip()):
            st.subheader("📜 Logs do gerador")
            st.code((result.stdout or "") + "\n" + (result.stderr or ""))

# ======================
# MAIN
# ======================
st.title("⚽ Cartola FC — Pro Lab 2026")
st.caption("Geração inteligente: histórico (2023–2025) + mercado 2026 | Capitão x1,5 + Técnico + Banco + Luxo")

if not DATA_TEAMS.exists():
    st.info("Nenhum time gerado ainda.")
    st.stop()

df = pd.read_csv(DATA_TEAMS)

# Normaliza colunas esperadas (pra não quebrar se algum CSV antigo estiver aí)
for col, default in [
    ("role", ""),
    ("is_captain", False),
    ("is_luxury", False),
    ("clube_abrev", ""),
    ("preco_main", 0.0),
    ("preco_bench", 0.0),
    ("preco_total", 0.0),
    ("score_total", 0.0),
    ("budget", 0.0),
    ("budget_ok", 1),
    ("budget_left", 0.0),
    ("min_cost_required", 0.0),
    ("captain_mult", 1.5),
]:
    if col not in df.columns:
        df[col] = default

# ======================
# KPIs
# ======================
st.subheader("📊 Resumo")

teams_n = int(df["team_id"].nunique()) if "team_id" in df.columns else 0
score_mean = float(df.groupby("team_id")["score_total"].first().mean()) if teams_n else 0.0
preco_main_mean = float(df.groupby("team_id")["preco_main"].first().mean()) if teams_n else 0.0
unique_players = int(df["atleta_id"].nunique()) if "atleta_id" in df.columns else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Times gerados", teams_n)
c2.metric("Score médio (cap x1,5)", f"{score_mean:.2f}")
c3.metric("Preço médio (titulares+TEC)", f"{preco_main_mean:.2f}")
c4.metric("Atletas únicos", unique_players)

# ======================
# TIMES
# ======================
st.subheader("🧩 Times gerados")

for team_id, team in df.groupby("team_id"):
    st.markdown(f"## 🟦 Time {team_id}")

    budget_team = float(team["budget"].iloc[0]) if "budget" in team.columns else 0.0
    preco_main = float(team["preco_main"].iloc[0])
    preco_bench = float(team["preco_bench"].iloc[0])
    preco_total = float(team["preco_total"].iloc[0])
    score_total = float(team["score_total"].iloc[0])

    budget_ok = int(team["budget_ok"].iloc[0]) if "budget_ok" in team.columns else 1
    budget_left = float(team["budget_left"].iloc[0]) if "budget_left" in team.columns else (budget_team - preco_main)
    min_required = float(team["min_cost_required"].iloc[0]) if "min_cost_required" in team.columns else 0.0
    cap_mult = float(team["captain_mult"].iloc[0]) if "captain_mult" in team.columns else 1.5

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Score total", f"{score_total:.2f}")
    k2.metric("Cartoletas disponíveis", f"{budget_team:.2f}")
    k3.metric("Gasto (titulares+TEC)", f"{preco_main:.2f}")
    k4.metric("Banco (não consome cartoletas)", f"{preco_bench:.2f}")
    k5.metric("Sobra (cartoletas)", f"{budget_left:.2f}")

    if not budget_ok:
        st.warning(
            f"Orçamento insuficiente para fechar a formação COM o filtro de cartoletas. "
            f"Gerado o time completo mais barato possível. "
            f"(mínimo estimado: {min_required:.2f} | capitão x{cap_mult:.1f})"
        )

    # Captain + Coach
    captain = team[(team["role"] == "starter") & (team["is_captain"] == True)]
    coach = team[team["role"] == "coach"]

    cap_name = captain["apelido"].iloc[0] if len(captain) else "—"
    cap_pos = captain["posicao"].iloc[0] if len(captain) else "—"
    cap_club = captain["clube_abrev"].iloc[0] if len(captain) else "—"
    tec_name = coach["apelido"].iloc[0] if len(coach) else "—"
    tec_club = coach["clube_abrev"].iloc[0] if len(coach) else "—"

    st.markdown(
        f"**👑 Capitão (x{cap_mult:.1f}):** {cap_name} ({cap_pos} — {cap_club})  \n"
        f"**🧑‍🏫 Técnico:** {tec_name} ({tec_club})"
    )

    # Starters table
    starters = team[team["role"] == "starter"].copy()
    starters["C"] = starters["is_captain"].apply(lambda x: "👑" if bool(x) else "")
    starters = starters[["C", "apelido", "clube_abrev", "posicao", "preco", "score"]].sort_values(
        "score", ascending=False
    )
    starters = starters.rename(columns={"clube_abrev": "time"})

    st.markdown("### Titulares")
    st.dataframe(starters, use_container_width=True)

    # Coach table
    coach_df = team[team["role"] == "coach"].copy()
    if len(coach_df):
        coach_df = coach_df[["apelido", "clube_abrev", "posicao", "preco", "score"]].rename(columns={"clube_abrev": "time"})
        st.markdown("### Técnico")
        st.dataframe(coach_df, use_container_width=True)

    # Bench table
    bench = team[team["role"] == "bench"].copy()
    if len(bench):
        bench["Luxo"] = bench["is_luxury"].apply(lambda x: "💎" if bool(x) else "")
        bench = bench[["Luxo", "apelido", "clube_abrev", "posicao", "preco", "score"]].sort_values(
            ["Luxo", "score"], ascending=[False, False]
        )
        bench = bench.rename(columns={"clube_abrev": "time"})
        st.markdown("### Banco de reservas + reserva de luxo")
        st.dataframe(bench, use_container_width=True)
    else:
        st.info("Banco não gerado (sem opções que respeitem a regra de preço por posição).")

    st.divider()

# ======================
# DOWNLOAD
# ======================
st.download_button(
    "⬇️ Baixar CSV",
    df.to_csv(index=False),
    file_name="teams_2026.csv",
    mime="text/csv"
)
