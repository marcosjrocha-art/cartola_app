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
budget = st.sidebar.slider("Orçamento total (C$)", 80.0, 200.0, 120.0)
formation = st.sidebar.selectbox("Formação", ["3-4-3","3-5-2","4-3-3","4-4-2","4-5-1","5-3-2","5-4-1"])
allow_repeat = st.sidebar.checkbox("Permitir repetir atletas entre times")
bench_ratio = st.sidebar.slider("Percentual do orçamento pro banco", 0.0, 0.40, 0.15)

if st.sidebar.button("🚀 Gerar times"):
    cmd = [
        "python",
        "scripts/team_generator_2026.py",
        "--n-times", str(n_times),
        "--budget", str(budget),
        "--formation", formation,
        "--bench-budget-ratio", str(bench_ratio),
        "--out", str(DATA_TEAMS)
    ]
    if allow_repeat:
        cmd.append("--allow-repeat")

    with st.spinner("Gerando times..."):
        result = subprocess.run(cmd)

    if result.returncode != 0:
        st.error("Erro ao gerar os times. Veja os logs no terminal.")
    else:
        st.success("Times gerados com sucesso!")

# ======================
# MAIN
# ======================
st.title("⚽ Cartola FC — Pro Lab 2026")
st.caption("Geração inteligente: histórico (2023–2025) + mercado 2026 | com Capitão + Técnico + Banco + Luxo")

if not DATA_TEAMS.exists():
    st.info("Nenhum time gerado ainda.")
    st.stop()

df = pd.read_csv(DATA_TEAMS)

# ======================
# KPIs
# ======================
st.subheader("📊 Resumo")

teams_n = df["team_id"].nunique()
score_mean = df.groupby("team_id")["score_total"].first().mean()
preco_mean = df.groupby("team_id")["preco_total"].first().mean()
unique_players = df["atleta_id"].nunique()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Times gerados", teams_n)
c2.metric("Score médio (c/ capitão)", f"{score_mean:.2f}")
c3.metric("Preço médio (total)", f"{preco_mean:.2f}")
c4.metric("Atletas únicos", unique_players)

# ======================
# TIMES
# ======================
st.subheader("🧩 Times gerados")

for team_id, team in df.groupby("team_id"):
    st.markdown(f"## 🟦 Time {team_id}")

    preco_main = float(team["preco_main"].iloc[0])
    preco_bench = float(team["preco_bench"].iloc[0])
    preco_total = float(team["preco_total"].iloc[0])
    score_total = float(team["score_total"].iloc[0])

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Score total (c/ capitão)", f"{score_total:.2f}")
    k2.metric("Preço principal", f"{preco_main:.2f}")
    k3.metric("Preço banco", f"{preco_bench:.2f}")
    k4.metric("Preço total", f"{preco_total:.2f}")

    # Captain + Coach
    captain = team[(team["role"] == "starter") & (team["is_captain"] == True)]
    coach = team[team["role"] == "coach"]

    cap_name = captain["apelido"].iloc[0] if len(captain) else "—"
    cap_pos = captain["posicao"].iloc[0] if len(captain) else "—"
    tec_name = coach["apelido"].iloc[0] if len(coach) else "—"

    st.markdown(f"**👑 Capitão:** {cap_name} ({cap_pos})  \n**🧑‍🏫 Técnico:** {tec_name}")

    # Starters table
    starters = team[team["role"] == "starter"].copy()
    starters["C"] = starters["is_captain"].apply(lambda x: "👑" if bool(x) else "")
    starters = starters[["C","apelido","posicao","preco","score"]].sort_values("score", ascending=False)

    st.markdown("### Titulares")
    st.dataframe(starters, use_container_width=True)

    # Bench table
    bench = team[team["role"] == "bench"].copy()
    if len(bench):
        bench["Luxo"] = bench["is_luxury"].apply(lambda x: "💎" if bool(x) else "")
        bench = bench[["Luxo","apelido","posicao","preco","score"]].sort_values(["Luxo","score"], ascending=[False, False])
        st.markdown("### Banco (reservas + reserva de luxo)")
        st.dataframe(bench, use_container_width=True)
    else:
        st.info("Banco não gerado (orçamento/mercado).")

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
