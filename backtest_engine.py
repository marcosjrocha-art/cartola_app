# scripts/backtest_engine.py
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

# =========================
# CONFIGURAÇÕES
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
CARTOLA_DATA = BASE_DIR / "caRtola" / "data" / "01_raw"
OUTPUT_FILE = BASE_DIR / "data" / "backtest_results.csv"
MODEL_FILE = BASE_DIR / "models" / "model.joblib"

ANOS_BACKTEST = [2023, 2024, 2025]
ORCAMENTO = 100.0
TIME_SIZE = 11

# =========================
# FUNÇÕES AUXILIARES
# =========================

def load_season_data(ano):
    ano_dir = CARTOLA_DATA / str(ano)
    dfs = []
    for csv in sorted(ano_dir.glob("rodada-*.csv")):
        df = pd.read_csv(csv)
        df["rodada"] = int(csv.stem.split("-")[1])
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def prepare_features(df):
    cols = [
        "atletas.preco_num",
        "atletas.variacao_num",
        "atletas.media_num",
        "atletas.jogos_num"
    ]
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
    return df, cols


def train_model(train_df):
    train_df, features = prepare_features(train_df)
    train_df = train_df.dropna(subset=["atletas.pontos_num"])

    X = train_df[features]
    y = train_df["atletas.pontos_num"]

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    return model, features


def generate_team(df, features, model):
    df = df.copy()
    df["pred"] = model.predict(df[features])

    df = df.sort_values("pred", ascending=False)

    team = []
    budget = ORCAMENTO

    for _, row in df.iterrows():
        price = row["atletas.preco_num"]
        if price <= budget:
            team.append(row)
            budget -= price
        if len(team) == TIME_SIZE:
            break

    return pd.DataFrame(team)


# =========================
# BACKTEST PRINCIPAL
# =========================

def run_backtest():
    print("🚀 Iniciando Backtest Avançado (2023–2025)")
    results = []

    for ano in ANOS_BACKTEST:
        print(f"\n📅 Temporada {ano}")
        season_df = load_season_data(ano)

        max_rodada = season_df["rodada"].max()

        for rodada in range(1, max_rodada + 1):
            print(f"  ▶ Rodada {rodada}")

            train_df = season_df[season_df["rodada"] < rodada]
            test_df = season_df[season_df["rodada"] == rodada]

            if len(train_df) < 500:
                continue

            model, features = train_model(train_df)

            team = generate_team(test_df, features, model)

            pontos = team["atletas.pontos_num"].sum()

            results.append({
                "ano": ano,
                "rodada": rodada,
                "pontos_time": pontos,
                "media_time": pontos / TIME_SIZE
            })

    result_df = pd.DataFrame(results)
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    result_df.to_csv(OUTPUT_FILE, index=False)

    print("\n✅ Backtest finalizado com sucesso!")
    print(f"📄 Resultado salvo em: {OUTPUT_FILE}")


# =========================
# EXECUÇÃO
# =========================

if __name__ == "__main__":
    run_backtest()
