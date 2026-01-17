import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
import joblib

RAW_BASE = Path("cartola/data/01_raw")
MODEL_PATH = Path("models/model.joblib")

TARGET_CANDIDATES = [
    "atletas.pontos_num",
    "pontos_num",
    "pontos"
]

MAX_ROWS = 300_000  # ⚡ performance garantida

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    return df

def detect_target(df: pd.DataFrame) -> str | None:
    for col in TARGET_CANDIDATES:
        col = col.lower()
        if col in df.columns:
            return col
    return None

def load_all_rounds(years=(2023, 2024, 2025)):
    dfs = []

    for year in years:
        year_path = RAW_BASE / str(year)
        if not year_path.exists():
            continue

        for csv in sorted(year_path.glob("rodada-*.csv")):
            try:
                df = pd.read_csv(csv)
                df = normalize_columns(df)

                target = detect_target(df)
                if not target:
                    continue  # ignora CSV sem pontos

                df["ano"] = year
                df["rodada"] = int(csv.stem.split("-")[1])
                df["__target__"] = df[target]

                dfs.append(df)

            except Exception as e:
                print(f"⚠️ Ignorado {csv.name}: {e}")

    if not dfs:
        raise RuntimeError("Nenhum CSV válido com coluna alvo encontrado.")

    df_all = pd.concat(dfs, ignore_index=True)

    if len(df_all) > MAX_ROWS:
        df_all = df_all.sample(MAX_ROWS, random_state=42)

    return df_all

def main():
    print("📥 Carregando dados históricos (caRtola)...")
    df = load_all_rounds()

    print(f"📊 Total de linhas após limpeza: {len(df)}")

    df = df.dropna(subset=["__target__"])

    features = df.select_dtypes(include=["int64", "float64"])
    y = features["__target__"]
    X = features.drop(columns=["__target__"])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", HistGradientBoostingRegressor(
                max_depth=6,
                max_iter=150,
                learning_rate=0.05,
                random_state=42
            ))
        ]
    )

    print("🧠 Treinando modelo ML...")
    model.fit(X_train, y_train)

    score = model.score(X_val, y_val)
    print(f"✅ R² validação: {score:.4f}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("💾 Modelo salvo em models/model.joblib")

if __name__ == "__main__":
    main()
