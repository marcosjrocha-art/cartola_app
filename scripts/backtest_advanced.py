import os
import glob
import math
import random
from typing import Dict, List, Tuple, Optional

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# =========================
# Config
# =========================
CARTOLA_REPO_DIR = "caRtola"
RAW_DIR = os.path.join(CARTOLA_REPO_DIR, "data", "01_raw")
OUT_DIR = os.path.join("data", "backtest_results")

YEARS = [2023, 2024, 2025]
BUDGET = 100.0

FORMATION = {"GOL": 1, "LAT": 2, "ZAG": 2, "MEI": 3, "ATA": 3, "TEC": 1}

POS_MAP = {
    "GOL": "GOL", "GOLEIRO": "GOL",
    "LAT": "LAT", "LATERAL": "LAT",
    "ZAG": "ZAG", "ZAGUEIRO": "ZAG",
    "MEI": "MEI", "MEIA": "MEI", "MEIO": "MEI",
    "ATA": "ATA", "ATACANTE": "ATA",
    "TEC": "TEC", "TECNICO": "TEC", "TÉCNICO": "TEC",
}


# =========================
# Helpers
# =========================
def bucket_from_pos(pos: str) -> Optional[str]:
    s = str(pos).upper().strip()
    if s in POS_MAP:
        return POS_MAP[s]
    # fallback por substring
    if "GOL" in s: return "GOL"
    if "LAT" in s: return "LAT"
    if "ZAG" in s: return "ZAG"
    if "MEI" in s or "MEIA" in s or "MEIO" in s: return "MEI"
    if "ATA" in s: return "ATA"
    if "TEC" in s or "TÉC" in s: return "TEC"
    return None


def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def safe_num(series: pd.Series, default=0.0) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce").fillna(default)
    return out


def load_year_rounds(year: int) -> List[Tuple[int, str]]:
    pattern = os.path.join(RAW_DIR, str(year), "rodada-*.csv")
    files = sorted(glob.glob(pattern))
    rounds = []
    for f in files:
        try:
            r = int(os.path.basename(f).replace("rodada-", "").replace(".csv", ""))
            rounds.append((r, f))
        except:
            continue
    rounds.sort(key=lambda x: x[0])
    return rounds


def load_round_df(year: int, rodada: int, filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df.columns = [str(c).strip().lower() for c in df.columns]

    col_id = pick_col(df, ["atleta_id", "atletaid", "id_atleta"])
    col_clube = pick_col(df, ["clube", "clube_nome", "time"])
    col_pos = pick_col(df, ["posicao", "posição"])
    col_preco = pick_col(df, ["preco_num", "preco", "preço"])
    col_media = pick_col(df, ["media_num", "media", "média"])
    col_pontos = pick_col(df, ["pontos_num", "pontos", "pontuacao", "pontuação"])
    col_jogos = pick_col(df, ["jogos_num", "jogos"])

    required = [col_id, col_clube, col_pos, col_preco, col_media, col_pontos]
    if any(c is None for c in required):
        raise RuntimeError(
            f"[{year} R{rodada}] CSV não tem colunas essenciais. "
            f"Tenho: {list(df.columns)}"
        )

    out = pd.DataFrame({
        "ano": year,
        "rodada": rodada,
        "atleta_id": df[col_id],
        "clube": df[col_clube].astype(str),
        "posicao": df[col_pos].astype(str),
        "preco": safe_num(df[col_preco], 0.0),
        "media": safe_num(df[col_media], 0.0),
        "target": safe_num(df[col_pontos], 0.0),
        "jogos": safe_num(df[col_jogos], 0).astype(int) if col_jogos else 0,
    })

    out["bucket"] = out["posicao"].apply(bucket_from_pos)
    out = out.dropna(subset=["bucket", "atleta_id"]).copy()

    # garantir atleta_id inteiro quando possível
    out["atleta_id"] = pd.to_numeric(out["atleta_id"], errors="coerce")
    out = out.dropna(subset=["atleta_id"])
    out["atleta_id"] = out["atleta_id"].astype(int)

    return out


def compute_lag_features(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features SEM vazamento:
    - pontos_ultima = target da rodada anterior do mesmo atleta
    - rolling_3_prev = média das 3 rodadas anteriores (antes da rodada atual)
    - tendencia_prev = (pontos_ultima - pontos_penultima)
    """
    d = df_all.sort_values(["atleta_id", "ano", "rodada"]).copy()

    d["pontos_ultima"] = d.groupby("atleta_id")["target"].shift(1).fillna(0.0)
    pontos_penultima = d.groupby("atleta_id")["target"].shift(2).fillna(0.0)
    d["tendencia_prev"] = d["pontos_ultima"] - pontos_penultima

    # rolling das 3 anteriores: shift(1) para não incluir a rodada atual
    d["rolling_3_prev"] = (
        d.groupby("atleta_id")["target"]
        .apply(lambda s: s.shift(1).rolling(3).mean())
        .reset_index(level=0, drop=True)
        .fillna(0.0)
    )

    return d


def heuristic_pred(df: pd.DataFrame, alpha: float = 0.75, bonus_cb: float = 0.35) -> pd.Series:
    pred_base = alpha * df["media"] + (1.0 - alpha) * df["pontos_ultima"]
    cb = pred_base / df["preco"].clip(lower=1.0)
    return pred_base + bonus_cb * cb


def train_ml(train_df: pd.DataFrame, n_estimators: int = 450, random_state: int = 42) -> Pipeline:
    features_num = ["preco", "media", "pontos_ultima", "rolling_3_prev", "tendencia_prev", "jogos"]
    features_cat = ["bucket", "clube"]

    X = train_df[features_num + features_cat].copy()
    y = train_df["target"].copy()

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), features_cat),
            ("num", "passthrough", features_num),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        min_samples_leaf=2,
        max_features="sqrt",
    )

    pipe = Pipeline(steps=[("pre", pre), ("model", model)])
    pipe.fit(X, y)
    return pipe


def build_team(df_round: pd.DataFrame, budget: float, strategy: str, avoid_ids: set, max_overlap: Optional[int], seed: int) -> pd.DataFrame:
    """
    strategy:
      - "cons": maior pred
      - "cb": maior pred/preco
      - "dif": aleatório controlado, penaliza repetidos de avoid_ids
    """
    random.seed(seed)
    df = df_round.copy()

    df["is_avoid"] = df["atleta_id"].isin(avoid_ids) if avoid_ids else False
    df["cb"] = df["pred"] / df["preco"].clip(lower=1.0)

    team_rows = []
    spent = 0.0
    overlap = 0

    # Monta por bucket, priorizando buckets com menos opções (evita travar)
    bucket_order = sorted(list(FORMATION.keys()), key=lambda b: (df[df["bucket"] == b].shape[0], b))

    for b in bucket_order:
        need = FORMATION[b]
        pool = df[df["bucket"] == b].copy()
        if pool.empty:
            return pd.DataFrame()

        if strategy == "cons":
            pool = pool.sort_values(["pred", "cb", "preco"], ascending=[False, False, True])
        elif strategy == "cb":
            pool = pool.sort_values(["cb", "pred", "preco"], ascending=[False, False, True])
        else:
            # diferentão: score com ruído e penalidade para repetidos
            noise = [random.uniform(-0.25, 0.25) for _ in range(len(pool))]
            pool["score"] = pool["pred"] * (1.0 + pd.Series(noise).values) * pool["is_avoid"].apply(lambda x: 0.88 if x else 1.0)
            pool = pool.sort_values(["score", "preco"], ascending=[False, True])

        picked_in_bucket = 0
        for _, row in pool.iterrows():
            if picked_in_bucket >= need:
                break

            if row["atleta_id"] in [r["atleta_id"] for r in team_rows]:
                continue

            price = float(row["preco"])
            if spent + price > budget:
                continue

            # regra de overlap (apenas para time 3)
            if max_overlap is not None and bool(row["is_avoid"]):
                if overlap + 1 > max_overlap:
                    continue

            team_rows.append(row.to_dict())
            spent += price
            picked_in_bucket += 1
            if bool(row["is_avoid"]):
                overlap += 1

        if picked_in_bucket < need:
            # não conseguiu preencher
            return pd.DataFrame()

    team = pd.DataFrame(team_rows)
    return team


def run_backtest_for_year(df_year: pd.DataFrame, year: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    df_year contém todas as rodadas do ano com features lagged já calculadas.
    Backtest: para rodada r, treina com dados < r (no mesmo ano) + todos anos anteriores,
    prediz rodada r, escala 3 times e mede pontos reais.
    """
    rounds = sorted(df_year["rodada"].unique().tolist())
    if len(rounds) < 2:
        raise RuntimeError(f"Ano {year} tem poucas rodadas no dataset.")

    logs = []
    picks = []

    # dataset "global" até o ano atual (com outros anos já incluídos no df_all)
    # aqui vamos treinar incrementalmente dentro do df_all, mas recebemos df_year e faremos o filtro no df_all no main
    # então o main passa df_all e df_year? Para simplificar, vamos usar df_year apenas e treinar dentro do mesmo ano
    # + dados anteriores (df_prev_years) será concatenado no main e passado como base_train.
    return pd.DataFrame(), pd.DataFrame()  # placeholder


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Carregar dados brutos dos anos (da pasta do caRtola)
    all_rows = []
    for y in YEARS:
        yr_rounds = load_year_rounds(y)
        if not yr_rounds:
            raise RuntimeError(
                f"Não encontrei CSVs de {y} em {os.path.join(RAW_DIR, str(y))}. "
                f"Verifique se você clonou o caRtola em ./caRtola"
            )
        for rodada, fp in yr_rounds:
            all_rows.append(load_round_df(y, rodada, fp))

    df_all = pd.concat(all_rows, ignore_index=True)

    # 2) Features SEM vazamento
    df_all = compute_lag_features(df_all)

    # 3) Vamos backtestar ano a ano, mas treinando com dados anteriores (anos < ano atual + rodadas anteriores no mesmo ano)
    results_all = []
    picks_all = []

    for year in YEARS:
        df_year = df_all[df_all["ano"] == year].copy()
        rounds = sorted(df_year["rodada"].unique().tolist())

        # Começa na rodada 2 para ter pelo menos lag básico
        for r in rounds:
            if r < 2:
                continue

            train_df = df_all[(df_all["ano"] < year) | ((df_all["ano"] == year) & (df_all["rodada"] < r))].copy()
            test_df = df_all[(df_all["ano"] == year) & (df_all["rodada"] == r)].copy()

            # Segurança: se treino pequeno, pula
            if len(train_df) < 1000 or len(test_df) < 50:
                continue

            # Previsões
            test_df["pred_heur"] = heuristic_pred(test_df, alpha=0.75, bonus_cb=0.35)

            ml_model = train_ml(train_df, n_estimators=450, random_state=42)
            features_num = ["preco", "media", "pontos_ultima", "rolling_3_prev", "tendencia_prev", "jogos"]
            features_cat = ["bucket", "clube"]
            X_test = test_df[features_num + features_cat].copy()
            test_df["pred_ml"] = ml_model.predict(X_test)

            # Híbrido: mais estável
            test_df["pred_hyb"] = 0.60 * test_df["pred_ml"] + 0.40 * test_df["pred_heur"]

            # Para escalar, escolhemos qual pred usar em cada estratégia
            engines = [
                ("HEUR", "pred_heur"),
                ("ML", "pred_ml"),
                ("HYB", "pred_hyb"),
            ]

            for engine_name, pred_col in engines:
                round_pool = test_df.copy()
                round_pool["pred"] = pd.to_numeric(round_pool[pred_col], errors="coerce").fillna(0.0)

                # Time 1
                t1 = build_team(round_pool, BUDGET, strategy="cons", avoid_ids=set(), max_overlap=None, seed=1000 + r)
                avoid = set(t1["atleta_id"].astype(int).tolist()) if not t1.empty else set()

                # Time 2
                t2 = build_team(round_pool, BUDGET, strategy="cb", avoid_ids=set(), max_overlap=None, seed=2000 + r)

                # Time 3 (diferentão) — limita repetidos do Time 1
                t3 = build_team(round_pool, BUDGET, strategy="dif", avoid_ids=avoid, max_overlap=6, seed=3000 + r)

                for team_idx, team_df in [(1, t1), (2, t2), (3, t3)]:
                    if team_df is None or team_df.empty or len(team_df) < 11:
                        # se falhar, registra como NaN
                        pts = float("nan")
                        cost = float("nan")
                    else:
                        pts = float(team_df["target"].sum())
                        cost = float(team_df["preco"].sum())

                        # log detalhado das escolhas (para auditoria)
                        pick_rows = team_df[["ano","rodada","atleta_id","clube","bucket","preco","target"]].copy()
                        pick_rows["engine"] = engine_name
                        pick_rows["time"] = team_idx
                        picks_all.append(pick_rows)

                    results_all.append({
                        "ano": year,
                        "rodada": r,
                        "engine": engine_name,   # HEUR / ML / HYB
                        "time": team_idx,        # 1/2/3
                        "pontos": pts,
                        "custo": cost,
                        "qtd_atletas_pool": int(len(round_pool)),
                        "train_rows": int(len(train_df)),
                    })

        # Salva por ano
        df_year_res = pd.DataFrame([x for x in results_all if x["ano"] == year])
        df_year_res.to_csv(os.path.join(OUT_DIR, f"backtest_{year}.csv"), index=False)

    # Salva consolidado
    df_res = pd.DataFrame(results_all)
    df_res.to_csv(os.path.join(OUT_DIR, "backtest_total.csv"), index=False)

    # Salva picks detalhados (auditoria)
    if picks_all:
        df_picks = pd.concat(picks_all, ignore_index=True)
        df_picks.to_csv(os.path.join(OUT_DIR, "backtest_picks.csv"), index=False)
    else:
        df_picks = pd.DataFrame()

    # Resumo final
    summary = (
        df_res.dropna(subset=["pontos"])
        .groupby(["ano","engine","time"], as_index=False)
        .agg(
            pontos_total=("pontos", "sum"),
            media_por_rodada=("pontos", "mean"),
            rodadas_validas=("pontos", "count"),
        )
        .sort_values(["ano","engine","time"])
    )
    summary.to_csv(os.path.join(OUT_DIR, "backtest_summary.csv"), index=False)

    print("OK ✅ Backtest concluído.")
    print(f"- {os.path.join(OUT_DIR, 'backtest_total.csv')}")
    print(f"- {os.path.join(OUT_DIR, 'backtest_summary.csv')}")
    print(f"- {os.path.join(OUT_DIR, 'backtest_picks.csv')} (auditoria)")

if __name__ == "__main__":
    main()
