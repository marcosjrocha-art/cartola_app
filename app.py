import math
import random
from typing import Dict, Optional, Set, Tuple

import pandas as pd
import requests
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# Config
# ============================================================
DEFAULT_API_URLS = [
    "https://api.cartola.globo.com/atletas/mercado",
    "https://api.cartolafc.globo.com/atletas/mercado",
]

FORMATION_433 = {"GOL": 1, "LAT": 2, "ZAG": 2, "MEI": 3, "ATA": 3, "TEC": 1}
BUDGET_DEFAULT = 100.0

POSITION_ALIASES = {
    "GOL": {"GOL", "GOLEIRO"},
    "LAT": {"LAT", "LATERAL"},
    "ZAG": {"ZAG", "ZAGUEIRO"},
    "MEI": {"MEI", "MEIA", "MEIO-CAMPO", "MEIO CAMPO"},
    "ATA": {"ATA", "ATACANTE"},
    "TEC": {"TEC", "TÉC", "TECNICO", "TÉCNICO", "TÉCNICO(A)"},
}

EXPECTED_COLS = [
    "atleta_id", "apelido", "nome", "slug", "foto",
    "clube_id", "clube",
    "posicao_id", "posicao",
    "status_id", "status",
    "preco", "media", "pontos_ultima", "jogos",
]

TARGET_ALIASES = {"target", "alvo", "pontos_alvo", "pontos_target", "pontos_prox", "pontos_proxima", "pontos_rodada"}


# ============================================================
# UI / CSS (layout profissional)
# ============================================================
st.set_page_config(
    page_title="Cartola FC • Gerador Pro (Heurística / ML)",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
/* Layout geral */
.block-container { padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1200px; }
h1, h2, h3 { letter-spacing: -0.02em; }
small { opacity: 0.85; }

/* Sidebar */
section[data-testid="stSidebar"] { border-right: 1px solid rgba(250,250,250,0.06); }
section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }

/* Cards */
.pro-card {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 18px;
  padding: 14px 14px;
}
.pro-card h4 { margin: 0 0 6px 0; font-size: 0.95rem; opacity: 0.9; }
.pro-card .big { font-size: 1.35rem; font-weight: 700; }
.pro-badge {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 999px;
  background: rgba(255,255,255,0.10);
  border: 1px solid rgba(255,255,255,0.14);
  font-size: 0.78rem;
  opacity: 0.92;
}

/* Tabelas */
[data-testid="stDataFrame"] { border-radius: 16px; overflow: hidden; }

/* Botões */
.stButton>button {
  border-radius: 14px;
  padding: 0.6rem 0.9rem;
  font-weight: 650;
}

/* Esconde footer/menu */
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# Helpers
# ============================================================
def safe_float(x, default=0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return default
        return float(x)
    except Exception:
        return default


def norm_upper(s: str) -> str:
    return (s or "").strip().upper()


def detect_position_bucket(pos_name: str) -> Optional[str]:
    up = norm_upper(pos_name)
    for bucket, aliases in POSITION_ALIASES.items():
        if up in aliases:
            return bucket
    if "GOL" in up: return "GOL"
    if "LAT" in up: return "LAT"
    if "ZAG" in up: return "ZAG"
    if "MEI" in up or "MEIA" in up or "MEIO" in up: return "MEI"
    if "ATA" in up: return "ATA"
    if "TEC" in up or "TÉC" in up: return "TEC"
    return None


def request_json(url: str, timeout: int = 15) -> dict:
    headers = {"User-Agent": "Mozilla/5.0 (Streamlit Cartola Pro App)"}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=300)
def fetch_market_data(api_url: str) -> dict:
    return request_json(api_url)


def ensure_expected_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in EXPECTED_COLS:
        if c not in df.columns:
            df[c] = pd.Series(dtype="object")
    return df


def parse_api_payload(payload: dict) -> Tuple[pd.DataFrame, dict]:
    atletas = payload.get("atletas", None)
    clubes = payload.get("clubes", {}) or {}
    posicoes = payload.get("posicoes", {}) or {}
    status = payload.get("status", {}) or {}

    diag = {
        "payload_keys": list(payload.keys()) if isinstance(payload, dict) else [],
        "atletas_type": str(type(atletas)),
        "atletas_count": len(atletas) if isinstance(atletas, list) else 0,
    }

    rows = []
    if isinstance(atletas, list):
        for a in atletas:
            clube_id = a.get("clube_id")
            pos_id = a.get("posicao_id")
            status_id = a.get("status_id")

            clube_nome = (clubes.get(str(clube_id)) or clubes.get(clube_id) or {}).get("nome", "")
            pos_nome = (posicoes.get(str(pos_id)) or posicoes.get(pos_id) or {}).get("nome", "")
            status_nome = (status.get(str(status_id)) or status.get(status_id) or {}).get("nome", "")

            preco_num = a.get("preco_num", a.get("preco", None))
            media_num = a.get("media_num", a.get("media", None))
            pontos_num = a.get("pontos_num", a.get("pontos", None))
            jogos_num = a.get("jogos_num", a.get("jogos", None))

            rows.append(
                {
                    "atleta_id": a.get("atleta_id"),
                    "apelido": a.get("apelido") or "",
                    "nome": a.get("nome") or "",
                    "slug": a.get("slug") or "",
                    "foto": a.get("foto") or "",
                    "clube_id": clube_id,
                    "clube": clube_nome or "",
                    "posicao_id": pos_id,
                    "posicao": pos_nome or "",
                    "status_id": status_id,
                    "status": status_nome or "",
                    "preco": safe_float(preco_num, 0.0),
                    "media": safe_float(media_num, 0.0),
                    "pontos_ultima": safe_float(pontos_num, 0.0),
                    "jogos": int(jogos_num or 0),
                }
            )

    df = pd.DataFrame(rows)
    df = ensure_expected_columns(df)

    for col in ["apelido", "clube", "posicao", "status"]:
        df[col] = df[col].fillna("").astype(str)

    for col in ["preco", "media", "pontos_ultima"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["jogos"] = pd.to_numeric(df["jogos"], errors="coerce").fillna(0).astype(int)
    df["bucket"] = df["posicao"].apply(detect_position_bucket)

    return df, diag


# ============================================================
# Previsão (Heurística)
# ============================================================
def add_prediction_columns_heuristic(df: pd.DataFrame, alpha_media: float, bonus_cb: float, min_price_for_cb: float = 1.0) -> pd.DataFrame:
    df = df.copy()
    df["pred_base"] = alpha_media * df["media"] + (1.0 - alpha_media) * df["pontos_ultima"]
    denom = df["preco"].clip(lower=min_price_for_cb)
    df["cb"] = df["pred_base"] / denom
    df["pred"] = df["pred_base"] + bonus_cb * df["cb"]
    return df


# ============================================================
# ML (treino via CSV)
# ============================================================
def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def find_target_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if c in TARGET_ALIASES:
            return c
    for c in df.columns:
        if "target" in c or "alvo" in c:
            return c
    return None


def prepare_training_df(train_df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    df = normalize_cols(train_df)

    # mapear nomes comuns
    rename_map = {}
    for c in df.columns:
        cu = c.lower()
        if cu in {"preço", "preco_num"}: rename_map[c] = "preco"
        if cu in {"média", "media_num"}: rename_map[c] = "media"
        if cu in {"pontos_ultima", "pontos_num", "pontos"}: rename_map[c] = "pontos_ultima"
        if cu in {"posição", "posicao"}: rename_map[c] = "posicao"
        if cu in {"time", "clube"}: rename_map[c] = "clube"
    if rename_map:
        df = df.rename(columns=rename_map)

    target_col = find_target_column(df)
    if not target_col:
        raise ValueError("CSV de treino sem coluna alvo. Crie 'target' com os pontos reais da rodada seguinte.")

    # bucket
    if "bucket" not in df.columns:
        if "posicao" in df.columns:
            df["bucket"] = df["posicao"].astype(str).apply(detect_position_bucket)
        else:
            raise ValueError("CSV de treino precisa ter 'posicao' ou 'bucket'.")

    required = {"preco", "media", "pontos_ultima", "jogos", "clube", "bucket"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV de treino faltando colunas: {sorted(list(missing))}")

    df["preco"] = pd.to_numeric(df["preco"], errors="coerce").fillna(0.0)
    df["media"] = pd.to_numeric(df["media"], errors="coerce").fillna(0.0)
    df["pontos_ultima"] = pd.to_numeric(df["pontos_ultima"], errors="coerce").fillna(0.0)
    df["jogos"] = pd.to_numeric(df["jogos"], errors="coerce").fillna(0).astype(int)
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    df = df.dropna(subset=[target_col])
    df = df[df["bucket"].notna()]
    return df, target_col


def train_model(train_df: pd.DataFrame, target_col: str, n_estimators: int = 450, random_state: int = 42):
    features_num = ["preco", "media", "pontos_ultima", "jogos"]
    features_cat = ["bucket", "clube"]

    X = train_df[features_num + features_cat].copy()
    y = train_df[target_col].copy()

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_state)

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
    )

    pipe = Pipeline(steps=[("pre", pre), ("model", model)])
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_val)
    mae = float(mean_absolute_error(y_val, preds))
    return pipe, mae


def apply_model_predictions(df_market: pd.DataFrame, model_pipe) -> pd.DataFrame:
    df = df_market.copy()
    df["bucket"] = df["posicao"].apply(detect_position_bucket)

    features_num = ["preco", "media", "pontos_ultima", "jogos"]
    features_cat = ["bucket", "clube"]
    X = df[features_num + features_cat].copy()

    df["pred"] = model_pipe.predict(X)
    df["pred"] = pd.to_numeric(df["pred"], errors="coerce").fillna(0.0)
    df["cb"] = df["pred"] / df["preco"].clip(lower=1.0)
    return df


# ============================================================
# Filtros e escalação
# ============================================================
def filter_players(df: pd.DataFrame, allowed_status: Set[str], selected_clubs: Optional[Set[str]], min_games: int, max_price: float, include_zero_price: bool) -> pd.DataFrame:
    dff = df.copy()
    dff["bucket"] = dff["posicao"].apply(detect_position_bucket)
    dff = dff[dff["bucket"].notna()]

    if allowed_status:
        allowed = {s.upper() for s in allowed_status}
        dff = dff[dff["status"].str.upper().isin(allowed)]

    if selected_clubs and len(selected_clubs) > 0:
        dff = dff[dff["clube"].isin(selected_clubs)]

    dff = dff[dff["jogos"] >= int(min_games)]
    dff = dff[dff["preco"] <= float(max_price)]

    if not include_zero_price:
        dff = dff[dff["preco"] > 0]

    dff = dff.dropna(subset=["atleta_id"]).drop_duplicates(subset=["atleta_id"])
    return dff


def min_cost_for_bucket(dff: pd.DataFrame, bucket: str, k: int) -> float:
    cand = dff[dff["bucket"] == bucket].sort_values("preco", ascending=True).head(k)
    if len(cand) < k:
        return float("inf")
    return float(cand["preco"].sum())


def min_remaining_cost(dff: pd.DataFrame, formation: Dict[str, int], already: Dict[str, int]) -> float:
    total = 0.0
    for b, k in formation.items():
        need = k - already.get(b, 0)
        if need > 0:
            total += min_cost_for_bucket(dff, b, need)
    return total


def build_team(dff: pd.DataFrame, formation: Dict[str, int], budget: float, strategy: str, avoid_ids: Set[int], max_overlap_with_avoids: Optional[int], seed: int) -> Tuple[pd.DataFrame, Dict[str, float]]:
    random.seed(seed)

    candidates = dff.copy()
    if avoid_ids:
        candidates["is_avoid"] = candidates["atleta_id"].astype(int).isin({int(x) for x in avoid_ids})
    else:
        candidates["is_avoid"] = False

    picked_rows = []
    picked_ids: Set[int] = set()
    picked_count: Dict[str, int] = {b: 0 for b in formation.keys()}
    overlap_count = 0
    remaining_budget = float(budget)

    bucket_order = sorted(list(formation.keys()), key=lambda b: candidates[candidates["bucket"] == b].shape[0])

    for bucket in bucket_order:
        need = formation[bucket]
        for _ in range(need):
            pool = candidates[(candidates["bucket"] == bucket) & (~candidates["atleta_id"].astype(int).isin(picked_ids))].copy()
            if pool.empty:
                return pd.DataFrame(), {"ok": 0.0, "reason": f"Sem opções para {bucket}"}

            if strategy == "conservador":
                pool = pool.sort_values(["pred", "cb", "preco"], ascending=[False, False, True])
            elif strategy == "custo":
                pool = pool.sort_values(["cb", "pred", "preco"], ascending=[False, False, True])
            else:
                noise = [random.uniform(-0.28, 0.28) for _ in range(len(pool))]
                pool["score"] = pool["pred"] * (1.0 + pd.Series(noise).values) * pool["is_avoid"].apply(lambda x: 0.90 if x else 1.0)
                pool = pool.sort_values(["score", "preco"], ascending=[False, True])

            chosen = None
            for _, row in pool.iterrows():
                atleta_id = int(row["atleta_id"])
                price = float(row["preco"])

                if max_overlap_with_avoids is not None and row.get("is_avoid", False):
                    if overlap_count + 1 > max_overlap_with_avoids:
                        continue
                if price > remaining_budget:
                    continue

                tmp_already = picked_count.copy()
                tmp_already[bucket] += 1
                tmp_dff = candidates[~candidates["atleta_id"].astype(int).isin(picked_ids | {atleta_id})]
                min_cost = min_remaining_cost(tmp_dff, formation, tmp_already)
                if remaining_budget - price < min_cost:
                    continue

                chosen = row
                break

            if chosen is None:
                return pd.DataFrame(), {"ok": 0.0, "reason": f"Orçamento inviável ao escolher {bucket}"}

            picked_rows.append(chosen)
            picked_ids.add(int(chosen["atleta_id"]))
            picked_count[bucket] += 1
            remaining_budget -= float(chosen["preco"])
            if bool(chosen.get("is_avoid", False)):
                overlap_count += 1

    team = pd.DataFrame(picked_rows).copy()
    team["pred"] = pd.to_numeric(team["pred"], errors="coerce").fillna(0.0)
    totals = {
        "ok": 1.0,
        "preco_total": float(team["preco"].sum()) if not team.empty else 0.0,
        "pred_total": float(team["pred"].sum()) if not team.empty else 0.0,
        "overlap": float(overlap_count),
        "restante": float(remaining_budget),
    }
    return team, totals


def format_team_table(team: pd.DataFrame) -> pd.DataFrame:
    if team.empty:
        return team
    cols = ["bucket", "apelido", "clube", "status", "preco", "media", "pontos_ultima", "pred", "jogos"]
    out = team[cols].copy()
    order = {"GOL": 0, "LAT": 1, "ZAG": 2, "MEI": 3, "ATA": 4, "TEC": 5}
    out["ord"] = out["bucket"].map(order).fillna(99).astype(int)
    out = out.sort_values(["ord", "pred"], ascending=[True, False]).drop(columns=["ord"])
    for c in ["preco", "media", "pontos_ultima", "pred"]:
        out[c] = out[c].map(lambda x: round(float(x), 2))
    return out


def card_metric(title: str, value: str, subtitle: str = ""):
    sub = f"<div style='opacity:0.72;font-size:0.85rem;margin-top:2px'>{subtitle}</div>" if subtitle else ""
    st.markdown(
        f"""
<div class="pro-card">
  <h4>{title}</h4>
  <div class="big">{value}</div>
  {sub}
</div>
""",
        unsafe_allow_html=True,
    )


def captain_badge(cap_name: str, cap_team: str, cap_pred: float):
    st.markdown(
        f"""<span class="pro-badge">⭐ Capitão sugerido: <b>{cap_name}</b> ({cap_team}) • previsão {cap_pred:.2f}</span>""",
        unsafe_allow_html=True,
    )


# ============================================================
# Sidebar (configuração profissional)
# ============================================================
with st.sidebar:
    st.markdown("### ⚙️ Configuração")
    api_url = st.selectbox("Endpoint (API)", DEFAULT_API_URLS, index=0)

    st.markdown("### 🧠 Motor de previsão")
    mode = st.radio("Escolha", ["Heurística (simples)", "ML (treinar com CSV)"], index=0)

    if mode == "Heurística (simples)":
        alpha = st.slider("Peso da média (alpha)", 0.0, 1.0, 0.75, 0.05)
        bonus_cb = st.slider("Bônus custo-benefício", 0.0, 2.0, 0.35, 0.05)
        train_file = None
        n_estimators = None
    else:
        st.caption("Envie um CSV histórico com coluna **target** (pontos reais).")
        train_file = st.file_uploader("CSV de treino", type=["csv"])
        n_estimators = st.slider("Complexidade do modelo (árvores)", 100, 1000, 450, 50)
        alpha = 0.75
        bonus_cb = 0.35

    st.markdown("### 🧩 Restrições")
    budget = st.number_input("Orçamento (C$)", min_value=50.0, max_value=200.0, value=BUDGET_DEFAULT, step=0.5)
    max_price = st.number_input("Preço máximo por atleta", min_value=1.0, max_value=120.0, value=60.0, step=1.0)
    min_games = st.number_input("Mínimo de jogos", min_value=0, max_value=38, value=1, step=1)

    # Dica: o maior motivo de “Após filtros = 0” é status não bater
    allowed_status_str = st.text_input("Status permitidos (vírgula)", value="Provável")
    allowed_status = {norm_upper(x) for x in allowed_status_str.split(",") if x.strip()}
    include_zero_price = st.toggle("Permitir preço 0", value=False)

    st.markdown("### 🎛️ Diversidade")
    max_overlap = st.slider("Máx. repetidos (Time 1 → Time 3)", 0, 11, 6)

    st.markdown("---")
    regenerate = st.button("🚀 Gerar 3 times", type="primary")
    st.caption("Dica: se **Após filtros = 0**, abra o Diagnóstico e veja os status reais retornados.")


# ============================================================
# Header (hero)
# ============================================================
st.markdown("## ⚽ Cartola FC — Gerador Pro de 3 times")
st.caption("Layout profissional • Heurística ou ML • 4-3-3 • 100 C$ • download dos times • diagnóstico inteligente")


# ============================================================
# Load API
# ============================================================
@st.cache_data(ttl=300)
def load_market(api_url: str) -> Tuple[pd.DataFrame, dict]:
    payload = fetch_market_data(api_url)
    df, diag = parse_api_payload(payload)
    return df, diag


try:
    df_raw, diag = load_market(api_url)
except Exception as e:
    st.error(f"Falha ao acessar a API: {e}")
    st.stop()

if diag.get("atletas_count", 0) == 0:
    st.warning("A API retornou lista vazia (mercado fechado ou instabilidade). Troque o endpoint e tente novamente.")
    st.stop()

# clube list
club_list = sorted([c for c in df_raw["clube"].dropna().unique().tolist() if str(c).strip()])

with st.sidebar:
    selected_clubs = st.multiselect("Filtrar clubes (opcional)", club_list, default=[])
selected_clubs_set = set(selected_clubs) if selected_clubs else set()

# ============================================================
# Prediction
# ============================================================
mae = None

if mode == "ML (treinar com CSV)":
    if "ml_model" not in st.session_state:
        st.session_state["ml_model"] = None
        st.session_state["ml_mae"] = None

    if train_file is None and st.session_state["ml_model"] is None:
        st.info("Envie um CSV de treino para ativar o modo ML.")
        st.stop()

    if train_file is not None:
        try:
            train_df_raw = pd.read_csv(train_file)
            train_df, target_col = prepare_training_df(train_df_raw)
            model_pipe, mae = train_model(train_df, target_col, n_estimators=int(n_estimators))
            st.session_state["ml_model"] = model_pipe
            st.session_state["ml_mae"] = mae
        except Exception as e:
            st.error(f"Erro ao treinar modelo ML: {e}")
            st.stop()

    model_pipe = st.session_state["ml_model"]
    mae = st.session_state["ml_mae"]

    if model_pipe is None:
        st.error("Modelo ML não disponível.")
        st.stop()

    df_pred = apply_model_predictions(df_raw, model_pipe)
else:
    df_pred = add_prediction_columns_heuristic(df_raw, alpha_media=float(alpha), bonus_cb=float(bonus_cb))

# ============================================================
# Filtering + Smart fallback when filters become 0
# ============================================================
df_ok = filter_players(
    df_pred,
    allowed_status=allowed_status,
    selected_clubs=selected_clubs_set,
    min_games=int(min_games),
    max_price=float(max_price),
    include_zero_price=bool(include_zero_price),
)

# Fallback automático: se após filtros zerou, tenta relaxar status (mantém demais filtros)
fallback_used = False
if df_ok.empty and len(df_pred) > 0:
    df_ok2 = filter_players(
        df_pred,
        allowed_status=set(),  # sem filtro por status
        selected_clubs=selected_clubs_set,
        min_games=int(min_games),
        max_price=float(max_price),
        include_zero_price=bool(include_zero_price),
    )
    if not df_ok2.empty:
        fallback_used = True
        df_ok = df_ok2

# ============================================================
# Top KPI Bar (cards)
# ============================================================
k1, k2, k3, k4, k5 = st.columns([1, 1, 1, 1, 1])
with k1:
    card_metric("Atletas recebidos", f"{len(df_raw)}", "mercado")
with k2:
    card_metric("Após filtros", f"{len(df_ok)}", "selecionáveis")
with k3:
    card_metric("Orçamento", f"{budget:.2f}", "C$")
with k4:
    card_metric("Formação", "4-3-3", "fixa")
with k5:
    if mode == "ML (treinar com CSV)":
        card_metric("Modo", "ML", f"MAE: {mae:.2f}" if mae is not None else "—")
    else:
        card_metric("Modo", "Heurística", f"alpha {alpha:.2f} • cb {bonus_cb:.2f}")

if fallback_used:
    st.info(
        "⚠️ Seus filtros de **status** não bateram com os status retornados pela API. "
        "Eu removi o filtro de status automaticamente para você não ficar com 0 atletas. "
        "Abra o **Diagnóstico** e veja os status reais."
    )

# ============================================================
# Diagnostics + Explore
# ============================================================
with st.expander("🧪 Diagnóstico da API (recomendado)", expanded=False):
    st.write("Endpoint:", api_url)
    st.write("Chaves no JSON:", diag.get("payload_keys"))
    st.write("Tipo de atletas:", diag.get("atletas_type"))
    st.write("Quantidade de atletas:", diag.get("atletas_count"))

    statuses = sorted([s for s in df_raw["status"].dropna().unique().tolist() if str(s).strip()])
    st.write("Status retornados pela API:", statuses)

    # mostra amostra de colunas
    st.write("Colunas do DataFrame:", list(df_raw.columns))

with st.expander("🔍 Explorar atletas (Top 200 por previsão)", expanded=False):
    show = df_ok.sort_values("pred", ascending=False).head(200).copy()
    show["preco"] = show["preco"].map(lambda x: round(float(x), 2))
    show["pred"] = show["pred"].map(lambda x: round(float(x), 2))
    st.dataframe(
        show[["apelido", "clube", "posicao", "status", "preco", "media", "pontos_ultima", "pred", "cb", "jogos"]],
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# Generate teams
# ============================================================
if not regenerate and "teams_done" not in st.session_state:
    st.markdown(
        "<div class='pro-card'><b>✅ Pronto.</b> Ajuste os parâmetros na lateral e clique em <b>🚀 Gerar 3 times</b>.</div>",
        unsafe_allow_html=True,
    )
    st.stop()

if df_ok.empty:
    st.error(
        "Depois dos filtros, não sobrou nenhum atleta. Tente:\n"
        "- Aumentar o preço máximo\n"
        "- Diminuir mínimo de jogos\n"
        "- Limpar filtro de clubes\n"
        "- Ajustar status (ou deixar em branco)"
    )
    st.stop()

seed = random.randint(1, 10_000_000)
t1, m1 = build_team(df_ok, FORMATION_433, float(budget), "conservador", avoid_ids=set(), max_overlap_with_avoids=None, seed=seed)
avoid = set(t1["atleta_id"].astype(int).tolist()) if not t1.empty else set()
t2, m2 = build_team(df_ok, FORMATION_433, float(budget), "custo", avoid_ids=set(), max_overlap_with_avoids=None, seed=seed + 1337)
t3, m3 = build_team(df_ok, FORMATION_433, float(budget), "dif", avoid_ids=avoid, max_overlap_with_avoids=int(max_overlap), seed=seed + 2025)

st.session_state["teams_done"] = True


def render_team_block(title: str, subtitle: str, team: pd.DataFrame, meta: Dict[str, float]):
    st.markdown(f"### {title}")
    st.caption(subtitle)

    if team.empty or meta.get("ok", 0.0) < 1.0:
        st.error(f"Não foi possível montar o time. Motivo: {meta.get('reason', 'desconhecido')}")
        return

    # Capitão: melhor pred (exceto TEC)
    players = team[team["bucket"] != "TEC"].copy()
    if not players.empty:
        cap = players.sort_values("pred", ascending=False).iloc[0]
        captain_badge(str(cap["apelido"]), str(cap["clube"]), float(cap["pred"]))

    a, b, c, d = st.columns([1, 1, 1, 1])
    with a: card_metric("Pontuação prevista", f"{meta['pred_total']:.2f}")
    with b: card_metric("Custo", f"{meta['preco_total']:.2f}", "C$")
    with c: card_metric("Sobra", f"{meta['restante']:.2f}", "C$")
    with d: card_metric("Repetidos", f"{int(meta.get('overlap', 0))}", "vs Time 1")

    # Tabela bonita
    st.dataframe(format_team_table(team), use_container_width=True, hide_index=True)

    # Download CSV do time
    csv = format_team_table(team).to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Baixar este time (CSV)",
        data=csv,
        file_name=f"{title.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )


# ============================================================
# Times (tabs)
# ============================================================
tab1, tab2, tab3 = st.tabs(["🟢 Time 1 (Conservador)", "🔵 Time 2 (Custo-benefício)", "🟣 Time 3 (Diferentão)"])
with tab1:
    render_team_block("Time 1 — Conservador", "Foco em maior pontuação prevista (mais “seguro”).", t1, m1)
with tab2:
    render_team_block("Time 2 — Custo-benefício", "Foco em pontuar bem gastando menos (eficiência).", t2, m2)
with tab3:
    render_team_block("Time 3 — Diferentão", "Mais variação (limita repetidos do Time 1).", t3, m3)

st.markdown("---")
st.markdown(
    """
<div class="pro-card">
  <b>Como a previsão funciona</b><br/>
  <span style="opacity:0.85">
  Heurística: combina média + última rodada e um bônus de custo-benefício. <br/>
  ML: treina um RandomForest no seu CSV histórico (com coluna <b>target</b>) e prevê a pontuação.
  </span>
</div>
""",
    unsafe_allow_html=True,
)
