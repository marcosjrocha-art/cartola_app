import math
import random
from typing import Dict, Optional, Set, Tuple

import pandas as pd
import requests
import streamlit as st


# =========================
# Config
# =========================
DEFAULT_API_URLS = [
    "https://api.cartola.globo.com/atletas/mercado",
    "https://api.cartolafc.globo.com/atletas/mercado",
]
BUDGET_DEFAULT = 100.0
FORMATION_433 = {
    "GOL": 1,
    "LAT": 2,
    "ZAG": 2,
    "MEI": 3,
    "ATA": 3,
    "TEC": 1,
}

# Posições comuns
POSITION_ALIASES = {
    "GOL": {"GOL", "GOLEIRO"},
    "LAT": {"LAT", "LATERAL"},
    "ZAG": {"ZAG", "ZAGUEIRO"},
    "MEI": {"MEI", "MEIA", "MEIO-CAMPO", "MEIO CAMPO"},
    "ATA": {"ATA", "ATACANTE"},
    "TEC": {"TEC", "TÉC", "TECNICO", "TÉCNICO", "TÉCNICO(A)"},
}

# Colunas mínimas que o app espera existir sempre
EXPECTED_COLS = [
    "atleta_id",
    "apelido",
    "nome",
    "slug",
    "foto",
    "clube_id",
    "clube",
    "posicao_id",
    "posicao",
    "status_id",
    "status",
    "preco",
    "media",
    "pontos_ultima",
    "jogos",
]


# =========================
# Util
# =========================
def safe_float(x, default=0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return default
        return float(x)
    except Exception:
        return default


def norm_upper(s: str) -> str:
    return (s or "").strip().upper()


def request_json(url: str, timeout: int = 15) -> dict:
    headers = {"User-Agent": "Mozilla/5.0 (Streamlit Cartola Scout App)"}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=300)
def fetch_market_data(api_url: str) -> dict:
    return request_json(api_url)


def ensure_expected_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Garante que o DataFrame tenha todas as colunas esperadas, mesmo vazio."""
    for c in EXPECTED_COLS:
        if c not in df.columns:
            df[c] = pd.Series(dtype="object")
    return df


def parse_api_payload(payload: dict) -> Tuple[pd.DataFrame, dict]:
    """
    Esperado (padrão do /atletas/mercado):
      payload["atletas"] : lista
      payload["clubes"]  : dict
      payload["posicoes"]: dict
      payload["status"]  : dict

    Retorna: (df, diag)
    """
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

            # Algumas variações possíveis no payload:
            preco_num = a.get("preco_num", None)
            if preco_num is None:
                preco_num = a.get("preco", None)

            media_num = a.get("media_num", None)
            if media_num is None:
                media_num = a.get("media", None)

            pontos_num = a.get("pontos_num", None)
            if pontos_num is None:
                pontos_num = a.get("pontos", None)

            jogos_num = a.get("jogos_num", None)
            if jogos_num is None:
                jogos_num = a.get("jogos", None)

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

    # Garante colunas mesmo se vier vazio
    df = ensure_expected_columns(df)

    # Normaliza strings
    for col in ["apelido", "clube", "posicao", "status"]:
        df[col] = df[col].fillna("").astype(str)

    # Normaliza numéricos
    for col in ["preco", "media", "pontos_ultima"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["jogos"] = pd.to_numeric(df["jogos"], errors="coerce").fillna(0).astype(int)

    return df, diag


def detect_position_bucket(pos_name: str) -> Optional[str]:
    up = norm_upper(pos_name)
    for bucket, aliases in POSITION_ALIASES.items():
        if up in aliases:
            return bucket
    # fallback por substring
    if "GOL" in up:
        return "GOL"
    if "LAT" in up:
        return "LAT"
    if "ZAG" in up:
        return "ZAG"
    if "MEI" in up or "MEIA" in up or "MEIO" in up:
        return "MEI"
    if "ATA" in up:
        return "ATA"
    if "TEC" in up or "TÉC" in up:
        return "TEC"
    return None


def add_prediction_columns(
    df: pd.DataFrame,
    alpha_media: float,
    bonus_cb: float,
    min_price_for_cb: float = 1.0,
) -> pd.DataFrame:
    """
    prev = alpha*media + (1-alpha)*pontos_ultima + bonus_cb*(prev/preco)
    """
    df = df.copy()

    # Fallback extra: se por algum motivo não existir "preco", tenta criar
    if "preco" not in df.columns:
        if "preco_num" in df.columns:
            df["preco"] = pd.to_numeric(df["preco_num"], errors="coerce").fillna(0.0)
        else:
            df["preco"] = 0.0

    for col in ["media", "pontos_ultima"]:
        if col not in df.columns:
            df[col] = 0.0

    df["preco"] = pd.to_numeric(df["preco"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df["media"] = pd.to_numeric(df["media"], errors="coerce").fillna(0.0)
    df["pontos_ultima"] = pd.to_numeric(df["pontos_ultima"], errors="coerce").fillna(0.0)

    df["pred_base"] = alpha_media * df["media"] + (1.0 - alpha_media) * df["pontos_ultima"]
    denom = df["preco"].clip(lower=min_price_for_cb)
    df["cb"] = df["pred_base"] / denom
    df["pred"] = df["pred_base"] + bonus_cb * df["cb"]
    return df


def filter_players(
    df: pd.DataFrame,
    allowed_status: Set[str],
    selected_clubs: Optional[Set[str]],
    min_games: int,
    max_price: float,
    include_zero_price: bool = False,
) -> pd.DataFrame:
    dff = df.copy()
    dff["bucket"] = dff["posicao"].apply(detect_position_bucket)
    dff = dff[dff["bucket"].notna()]

    if allowed_status:
        dff = dff[dff["status"].str.upper().isin({s.upper() for s in allowed_status})]

    if selected_clubs and len(selected_clubs) > 0:
        dff = dff[dff["clube"].isin(selected_clubs)]

    dff = dff[dff["jogos"] >= int(min_games)]
    dff = dff[dff["preco"] <= float(max_price)]

    if not include_zero_price:
        dff = dff[dff["preco"] > 0]

    dff = dff.dropna(subset=["atleta_id"]).drop_duplicates(subset=["atleta_id"])
    return dff


# =========================
# Escalação (heurística com viabilidade de orçamento)
# =========================
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


def build_team(
    dff: pd.DataFrame,
    formation: Dict[str, int],
    budget: float,
    strategy: str,
    avoid_ids: Set[int],
    max_overlap_with_avoids: Optional[int],
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
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
            pool = candidates[
                (candidates["bucket"] == bucket)
                & (~candidates["atleta_id"].astype(int).isin(picked_ids))
            ].copy()

            if pool.empty:
                return pd.DataFrame(), {"ok": 0.0, "reason": f"Sem opções para {bucket}"}

            if strategy == "conservador":
                pool = pool.sort_values(["pred", "cb", "preco"], ascending=[False, False, True])
            elif strategy == "custo-beneficio":
                pool = pool.sort_values(["cb", "pred", "preco"], ascending=[False, False, True])
            else:
                noise = [random.uniform(-0.35, 0.35) for _ in range(len(pool))]
                pool["score_noise"] = pool["pred"] * (1.0 + pd.Series(noise).values)
                pool["penalty"] = pool["is_avoid"].apply(lambda x: 0.90 if x else 1.0)
                pool["score"] = pool["score_noise"] * pool["penalty"]
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
    out["preco"] = out["preco"].map(lambda x: round(float(x), 2))
    out["media"] = out["media"].map(lambda x: round(float(x), 2))
    out["pontos_ultima"] = out["pontos_ultima"].map(lambda x: round(float(x), 2))
    out["pred"] = out["pred"].map(lambda x: round(float(x), 2))
    return out


# =========================
# UI
# =========================
st.set_page_config(page_title="Cartola FC - 3 Times Sugeridos", layout="wide")

st.title("⚽ Cartola FC — Gerador de 3 times (previsão simples)")
st.caption(
    "Gera 3 sugestões de escalação (4-3-3) com base em média + última rodada + custo-benefício. "
    "Se a API não retornar atletas, o app mostra diagnóstico para você."
)

with st.sidebar:
    st.header("Fonte dos dados (API)")
    api_url = st.selectbox("Endpoint", DEFAULT_API_URLS, index=0)

    st.divider()
    st.header("Parâmetros de previsão")
    alpha = st.slider("Peso da média (alpha)", 0.0, 1.0, 0.75, 0.05)
    bonus_cb = st.slider("Bônus de custo-benefício", 0.0, 2.0, 0.35, 0.05)

    st.divider()
    st.header("Restrições e filtros")
    budget = st.number_input("Orçamento (cartoletas)", min_value=50.0, max_value=200.0, value=BUDGET_DEFAULT, step=0.5)
    max_price = st.number_input("Preço máximo por atleta", min_value=1.0, max_value=100.0, value=60.0, step=1.0)
    min_games = st.number_input("Mín. jogos (para evitar sem histórico)", min_value=0, max_value=38, value=1, step=1)

    st.caption("Dica: para segurança, deixe apenas 'Provável'.")
    allowed_status_str = st.text_input(
        "Status permitidos (separados por vírgula)",
        value="Provável",
    )
    allowed_status = {norm_upper(x) for x in allowed_status_str.split(",") if x.strip()}

    include_zero_price = st.toggle("Permitir atletas com preço 0 (recomendado: não)", value=False)

    st.divider()
    st.header("Diversidade entre os 3 times")
    max_overlap = st.slider("Máx. repetidos do Time 1 no Time 3", 0, 11, 6)

    regenerate = st.button("🔁 Gerar (ou regerar) os 3 times", type="primary")


@st.cache_data(ttl=300)
def load_data_from_api(api_url: str) -> Tuple[pd.DataFrame, dict]:
    payload = fetch_market_data(api_url)
    df, diag = parse_api_payload(payload)
    return df, diag


# =========================
# Carregamento (API)
# =========================
try:
    df_raw, diag = load_data_from_api(api_url)
except Exception as e:
    st.error(f"Falha ao acessar a API: {e}")
    st.stop()

# Diagnóstico visível (ajuda muito quando a API muda ou volta vazia)
with st.expander("🧪 Diagnóstico da API (clique para abrir)", expanded=False):
    st.write("Endpoint:", api_url)
    st.write("Chaves no JSON:", diag.get("payload_keys"))
    st.write("Tipo de payload['atletas']:", diag.get("atletas_type"))
    st.write("Quantidade de atletas recebidos:", diag.get("atletas_count"))
    st.write("Colunas no DataFrame:", list(df_raw.columns))

# Se não veio atleta nenhum, para aqui com mensagem amigável
if diag.get("atletas_count", 0) == 0:
    st.warning(
        "A API não retornou atletas agora (lista vazia). Isso pode acontecer quando o mercado está fechado "
        "ou por instabilidade do endpoint. Tente o outro endpoint na lateral e gere novamente."
    )
    st.stop()

# Filtro por clube
club_list = sorted([c for c in df_raw["clube"].dropna().unique().tolist() if str(c).strip()])
with st.sidebar:
    selected_clubs = st.multiselect("Filtrar clubes", club_list, default=[])

selected_clubs_set = set(selected_clubs) if selected_clubs else set()

# Previsão + filtros
df_pred = add_prediction_columns(df_raw, alpha_media=alpha, bonus_cb=bonus_cb)
df_ok = filter_players(
    df_pred,
    allowed_status=allowed_status,
    selected_clubs=selected_clubs_set,
    min_games=int(min_games),
    max_price=float(max_price),
    include_zero_price=bool(include_zero_price),
)

# Diagnóstico rápido
col1, col2, col3, col4 = st.columns(4)
col1.metric("Atletas recebidos", len(df_raw))
col2.metric("Atletas após filtros", len(df_ok))
col3.metric("Orçamento", f"{budget:.2f}")
col4.metric("Formação", "4-3-3")

with st.expander("Ver atletas filtrados (top 200 por previsão)", expanded=False):
    show = df_ok.copy()
    show = show.sort_values("pred", ascending=False).head(200)
    st.dataframe(
        show[["apelido", "clube", "posicao", "status", "preco", "media", "pontos_ultima", "pred", "cb", "jogos"]],
        use_container_width=True,
        hide_index=True,
    )

# Se não clicou gerar ainda
if not regenerate and "teams_done" not in st.session_state:
    st.info("Clique em **Gerar** para montar os 3 times.")
    st.stop()

# Se filtros ficaram restritivos demais
if df_ok.empty:
    st.error(
        "Depois dos filtros, não sobrou nenhum atleta. Tente:\n"
        "- Aumentar 'Preço máximo'\n"
        "- Diminuir 'Mín. jogos'\n"
        "- Ajustar 'Status permitidos' (ex.: incluir 'Dúvida')\n"
        "- Remover filtro de clubes"
    )
    st.stop()

# =========================
# Gerar 3 times
# =========================
seed = random.randint(1, 10_000_000)

t1, t1_meta = build_team(df_ok, FORMATION_433, budget, "conservador", avoid_ids=set(), max_overlap_with_avoids=None, seed=seed)
avoid = set(t1["atleta_id"].astype(int).tolist()) if not t1.empty else set()

t2, t2_meta = build_team(df_ok, FORMATION_433, budget, "custo-beneficio", avoid_ids=set(), max_overlap_with_avoids=None, seed=seed + 1337)
t3, t3_meta = build_team(df_ok, FORMATION_433, budget, "diferentao", avoid_ids=avoid, max_overlap_with_avoids=max_overlap, seed=seed + 2025)

st.session_state["teams_done"] = True


def render_team(title: str, team: pd.DataFrame, meta: Dict[str, float]):
    st.subheader(title)

    if team.empty or meta.get("ok", 0.0) < 1.0:
        st.error(f"Não foi possível montar o time. Motivo: {meta.get('reason', 'desconhecido')}")
        return

    team_players = team[team["bucket"] != "TEC"].copy()
    cap = None
    if not team_players.empty:
        cap = team_players.sort_values("pred", ascending=False).iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pontuação prevista", f"{meta['pred_total']:.2f}")
    c2.metric("Custo (C$)", f"{meta['preco_total']:.2f}")
    c3.metric("Sobra (C$)", f"{meta['restante']:.2f}")
    c4.metric("Repetidos (vs Time 1)", f"{int(meta.get('overlap', 0))}")

    if cap is not None:
        st.caption(f"⭐ Capitão sugerido: **{cap['apelido']}** ({cap['clube']}) — previsão {cap['pred']:.2f}")

    st.dataframe(format_team_table(team), use_container_width=True, hide_index=True)


tab1, tab2, tab3 = st.tabs(["Time 1 — Conservador", "Time 2 — Custo-benefício", "Time 3 — Diferentão"])

with tab1:
    render_team("Time 1 — Foco em pontuação (mais “seguro”)", t1, t1_meta)

with tab2:
    render_team("Time 2 — Foco em custo-benefício", t2, t2_meta)

with tab3:
    render_team("Time 3 — Mais diferente (limitando repetidos do Time 1)", t3, t3_meta)

st.divider()
st.caption(
    "Como funciona a previsão: `pred_base = alpha*média + (1-alpha)*pontos_ultima` e depois adiciona um bônus de "
    "custo-benefício (`pred_base/preço`) ajustável. Os times são montados por heurística respeitando orçamento e formação."
)
