#!/usr/bin/env python3
import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    import requests
except Exception:
    requests = None


# =========================
# Cartola API
# =========================
API_MERCADO_STATUS = "https://api.cartolafc.globo.com/mercado/status"
API_ATLETAS_MERCADO = "https://api.cartolafc.globo.com/atletas/mercado"

# posicao_id (Cartola)
# 1 = GOL, 2 = LAT, 3 = ZAG, 4 = MEI, 5 = ATA, 6 = TEC
POS_MAP = {1: "GOL", 2: "LAT", 3: "ZAG", 4: "MEI", 5: "ATA", 6: "TEC"}

# formação -> (ZAG, LAT, MEI, ATA) + 1 GOL + 1 TEC
FORMATION_MAP = {
    "3-4-3": {"ZAG": 3, "LAT": 0, "MEI": 4, "ATA": 3, "GOL": 1, "TEC": 1},
    "3-5-2": {"ZAG": 3, "LAT": 0, "MEI": 5, "ATA": 2, "GOL": 1, "TEC": 1},
    "4-3-3": {"ZAG": 2, "LAT": 2, "MEI": 3, "ATA": 3, "GOL": 1, "TEC": 1},
    "4-4-2": {"ZAG": 2, "LAT": 2, "MEI": 4, "ATA": 2, "GOL": 1, "TEC": 1},
    "4-5-1": {"ZAG": 2, "LAT": 2, "MEI": 5, "ATA": 1, "GOL": 1, "TEC": 1},
    "5-3-2": {"ZAG": 3, "LAT": 2, "MEI": 3, "ATA": 2, "GOL": 1, "TEC": 1},
    "5-4-1": {"ZAG": 3, "LAT": 2, "MEI": 4, "ATA": 1, "GOL": 1, "TEC": 1},
}

CAPTAIN_MULT = 1.5  # <<< capitão vale x1,5


def info(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[WARN] {msg}", flush=True)


def fetch_json(url: str, timeout: float = 12.0) -> dict:
    if requests is None:
        raise RuntimeError("requests não instalado")
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def get_market_status() -> Tuple[int, int]:
    data = fetch_json(API_MERCADO_STATUS, timeout=10.0)
    temporada = int(data.get("temporada", 2026))
    rodada = int(data.get("rodada_atual", 1))
    return temporada, rodada


def load_market_players() -> pd.DataFrame:
    """
    Baixa atletas do mercado + clubes/posições.
    Retorna df com:
      atleta_id, apelido, nome, posicao_id, posicao, preco, status_id, clube_id, clube_abrev
    """
    data = fetch_json(API_ATLETAS_MERCADO, timeout=15.0)

    atletas = data.get("atletas", [])
    if not atletas:
        raise RuntimeError("API retornou 0 atletas no mercado.")

    clubes = data.get("clubes", {}) or {}
    clube_abrev_map: Dict[int, str] = {}
    for k, v in clubes.items():
        try:
            cid = int(k)
        except Exception:
            try:
                cid = int(v.get("id"))
            except Exception:
                continue
        ab = str(v.get("abreviacao") or v.get("sigla") or "").strip()
        if ab:
            clube_abrev_map[cid] = ab

    rows = []
    for a in atletas:
        atleta_id = a.get("atleta_id")
        pos_id = a.get("posicao_id")
        if atleta_id is None or pos_id is None:
            continue

        clube_id = int(a.get("clube_id", 0) or 0)
        rows.append(
            {
                "atleta_id": int(atleta_id),
                "apelido": str(a.get("apelido", "")),
                "nome": str(a.get("nome", "")),
                "posicao_id": int(pos_id),
                "posicao": POS_MAP.get(int(pos_id), f"POS_{pos_id}"),
                "preco": float(a.get("preco_num", 0.0) or 0.0),
                "status_id": int(a.get("status_id", 0) or 0),
                "clube_id": clube_id,
                "clube_abrev": str(clube_abrev_map.get(clube_id, "")).strip(),
            }
        )

    df = pd.DataFrame(rows)

    # Filtra somente posições conhecidas e preço positivo
    df = df[df["posicao_id"].isin([1, 2, 3, 4, 5, 6])].copy()
    df = df[df["preco"] > 0].copy()

    # fallback: se não tiver abreviação, usa "T<id>"
    df["clube_abrev"] = df["clube_abrev"].fillna("")
    df.loc[df["clube_abrev"] == "", "clube_abrev"] = df.loc[df["clube_abrev"] == "", "clube_id"].apply(lambda x: f"T{int(x)}")

    return df


def load_history_scores(history_root: Path, years: List[int]) -> pd.DataFrame:
    candidates = []
    for base in [history_root, history_root / "01_raw"]:
        for y in years:
            p = base / str(y)
            if p.exists():
                candidates.append(p)

    if not candidates:
        warn("Histórico não encontrado. Vou usar score=0 e heurística por preço/posição.")
        return pd.DataFrame(columns=["atleta_id", "score_modelo"])

    rows = []
    for p in candidates:
        for f in sorted(p.glob("rodada-*.csv")):
            try:
                df = pd.read_csv(f)
            except Exception:
                continue

            id_col = None
            for c in ["atletas.atleta_id", "atletas_atleta_id", "atleta_id"]:
                if c in df.columns:
                    id_col = c
                    break
            if id_col is None:
                continue

            pts_col = None
            for c in ["atletas.pontos_num", "pontos_num", "pontos", "score"]:
                if c in df.columns:
                    pts_col = c
                    break
            if pts_col is None:
                continue

            sub = df[[id_col, pts_col]].copy()
            sub.columns = ["atleta_id", "pontos"]
            sub["atleta_id"] = pd.to_numeric(sub["atleta_id"], errors="coerce")
            sub["pontos"] = pd.to_numeric(sub["pontos"], errors="coerce")
            sub = sub.dropna()
            if len(sub):
                rows.append(sub)

    if not rows:
        warn("Histórico lido, mas sem colunas compatíveis. Vou usar score=0.")
        return pd.DataFrame(columns=["atleta_id", "score_modelo"])

    hist = pd.concat(rows, ignore_index=True)
    hist["atleta_id"] = hist["atleta_id"].astype(int)
    model = hist.groupby("atleta_id")["pontos"].mean().reset_index()
    model.columns = ["atleta_id", "score_modelo"]
    return model


def make_scored_market(market: pd.DataFrame, hist_scores: pd.DataFrame) -> pd.DataFrame:
    df = market.merge(hist_scores, on="atleta_id", how="left")
    df["score_modelo"] = df["score_modelo"].fillna(0.0)

    # Heurística leve (rápida/estável)
    df["valor"] = df["score_modelo"] / (df["preco"].replace(0, 0.01))
    df["score"] = df["score_modelo"] + 0.05 * df["valor"]

    return df


@dataclass
class Pick:
    atleta_id: int
    apelido: str
    posicao: str
    posicao_id: int
    preco: float
    score: float
    clube_id: int
    clube_abrev: str


def _pos_pool(
    scored_market: pd.DataFrame,
    pos_name: str,
    used_team_ids: set,
    global_counts: Dict[int, int],
    max_repeat: int,
    allow_repeat_across_teams: bool,
) -> pd.DataFrame:
    """
    Regras:
    - NUNCA repetir dentro do mesmo time (sempre filtra used_team_ids)
    - Repetição entre times:
        - se allow_repeat_across_teams=False => jogador só pode aparecer 1 vez no total
        - se True => jogador pode repetir ATÉ max_repeat vezes no total
    """
    df_pos = scored_market[scored_market["posicao"] == pos_name].copy()

    if used_team_ids:
        df_pos = df_pos[~df_pos["atleta_id"].isin(list(used_team_ids))].copy()

    if not allow_repeat_across_teams:
        if global_counts:
            used_global = [aid for aid, c in global_counts.items() if c >= 1]
            if used_global:
                df_pos = df_pos[~df_pos["atleta_id"].isin(used_global)].copy()
    else:
        # permite repetir, mas respeita max_repeat
        if global_counts and max_repeat > 0:
            blocked = [aid for aid, c in global_counts.items() if c >= max_repeat]
            if blocked:
                df_pos = df_pos[~df_pos["atleta_id"].isin(blocked)].copy()

    return df_pos


def _min_cost_for_k(df_pos: pd.DataFrame, k: int) -> Optional[float]:
    if k <= 0:
        return 0.0
    if df_pos.empty or len(df_pos) < k:
        return None
    return float(df_pos["preco"].nsmallest(k).sum())


def estimate_min_cost_to_complete(
    scored_market: pd.DataFrame,
    formation: str,
    used_team_ids: set,
    global_counts: Dict[int, int],
    max_repeat: int,
    allow_repeat_across_teams: bool,
) -> Optional[float]:
    need = FORMATION_MAP.get(formation)
    if not need:
        return None

    total = 0.0
    for pos_name, k in need.items():
        if int(k) <= 0:
            continue

        pool = _pos_pool(
            scored_market=scored_market,
            pos_name=pos_name,
            used_team_ids=used_team_ids,
            global_counts=global_counts,
            max_repeat=max_repeat,
            allow_repeat_across_teams=allow_repeat_across_teams,
        )
        mc = _min_cost_for_k(pool, int(k))
        if mc is None:
            return None
        total += mc

    return float(total)


def pick_best_feasible(
    scored_market: pd.DataFrame,
    formation: str,
    pos_name: str,
    budget_left: float,
    used_team_ids: set,
    global_counts: Dict[int, int],
    max_repeat: int,
    allow_repeat_across_teams: bool,
    remaining_need: Dict[str, int],
) -> Optional[Pick]:
    df_pos = _pos_pool(
        scored_market=scored_market,
        pos_name=pos_name,
        used_team_ids=used_team_ids,
        global_counts=global_counts,
        max_repeat=max_repeat,
        allow_repeat_across_teams=allow_repeat_across_teams,
    )
    if df_pos.empty:
        return None

    df_pos = df_pos.sort_values(["score", "score_modelo", "preco"], ascending=[False, False, True]).copy()

    for _, row in df_pos.iterrows():
        preco = float(row["preco"])
        if preco > budget_left:
            continue

        tmp_used_team = set(used_team_ids)
        tmp_used_team.add(int(row["atleta_id"]))

        tmp_need = dict(remaining_need)
        tmp_need[pos_name] = max(0, int(tmp_need.get(pos_name, 0)) - 1)

        min_rest = 0.0
        possible = True
        for p, k in tmp_need.items():
            if int(k) <= 0:
                continue
            pool = _pos_pool(
                scored_market=scored_market,
                pos_name=p,
                used_team_ids=tmp_used_team,
                global_counts=global_counts,
                max_repeat=max_repeat,
                allow_repeat_across_teams=allow_repeat_across_teams,
            )
            mc = _min_cost_for_k(pool, int(k))
            if mc is None:
                possible = False
                break
            min_rest += mc

        if not possible:
            continue

        if preco + min_rest <= budget_left:
            return Pick(
                atleta_id=int(row["atleta_id"]),
                apelido=str(row["apelido"]),
                posicao=str(row["posicao"]),
                posicao_id=int(row["posicao_id"]),
                preco=float(row["preco"]),
                score=float(row["score"]),
                clube_id=int(row.get("clube_id", 0) or 0),
                clube_abrev=str(row.get("clube_abrev", "")).strip(),
            )

    return None


def pick_cheapest(
    scored_market: pd.DataFrame,
    pos_name: str,
    used_team_ids: set,
    global_counts: Dict[int, int],
    max_repeat: int,
    allow_repeat_across_teams: bool,
    price_cap: Optional[float] = None,
) -> Optional[Pick]:
    df_pos = _pos_pool(
        scored_market=scored_market,
        pos_name=pos_name,
        used_team_ids=used_team_ids,
        global_counts=global_counts,
        max_repeat=max_repeat,
        allow_repeat_across_teams=allow_repeat_across_teams,
    )
    if price_cap is not None:
        df_pos = df_pos[df_pos["preco"] <= float(price_cap)].copy()
    if df_pos.empty:
        return None

    df_pos = df_pos.sort_values(["preco", "score"], ascending=[True, False]).copy()
    row = df_pos.iloc[0]
    return Pick(
        atleta_id=int(row["atleta_id"]),
        apelido=str(row["apelido"]),
        posicao=str(row["posicao"]),
        posicao_id=int(row["posicao_id"]),
        preco=float(row["preco"]),
        score=float(row["score"]),
        clube_id=int(row.get("clube_id", 0) or 0),
        clube_abrev=str(row.get("clube_abrev", "")).strip(),
    )


def build_team(
    scored_market: pd.DataFrame,
    formation: str,
    budget_total: float,
    allow_repeat_across_teams: bool,
    max_repeat: int,
    global_counts: Dict[int, int],
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Orçamento (cartoletas) vale APENAS para titulares + técnico.
    Banco NÃO consome cartoletas.

    Repetição:
    - Dentro do mesmo time: NUNCA repete (sempre bloqueado)
    - Entre times:
        - allow_repeat_across_teams=False => no máximo 1 vez no total
        - True => até max_repeat vezes no total
    """
    random.seed(seed)

    need = FORMATION_MAP.get(formation)
    if not need:
        raise ValueError(f"Formação inválida: {formation}")

    used_team_ids = set()

    min_cost_required = estimate_min_cost_to_complete(
        scored_market=scored_market,
        formation=formation,
        used_team_ids=used_team_ids,
        global_counts=global_counts,
        max_repeat=max_repeat,
        allow_repeat_across_teams=allow_repeat_across_teams,
    )
    if min_cost_required is None:
        raise RuntimeError("Não há atletas suficientes no mercado para completar a formação (com essas regras).")

    budget_ok = 1
    if float(budget_total) < float(min_cost_required):
        budget_ok = 0
        warn(
            f"Orçamento insuficiente para fechar formação. "
            f"budget={budget_total:.2f} min_required={min_cost_required:.2f}. "
            "Vou gerar o time completo mais barato possível."
        )

    rows = []
    main_picks: List[Pick] = []
    budget_left = float(budget_total)

    remaining_need = dict(need)
    pick_order = ["GOL", "ZAG", "LAT", "MEI", "ATA", "TEC"]

    for pos_name in pick_order:
        k = int(need.get(pos_name, 0))
        if k <= 0:
            continue

        for _ in range(k):
            p = None

            if budget_ok == 1:
                p = pick_best_feasible(
                    scored_market=scored_market,
                    formation=formation,
                    pos_name=pos_name,
                    budget_left=budget_left,
                    used_team_ids=used_team_ids,
                    global_counts=global_counts,
                    max_repeat=max_repeat,
                    allow_repeat_across_teams=allow_repeat_across_teams,
                    remaining_need=remaining_need,
                )

                if p is None:
                    cheap = pick_cheapest(
                        scored_market,
                        pos_name,
                        used_team_ids,
                        global_counts,
                        max_repeat,
                        allow_repeat_across_teams,
                        price_cap=None,
                    )
                    if cheap is None:
                        raise RuntimeError(f"Sem opções para {pos_name}.")
                    if cheap.preco > budget_left:
                        budget_ok = 0
                        warn(f"Budget estourou ao tentar completar {pos_name}. Mudando para modo mais barato possível.")
                        p = None
                    else:
                        p = cheap

            if budget_ok == 0 and p is None:
                p = pick_cheapest(
                    scored_market,
                    pos_name,
                    used_team_ids,
                    global_counts,
                    max_repeat,
                    allow_repeat_across_teams,
                    price_cap=None,
                )
                if p is None:
                    raise RuntimeError(f"Não consegui completar {pos_name} (sem opções).")

            used_team_ids.add(p.atleta_id)
            remaining_need[pos_name] = max(0, int(remaining_need.get(pos_name, 0)) - 1)
            budget_left -= float(p.preco)
            main_picks.append(p)

    starters_no_tec = [p for p in main_picks if p.posicao != "TEC"]
    if not starters_no_tec:
        raise RuntimeError("Time sem titulares (algo deu muito errado).")
    captain = max(starters_no_tec, key=lambda x: x.score)

    for p in main_picks:
        rows.append(
            {
                "atleta_id": p.atleta_id,
                "apelido": p.apelido,
                "posicao": p.posicao,
                "posicao_id": p.posicao_id,
                "preco": p.preco,
                "score": p.score,
                "clube_id": p.clube_id,
                "clube_abrev": p.clube_abrev,
                "role": "coach" if p.posicao == "TEC" else "starter",
                "is_captain": bool(p.atleta_id == captain.atleta_id) if p.posicao != "TEC" else False,
                "is_luxury": False,
            }
        )

    # -------------------------
    # Banco (não consome cartoletas)
    # regra: reserva <= mais barato titular da mesma posição
    # -------------------------
    bench_picks: List[Pick] = []

    starters_df = pd.DataFrame([r for r in rows if r["role"] == "starter"])
    cheapest_by_pos: Dict[str, float] = {}
    if not starters_df.empty:
        cheapest_by_pos = starters_df.groupby("posicao")["preco"].min().to_dict()

    for pos_name in ["GOL", "ZAG", "LAT", "MEI", "ATA"]:
        if pos_name not in cheapest_by_pos:
            continue
        price_cap = float(cheapest_by_pos[pos_name])

        p = pick_cheapest(
            scored_market=scored_market,
            pos_name=pos_name,
            used_team_ids=used_team_ids,
            global_counts=global_counts,
            max_repeat=max_repeat,
            allow_repeat_across_teams=allow_repeat_across_teams,
            price_cap=price_cap,
        )
        if p is None:
            warn(f"Banco: sem opção para {pos_name} <= {price_cap:.2f}. Pulando.")
            continue

        used_team_ids.add(p.atleta_id)
        bench_picks.append(p)

    luxury_id = None
    if bench_picks:
        luxury_id = max(bench_picks, key=lambda x: x.score).atleta_id

    for p in bench_picks:
        rows.append(
            {
                "atleta_id": p.atleta_id,
                "apelido": p.apelido,
                "posicao": p.posicao,
                "posicao_id": p.posicao_id,
                "preco": p.preco,
                "score": p.score,
                "clube_id": p.clube_id,
                "clube_abrev": p.clube_abrev,
                "role": "bench",
                "is_captain": False,
                "is_luxury": bool(luxury_id is not None and p.atleta_id == luxury_id),
            }
        )

    df_team = pd.DataFrame(rows)

    preco_main = float(df_team[df_team["role"].isin(["starter", "coach"])]["preco"].sum())
    preco_bench = float(df_team[df_team["role"] == "bench"]["preco"].sum())
    preco_total = float(preco_main + preco_bench)

    starters_only = df_team[df_team["role"] == "starter"].copy()
    coach_only = df_team[df_team["role"] == "coach"].copy()

    score_sum = float(starters_only["score"].sum() + coach_only["score"].sum())
    cap_score = float(starters_only[starters_only["is_captain"] == True]["score"].sum())

    # Capitão x1,5 => adiciona +0,5 do capitão
    score_total = float(score_sum + (CAPTAIN_MULT - 1.0) * cap_score)

    budget_ok_final = 1 if (preco_main <= float(budget_total) and float(budget_total) >= float(min_cost_required)) else 0
    budget_left_final = float(float(budget_total) - preco_main)

    kpis = {
        "preco_main": preco_main,
        "preco_bench": preco_bench,
        "preco_total": preco_total,
        "score_total": score_total,
        "budget_ok": int(budget_ok_final),
        "min_cost_required": float(min_cost_required),
        "budget_left": float(budget_left_final),
        "captain_mult": float(CAPTAIN_MULT),
    }
    return df_team, kpis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-times", type=int, required=True)
    ap.add_argument("--budget", type=float, required=True)
    ap.add_argument("--formation", type=str, required=True)

    # Mantido só pra compatibilidade (não usado; banco não consome cartoletas)
    ap.add_argument("--bench-budget-ratio", type=float, default=0.0)

    # Se ligado, permite repetição ENTRE times, respeitando max-repeat
    ap.add_argument("--allow-repeat", action="store_true")

    # Limite de vezes que o MESMO atleta pode aparecer NO TOTAL entre os times
    # (Quando allow-repeat desligado, equivale a 1)
    ap.add_argument("--max-repeat", type=int, default=1)

    ap.add_argument("--history-root", type=str, default="cartola/data")
    ap.add_argument("--years", type=str, default="2023,2024,2025")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    n_times = int(args.n_times)
    budget = float(args.budget)
    formation = str(args.formation).strip()

    allow_repeat_across_teams = bool(args.allow_repeat)
    max_repeat = int(args.max_repeat) if int(args.max_repeat) > 0 else 1
    if not allow_repeat_across_teams:
        max_repeat = 1

    years = [int(x.strip()) for x in str(args.years).split(",") if x.strip().isdigit()]
    history_root = Path(args.history_root)

    if formation not in FORMATION_MAP:
        raise SystemExit(f"[ERRO] Formação inválida: {formation}")
    if n_times < 1:
        raise SystemExit("[ERRO] n-times precisa ser >= 1")

    info("Iniciando geração de times 2026 via API do Cartola...")
    temporada, rodada = get_market_status()
    info(f"API status: temporada={temporada} rodada_atual={rodada}")

    market = load_market_players()
    info(f"Atletas no mercado (filtrados): {len(market)}")

    hist = load_history_scores(history_root, years)
    info(f"Histórico carregado: {len(hist)} atletas com média")

    scored = make_scored_market(market, hist)

    out_rows = []
    ok = 0
    failed = 0

    # contagem global de aparições por atleta entre times
    global_counts: Dict[int, int] = {}

    info(f"Repetição entre times: {'ON' if allow_repeat_across_teams else 'OFF'} | max_repeat={max_repeat}")

    for i in range(1, n_times + 1):
        try:
            team_df, kpis = build_team(
                scored_market=scored,
                formation=formation,
                budget_total=budget,
                allow_repeat_across_teams=allow_repeat_across_teams,
                max_repeat=max_repeat,
                global_counts=global_counts,
                seed=1234 + i,
            )

            team_df = team_df.copy()
            team_df["team_id"] = int(i)
            team_df["formation"] = formation
            team_df["budget"] = float(budget)

            team_df["preco_main"] = float(kpis["preco_main"])
            team_df["preco_bench"] = float(kpis["preco_bench"])
            team_df["preco_total"] = float(kpis["preco_total"])
            team_df["score_total"] = float(kpis["score_total"])

            team_df["budget_ok"] = int(kpis["budget_ok"])
            team_df["min_cost_required"] = float(kpis["min_cost_required"])
            team_df["budget_left"] = float(kpis["budget_left"])
            team_df["captain_mult"] = float(kpis["captain_mult"])

            out_rows.append(team_df)

            # atualiza contagem global (considera todos: titulares, técnico e banco)
            for aid in team_df["atleta_id"].astype(int).tolist():
                global_counts[aid] = int(global_counts.get(aid, 0) + 1)

            ok += 1
            info(
                f"Time {i}: atletas={len(team_df)} "
                f"preco_main={kpis['preco_main']:.2f} (budget={budget:.2f}, ok={kpis['budget_ok']}) "
                f"score_total={kpis['score_total']:.2f} (capx{CAPTAIN_MULT}) "
                f"banco={kpis['preco_bench']:.2f}"
            )

        except Exception as e:
            failed += 1
            warn(f"Time {i} falhou: {e}")

    if not out_rows:
        raise SystemExit("[ERRO] Nenhum time gerado.")

    df_out = pd.concat(out_rows, ignore_index=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)

    info(f"Arquivo salvo: {out_path} (linhas={len(df_out)})")
    info(f"Gerados: ok={ok}/{n_times} | falhas={failed}")


if __name__ == "__main__":
    main()
