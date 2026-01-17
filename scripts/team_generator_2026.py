#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Geração de times Cartola 2026 usando:
- histórico (CSV rodadas 2023-2025) para aprender score previsto por atleta
- mercado 2026 via API do Cartola (atletas disponíveis)

Entrega:
- N times com formação escolhida e budget
- Capitão = maior score previsto no XI
- Técnico (TEC) escolhido do mercado (posicao_id=6)
- Reservas: 1 DEF + 1 MEI + 1 ATA
- Reserva de luxo: maior score previsto entre reservas
- CSV de saída com colunas detalhadas

Uso:
  python scripts/team_generator_2026.py \
    --history-root cartola/data/01_raw \
    --years 2023 2024 2025 \
    --n-times 3 \
    --budget 100 \
    --formation 4-3-3 \
    --out data/teams_2026.csv

Obs:
- Se o atleta não aparece no histórico, ele recebe score global (média geral).
- O algoritmo é heurístico (rápido/estável), com checagem de orçamento mínimo restante.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

try:
    import requests
except Exception:
    requests = None


# =========================
# Logs simples e claros
# =========================
def info(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)

def warn(msg: str) -> None:
    print(f"[WARN] {msg}", flush=True)

def err(msg: str) -> None:
    print(f"[ERRO] {msg}", flush=True)


# =========================
# Constantes Cartola
# =========================
# posicao_id:
# 1 GOL, 2 LAT, 3 ZAG, 4 MEI, 5 ATA, 6 TEC
POS_GOL = 1
POS_LAT = 2
POS_ZAG = 3
POS_MEI = 4
POS_ATA = 5
POS_TEC = 6

# Setores para reserva (simplificado)
SECTOR_DEF = {POS_LAT, POS_ZAG}
SECTOR_MEI = {POS_MEI}
SECTOR_ATA = {POS_ATA}

DEFAULT_API_BASE = "https://api.cartola.globo.com"


FORMATION_MAP: Dict[str, Dict[int, int]] = {
    # formação: {posicao_id: quantidade}
    "4-3-3": {POS_GOL: 1, POS_LAT: 2, POS_ZAG: 2, POS_MEI: 3, POS_ATA: 3},
    "4-4-2": {POS_GOL: 1, POS_LAT: 2, POS_ZAG: 2, POS_MEI: 4, POS_ATA: 2},
    "3-4-3": {POS_GOL: 1, POS_ZAG: 3, POS_MEI: 4, POS_ATA: 3},
    "3-5-2": {POS_GOL: 1, POS_ZAG: 3, POS_MEI: 5, POS_ATA: 2},
    "5-3-2": {POS_GOL: 1, POS_LAT: 2, POS_ZAG: 3, POS_MEI: 3, POS_ATA: 2},
    "5-4-1": {POS_GOL: 1, POS_LAT: 2, POS_ZAG: 3, POS_MEI: 4, POS_ATA: 1},
}


# =========================
# Utilidades
# =========================
def parse_years(values: List[str]) -> List[int]:
    out = []
    for v in values:
        v = str(v).strip()
        if not v:
            continue
        out.append(int(v))
    return out

def safe_float(x, default=0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip().replace(",", ".")
        if s == "" or s.lower() == "nan":
            return default
        return float(s)
    except Exception:
        return default

def safe_int(x, default=0) -> int:
    try:
        if x is None:
            return default
        if isinstance(x, int):
            return int(x)
        if isinstance(x, float) and not math.isnan(x):
            return int(x)
        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return default
        return int(float(s))
    except Exception:
        return default

def formation_to_counts(formation: str) -> Dict[int, int]:
    if formation not in FORMATION_MAP:
        raise ValueError(f"Formação inválida: {formation}. Opções: {sorted(FORMATION_MAP.keys())}")
    return dict(FORMATION_MAP[formation])

def list_round_files(history_root: str, year: int) -> List[str]:
    # Esperado: history_root/{year}/rodada-*.csv  (ex.: cartola/data/01_raw/2023/rodada-1.csv)
    base = os.path.join(history_root, str(year))
    if not os.path.isdir(base):
        return []
    files = []
    for fn in os.listdir(base):
        if fn.startswith("rodada-") and fn.endswith(".csv"):
            files.append(os.path.join(base, fn))
    # ordena por rodada numérica
    def rodada_key(p: str) -> int:
        m = re.search(r"rodada-(\d+)\.csv$", p)
        return int(m.group(1)) if m else 9999
    files.sort(key=rodada_key)
    return files


# =========================
# Histórico -> "modelo" simples (score previsto)
# =========================
@dataclass
class HistoryModel:
    global_mean: float
    atleta_mean: Dict[int, float]
    atleta_n: Dict[int, int]

def build_history_model(history_root: str, years: List[int], min_games: int = 1, shrink_k: int = 5) -> HistoryModel:
    """
    Aprende score por atleta usando média de pontos históricos com suavização:
      pred = (n/(n+k))*mean + (k/(n+k))*global_mean
    """
    info(f"Carregando histórico: {history_root} years={years}")
    rows = 0
    sum_points = 0.0
    sum_count = 0

    atleta_sum: Dict[int, float] = {}
    atleta_cnt: Dict[int, int] = {}

    # colunas históricas típicas (do seu print)
    col_id = "atletas.atleta_id"
    col_pts = "atletas.pontos_num"

    for y in years:
        files = list_round_files(history_root, y)
        info(f"{y}: {len(files)} arquivos")
        for fp in files:
            try:
                df = pd.read_csv(fp)
            except Exception as e:
                warn(f"Falha ao ler {fp}: {e}")
                continue

            if col_id not in df.columns or col_pts not in df.columns:
                # tenta fallback com nomes alternativos
                # (não mexe no seu histórico se estiver diferente; só ignora)
                continue

            ids = df[col_id].apply(safe_int)
            pts = df[col_pts].apply(safe_float)

            for aid, p in zip(ids.tolist(), pts.tolist()):
                if aid <= 0:
                    continue
                atleta_sum[aid] = atleta_sum.get(aid, 0.0) + float(p)
                atleta_cnt[aid] = atleta_cnt.get(aid, 0) + 1
                sum_points += float(p)
                sum_count += 1
                rows += 1

    global_mean = (sum_points / sum_count) if sum_count > 0 else 0.0
    info(f"Histórico carregado: linhas={rows:,} global_mean={global_mean:.4f}")

    atleta_mean: Dict[int, float] = {}
    atleta_n: Dict[int, int] = {}
    for aid, cnt in atleta_cnt.items():
        if cnt < min_games:
            continue
        mean = atleta_sum[aid] / cnt
        # shrink
        k = shrink_k
        pred = (cnt / (cnt + k)) * mean + (k / (cnt + k)) * global_mean
        atleta_mean[aid] = float(pred)
        atleta_n[aid] = int(cnt)

    info(f"Atletas com histórico útil: {len(atleta_mean):,}")
    return HistoryModel(global_mean=global_mean, atleta_mean=atleta_mean, atleta_n=atleta_n)


# =========================
# API Cartola
# =========================
def http_get_json(url: str, timeout: int = 20) -> dict:
    if requests is None:
        raise RuntimeError("requests não está disponível no ambiente. Instale: pip install requests")
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "cartola-prolab/1.0"})
    r.raise_for_status()
    return r.json()

def fetch_mercado(api_base: str) -> dict:
    # endpoint público clássico
    return http_get_json(f"{api_base}/atletas/mercado")

def mercado_to_df(mercado_json: dict) -> pd.DataFrame:
    atletas = mercado_json.get("atletas", [])
    if not isinstance(atletas, list):
        atletas = []
    rows = []
    for a in atletas:
        try:
            aid = safe_int(a.get("atleta_id"))
            pos = safe_int(a.get("posicao_id"))
            preco = safe_float(a.get("preco_num"))
            status = safe_int(a.get("status_id"))
            nome = str(a.get("nome") or "")
            apelido = str(a.get("apelido") or "")
            clube_id = safe_int(a.get("clube_id"))
            foto = str(a.get("foto") or "")
            rows.append(
                {
                    "atleta_id": aid,
                    "posicao_id": pos,
                    "preco_num": preco,
                    "status_id": status,
                    "nome": nome,
                    "apelido": apelido,
                    "clube_id": clube_id,
                    "foto": foto,
                }
            )
        except Exception:
            continue
    df = pd.DataFrame(rows)
    return df


# =========================
# Seleção do time
# =========================
@dataclass
class PickResult:
    ok: bool
    reason: str
    starters: pd.DataFrame
    tecnico: Optional[pd.Series]
    captain_id: Optional[int]
    reserves: pd.DataFrame
    luxury_reserve_id: Optional[int]
    budget_total: float
    budget_used: float
    budget_left: float
    score_total: float

def _min_cost_for_remaining(candidates_by_pos: Dict[int, pd.DataFrame],
                            remaining_counts: Dict[int, int],
                            used: Set[int]) -> float:
    """
    Calcula um limite inferior do custo mínimo para completar o time
    (usado para checagem de orçamento durante o greedy).
    """
    total = 0.0
    for pos, need in remaining_counts.items():
        if need <= 0:
            continue
        df = candidates_by_pos.get(pos)
        if df is None or df.empty:
            return float("inf")
        # pega os mais baratos que não estão em used
        df2 = df[~df["atleta_id"].isin(used)].sort_values("preco_num", ascending=True)
        if len(df2) < need:
            return float("inf")
        total += float(df2["preco_num"].head(need).sum())
    return float(total)

def _pick_one_position(pos: int,
                       need: int,
                       candidates: pd.DataFrame,
                       used: Set[int],
                       counts_remaining: Dict[int, int],
                       candidates_by_pos: Dict[int, pd.DataFrame],
                       budget_left: float) -> Tuple[List[pd.Series], float, str]:
    """
    Seleciona 'need' atletas de uma posição com greedy com checagem de orçamento mínimo restante.
    Prioriza maior score_pred e, em empate, menor preço.
    """
    picked: List[pd.Series] = []

    if need <= 0:
        return picked, budget_left, "ok"

    # ordena por score alto, preço baixo
    df = candidates[~candidates["atleta_id"].isin(used)].copy()
    if df.empty:
        return picked, budget_left, f"sem candidatos pos={pos}"

    df = df.sort_values(["score_pred", "preco_num"], ascending=[False, True])

    for _, row in df.iterrows():
        if len(picked) >= need:
            break

        aid = int(row["atleta_id"])
        price = float(row["preco_num"])
        if price > budget_left:
            continue

        # simula escolher esse jogador e checa se ainda dá pra completar o resto
        used_tmp = set(used)
        used_tmp.add(aid)

        counts_tmp = dict(counts_remaining)
        counts_tmp[pos] = max(0, counts_tmp.get(pos, 0) - 1)

        min_rest = _min_cost_for_remaining(candidates_by_pos, counts_tmp, used_tmp)
        if price + min_rest > budget_left + price:  # equivalente a min_rest > budget_left - price
            # não dá pra completar depois
            continue

        # escolhe
        picked.append(row)
        used.add(aid)
        budget_left -= price
        counts_remaining[pos] = counts_tmp[pos]

    if len(picked) < need:
        return picked, budget_left, f"faltou pos={pos} ({len(picked)}/{need})"
    return picked, budget_left, "ok"

def _choose_tecnico(df_market: pd.DataFrame,
                    used: Set[int],
                    budget_left: float) -> Optional[pd.Series]:
    tec = df_market[df_market["posicao_id"] == POS_TEC].copy()
    tec = tec[~tec["atleta_id"].isin(used)]
    tec = tec[tec["preco_num"] <= budget_left]
    if tec.empty:
        return None
    tec = tec.sort_values(["score_pred", "preco_num"], ascending=[False, True])
    return tec.iloc[0]

def _choose_reserves(df_market: pd.DataFrame,
                     used: Set[int],
                     budget_left: float) -> Tuple[pd.DataFrame, float]:
    """
    Reservas:
      - 1 DEF (LAT/ZAG)
      - 1 MEI
      - 1 ATA
    Dentro do orçamento restante.
    """
    reserves = []
    budget = budget_left

    def pick_from(pos_set: Set[int], label: str) -> Optional[pd.Series]:
        nonlocal budget
        cand = df_market[df_market["posicao_id"].isin(pos_set)].copy()
        cand = cand[~cand["atleta_id"].isin(used)]
        cand = cand[cand["preco_num"] <= budget]
        if cand.empty:
            return None
        cand = cand.sort_values(["score_pred", "preco_num"], ascending=[False, True])
        row = cand.iloc[0]
        aid = int(row["atleta_id"])
        price = float(row["preco_num"])
        used.add(aid)
        budget -= price
        row = row.copy()
        row["reserva_setor"] = label
        return row

    r_def = pick_from(SECTOR_DEF, "DEF")
    if r_def is not None:
        reserves.append(r_def)

    r_mei = pick_from(SECTOR_MEI, "MEI")
    if r_mei is not None:
        reserves.append(r_mei)

    r_ata = pick_from(SECTOR_ATA, "ATA")
    if r_ata is not None:
        reserves.append(r_ata)

    if reserves:
        df_res = pd.DataFrame(reserves)
    else:
        df_res = pd.DataFrame(columns=list(df_market.columns) + ["reserva_setor"])
    return df_res, budget

def generate_one_team(df_market: pd.DataFrame,
                      formation: str,
                      budget: float,
                      used_global: Set[int],
                      allow_repeat: bool,
                      require_complete: bool = True) -> PickResult:
    counts = formation_to_counts(formation)

    # candidatos por posição (só "disponíveis" de mercado — aqui não filtramos status_id forte)
    candidates_by_pos: Dict[int, pd.DataFrame] = {}
    for pos in counts.keys():
        candidates_by_pos[pos] = df_market[df_market["posicao_id"] == pos].copy()

    used: Set[int] = set()
    if not allow_repeat:
        used = set(used_global)

    budget_left = float(budget)
    picked_rows: List[pd.Series] = []

    counts_remaining = dict(counts)

    # escolhe por ordem de "restrição": GOL primeiro, depois DEF, MEI, ATA
    order = [POS_GOL, POS_LAT, POS_ZAG, POS_MEI, POS_ATA]
    for pos in order:
        if pos not in counts_remaining:
            continue
        need = counts_remaining[pos]
        cand = candidates_by_pos.get(pos)
        if cand is None or cand.empty:
            return PickResult(
                ok=False,
                reason=f"Sem candidatos para posicao {pos}",
                starters=pd.DataFrame(),
                tecnico=None,
                captain_id=None,
                reserves=pd.DataFrame(),
                luxury_reserve_id=None,
                budget_total=budget,
                budget_used=0.0,
                budget_left=budget,
                score_total=0.0,
            )

        picked, budget_left, status = _pick_one_position(
            pos=pos,
            need=need,
            candidates=cand,
            used=used,
            counts_remaining=counts_remaining,
            candidates_by_pos=candidates_by_pos,
            budget_left=budget_left,
        )
        picked_rows.extend(picked)
        if status != "ok":
            if require_complete:
                return PickResult(
                    ok=False,
                    reason=status,
                    starters=pd.DataFrame(),
                    tecnico=None,
                    captain_id=None,
                    reserves=pd.DataFrame(),
                    luxury_reserve_id=None,
                    budget_total=budget,
                    budget_used=float(budget - budget_left),
                    budget_left=budget_left,
                    score_total=0.0,
                )
            else:
                warn(status)

    starters = pd.DataFrame(picked_rows).copy()
    if starters.empty:
        return PickResult(
            ok=False,
            reason="Nenhum atleta selecionado",
            starters=pd.DataFrame(),
            tecnico=None,
            captain_id=None,
            reserves=pd.DataFrame(),
            luxury_reserve_id=None,
            budget_total=budget,
            budget_used=0.0,
            budget_left=budget,
            score_total=0.0,
        )

    # Técnico
    tec = _choose_tecnico(df_market, used=used, budget_left=budget_left)
    if tec is not None:
        budget_left -= float(tec["preco_num"])
        used.add(int(tec["atleta_id"]))

    # Reservas
    reserves, budget_left = _choose_reserves(df_market, used=used, budget_left=budget_left)

    # Capitão = maior score previsto no time titular (não inclui TEC nem reservas)
    starters_sorted = starters.sort_values(["score_pred", "preco_num"], ascending=[False, True])
    captain_id = int(starters_sorted.iloc[0]["atleta_id"]) if len(starters_sorted) > 0 else None

    # Reserva de luxo = maior score previsto entre reservas
    luxury_id = None
    if reserves is not None and not reserves.empty:
        r_sorted = reserves.sort_values(["score_pred", "preco_num"], ascending=[False, True])
        luxury_id = int(r_sorted.iloc[0]["atleta_id"])

    # score total (só titulares + técnico + reservas? aqui reporto separado: score_total = titulares + técnico)
    score_total = float(starters["score_pred"].sum())
    if tec is not None:
        score_total += float(tec["score_pred"])

    budget_used = float(budget - budget_left)

    # Atualiza used_global se não permitir repetição
    if not allow_repeat:
        used_global.update(set(starters["atleta_id"].astype(int).tolist()))
        if tec is not None:
            used_global.add(int(tec["atleta_id"]))
        if reserves is not None and not reserves.empty:
            used_global.update(set(reserves["atleta_id"].astype(int).tolist()))

    return PickResult(
        ok=True,
        reason="ok",
        starters=starters,
        tecnico=tec,
        captain_id=captain_id,
        reserves=reserves,
        luxury_reserve_id=luxury_id,
        budget_total=budget,
        budget_used=budget_used,
        budget_left=budget_left,
        score_total=score_total,
    )


# =========================
# Montagem do dataset final
# =========================
def attach_predictions(df_market: pd.DataFrame, model: HistoryModel) -> pd.DataFrame:
    df = df_market.copy()
    preds = []
    ns = []
    for aid in df["atleta_id"].tolist():
        aid = int(aid)
        pred = model.atleta_mean.get(aid, model.global_mean)
        n = model.atleta_n.get(aid, 0)
        preds.append(float(pred))
        ns.append(int(n))
    df["score_pred"] = preds
    df["hist_n"] = ns
    return df

def team_to_rows(team_idx: int, result: PickResult) -> List[dict]:
    rows: List[dict] = []

    # starters
    for _, r in result.starters.iterrows():
        rows.append(
            {
                "team_id": team_idx,
                "slot": "starter",
                "posicao_id": int(r["posicao_id"]),
                "atleta_id": int(r["atleta_id"]),
                "nome": str(r.get("nome", "")),
                "apelido": str(r.get("apelido", "")),
                "clube_id": int(r.get("clube_id", 0)),
                "preco_num": float(r.get("preco_num", 0.0)),
                "score_pred": float(r.get("score_pred", 0.0)),
                "is_captain": 1 if (result.captain_id is not None and int(r["atleta_id"]) == int(result.captain_id)) else 0,
                "is_tecnico": 0,
                "is_reserve": 0,
                "reserva_setor": "",
                "is_luxury_reserve": 0,
            }
        )

    # tecnico
    if result.tecnico is not None:
        t = result.tecnico
        rows.append(
            {
                "team_id": team_idx,
                "slot": "tecnico",
                "posicao_id": int(t["posicao_id"]),
                "atleta_id": int(t["atleta_id"]),
                "nome": str(t.get("nome", "")),
                "apelido": str(t.get("apelido", "")),
                "clube_id": int(t.get("clube_id", 0)),
                "preco_num": float(t.get("preco_num", 0.0)),
                "score_pred": float(t.get("score_pred", 0.0)),
                "is_captain": 0,
                "is_tecnico": 1,
                "is_reserve": 0,
                "reserva_setor": "",
                "is_luxury_reserve": 0,
            }
        )

    # reserves
    if result.reserves is not None and not result.reserves.empty:
        for _, r in result.reserves.iterrows():
            aid = int(r["atleta_id"])
            rows.append(
                {
                    "team_id": team_idx,
                    "slot": "reserve",
                    "posicao_id": int(r["posicao_id"]),
                    "atleta_id": aid,
                    "nome": str(r.get("nome", "")),
                    "apelido": str(r.get("apelido", "")),
                    "clube_id": int(r.get("clube_id", 0)),
                    "preco_num": float(r.get("preco_num", 0.0)),
                    "score_pred": float(r.get("score_pred", 0.0)),
                    "is_captain": 0,
                    "is_tecnico": 0,
                    "is_reserve": 1,
                    "reserva_setor": str(r.get("reserva_setor", "")),
                    "is_luxury_reserve": 1 if (result.luxury_reserve_id is not None and aid == int(result.luxury_reserve_id)) else 0,
                }
            )

    return rows


# =========================
# CLI
# =========================
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Gerador de times Cartola 2026 (PRO)")
    p.add_argument("--api-base", default=DEFAULT_API_BASE, help="Base da API do Cartola")
    p.add_argument("--history-root", required=True, help="Pasta do histórico (ex.: cartola/data/01_raw)")
    p.add_argument("--years", nargs="+", required=True, help="Anos do histórico (ex.: 2023 2024 2025)")
    p.add_argument("--n-times", type=int, default=3, help="Quantidade de times")
    p.add_argument("--budget", type=float, default=100.0, help="Orçamento (C$)")
    p.add_argument("--formation", type=str, default="4-3-3", help=f"Formação ({', '.join(sorted(FORMATION_MAP.keys()))})")
    p.add_argument("--out", type=str, default="data/teams_2026.csv", help="Arquivo CSV de saída")
    p.add_argument("--allow-repeat", action="store_true", help="Permitir repetir atletas entre times")
    p.add_argument("--min-games", type=int, default=1, help="Mínimo de jogos no histórico para considerar atleta")
    p.add_argument("--shrink-k", type=int, default=5, help="Parâmetro de suavização do score histórico")
    p.add_argument("--require-complete", action="store_true", help="Falhar se não montar time completo (default: tenta)")
    return p

def main() -> int:
    args = build_argparser().parse_args()

    years = parse_years(args.years)
    n_times = int(args.n_times)
    if n_times < 1:
        err("n-times deve ser >= 1")
        return 2

    formation = str(args.formation).strip()
    _ = formation_to_counts(formation)  # valida

    budget = float(args.budget)
    if budget <= 0:
        err("budget deve ser > 0")
        return 2

    # 1) Treina modelo simples pelo histórico
    model = build_history_model(
        history_root=args.history_root,
        years=years,
        min_games=int(args.min_games),
        shrink_k=int(args.shrink_k),
    )

    # 2) Mercado 2026 via API
    info("Buscando mercado 2026 via API do Cartola...")
    try:
        mercado = fetch_mercado(args.api_base)
    except Exception as e:
        err(f"Falha ao acessar API do Cartola: {e}")
        return 3

    status = mercado.get("status_mercado", None)
    rodada = mercado.get("rodada_atual", None)
    temporada = mercado.get("temporada", None)

    info(f"API status: temporada={temporada} rodada_atual={rodada} status_mercado={status}")

    df_market = mercado_to_df(mercado)
    if df_market.empty:
        err("API retornou mercado vazio.")
        return 4

    # 3) Junta predição
    df_market = attach_predictions(df_market, model)

    # 4) Gera times
    used_global: Set[int] = set()
    out_rows: List[dict] = []
    summary_rows: List[dict] = []

    ok_count = 0
    fail_count = 0

    for i in range(1, n_times + 1):
        info(f"=== Gerando TIME {i}/{n_times} (formation={formation} budget={budget:.2f}) ===")
        res = generate_one_team(
            df_market=df_market,
            formation=formation,
            budget=budget,
            used_global=used_global,
            allow_repeat=bool(args.allow_repeat),
            require_complete=bool(args.require_complete),
        )
        if not res.ok:
            warn(f"TIME {i}: falhou: {res.reason}")
            fail_count += 1
            continue

        ok_count += 1
        out_rows.extend(team_to_rows(i, res))

        # summary
        captain_name = ""
        if res.captain_id is not None:
            cap = res.starters[res.starters["atleta_id"].astype(int) == int(res.captain_id)]
            if not cap.empty:
                captain_name = str(cap.iloc[0].get("apelido") or cap.iloc[0].get("nome") or "")

        tec_name = ""
        if res.tecnico is not None:
            tec_name = str(res.tecnico.get("apelido") or res.tecnico.get("nome") or "")

        lux_name = ""
        if res.luxury_reserve_id is not None and res.reserves is not None and not res.reserves.empty:
            lux = res.reserves[res.reserves["atleta_id"].astype(int) == int(res.luxury_reserve_id)]
            if not lux.empty:
                lux_name = str(lux.iloc[0].get("apelido") or lux.iloc[0].get("nome") or "")

        summary_rows.append(
            {
                "team_id": i,
                "formation": formation,
                "budget_total": res.budget_total,
                "budget_used": round(res.budget_used, 2),
                "budget_left": round(res.budget_left, 2),
                "score_total_pred": round(res.score_total, 4),
                "captain_id": res.captain_id or 0,
                "captain_name": captain_name,
                "tecnico_id": int(res.tecnico["atleta_id"]) if res.tecnico is not None else 0,
                "tecnico_name": tec_name,
                "luxury_reserve_id": res.luxury_reserve_id or 0,
                "luxury_reserve_name": lux_name,
                "n_starters": int(len(res.starters)),
                "n_reserves": int(len(res.reserves)) if res.reserves is not None else 0,
            }
        )

    if not out_rows:
        err("Nenhum time foi gerado. Tente aumentar budget, permitir repetição ou mudar formação.")
        return 5

    # 5) Salva CSV (detalhado) + summary
    out_path = str(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df_out = pd.DataFrame(out_rows)
    df_out.to_csv(out_path, index=False)

    summary_path = out_path.replace(".csv", "_summary.csv")
    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(summary_path, index=False)

    info("=== RESUMO ===")
    info(f"Gerados: ok={ok_count}/{n_times} falhas={fail_count}")
    info(f"Arquivo salvo: {out_path}")
    info(f"Resumo salvo:  {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
