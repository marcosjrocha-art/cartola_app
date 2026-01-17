import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


# =========================
# Logs
# =========================
def info(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[WARN] {msg}", flush=True)


def err(msg: str) -> None:
    print(f"[ERRO] {msg}", flush=True)


# =========================
# Config / helpers
# =========================
POS_ID_TO_NAME = {
    1: "GOL",
    2: "LAT",
    3: "ZAG",
    4: "MEI",
    5: "ATA",
    6: "TEC",
}

FORMATION_MAP = {
    "3-4-3": (3, 0, 4, 3),
    "3-5-2": (3, 0, 5, 2),
    "4-3-3": (2, 2, 3, 3),
    "4-4-2": (2, 2, 4, 2),
    "4-5-1": (2, 2, 5, 1),
    "5-3-2": (3, 2, 3, 2),
    "5-4-1": (3, 2, 4, 1),
}

# Banco (pro): 1 GOL + 1 DEF (LAT/ZAG) + 1 MEI + 1 ATA + LUXO (qualquer exceto TEC)
BENCH_BASE = ["GOL", "DEF", "MEI", "ATA"]


def pos_name_from_row(df: pd.DataFrame) -> pd.Series:
    # Prefer posicao_id se existir
    if "atletas.posicao_id" in df.columns:
        return df["atletas.posicao_id"].map(POS_ID_TO_NAME).fillna("UNK")
    # fallback
    return pd.Series(["UNK"] * len(df))


def normalize_def(pos: str) -> str:
    return "DEF" if pos in ("LAT", "ZAG") else pos


def safe_float(x, default=0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, float) and math.isnan(x):
            return default
        return float(x)
    except Exception:
        return default


def find_round_files(data_root: Path, year: int) -> List[Path]:
    # esperado: {data_root}/01_raw/{year}/rodada-1.csv ...
    base = data_root / "01_raw" / str(year)
    if not base.exists():
        return []
    files = sorted(base.glob("rodada-*.csv"), key=lambda p: int(p.stem.split("-")[1]))
    return files


def get_round_id_from_path(p: Path) -> int:
    # rodada-12.csv -> 12
    return int(p.stem.split("-")[1])


# =========================
# Modelo simples (heurístico) para "prever"
# =========================
def fit_simple_model(hist: pd.DataFrame) -> Dict[str, float]:
    """
    Modelo simples e robusto (sem depender de sklearn):
    score_pred = w_media * media_num + w_jogos * jogos_num + w_preco * preco_num + w_minval * minimo_para_valorizar + bias
    """
    # Colunas candidatas (histórico)
    candidates = {
        "media": "atletas.media_num",
        "jogos": "atletas.jogos_num",
        "preco": "atletas.preco_num",
        "minval": "atletas.minimo_para_valorizar",
    }

    X = {}
    for k, col in candidates.items():
        X[k] = hist[col] if col in hist.columns else pd.Series([0.0] * len(hist))

    y_col = "atletas.pontos_num"
    y = hist[y_col] if y_col in hist.columns else pd.Series([0.0] * len(hist))

    # limpa NaN
    df = pd.DataFrame({k: pd.to_numeric(v, errors="coerce").fillna(0.0) for k, v in X.items()})
    y = pd.to_numeric(y, errors="coerce").fillna(0.0)

    # pesos fixos (funciona bem o suficiente e não quebra):
    # - media pesa mais
    # - jogos dá leve confiança
    # - preco penaliza um pouco
    # - minimo_para_valorizar dá leve sinal
    weights = {
        "media": 1.00,
        "jogos": 0.05,
        "preco": -0.02,
        "minval": 0.03,
        "bias": 0.00,
    }

    # Ajuste leve pelo “alvo” do dataset (se tudo muito negativo/positivo)
    # só pra centralizar (evita distorção em anos com pontuação média diferente)
    y_mean = float(y.mean()) if len(y) else 0.0
    weights["bias"] = 0.15 * y_mean

    return weights


def predict_score(row: pd.Series, w: Dict[str, float]) -> float:
    media = safe_float(row.get("atletas.media_num", 0.0))
    jogos = safe_float(row.get("atletas.jogos_num", 0.0))
    preco = safe_float(row.get("atletas.preco_num", 0.0))
    minval = safe_float(row.get("atletas.minimo_para_valorizar", 0.0))
    return w["bias"] + w["media"] * media + w["jogos"] * jogos + w["preco"] * preco + w["minval"] * minval


# =========================
# Escalação (greedy)
# =========================
@dataclass(frozen=True)
class Pick:
    atleta_id: int
    apelido: str
    pos: str            # GOL/LAT/ZAG/MEI/ATA/TEC
    pos_group: str      # GOL/DEF/MEI/ATA/TEC
    clube_id: int
    preco: float
    pred: float         # previsão usada na escalação


def pick_team_and_bench(
    mercado_df: pd.DataFrame,
    formation: str,
    budget: float,
    allow_repeat: bool,
    used_global: Set[int],
    top_pool: int,
) -> Tuple[List[Pick], List[Pick], Optional[int]]:
    """
    Retorna:
    - titulares+tec (12 Picks) (role principal)
    - banco base + luxo (5 Picks; TEC não entra no banco)
    - captain_id (maior pred entre titulares exceto TEC)
    """
    if formation not in FORMATION_MAP:
        raise ValueError(f"Formação inválida: {formation}")

    zag_n, lat_n, mei_n, ata_n = FORMATION_MAP[formation]

    # gera pred
    w = fit_simple_model(mercado_df)
    df = mercado_df.copy()
    df["pred"] = df.apply(lambda r: predict_score(r, w), axis=1)

    # base ids/clubes
    used_ids: Set[int] = set(used_global) if not allow_repeat else set()
    club_count: Dict[int, int] = {}

    def can_take(row) -> bool:
        aid = int(row["atletas.atleta_id"])
        if aid in used_ids:
            return False
        preco = safe_float(row.get("atletas.preco_num", 0.0))
        if preco < 0:
            return False
        clube = int(row.get("atletas.clube_id", 0) or 0)
        if clube != 0 and club_count.get(clube, 0) >= 3:
            return False
        return True

    def take(row) -> Pick:
        aid = int(row["atletas.atleta_id"])
        used_ids.add(aid)
        clube = int(row.get("atletas.clube_id", 0) or 0)
        if clube != 0:
            club_count[clube] = club_count.get(clube, 0) + 1
        pos = POS_ID_TO_NAME.get(int(row["atletas.posicao_id"]), "UNK")
        pg = normalize_def(pos)
        return Pick(
            atleta_id=aid,
            apelido=str(row.get("atletas.apelido") or row.get("atletas.nome") or f"ID{aid}"),
            pos=pos,
            pos_group=pg,
            clube_id=clube,
            preco=safe_float(row.get("atletas.preco_num", 0.0)),
            pred=safe_float(row.get("pred", 0.0)),
        )

    # pools por pos
    df = df[df["atletas.posicao_id"].isin([1,2,3,4,5,6])].copy()
    df = df.sort_values("pred", ascending=False).head(max(1, int(top_pool)))

    pools = {
        "GOL": df[df["atletas.posicao_id"] == 1],
        "LAT": df[df["atletas.posicao_id"] == 2],
        "ZAG": df[df["atletas.posicao_id"] == 3],
        "MEI": df[df["atletas.posicao_id"] == 4],
        "ATA": df[df["atletas.posicao_id"] == 5],
        "TEC": df[df["atletas.posicao_id"] == 6],
    }

    budget_left = float(budget)
    picks_main: List[Pick] = []

    def greedy_pick(pos_key: str, need: int) -> None:
        nonlocal budget_left, picks_main
        if need <= 0:
            return
        pool = pools[pos_key]
        for _, row in pool.iterrows():
            if need <= 0:
                break
            preco = safe_float(row.get("atletas.preco_num", 0.0))
            if preco > budget_left + 1e-9:
                continue
            if not can_take(row):
                continue
            picks_main.append(take(row))
            budget_left -= preco
            need -= 1
        if need > 0:
            raise RuntimeError(f"Não conseguiu completar posição {pos_key} (faltou {need}).")

    # titulares + TEC
    greedy_pick("GOL", 1)
    greedy_pick("ZAG", zag_n)
    greedy_pick("LAT", lat_n)
    greedy_pick("MEI", mei_n)
    greedy_pick("ATA", ata_n)
    greedy_pick("TEC", 1)

    # capitão = maior pred (exceto TEC)
    starters = [p for p in picks_main if p.pos != "TEC"]
    captain_id = max(starters, key=lambda p: p.pred).atleta_id if starters else None

    # ===== Banco PRO =====
    # Tentamos manter orçamento suficiente pro banco, mas sem travar:
    # - se faltar grana, pega mais barato no mercado (ainda respeitando used_ids)
    bench: List[Pick] = []

    # cria um pool “amplo” (não só top_pool) pra banco barato funcionar
    df2 = mercado_df.copy()
    w2 = fit_simple_model(df2)
    df2["pred"] = df2.apply(lambda r: predict_score(r, w2), axis=1)
    df2 = df2[df2["atletas.posicao_id"].isin([1,2,3,4,5,6])].copy()

    def pick_bench_from_group(group: str) -> None:
        nonlocal budget_left, bench
        # group: GOL/DEF/MEI/ATA
        if group == "DEF":
            pool = df2[df2["atletas.posicao_id"].isin([2,3])].copy()
        elif group == "GOL":
            pool = df2[df2["atletas.posicao_id"] == 1].copy()
        elif group == "MEI":
            pool = df2[df2["atletas.posicao_id"] == 4].copy()
        elif group == "ATA":
            pool = df2[df2["atletas.posicao_id"] == 5].copy()
        else:
            pool = df2.copy()

        # para banco: ordena por (pred/preco) e tenta caber
        pool["val"] = pool["pred"] / (pd.to_numeric(pool["atletas.preco_num"], errors="coerce").fillna(0.0) + 1.0)
        pool = pool.sort_values("val", ascending=False)

        for _, row in pool.iterrows():
            preco = safe_float(row.get("atletas.preco_num", 0.0))
            if preco > budget_left + 1e-9:
                continue
            if not can_take(row):
                continue
            p = take(row)
            if group == "DEF" and normalize_def(p.pos) != "DEF":
                continue
            if group != "DEF" and p.pos != group:
                continue
            bench.append(p)
            budget_left -= preco
            return

        # fallback barato
        pool2 = pool.sort_values("atletas.preco_num", ascending=True)
        for _, row in pool2.iterrows():
            preco = safe_float(row.get("atletas.preco_num", 0.0))
            if preco > budget_left + 1e-9:
                continue
            if not can_take(row):
                continue
            p = take(row)
            if group == "DEF" and normalize_def(p.pos) != "DEF":
                continue
            if group != "DEF" and p.pos != group:
                continue
            bench.append(p)
            budget_left -= preco
            return

        warn(f"Não consegui pegar reserva {group} (orçamento/mercado).")

    for g in BENCH_BASE:
        pick_bench_from_group(g)

    # Luxo: melhor pred (qualquer exceto TEC) que caiba
    luxo = None
    pool_luxo = df2[df2["atletas.posicao_id"].isin([1,2,3,4,5])].copy()
    pool_luxo["val"] = pool_luxo["pred"] / (pd.to_numeric(pool_luxo["atletas.preco_num"], errors="coerce").fillna(0.0) + 1.0)
    pool_luxo = pool_luxo.sort_values("val", ascending=False)

    for _, row in pool_luxo.iterrows():
        preco = safe_float(row.get("atletas.preco_num", 0.0))
        if preco > budget_left + 1e-9:
            continue
        if not can_take(row):
            continue
        luxo = take(row)
        budget_left -= preco
        break

    if luxo is not None:
        bench.append(luxo)

    # atualiza global
    if not allow_repeat:
        used_global.update(used_ids)

    return picks_main, bench, captain_id


# =========================
# Scoring PRO (substituições)
# =========================
def round_points_map(round_df: pd.DataFrame) -> Dict[int, float]:
    """
    Mapa atleta_id -> pontos_num (float)
    Se pontos_num vier NaN, tratamos como ausente.
    """
    out: Dict[int, float] = {}
    if "atletas.atleta_id" not in round_df.columns:
        return out
    pts_col = "atletas.pontos_num"
    if pts_col not in round_df.columns:
        # sem pontos, zero tudo
        for aid in round_df["atletas.atleta_id"].dropna().astype(int).tolist():
            out[aid] = 0.0
        return out

    for _, r in round_df.iterrows():
        try:
            aid = int(r["atletas.atleta_id"])
            pts = pd.to_numeric(r.get(pts_col), errors="coerce")
            if pd.isna(pts):
                # não marca (ausente)
                continue
            out[aid] = float(pts)
        except Exception:
            continue
    return out


def did_play(aid: int, pts_map: Dict[int, float]) -> bool:
    # regra robusta: se não está no mapa => consideramos "não jogou"
    return aid in pts_map


def apply_pro_substitutions(
    starters: List[Pick],
    bench: List[Pick],
    pts_map: Dict[int, float],
) -> Tuple[List[Tuple[Pick, float]], List[str]]:
    """
    Retorna:
    - lista final de "quem pontuou" (Pick, pontos)
    - logs do que substituiu
    """
    logs: List[str] = []
    used_bench_ids: Set[int] = set()

    # separa reservas por grupo
    bench_by_group: Dict[str, List[Pick]] = {"GOL": [], "DEF": [], "MEI": [], "ATA": [], "ANY": []}
    for b in bench:
        if b.pos == "TEC":
            continue
        g = normalize_def(b.pos)
        if g not in bench_by_group:
            bench_by_group[g] = []
        bench_by_group[g].append(b)
        bench_by_group["ANY"].append(b)

    # ordena reservas por melhor pontuação REAL na rodada (se disponível), senão 0
    def bench_sort_key(p: Pick) -> float:
        return pts_map.get(p.atleta_id, -9999.0)

    for k in bench_by_group:
        bench_by_group[k].sort(key=bench_sort_key, reverse=True)

    final: List[Tuple[Pick, float]] = []
    missing: List[Pick] = []

    # 1) marca quem jogou / quem não jogou
    for s in starters:
        if did_play(s.atleta_id, pts_map):
            final.append((s, pts_map[s.atleta_id]))
        else:
            missing.append(s)

    # 2) tenta substituir por posição
    def pop_best(group: str) -> Optional[Pick]:
        arr = bench_by_group.get(group, [])
        for p in arr:
            if p.atleta_id in used_bench_ids:
                continue
            if not did_play(p.atleta_id, pts_map):
                continue
            used_bench_ids.add(p.atleta_id)
            return p
        return None

    still_missing: List[Pick] = []
    for s in missing:
        g = normalize_def(s.pos)
        rep = pop_best(g)
        if rep is not None:
            final.append((rep, pts_map[rep.atleta_id]))
            logs.append(f"Substituiu {s.pos} {s.apelido} (não jogou) -> {rep.pos} {rep.apelido}")
        else:
            still_missing.append(s)

    # 3) reserva de luxo / qualquer restante (melhor pontuação disponível)
    for s in still_missing:
        rep = pop_best("ANY")
        if rep is not None:
            final.append((rep, pts_map[rep.atleta_id]))
            logs.append(f"Luxo/qualquer: {s.pos} {s.apelido} (não jogou) -> {rep.pos} {rep.apelido}")
        else:
            # sem substituto, fica zerado
            final.append((s, 0.0))
            logs.append(f"Sem substituto: {s.pos} {s.apelido} => 0.0")

    return final, logs


def compute_team_points_pro(
    picks_main: List[Pick],
    bench: List[Pick],
    captain_id: Optional[int],
    round_df: pd.DataFrame,
) -> Tuple[float, Dict[str, int], List[str]]:
    """
    Pontuação PRO:
    - Titulares pontuam se jogaram; se não, entram reservas por posição.
    - Capitão dobra (somente se o capitão jogou no final)
    - Técnico sempre conta se existir pontuação; se não tiver no mapa => 0.
    """
    pts_map = round_points_map(round_df)

    coach = next((p for p in picks_main if p.pos == "TEC"), None)
    starters = [p for p in picks_main if p.pos != "TEC"]

    final_players, logs = apply_pro_substitutions(starters, bench, pts_map)

    total = 0.0
    used_count = {
        "starters_used": 0,
        "bench_used": 0,
        "missing_after": 0,
    }

    final_ids = [p.atleta_id for (p, _) in final_players]
    starter_ids = set(p.atleta_id for p in starters)
    used_count["starters_used"] = sum(1 for x in final_ids if x in starter_ids)
    used_count["bench_used"] = len(final_ids) - used_count["starters_used"]

    # soma titulares (com substituições)
    pts_sum = 0.0
    cap_extra = 0.0
    cap_final_played = False

    for p, pts in final_players:
        pts_sum += pts
        if captain_id is not None and p.atleta_id == captain_id:
            cap_final_played = True
            cap_extra = pts  # dobra = +1x
    total += pts_sum
    if cap_final_played:
        total += cap_extra

    # técnico
    if coach is not None:
        total += pts_map.get(coach.atleta_id, 0.0)

    # missing_after = quantos terminaram com 0 por falta total de substituição E titular ausente
    used_count["missing_after"] = sum(1 for p, pts in final_players if pts == 0.0 and (p.atleta_id in starter_ids) and (p.atleta_id not in pts_map))

    return total, used_count, logs


# =========================
# Backtest runner
# =========================
def run_backtest_year(
    data_root: Path,
    year: int,
    formation: str,
    budget: float,
    n_teams: int,
    allow_repeat: bool,
    top_pool: int,
    verbose: bool,
) -> Tuple[float, float, float, int, int, int]:
    """
    Para cada rodada:
    - usa o "mercado" = rodada atual como base de features
    - monta N times
    - pontua com lógica PRO (reservas entram se titular não jogou)
    Retorna:
    team1_total, team2_total, team3_total, mean_total, rounds_ok, rounds_failed
    (os 3 primeiros só fazem sentido se n_teams>=3, mas mantemos compatibilidade)
    """
    files = find_round_files(data_root, year)
    if not files:
        raise RuntimeError(f"Nenhuma rodada encontrada para {year} em {data_root}.")

    used_global: Set[int] = set()
    team_totals = [0.0 for _ in range(n_teams)]
    rounds_ok = 0
    rounds_failed = 0

    for rf in files:
        r = get_round_id_from_path(rf)
        try:
            rodada_df = pd.read_csv(rf)
            # garante colunas mínimas
            needed = ["atletas.atleta_id", "atletas.posicao_id", "atletas.clube_id", "atletas.preco_num", "atletas.apelido", "atletas.media_num", "atletas.jogos_num"]
            for c in needed:
                if c not in rodada_df.columns:
                    # não trava: cria coluna vazia
                    rodada_df[c] = 0

            # monta times (usando essa rodada como “mercado”)
            for t in range(n_teams):
                picks_main, bench, captain_id = pick_team_and_bench(
                    mercado_df=rodada_df,
                    formation=formation,
                    budget=budget,
                    allow_repeat=allow_repeat,
                    used_global=used_global,
                    top_pool=top_pool,
                )
                pts, counts, logs = compute_team_points_pro(
                    picks_main=picks_main,
                    bench=bench,
                    captain_id=captain_id,
                    round_df=rodada_df,
                )
                team_totals[t] += pts

                if verbose and (counts["bench_used"] > 0 or counts["missing_after"] > 0):
                    info(f"{year} R{r:02d} T{t+1}: pts={pts:.2f} starters={counts['starters_used']} bench={counts['bench_used']} missing={counts['missing_after']}")
                    for lg in logs[:10]:
                        warn("  " + lg)

            rounds_ok += 1
        except Exception as e:
            rounds_failed += 1
            warn(f"{year} R{r}: falhou ({e})")

    mean_total = float(sum(team_totals) / max(1, n_teams))
    # compat: devolve 3 colunas, preenchendo com 0 se não existir
    t1 = team_totals[0] if n_teams >= 1 else 0.0
    t2 = team_totals[1] if n_teams >= 2 else 0.0
    t3 = team_totals[2] if n_teams >= 3 else 0.0
    return t1, t2, t3, mean_total, rounds_ok, rounds_failed, len(files)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="cartola/data")
    ap.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025])
    ap.add_argument("--formation", type=str, default="4-3-3")
    ap.add_argument("--budget", type=float, default=100.0)
    ap.add_argument("--n-teams", type=int, default=3)
    ap.add_argument("--top-pool", type=int, default=300)
    ap.add_argument("--allow-repeat", action="store_true")
    ap.add_argument("--out", type=str, default="data/backtest_seasons.csv")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    years = list(args.years)
    n_teams = int(args.n_teams)
    if n_teams < 1:
        raise SystemExit("n-teams precisa ser >= 1")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    info(f"DATA_ROOT: {data_root.resolve()}")
    info(f"Config: years={years} formation={args.formation} budget={args.budget} n_teams={n_teams} top_pool={args.top_pool} allow_repeat={args.allow_repeat}")

    rows = []
    for y in years:
        info(f"=== BACKTEST {y} ===")
        t1, t2, t3, mean_total, rounds_ok, rounds_failed, rounds_found = run_backtest_year(
            data_root=data_root,
            year=int(y),
            formation=args.formation,
            budget=float(args.budget),
            n_teams=n_teams,
            allow_repeat=bool(args.allow_repeat),
            top_pool=int(args.top_pool),
            verbose=bool(args.verbose),
        )
        info(f"{y}: rodadas_encontradas={rounds_found} ok={rounds_ok} falhas={rounds_failed} mean={mean_total:.2f}")

        rows.append({
            "year": float(y),
            "team_1": float(t1),
            "team_2": float(t2),
            "team_3": float(t3),
            "mean": float(mean_total),
            "rounds_ok": float(rounds_ok),
            "rounds_failed": float(rounds_failed),
            "n_teams": float(n_teams),
            "formation": args.formation,
            "budget": float(args.budget),
            "top_pool": float(args.top_pool),
            "allow_repeat": float(1.0 if args.allow_repeat else 0.0),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    info("=== BACKTEST FINALIZADO (PRO) ===")
    info(f"Arquivo salvo (rodadas): {out_path}")
    print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
