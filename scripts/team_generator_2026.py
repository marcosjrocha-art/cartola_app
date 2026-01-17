import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import requests

# =========================
# Config
# =========================
CARTOLA_MERCADO_URL = "https://api.cartolafc.globo.com/atletas/mercado"
OUT_CSV_DEFAULT = "data/teams_2026.csv"

POS_MAP = {
    1: "GOL",
    2: "LAT",
    3: "ZAG",
    4: "MEI",
    5: "ATA",
    6: "TEC",
}

# Formação -> (ZAG, LAT, MEI, ATA)
FORMATION_MAP = {
    "3-4-3": (3, 0, 4, 3),
    "3-5-2": (3, 0, 5, 2),
    "4-3-3": (2, 2, 3, 3),
    "4-4-2": (2, 2, 4, 2),
    "4-5-1": (2, 2, 5, 1),
    "5-3-2": (3, 2, 3, 2),
    "5-4-1": (3, 2, 4, 1),
}

# Banco: 1 GOL + 1 DEF (LAT/ZAG) + 1 MEI + 1 ATA
BENCH_SPEC = ["GOL", "DEF", "MEI", "ATA"]

CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "mercado_2026.json"


# =========================
# Utils / Logs
# =========================
def info(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[WARN] {msg}", flush=True)


def err(msg: str) -> None:
    print(f"[ERRO] {msg}", flush=True)


def clamp(x: float, a: float, b: float) -> float:
    return max(a, min(b, x))


@dataclass(frozen=True)
class Player:
    atleta_id: int
    apelido: str
    posicao: str
    clube_id: int
    preco: float
    score: float


def fetch_mercado(use_cache: bool = True, timeout: int = 20) -> dict:
    """
    Busca mercado da API do Cartola.
    Cache em disco para estabilidade.
    """
    if use_cache and CACHE_FILE.exists():
        try:
            data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
            return data
        except Exception:
            pass

    r = requests.get(CARTOLA_MERCADO_URL, timeout=timeout)
    r.raise_for_status()
    data = r.json()

    try:
        CACHE_FILE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

    return data


def normalize_scores(players: List[Player]) -> List[Player]:
    """
    Normaliza score em [0, 1] só pra usar como ranking estável,
    mantendo a ordem relativa (não muda "quem é melhor", apenas escala).
    """
    if not players:
        return players
    scores = [p.score for p in players]
    mn, mx = min(scores), max(scores)
    if mx - mn < 1e-9:
        return players
    out = []
    for p in players:
        ns = (p.score - mn) / (mx - mn)
        out.append(Player(p.atleta_id, p.apelido, p.posicao, p.clube_id, p.preco, ns))
    return out


def build_player_pool(api_data: dict) -> List[Player]:
    """
    Constrói lista de Player a partir do payload do Cartola.
    Heurística de score "aprendida": usa principalmente média (media_num) e jogos,
    com pequenos ajustes para preço e variação.
    """
    atletas = api_data.get("atletas", [])
    if not atletas:
        raise RuntimeError("API retornou mercado vazio.")

    players: List[Player] = []

    for a in atletas:
        try:
            pos_id = int(a.get("posicao_id", 0))
            pos = POS_MAP.get(pos_id)
            if not pos:
                continue

            status_id = int(a.get("status_id", 0))  # 7 costuma ser "provável"
            # Se quiser ser mais agressivo: comente o filtro abaixo
            if status_id not in (7,):
                continue

            atleta_id = int(a.get("atleta_id"))
            apelido = str(a.get("apelido") or a.get("nome") or f"ID{atleta_id}")
            clube_id = int(a.get("clube_id") or 0)

            preco = float(a.get("preco_num") or 0.0)
            media = float(a.get("media_num") or 0.0)
            jogos = float(a.get("jogos_num") or 0.0)
            variacao = float(a.get("variacao_num") or 0.0)

            # ===== Score "aprendido" (heurística robusta) =====
            # Base: média
            score = media

            # Confiança por jogos (satura rápido)
            score *= (0.75 + 0.25 * clamp(jogos / 10.0, 0.0, 1.0))

            # Leve bônus por tendência positiva
            score += 0.05 * variacao

            # Penaliza muito caro (pra caber orçamento) sem matar craques
            score -= 0.01 * max(0.0, preco - 10.0)

            players.append(Player(atleta_id, apelido, pos, clube_id, preco, score))
        except Exception:
            continue

    if not players:
        raise RuntimeError("Após filtros, não sobraram atletas no mercado (status/prováveis).")

    return players


def pos_group(pos: str) -> str:
    if pos in ("LAT", "ZAG"):
        return "DEF"
    return pos


def best_pick(
    candidates: List[Player],
    used_ids: Set[int],
    used_clubes: Optional[Dict[int, int]],
    budget_left: float,
    allow_repeat: bool,
    max_per_clube: int = 3,
) -> Optional[Player]:
    """
    Escolhe 1 jogador maximizando score/(preco+1) dentro do orçamento e restrições.
    """
    best = None
    best_val = -1e18

    for p in candidates:
        if p.atleta_id in used_ids:
            continue
        if p.preco > budget_left + 1e-9:
            continue
        if (not allow_repeat) and used_clubes is not None:
            cnt = used_clubes.get(p.clube_id, 0)
            if p.clube_id != 0 and cnt >= max_per_clube:
                continue

        val = p.score / (p.preco + 1.0)
        if val > best_val:
            best_val = val
            best = p

    return best


def pick_many(
    pool: List[Player],
    need: int,
    used_ids: Set[int],
    used_clubes: Optional[Dict[int, int]],
    budget_left: float,
    allow_repeat: bool,
    max_per_clube: int = 3,
) -> Tuple[List[Player], float]:
    """
    Pega N jogadores de um pool, respeitando orçamento, usando greedy por score/(preco+1).
    """
    chosen: List[Player] = []
    remaining = budget_left

    for _ in range(need):
        p = best_pick(pool, used_ids, used_clubes, remaining, allow_repeat, max_per_clube=max_per_clube)
        if p is None:
            break
        chosen.append(p)
        used_ids.add(p.atleta_id)
        if used_clubes is not None and p.clube_id != 0:
            used_clubes[p.clube_id] = used_clubes.get(p.clube_id, 0) + 1
        remaining -= p.preco

    return chosen, remaining


def choose_captain(starters: List[Player]) -> Optional[int]:
    """
    Capitão = maior score entre titulares (exclui TEC por construção).
    """
    if not starters:
        return None
    return max(starters, key=lambda p: p.score).atleta_id


def team_from_market(
    players: List[Player],
    formation: str,
    budget: float,
    allow_repeat: bool,
    used_global: Set[int],
    bench_budget_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Monta 1 time:
    - time principal: 11 + TEC (12)
    - banco: 4 reservas (GOL, DEF, MEI, ATA)
    - luxo: 1 reserva extra (qualquer, sem TEC)
    """
    if formation not in FORMATION_MAP:
        raise ValueError(f"Formação inválida: {formation}")

    zag_n, lat_n, mei_n, ata_n = FORMATION_MAP[formation]

    # Pools por posição
    gol = [p for p in players if p.posicao == "GOL"]
    lat = [p for p in players if p.posicao == "LAT"]
    zag = [p for p in players if p.posicao == "ZAG"]
    mei = [p for p in players if p.posicao == "MEI"]
    ata = [p for p in players if p.posicao == "ATA"]
    tec = [p for p in players if p.posicao == "TEC"]

    # Ordena por score desc (melhor estabilidade)
    for arr in (gol, lat, zag, mei, ata, tec):
        arr.sort(key=lambda p: p.score, reverse=True)

    # Restrição: máximo 3 por clube (padrão Cartola)
    used_clubes: Dict[int, int] = {}

    # Se não permitir repetição entre times, já bloqueia global
    used_ids = set(used_global) if not allow_repeat else set()

    # Budget split: principal + banco (pra caber com 100 C$)
    budget_main = budget * (1.0 - bench_budget_ratio)
    budget_bench = budget - budget_main

    # ===== Time principal =====
    chosen_main: List[Player] = []

    # 1 GOL
    x, budget_main = pick_many(gol, 1, used_ids, used_clubes, budget_main, allow_repeat=False)
    chosen_main += x

    # DEF (ZAG + LAT conforme formação)
    x, budget_main = pick_many(zag, zag_n, used_ids, used_clubes, budget_main, allow_repeat=False)
    chosen_main += x
    if lat_n > 0:
        x, budget_main = pick_many(lat, lat_n, used_ids, used_clubes, budget_main, allow_repeat=False)
        chosen_main += x

    # MEI / ATA
    x, budget_main = pick_many(mei, mei_n, used_ids, used_clubes, budget_main, allow_repeat=False)
    chosen_main += x
    x, budget_main = pick_many(ata, ata_n, used_ids, used_clubes, budget_main, allow_repeat=False)
    chosen_main += x

    # TEC
    x, budget_main = pick_many(tec, 1, used_ids, used_clubes, budget_main, allow_repeat=False)
    chosen_main += x

    if len(chosen_main) < (1 + zag_n + lat_n + mei_n + ata_n + 1):
        raise RuntimeError("Não conseguiu montar o time principal dentro do orçamento/mercado (faltou posição).")

    # Split starters vs coach
    coach = [p for p in chosen_main if p.posicao == "TEC"][0]
    starters = [p for p in chosen_main if p.posicao != "TEC"]

    captain_id = choose_captain(starters)

    # ===== Banco =====
    chosen_bench: List[Player] = []
    # 1 GOL
    x, budget_bench = pick_many(gol, 1, used_ids, None, budget_bench, allow_repeat=True)
    chosen_bench += x

    # 1 DEF (pega melhor entre LAT/ZAG)
    def_pool = sorted(lat + zag, key=lambda p: p.score, reverse=True)
    x, budget_bench = pick_many(def_pool, 1, used_ids, None, budget_bench, allow_repeat=True)
    chosen_bench += x

    # 1 MEI
    x, budget_bench = pick_many(mei, 1, used_ids, None, budget_bench, allow_repeat=True)
    chosen_bench += x

    # 1 ATA
    x, budget_bench = pick_many(ata, 1, used_ids, None, budget_bench, allow_repeat=True)
    chosen_bench += x

    # Se não conseguiu algum reserva por orçamento, tenta completar com mais baratos do mercado (qualquer pos exceto TEC)
    if len(chosen_bench) < 4:
        warn("Banco incompleto no orçamento. Tentando completar com jogadores baratos...")
        any_no_tec = [p for p in players if p.posicao != "TEC"]
        any_no_tec.sort(key=lambda p: (p.preco, -p.score))
        need = 4 - len(chosen_bench)
        for p in any_no_tec:
            if need <= 0:
                break
            if p.atleta_id in used_ids:
                continue
            if p.preco > budget_bench + 1e-9:
                continue
            chosen_bench.append(p)
            used_ids.add(p.atleta_id)
            budget_bench -= p.preco
            need -= 1

    if len(chosen_bench) < 4:
        warn("Mesmo assim banco ficou incompleto (mercado/orçamento).")

    # ===== Reserva de Luxo =====
    luxury = None
    luxury_pool = [p for p in players if p.posicao != "TEC"]
    luxury_pool.sort(key=lambda p: p.score, reverse=True)
    for p in luxury_pool:
        if p.atleta_id in used_ids:
            continue
        if p.preco <= budget_bench + 1e-9:
            luxury = p
            used_ids.add(p.atleta_id)
            budget_bench -= p.preco
            break

    # Atualiza global (se não permitir repetição entre times)
    if not allow_repeat:
        used_global.update(used_ids)

    # ===== Totais =====
    preco_main = sum(p.preco for p in chosen_main)
    preco_bench = sum(p.preco for p in chosen_bench) + (luxury.preco if luxury else 0.0)
    preco_total = preco_main + preco_bench

    # Score total considera CAPITÃO dobrado (apenas titulares, sem banco)
    score_starters = sum(p.score for p in starters)
    score_coach = coach.score
    score_capt = 0.0
    if captain_id is not None:
        cap = next((p for p in starters if p.atleta_id == captain_id), None)
        if cap:
            score_capt = cap.score  # extra (dobro)
    score_total = score_starters + score_coach + score_capt

    # ===== Saída tabular =====
    rows = []

    def add_rows(role: str, arr: List[Player], is_lux: bool = False) -> None:
        for p in arr:
            rows.append({
                "role": role,
                "is_luxury": bool(is_lux),
                "is_captain": bool(p.atleta_id == captain_id and role == "starter"),
                "atleta_id": p.atleta_id,
                "apelido": p.apelido,
                "posicao": p.posicao,
                "clube_id": p.clube_id,
                "preco": float(p.preco),
                "score": float(p.score),
                # Totais repetidos por linha (facilita no app)
                "preco_main": float(preco_main),
                "preco_bench": float(preco_bench),
                "preco_total": float(preco_total),
                "score_total": float(score_total),
            })

    # starters (sem TEC)
    add_rows("starter", starters, is_lux=False)
    # coach
    add_rows("coach", [coach], is_lux=False)
    # bench
    add_rows("bench", chosen_bench, is_lux=False)
    # luxury
    if luxury:
        add_rows("bench", [luxury], is_lux=True)

    meta = {
        "preco_main": preco_main,
        "preco_bench": preco_bench,
        "preco_total": preco_total,
        "score_total": score_total,
        "captain_id": float(captain_id) if captain_id is not None else math.nan,
    }

    return pd.DataFrame(rows), meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-times", type=int, default=3)
    ap.add_argument("--budget", type=float, default=100.0)
    ap.add_argument("--formation", type=str, default="4-3-3")
    ap.add_argument("--allow-repeat", action="store_true")
    ap.add_argument("--out", type=str, default=OUT_CSV_DEFAULT)
    ap.add_argument("--bench-budget-ratio", type=float, default=0.15)
    args = ap.parse_args()

    n_times = int(args.n_times)
    if n_times < 1:
        raise SystemExit("n-times precisa ser >= 1")

    budget = float(args.budget)
    if budget <= 0:
        raise SystemExit("budget precisa ser > 0")

    bench_ratio = float(args.bench_budget_ratio)
    bench_ratio = clamp(bench_ratio, 0.0, 0.40)

    info("Baixando mercado 2026 via API do Cartola...")
    api_data = fetch_mercado(use_cache=True)
    status = api_data.get("status_mercado", {})
    info(f"API status: temporada={status.get('temporada')} rodada_atual={status.get('rodada_atual')}")

    info("Montando pool de atletas prováveis...")
    players = build_player_pool(api_data)
    info(f"Atletas disponíveis (filtrados): {len(players)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    used_global: Set[int] = set()
    all_rows = []
    ok = 0
    fails = 0

    for tid in range(1, n_times + 1):
        try:
            df_team, meta = team_from_market(
                players=players,
                formation=args.formation,
                budget=budget,
                allow_repeat=args.allow_repeat,
                used_global=used_global,
                bench_budget_ratio=bench_ratio,
            )
            df_team.insert(0, "team_id", tid)
            all_rows.append(df_team)
            ok += 1
            info(f"Time {tid}: OK | preco_total={meta['preco_total']:.2f} score_total={meta['score_total']:.2f}")
        except Exception as e:
            fails += 1
            warn(f"Time {tid}: falhou ({e})")

    if ok == 0:
        raise RuntimeError("Nenhum time foi gerado. Ajuste orçamento/formação/filtro do mercado.")

    df_out = pd.concat(all_rows, ignore_index=True)
    df_out.to_csv(out_path, index=False)

    info("=== RESUMO ===")
    info(f"Gerados: {ok}/{n_times} | falhas={fails}")
    info(f"Arquivo salvo: {out_path}")


if __name__ == "__main__":
    main()
