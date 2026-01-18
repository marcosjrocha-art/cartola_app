#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd

try:
    import requests
except Exception:
    requests = None


API_ESQUEMAS_URL = "https://api.cartolafc.globo.com/esquemas"

FALLBACK_ESQUEMAS = {
    "3-4-3": 1,
    "3-5-2": 2,
    "4-3-3": 3,
    "4-4-2": 4,
    "4-5-1": 5,
    "5-3-2": 6,
    "5-4-1": 7,
}


def fetch_esquemas_map(timeout: float = 8.0) -> Dict[str, int]:
    if requests is None:
        return dict(FALLBACK_ESQUEMAS)

    try:
        r = requests.get(API_ESQUEMAS_URL, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        out = {}
        for e in data.get("esquemas", []):
            nome = str(e.get("nome", "")).strip()
            eid = e.get("esquema_id")
            if nome and isinstance(eid, int):
                out[nome] = eid
        return out or dict(FALLBACK_ESQUEMAS)
    except Exception:
        return dict(FALLBACK_ESQUEMAS)


def split_roles(team_df: pd.DataFrame):
    """
    Retorna (starters, bench, coach) de forma robusta
    """
    if "role" in team_df.columns:
        starters = team_df[team_df["role"] == "starter"].copy()
        bench = team_df[team_df["role"] == "bench"].copy()
        coach = team_df[team_df["role"] == "coach"].copy()
        return starters, bench, coach

    # Fallback inteligente (CSV antigo ou reduzido)
    coach = team_df[team_df.get("posicao_id") == 6].copy()
    starters = team_df[team_df.get("posicao_id") != 6].copy()
    bench = team_df.iloc[0:0].copy()  # vazio

    return starters, bench, coach


def build_payload_for_team(
    team_df: pd.DataFrame,
    formation: str,
    esquema_map: Dict[str, int],
    include_bench: bool = True
) -> Dict[str, Any]:

    starters, bench, coach = split_roles(team_df)

    atletas_main: List[int] = []
    atletas_main.extend(starters["atleta_id"].astype(int).tolist())
    atletas_main.extend(coach["atleta_id"].astype(int).tolist())

    # Capitão = maior score previsto
    if "is_captain" in starters.columns and starters["is_captain"].any():
        cap = starters[starters["is_captain"] == True].iloc[0]
    else:
        cap = starters.sort_values("score", ascending=False).iloc[0]

    capitao_id = int(cap["atleta_id"])

    esquema_id = (
        esquema_map.get(formation)
        or FALLBACK_ESQUEMAS.get(formation)
        or 1
    )

    payload: Dict[str, Any] = {
        "esquema_id": int(esquema_id),
        "atletas": atletas_main,
        "capitao": capitao_id,
    }

    if include_bench and len(bench):
        payload["reservas"] = bench["atleta_id"].astype(int).tolist()

        if "is_luxury" in bench.columns and bench["is_luxury"].any():
            luxo = bench[bench["is_luxury"] == True].iloc[0]
            payload["reserva_de_luxo"] = int(luxo["atleta_id"])

    payload["meta"] = {
        "team_id": int(team_df["team_id"].iloc[0]),
        "score_total": float(team_df.get("score_total", 0).iloc[0]),
        "preco_total": float(team_df.get("preco_total", 0).iloc[0]),
    }

    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--formation", required=True)
    ap.add_argument("--team-id", type=int)
    ap.add_argument("--no-bench", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    esquema_map = fetch_esquemas_map()

    payloads = []

    for team_id, team_df in df.groupby("team_id"):
        if args.team_id is not None and int(team_id) != args.team_id:
            continue

        payloads.append(
            build_payload_for_team(
                team_df,
                formation=args.formation,
                esquema_map=esquema_map,
                include_bench=not args.no_bench,
            )
        )

    out = payloads[0] if len(payloads) == 1 else payloads
    Path(args.out_json).write_text(
        json.dumps(out, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"[OK] Exportado {len(payloads)} time(s) → {args.out_json}")


if __name__ == "__main__":
    main()
