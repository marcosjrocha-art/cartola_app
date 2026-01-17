#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import pandas as pd


def ensure_team_columns(df: pd.DataFrame, n_teams: int) -> pd.DataFrame:
    """
    Garante que a tabela tenha apenas team_1..team_n e 'mean' calculado corretamente.
    Evita quebrar quando n_teams muda (1, 2, 3, ...).
    """
    if df is None or df.empty:
        return pd.DataFrame()

    team_cols = [f"team_{i}" for i in range(1, n_teams + 1)]
    cols_present = [c for c in team_cols if c in df.columns]

    # se o df veio "longo" (team, value), tenta pivotar
    if "team" in df.columns and "value" in df.columns:
        pv = df.pivot_table(index=[c for c in df.columns if c not in ("team", "value")], columns="team", values="value", aggfunc="first")
        pv.columns = [f"team_{int(x)}" for x in pv.columns]
        pv = pv.reset_index()
        df = pv
        cols_present = [c for c in team_cols if c in df.columns]

    # filtra apenas o que existe
    out = df.copy()

    # calcula mean só das colunas presentes
    if cols_present:
        out["mean"] = out[cols_present].mean(axis=1)
    else:
        out["mean"] = 0.0

    # reordena: cols não-team primeiro (se houver), depois teams, depois mean
    non_team = [c for c in out.columns if not c.startswith("team_") and c != "mean"]
    ordered = non_team + cols_present + ["mean"]
    ordered = [c for c in ordered if c in out.columns]
    out = out[ordered]

    return out


def safe_metric_cards(summary: dict, n_teams: int) -> dict:
    """
    Padroniza o summary para não depender de team_3 etc.
    """
    if summary is None:
        summary = {}

    out = dict(summary)
    out["n_teams"] = n_teams

    # se vier com chaves fixas, não quebra
    for i in range(1, n_teams + 1):
        out.setdefault(f"team_{i}", 0.0)

    out.setdefault("mean", float(pd.Series([out.get(f"team_{i}", 0.0) for i in range(1, n_teams + 1)]).mean()))
    return out
