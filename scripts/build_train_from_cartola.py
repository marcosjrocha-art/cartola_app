import os
import glob
import pandas as pd

# posicao_id do Cartola (padrão)
# 1 GOL, 2 LAT, 3 ZAG, 4 MEI, 5 ATA, 6 TEC
POS_ID_TO_BUCKET = {
    1: "GOL",
    2: "LAT",
    3: "ZAG",
    4: "MEI",
    5: "ATA",
    6: "TEC",
}

def find_repo_dir() -> str:
    # você já tem "cartola", mas também aceitamos "caRtola"
    for d in ["cartola", "caRtola", "caRtola-main", "caRtola-master"]:
        if os.path.isdir(d):
            return d
    return ""

def list_csvs(repo_dir: str):
    pattern = os.path.join(repo_dir, "data", "01_raw", "*", "rodada-*.csv")
    return sorted(glob.glob(pattern))

def col_pick(df: pd.DataFrame, *names):
    # retorna a primeira coluna existente
    for n in names:
        if n in df.columns:
            return n
    return None

def main():
    repo_dir = find_repo_dir()
    if not repo_dir:
        raise SystemExit(
            "Não encontrei a pasta do caRtola.\n"
            "Dentro de ~/cartola_app rode:\n"
            "  git clone https://github.com/henriquepgomide/caRtola.git cartola\n"
        )

    files = list_csvs(repo_dir)
    if not files:
        raise SystemExit(
            f"Encontrei '{repo_dir}', mas não achei CSVs em {repo_dir}/data/01_raw/*/rodada-*.csv\n"
            f"Teste:\n  ls -la {repo_dir}/data/01_raw/2026 | head\n"
        )

    rows = []
    skipped = 0

    for f in files:
        # ano é o nome da pasta
        try:
            ano = int(os.path.basename(os.path.dirname(f)))
        except Exception:
            ano = None

        # rodada vem do arquivo rodada-N.csv
        try:
            rodada = int(os.path.basename(f).split("-")[1].replace(".csv", ""))
        except Exception:
            rodada = None

        df = pd.read_csv(f)
        df.columns = [str(c).strip().lower() for c in df.columns]

        # suportar 2 formatos:
        # - antigo: atleta_id, preco_num, etc
        # - caRtola atual: atletas.atleta_id, atletas.preco_num, etc
        c_atleta = col_pick(df, "atleta_id", "atletas.atleta_id")
        c_clube  = col_pick(df, "clube_id", "atletas.clube_id")
        c_posid  = col_pick(df, "posicao_id", "atletas.posicao_id")
        c_preco  = col_pick(df, "preco_num", "atletas.preco_num")
        c_media  = col_pick(df, "media_num", "atletas.media_num")
        c_pts    = col_pick(df, "pontos_num", "atletas.pontos_num")
        c_jogos  = col_pick(df, "jogos_num", "atletas.jogos_num")

        required = [c_atleta, c_clube, c_posid, c_preco, c_media, c_pts]
        if any(c is None for c in required):
            skipped += 1
            continue

        out = pd.DataFrame({
            "ano": ano,
            "rodada": rodada,
            "atleta_id": pd.to_numeric(df[c_atleta], errors="coerce"),
            "clube": pd.to_numeric(df[c_clube], errors="coerce").astype("Int64").astype(str),
            "posicao_id": pd.to_numeric(df[c_posid], errors="coerce").astype("Int64"),
            "preco": pd.to_numeric(df[c_preco], errors="coerce"),
            "media": pd.to_numeric(df[c_media], errors="coerce"),
            "target": pd.to_numeric(df[c_pts], errors="coerce"),
            "jogos": pd.to_numeric(df[c_jogos], errors="coerce") if c_jogos else 0,
        })

        out = out.dropna(subset=["atleta_id", "posicao_id", "target"])
        out["atleta_id"] = out["atleta_id"].astype(int)
        out["posicao_id"] = out["posicao_id"].astype(int)
        out["bucket"] = out["posicao_id"].map(POS_ID_TO_BUCKET)

        out = out.dropna(subset=["bucket"])

        # tipos
        for c in ["preco", "media", "target"]:
            out[c] = out[c].fillna(0.0)
        out["jogos"] = pd.to_numeric(out["jogos"], errors="coerce").fillna(0).astype(int)

        rows.append(out[[
            "ano","rodada","atleta_id","clube","bucket",
            "preco","media","target","jogos"
        ]])

    if not rows:
        raise SystemExit(
            "Não consegui ler nenhum CSV válido.\n"
            "Me mande o output de:\n"
            f"  ls -la {repo_dir}/data/01_raw/2026 | head\n"
            f"  head -n 1 {repo_dir}/data/01_raw/2026/rodada-1.csv\n"
        )

    full = pd.concat(rows, ignore_index=True)
    full = full.sort_values(["atleta_id","ano","rodada"])

    # ===== Features SEM vazamento =====
    full["pontos_ultima"] = full.groupby("atleta_id")["target"].shift(1).fillna(0.0)

    full["rolling_3_prev"] = (
        full.groupby("atleta_id")["pontos_ultima"]
            .rolling(3).mean()
            .reset_index(level=0, drop=True)
            .fillna(0.0)
    )

    full["pontos_retrasada"] = full.groupby("atleta_id")["target"].shift(2).fillna(0.0)
    full["tendencia_prev"] = (full["pontos_ultima"] - full["pontos_retrasada"]).fillna(0.0)
    full = full.drop(columns=["pontos_retrasada"])

    os.makedirs("data", exist_ok=True)
    out_path = "data/train.csv"
    full.to_csv(out_path, index=False)

    print("OK ✅")
    print(f"Repo detectado: {repo_dir}")
    print(f"Arquivos CSV encontrados: {len(files)} | pulados (formato diferente): {skipped}")
    print(f"Linhas no train.csv: {len(full)}")
    print(f"Arquivo gerado: {out_path}")

if __name__ == "__main__":
    main()
