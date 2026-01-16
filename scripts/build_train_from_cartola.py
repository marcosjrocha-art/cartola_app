import os
import glob
import argparse
import pandas as pd

# Mapeia nomes/variações do caRtola para o formato do seu app
POS_MAP = {
    "gol": "GOL",
    "goleiro": "GOL",
    "lat": "LAT",
    "lateral": "LAT",
    "zag": "ZAG",
    "zagueiro": "ZAG",
    "mei": "MEI",
    "meia": "MEI",
    "ata": "ATA",
    "atacante": "ATA",
    "tec": "TEC",
    "técnico": "TEC",
    "tecnico": "TEC",
}

def norm_pos(x: str) -> str:
    s = str(x).strip().lower()
    return POS_MAP.get(s, str(x).strip().upper())

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def main(cartola_dir: str, years: str, out_csv: str):
    years_list = [y.strip() for y in years.split(",") if y.strip()]
    if not years_list:
        raise SystemExit("Informe pelo menos 1 ano em --years (ex: 2022,2023).")

    # 1) Coletar todos os CSVs do caRtola
    files = []
    for y in years_list:
        pattern = os.path.join(cartola_dir, "data", "01_raw", y, "rodada-*.csv")
        files.extend(sorted(glob.glob(pattern)))

    if not files:
        raise SystemExit(f"Nenhum arquivo encontrado. Verifique cartola_dir={cartola_dir} e years={years_list}")

    all_rows = []
    for f in files:
        rodada = os.path.basename(f).replace("rodada-", "").replace(".csv", "")
        try:
            rodada = int(rodada)
        except:
            rodada = None

        df = pd.read_csv(f)
        df.columns = [str(c).strip().lower() for c in df.columns]

        # Colunas típicas (podem variar por ano):
        col_preco = pick_col(df, ["preco_num", "preco", "preço", "valor_num", "valor"])
        col_media = pick_col(df, ["media_num", "media", "média"])
        col_ult = pick_col(df, ["pontos_num", "pontos", "pontuacao", "pontuação"])
        col_jogos = pick_col(df, ["jogos_num", "jogos"])
        col_pos = pick_col(df, ["posicao", "posição", "posicao_nome", "posicao_id", "posicao_abreviacao"])
        col_clube = pick_col(df, ["clube", "clube_nome", "time", "clube_id"])

        # Target (pontos da rodada) — usamos como target e também como "pontos_ultima" quando deslocarmos
        col_target = pick_col(df, ["pontos_num", "pontos", "pontuacao", "pontuação"])

        # Se algo essencial faltar, pula arquivo
        essentials = [col_preco, col_media, col_pos, col_clube, col_target]
        if any(c is None for c in essentials):
            continue

        out = pd.DataFrame({
            "rodada": rodada,
            "preco": pd.to_numeric(df[col_preco], errors="coerce"),
            "media": pd.to_numeric(df[col_media], errors="coerce"),
            "jogos": pd.to_numeric(df[col_jogos], errors="coerce") if col_jogos else 0,
            "posicao_raw": df[col_pos].astype(str),
            "clube": df[col_clube].astype(str),
            "target": pd.to_numeric(df[col_target], errors="coerce"),
        })

        out["bucket"] = out["posicao_raw"].apply(norm_pos)
        all_rows.append(out)

    full = pd.concat(all_rows, ignore_index=True).dropna(subset=["target"])
    full["jogos"] = full["jogos"].fillna(0).astype(int)
    full["preco"] = full["preco"].fillna(0.0)
    full["media"] = full["media"].fillna(0.0)

    # 2) Criar "pontos_ultima" como target da rodada anterior (por clube+posicao como aproximação)
    # Obs: ideal seria por atleta_id, mas nem sempre vem consistente em todos os anos.
    full = full.sort_values(["clube", "bucket", "rodada"]).reset_index(drop=True)
    full["pontos_ultima"] = full.groupby(["clube", "bucket"])["target"].shift(1).fillna(0.0)

    # 3) Dataset final no formato do seu app (ML)
    train = full[["preco", "media", "pontos_ultima", "jogos", "clube", "bucket", "target"]].copy()

    # Limpeza final
    train = train.replace([float("inf"), float("-inf")], pd.NA).dropna()
    train = train[(train["preco"] >= 0) & (train["media"] >= 0)]
    train.to_csv(out_csv, index=False)
    print(f"OK: gerado {out_csv} com {len(train)} linhas")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cartola_dir", default="caRtola", help="Pasta onde o repo caRtola foi clonado")
    ap.add_argument("--years", default="2022,2023", help="Anos separados por vírgula (ex: 2019,2020,2021,2022,2023)")
    ap.add_argument("--out_csv", default="data/train.csv", help="Saída do dataset")
    args = ap.parse_args()
    main(args.cartola_dir, args.years, args.out_csv)
