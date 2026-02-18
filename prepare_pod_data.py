import os
import pandas as pd
from bot_cripto.core.config import get_settings
from bot_cripto.data.ingestion import BinanceFetcher
from bot_cripto.features.engineer import FeatureEngineer

def prepare_coin(symbol="SOL/USDT", timeframe="1h"):
    print(f"--- Preparando {symbol} ({timeframe}) ---")
    settings = get_settings()
    settings.ensure_dirs()
    
    # 1. Ingestión (calculamos días para unos 17000-18000 registros)
    # 1h: 17000 / 24 = 708 días
    # 5m: 17000 / (24*12) = 59 días
    days = 750 if timeframe == "1h" else 65
    
    fetcher = BinanceFetcher(settings)
    df_raw = fetcher.fetch_history(symbol, timeframe, days=days)
    if df_raw.empty:
        print(f"ERROR: No se descargaron datos para {symbol}")
        return
    
    # Guardar RAW
    raw_path = fetcher.save_data(df_raw, symbol, timeframe)
    print(f"Datos RAW guardados en: {raw_path}")
    
    # 2. Features
    engineer = FeatureEngineer()
    df_features = engineer.generate_all(df_raw)
    
    # 3. Guardar PROCESSED
    safe_symbol = symbol.replace("/", "_")
    out_path = settings.data_dir_processed / f"{safe_symbol}_{timeframe}_features.parquet"
    df_features.to_parquet(out_path)
    print(f"Features generadas en: {out_path} ({len(df_features)} filas)")

if __name__ == "__main__":
    # Preparar SOL 1h
    prepare_coin("SOL/USDT", "1h")
    # Preparar 5m para el "gatillo"
    prepare_coin("BTC/USDT", "5m")
    prepare_coin("SOL/USDT", "5m")
