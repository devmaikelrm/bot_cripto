import pandas as pd
import numpy as np
from pathlib import Path
from bot_cripto.core.config import get_settings

def create_dummy_data():
    settings = get_settings()
    settings.ensure_dirs()
    
    dates = pd.date_range(start="2024-01-01", periods=1000, freq="5min")
    close = np.linspace(40000, 45000, 1000) + np.random.normal(0, 100, 1000)
    high = close + 50
    low = close - 50
    open_ = close + np.random.normal(0, 10, 1000)
    volume = np.random.randint(100, 10000, 1000).astype(float)
    
    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    }, index=dates)
    df.index.name = "date"
    
    path = settings.data_dir_raw / "BTC_USDT_5m.parquet"
    df.to_parquet(path)
    print(f"Dummy data created at {path}")

if __name__ == "__main__":
    create_dummy_data()
