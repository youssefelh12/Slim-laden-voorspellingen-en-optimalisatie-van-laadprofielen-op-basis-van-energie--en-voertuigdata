import pandas as pd
from pathlib import Path
from typing import Iterable

CUT_OFF = pd.Timestamp("2025-05-21")   # last confirmed date

def read_layout1(path: Path | str) -> pd.DataFrame:
    """
    Read one Layout-1 file (CSV or Excel) and return a
    DataFrame with a parsed 'Datum' column.
    """
    path = Path(path)
    try:
        df = pd.read_csv(path, parse_dates=['Datum'])
    except Exception:
        df = pd.read_excel(path, parse_dates=['Datum'])
    return df


def calculate_daily_sums_multi_year(
    layout_paths: Iterable[str | Path],
    output_path: str | Path
):
    """
    Combine multiple Layout-1 files (e.g. 2023, 2024, 2025),
    build a full-calendar daily summary, drop rows after CUT_OFF,
    and save to CSV.
    """
    # ----------------------------
    # 1. Read & concatenate files
    # ----------------------------
    frames = [read_layout1(p) for p in layout_paths]
    df = pd.concat(frames, ignore_index=True)

    # Verify required columns exist
    required_cols = {'Datum', 'Vakantie', 'Thuiswerk'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Invoerbestanden moeten de kolommen {required_cols} bevatten")

    # ----------------------------
    # 2. Filter out unconfirmed future data
    # ----------------------------
    df = df[df['Datum'] <= CUT_OFF]

    # ----------------------------
    # 3. Aggregate per day
    # ----------------------------
    daily_sums = (
        df.groupby('Datum')[['Vakantie', 'Thuiswerk']]
          .sum()
          .reset_index()
    )

    # ----------------------------
    # 4. Build a full calendar over the *remaining* range
    # ----------------------------
    start_date = daily_sums['Datum'].min()
    end_date   = daily_sums['Datum'].max()
    full_calendar = pd.DataFrame({
        'Datum': pd.date_range(start=start_date, end=end_date, freq='D')
    })

    merged = full_calendar.merge(daily_sums, on='Datum', how='left')
    merged[['Vakantie', 'Thuiswerk']] = (
        merged[['Vakantie', 'Thuiswerk']].fillna(0).astype(int)
    )

    merged = merged.rename(columns={
        'Vakantie':   'Totaal_Vakantiedagen',
        'Thuiswerk':  'Totaal_Thuiswerkdagen'
    })

    # ----------------------------
    # 5. Save
    # ----------------------------
    output_path = Path(output_path)
    merged.to_csv(output_path, index=False, date_format='%Y-%m-%d')
    print(f"Samengevoegde kalender opgeslagen naar: {output_path.resolve()}")


if __name__ == "__main__":
    # ⇣ Vul hier de juiste bestandsnamen in ⇣
    input_files = [
        "./data/Overzicht_2023_thuiswerk_vakantie_layout1.xlsx",
        "./data/Overzicht_2024_thuiswerk_vakantie_layout1.xlsx",
        "./data/Overzicht_2025_thuiswerk_vakantie_layout1.xlsx",
    ]
    output_file = "layout1_full_calendar_2023-2025.csv"
    calculate_daily_sums_multi_year(input_files, output_file)
