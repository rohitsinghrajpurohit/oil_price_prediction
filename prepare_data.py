"""Raw Oil_Data.csv ko clean karta hai -> data/clean_oil_data.csv

Cleaning steps:
1. Khaali rows hatao (file me ~79 blank rows hain)
2. Date ko proper date me badlo (14-Oct-22 -> 2022-10-14)
3. Volume "298.03K" -> 298030 (jahan Volume 0 hai wo missing data hai -> NaN)
4. Chg% "-3.93%" -> -3.93
5. Purani date se nayi date ki taraf sort karo
"""
import pandas as pd


def parse_volume(v):
    s = str(v).strip().upper().replace(",", "")
    if s in ("", "0", "-", "NAN"):
        return None
    mult = 1.0
    if s.endswith("K"):
        mult, s = 1e3, s[:-1]
    elif s.endswith("M"):
        mult, s = 1e6, s[:-1]
    return float(s) * mult


def main():
    df = pd.read_csv("data/Oil_Data.csv").dropna(how="all")
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%b-%y")
    df["Volume"] = pd.to_numeric(df["Volume"].apply(parse_volume))
    df["Chg%"] = df["Chg%"].str.replace("%", "", regex=False).astype(float)
    df = df.rename(columns={"Chg%": "Change_pct"}).sort_values("Date").reset_index(drop=True)
    df.to_csv("data/clean_oil_data.csv", index=False)
    print(f"Clean data saved: {len(df)} rows, {df.Date.min().date()} to {df.Date.max().date()}")
    print(f"Volume missing in {df.Volume.isna().sum()} rows")


if __name__ == "__main__":
    main()
