"""
automate_Kalyca-Venezia.py  |  Hands-on 2

Preprocessing otomatis dataset Influencer Marketing ROI.
Ini adalah konversi dari notebook Eksperimen_Kalyca-Venezia.ipynb (bagian 5): tahapannya
sama persis, hanya strukturnya dipecah menjadi fungsi agar bisa dijalankan otomatis.

Pemakaian (dari folder mana pun):
    python automate_Kalyca-Venezia.py
    python automate_Kalyca-Venezia.py --input <csv_mentah> --output <folder_hasil>

Hasil: train.csv, test.csv, dan preprocessor.joblib di folder output.
Skrip berhenti dengan error (kode keluar bukan 0) jika ada pemeriksaan yang gagal,
sehingga GitHub Actions menandai prosesnya gagal dan tidak diam-diam menghasilkan data rusak.
"""
import argparse
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ------------------------------------------------------------------ KONFIGURASI
RANDOM_STATE = 42       # supaya pembagian data selalu sama
TEST_SIZE = 0.2         # 80% latih, 20% uji
MIN_RETENSI = 0.95      # minimal 95% baris harus bertahan setelah pembersihan

FITUR_KATEGORI = ["platform", "influencer_category", "campaign_type"]
FITUR_NUMERIK = ["estimated_reach", "campaign_duration_days"]
TARGET = "engagements"
KOLOM_TANGGAL = ["start_date", "end_date"]
KOLOM_WAJIB = {
    "campaign_id", "platform", "influencer_category", "campaign_type", "start_date",
    "engagements", "estimated_reach", "product_sales", "campaign_duration_days", "end_date",
}

FOLDER_SKRIP = Path(__file__).resolve().parent
INPUT_DEFAULT = FOLDER_SKRIP.parent / "influencer_marketing_roi_raw" / "influencer_marketing_roi_dataset.csv"
OUTPUT_DEFAULT = FOLDER_SKRIP / "influencer_marketing_roi_preprocessing"


# ------------------------------------------------------------------ FUNGSI BANTU
def ke_ordinal(teks):
    # Mengubah teks tanggal menjadi nomor hari (bilangan bulat).
    # Memakai modul datetime Python, bukan pd.to_datetime, karena pandas hanya mendukung
    # tahun 1677 sampai 2262 dan akan diam-diam mengubah tanggal lain menjadi NaT.
    try:
        return date.fromisoformat(str(teks)[:10]).toordinal()
    except ValueError:
        return np.nan


def laporkan(langkah, sebelum, sesudah):
    print(f"  {langkah:<34} dibuang: {sebelum - sesudah:>7}   tersisa: {sesudah:>7}")


# ------------------------------------------------------------------ BAGIAN 3: MEMUAT DATA
def muat_data(path_csv):
    path_csv = Path(path_csv)
    if not path_csv.exists():
        raise FileNotFoundError(f"Dataset mentah tidak ditemukan: {path_csv}")
    df = pd.read_csv(path_csv)
    kurang = KOLOM_WAJIB - set(df.columns)
    if kurang:
        raise ValueError(f"Kolom hilang di dataset: {sorted(kurang)}")
    print(f"Memuat data: {path_csv.name} | shape {df.shape}")
    return df


# ------------------------------------------------------------------ NOTEBOOK 5.1 - 5.6
def bersihkan_data(df):
    print("\nPembersihan data:")
    data = df.copy()                                                    # 5.1 salin data mentah
    n = len(data)

    data = data.dropna()                                                # 5.2 missing value
    laporkan("Missing value", n, len(data)); n = len(data)

    for kol in FITUR_KATEGORI:                                          # 5.3 teks kategorikal
        data[kol] = data[kol].str.strip()

    data = data.assign(_mulai=data["start_date"].map(ke_ordinal),       # 5.4 validasi tanggal
                       _selesai=data["end_date"].map(ke_ordinal))
    data = data.dropna(subset=["_mulai", "_selesai"])
    laporkan("Teks tanggal rusak", n, len(data)); n = len(data)
    data = data[data["_selesai"] >= data["_mulai"]]
    laporkan("end_date < start_date", n, len(data)); n = len(data)
    data = data.drop(columns=["_mulai", "_selesai"])

    data = data.drop_duplicates()                                       # 5.5 duplikat
    laporkan("Baris duplikat", n, len(data)); n = len(data)

    print("  Outlier (aturan IQR, hanya dilaporkan, tidak dibuang):")   # 5.6 outlier dan validasi
    for kol in [TARGET, "estimated_reach", "campaign_duration_days"]:
        q1, q3 = data[kol].quantile([0.25, 0.75])
        iqr = q3 - q1
        n_outlier = int(((data[kol] < q1 - 1.5 * iqr) | (data[kol] > q3 + 1.5 * iqr)).sum())
        print(f"    {kol:<24} outlier = {n_outlier}")
    valid = (data["estimated_reach"] > 0) & (data["campaign_duration_days"] > 0) & (data[TARGET] >= 0)
    data = data[valid]
    laporkan("Nilai tidak valid", n, len(data))

    retensi = len(data) / len(df)
    print(f"  Data dipertahankan: {len(data)} dari {len(df)} ({retensi:.2%})")
    if retensi < MIN_RETENSI:
        raise ValueError(f"Hanya {retensi:.2%} data yang bertahan (minimal {MIN_RETENSI:.0%}). "
                         "Periksa dataset mentah dan langkah pembersihan.")
    return data


# ------------------------------------------------------------------ NOTEBOOK 5.7 - 5.10
def bangun_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("kategori", OneHotEncoder(handle_unknown="ignore", sparse_output=False), FITUR_KATEGORI),
            ("numerik", StandardScaler(), FITUR_NUMERIK),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def siapkan_fitur(data):
    # 5.7 dan 5.8: kolom tanggal, product_sales, dan campaign_id tidak dipakai sebagai fitur
    X = data[FITUR_KATEGORI + FITUR_NUMERIK]
    y = data[TARGET]
    assert not {"engagements", "product_sales"} & set(X.columns)
    assert not set(KOLOM_TANGGAL) & set(X.columns)

    # 5.9: bagi data SEBELUM encoding/standarisasi agar data uji tidak "bocor"
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # 5.10: encoder dan scaler dipelajari (fit) hanya dari data latih
    preprocessor = bangun_preprocessor()
    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    kolom_onehot = [c for c in X_train_p.columns if c not in FITUR_NUMERIK]
    for bagian in (X_train_p, X_test_p):
        bagian[kolom_onehot] = bagian[kolom_onehot].astype(int)
        bagian[FITUR_NUMERIK] = bagian[FITUR_NUMERIK].round(6)

    print(f"\nPembagian data: latih {X_train_p.shape} | uji {X_test_p.shape}")
    print(f"Jumlah fitur akhir: {X_train_p.shape[1]}")
    return X_train_p, X_test_p, y_train, y_test, preprocessor


# ------------------------------------------------------------------ NOTEBOOK 5.11
def simpan_hasil(output_dir, X_train_p, X_test_p, y_train, y_test, preprocessor):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_out = X_train_p.copy()
    train_out[TARGET] = y_train.values
    test_out = X_test_p.copy()
    test_out[TARGET] = y_test.values

    train_out.to_csv(output_dir / "train.csv", index=False)
    test_out.to_csv(output_dir / "test.csv", index=False)
    joblib.dump(preprocessor, output_dir / "preprocessor.joblib")

    print("\nBerkas disimpan di:", output_dir)
    for berkas in sorted(output_dir.iterdir()):
        if berkas.name != ".gitkeep":
            print(f"  {berkas.name:<20} {berkas.stat().st_size / 1024 / 1024:6.2f} MB")


# ------------------------------------------------------------------ NOTEBOOK 5.12
def verifikasi_hasil(output_dir, jumlah_bersih, y_train):
    output_dir = Path(output_dir)
    train = pd.read_csv(output_dir / "train.csv")
    test = pd.read_csv(output_dir / "test.csv")

    pemeriksaan = {
        "Jumlah latih + uji = jumlah data bersih": len(train) + len(test) == jumlah_bersih,
        "Tidak ada NaN di train.csv dan test.csv": train.isnull().sum().sum() + test.isnull().sum().sum() == 0,
        "Kolom train.csv sama dengan test.csv": list(train.columns) == list(test.columns),
        "Fitur numerik terstandarisasi (mean ~ 0, std ~ 1)": bool(
            all(abs(train[c].mean()) < 1e-4 and abs(train[c].std() - 1) < 1e-3 for c in FITUR_NUMERIK)),
        "Target tidak diskalakan (sama dengan nilai asli)": bool(
            np.array_equal(train[TARGET].values, y_train.values)),
        "preprocessor.joblib dapat dimuat ulang": joblib.load(output_dir / "preprocessor.joblib") is not None,
    }

    print("\nVerifikasi hasil:")
    for nama, lolos in pemeriksaan.items():
        print(("  [PASS] " if lolos else "  [FAIL] ") + nama)
    gagal = [nama for nama, lolos in pemeriksaan.items() if not lolos]
    if gagal:
        raise AssertionError(f"Pemeriksaan gagal: {gagal}")


# ------------------------------------------------------------------ FUNGSI UTAMA
def preprocess_data(input_path=INPUT_DEFAULT, output_dir=OUTPUT_DEFAULT, simpan=True):
    # Menjalankan seluruh preprocessing dan mengembalikan data yang siap dilatih:
    # X_train, X_test, y_train, y_test, preprocessor
    df = muat_data(input_path)
    data = bersihkan_data(df)
    X_train_p, X_test_p, y_train, y_test, preprocessor = siapkan_fitur(data)
    if simpan:
        simpan_hasil(output_dir, X_train_p, X_test_p, y_train, y_test, preprocessor)
        verifikasi_hasil(output_dir, len(data), y_train)
    return X_train_p, X_test_p, y_train, y_test, preprocessor


def main():
    parser = argparse.ArgumentParser(description="Preprocessing otomatis dataset Influencer Marketing ROI")
    parser.add_argument("--input", type=Path, default=INPUT_DEFAULT, help="CSV dataset mentah")
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT, help="folder hasil preprocessing")
    args = parser.parse_args()

    preprocess_data(args.input, args.output)
    print("\nSELESAI. Dataset siap dilatih.")


if __name__ == "__main__":
    main()
