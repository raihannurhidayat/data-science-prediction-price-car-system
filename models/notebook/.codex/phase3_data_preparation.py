from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Car_sales.xls"
OUTPUT_DIR = BASE_DIR / "outputs" / "phase3"

TARGET_COLUMN = "Price_in_thousands"
FEATURE_COLUMNS = [
    "Engine_size",
    "Horsepower",
    "Curb_weight",
    "Fuel_efficiency",
]
RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_car_sales(path: Path) -> pd.DataFrame:
    """The provided .xls file contains comma-separated text."""
    return pd.read_csv(path)


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict]:
    missing_before = df.isna().sum().to_dict()

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    rows_before = len(df)
    prepared = df.dropna(subset=[TARGET_COLUMN]).copy()
    rows_after_target_drop = len(prepared)

    imputation_values: dict[str, float] = {}
    for column in FEATURE_COLUMNS:
        median_value = float(prepared[column].median())
        imputation_values[column] = median_value
        prepared[column] = prepared[column].fillna(median_value)

    x = prepared[FEATURE_COLUMNS]
    y = prepared[TARGET_COLUMN]

    summary = {
        "source_file": str(DATA_PATH.name),
        "source_format_note": "File uses .xls extension but contains CSV text.",
        "rows_before": rows_before,
        "columns_before": len(df.columns),
        "missing_before": missing_before,
        "target_column": TARGET_COLUMN,
        "feature_columns": FEATURE_COLUMNS,
        "rows_dropped_missing_target": rows_before - rows_after_target_drop,
        "rows_after_target_drop": rows_after_target_drop,
        "imputation_strategy": "Median imputation for selected independent variables after dropping missing target rows.",
        "imputation_values": imputation_values,
        "missing_after_preparation": prepared[required_columns].isna().sum().to_dict(),
        "test_size": TEST_SIZE,
        "train_size": 1 - TEST_SIZE,
        "random_state": RANDOM_STATE,
    }
    return x, y, summary


def write_report(summary: dict, train_rows: int, test_rows: int) -> None:
    report = f"""# Phase 3 - Data Preparation

## Scope

Phase 3 mengeksekusi acceptance criteria PRD untuk Data Preparation berdasarkan ketentuan PDF:

- Menghapus missing value jika row minor.
- Mengisi missing value untuk kolom kritis yang tetap dipakai.
- Menentukan variabel independen dan dependen.
- Memisahkan data training 80% dan testing 20%.

## Keputusan Preparation

- File sumber: `{summary["source_file"]}`.
- Catatan format: {summary["source_format_note"]}
- Target/dependent variable: `{summary["target_column"]}`.
- Independent variables: {", ".join(f"`{column}`" for column in summary["feature_columns"])}.
- Baris awal: {summary["rows_before"]}.
- Baris dengan target kosong yang dihapus: {summary["rows_dropped_missing_target"]}.
- Baris setelah target drop: {summary["rows_after_target_drop"]}.
- Strategi imputasi: {summary["imputation_strategy"]}

## Nilai Imputasi Median

| Kolom | Nilai Median |
|---|---:|
"""
    for column, value in summary["imputation_values"].items():
        report += f"| `{column}` | {value:.4f} |\n"

    report += f"""
## Hasil Split

| Dataset | Jumlah Baris |
|---|---:|
| Training | {train_rows} |
| Testing | {test_rows} |

Rasio split memakai `train_test_split(test_size={summary["test_size"]}, random_state={summary["random_state"]})`.

## Output Files

- `clean_dataset.csv`: dataset final berisi X dan y setelah drop/imputasi.
- `X_train.csv`, `X_test.csv`: independent variables untuk modeling.
- `y_train.csv`, `y_test.csv`: dependent variable untuk modeling.
- `preprocessing_summary.json`: ringkasan evidence preparation untuk audit.
"""
    (OUTPUT_DIR / "phase3_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_car_sales(DATA_PATH)
    x, y, summary = prepare_data(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    clean_dataset = x.copy()
    clean_dataset[TARGET_COLUMN] = y

    clean_dataset.to_csv(OUTPUT_DIR / "clean_dataset.csv", index=False)
    x_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
    x_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)
    y_train.to_frame(TARGET_COLUMN).to_csv(OUTPUT_DIR / "y_train.csv", index=False)
    y_test.to_frame(TARGET_COLUMN).to_csv(OUTPUT_DIR / "y_test.csv", index=False)

    summary["train_rows"] = len(x_train)
    summary["test_rows"] = len(x_test)
    summary["output_files"] = [
        "clean_dataset.csv",
        "X_train.csv",
        "X_test.csv",
        "y_train.csv",
        "y_test.csv",
        "preprocessing_summary.json",
        "phase3_report.md",
    ]

    (OUTPUT_DIR / "preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    write_report(summary, len(x_train), len(x_test))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
