# Data Analyst Studio

A Streamlit workspace for data quality assessment, safe data cleaning, exploratory analysis, SQL querying and export.

## Features

- CSV, Excel and Parquet upload
- Upload limit configured up to 2 GB
- Column-name standardization
- Whitespace and null-token normalization
- Exact duplicate removal
- Mixed date-format detection and parsing
- Currency / numeric-text / percentage conversion
- Missing-value policy controls
- Data profile and issue report
- Business-rule validation
- Insurance-oriented derived fields when applicable:
  - `claim_delay_days`
  - `indemnification_rate`
- KPI cards and EDA charts
- Correlation matrix
- Rule-based analytical notes
- Quick Ask for common questions
- SQL querying through DuckDB
- Export:
  - cleaned CSV
  - Excel analysis workbook
  - cleaning log
  - SQL query results

## Project structure

```text
data-analyst-studio/
├── .streamlit/
│   └── config.toml
├── sample_data/
│   └── dirty_insurance_claims.csv
├── tests/
│   └── test_smoke.py
├── .gitignore
├── app.py
├── README.md
├── requirements.txt
├── run_mac.command
└── run_windows.bat
```

## Run on macOS

```bash
cd ~/Downloads/data-analyst-studio
./run_mac.command
```

If macOS blocks execution:

```bash
chmod +x run_mac.command
./run_mac.command
```

Then open:

```text
http://localhost:8501
```

## Run on Windows

Double-click:

```text
run_windows.bat
```

or:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Manual setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

## GitHub

Create a repository, then run:

```bash
git init
git add .
git commit -m "Initial Data Analyst Studio"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

## Cleaning philosophy

The application separates cleaning into two categories:

**Safe automatic cleaning**
- structural standardization
- null-token normalization
- exact duplicate removal
- high-confidence date parsing
- high-confidence numeric/currency conversion

**User-controlled cleaning**
- missing-value imputation
- row deletion
- business-rule interpretation

This avoids silently changing business meaning.

## Notes on large files

The Streamlit upload limit is set to 2 GB. Practical dataset size still depends on available RAM because Pandas loads the dataset into memory. For multi-gigabyte production workloads, a future version should move ingestion and profiling to DuckDB or Polars streaming.

## License

Use and modify this project for personal, educational or portfolio purposes.
