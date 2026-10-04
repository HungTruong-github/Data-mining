# Raw Data Provenance & Verification

- **Dataset Name:** Online Retail Dataset
- **Repository Source:** UCI Machine Learning Repository
- **URL:** [https://archive.ics.uci.edu/dataset/352/online+retail](https://archive.ics.uci.edu/dataset/352/online+retail)
- **Local File Path:** `data/raw/Online Retail.xlsx`
- **File Format:** Microsoft Excel (.xlsx)
- **Size (bytes):** 23,715,344 bytes (22.62 MB)
- **SHA-256 Checksum:** `43465a06f2ccf7c8b5bd2892bc7defb52f97487934fe93b16ae4c3936424676d`
- **Period Covered:** 2010-12-01 08:26:00 to 2011-12-09 12:50:00 (1 year 8 days)
- **Total Transactions:** 541,909 rows, 8 columns
- **Note on Exclusion from ZIP:**
  Due to email and upload size constraints (target < 20 MB), the raw Excel file (~22.6 MB) is intentionally excluded from `review_bundle.zip`. Reviewers can place the original `Online Retail.xlsx` into `data/raw/` and execute `python run_pipeline.py` to reproduce the entire project end-to-end.
