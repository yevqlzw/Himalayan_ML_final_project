# Who will reach the summit?

Predicting the success and risks of Himalayan expeditions with the Himalayan Database.
Final machine learning project.

## Data

- **Raw tables:** Himalayan Database 2.75, autumn-winter 2025 update (https://www.himalayandatabase.com).
  We use `exped`, `members` and `peaks`, each as a `.DBF` + `.FPT` pair, placed in `data/raw/HIMDATA/` (not in git).
- **`docs/himalayan_fields_guide.txt`:** plain-text copy of Appendix B "Database Structure" (pages 60-66)
  of the Himalayan Database Guide, June 2026 edition, the official manual of the database. It describes every
  field and code of the three tables. It is an input file: the code only reads it (to build
  `docs/data_dictionary.csv`) and cannot recreate it, so do not delete it.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate        
pip install -r requirements.txt

# put the database tables into data/raw/HIMDATA/ (exped, members, peaks: .DBF + .FPT)
python -m src.load               # -> data/processed/{exped,members,peaks}

python -m src.data_dictionary    # -> docs/data_dictionary.csv

python -m src.dataset            # -> data/processed/dataset_v1.parquet

python -m src.features           # -> data/processed/features_v1.parquet (+ features_v1_preview.csv for viewing)

python -m pytest tests           # leakage, features, hand-checked records
```

## Data flow

```
raw DBF --load--> exped / members / peaks.parquet   (whole database, for death risk)
                        |
                     dataset --> dataset_v1.parquet   (filters, all allowed columns incl. names)
                        |
                     features --> features_v1.parquet  <-- the model is trained on this
```

## Structure

```
data/raw/HIMDATA/    original DBF files (not in git)
data/processed/      parquet, dataset_v1, features_v1
docs/                fields guide (input) and data dictionary (generated)
notebooks/           03_A_death_risk.ipynb
src/                 load, data_dictionary, leakage, dataset, features, death_risk
tests/               leakage, historical feature and death risk checks
reports/figures/     figures for the report
```

## Decision log

| Step | Problem | Solution |
|---|---|---|
| 1. Loading | The data version must be fixed so that all numbers are reproducible. | Himalayan Database 2.75 (autumn-winter 2025): exped 11,854, members 94,999, peaks 503, years 1905-2025. `load.py` checks these counts with assert. |
| 1. Loading | `AGE` and `BIRTHDATE` are empty in the public version. | Age is taken from `CALCAGE` (0 -> NaN). |
| 1. Loading | `EXPID` is not unique: it holds only two digits of the year, so 5 IDs repeat 100 years apart (`EVER21101` is both 1921 and 2021). Joining on `EXPID` alone gives 95,118 rows instead of 94,999. | Keys: exped (`EXPID`, `YEAR`), members (`EXPID`, `MYEAR`, `MEMBID`); join on both columns. Within 1990-2025 `EXPID` is unique, so `GroupKFold` by `EXPID` is valid. |
| 1. Loading | In the DBF files years are stored as text, empty values appear as `""`, `0` or `None`, and logical fields can be unknown. | Years and numbers -> `Int64` (an empty or 0 year -> NA); text trimmed, `""` -> NaN; dates -> `datetime64[ms]`. Logical -> `bool`, or `boolean` with NA where unknown values exist (`COMRTE` 20%, `STDRTE` 19%, `PRIMREF` 31%, `MSPEED` 46%), so that "unknown" does not turn into False. "0 = Unknown" codes are left as is. |
| 2. Data dictionary | Some fields are filled only after the climb and reveal the outcome to the model. | `FORBIDDEN` in `src/leakage.py`: fields from the plan plus fields added after checking the Guide. Dictionary: 110 after / 49 before / 10 technical. |
| 2. Data dictionary | `O2CLIMB` records actual oxygen use, not a plan: among expeditions that turned back below 7000 m it is True in only 1% (5 of 460), above 8000 m in 50%, among successful ones in 71%. | Excluded from the prediction model. |
| 2. Data dictionary | 20 more fields are not known before the start: `BCDATE` (empty if base camp was not reached); `BCONLY`, `NOTTOBC`; current peak status (`PSTATUS = Unclimbed` means there was no success); first ascent data (for 378 expeditions it is the expedition itself); peak memos. | All added to `FORBIDDEN`; `BCONLY` and `NOTTOBC` are used only as filters. `EXPID`, `MEMBID` -> `TECHNICAL`. |
| 3. Dataset | A derived column could carry a forbidden field under another name (`highpoint_bin`). | `check_no_leakage` blocks a column whose name (case-insensitive) equals a `FORBIDDEN` or `TECHNICAL` field or starts with `<FIELD>_`. |
| 3. Dataset | We need one "member x expedition" table with only the people who actually attempted the summit. | Filters: 94,999 -> 78,489 (1990-2025) -> 58,100 (`HIRED == False`) -> 54,674 (no `MSMTTERM == 17`) -> 54,252 (no `NOTTOBC`) -> 52,244 (no `BCONLY`) -> **52,125** (no `MDISPUTED`/`MCLAIMED`). Train 37,969 / val 5,042 / test 9,114 rows. |
| 3. Dataset | The partner could accidentally feed a forbidden field to the model. | `dataset_v1` keeps only allowed columns (45): keys, targets `MSUCCESS` and `DEATH`, before fields, `PEAK_GROUP`, `ANOMALOUS_SEASON`. Dropped: duplicates `MYEAR`/`MSEASON`, empty `AGE`/`BIRTHDATE`, `HIRED` (constant after filtering), `ROUTE2-4`. |
| 3. Dataset | `ROUTE1` contains after-the-fact notes: "(to 6000m)" appears only in unsuccessful expeditions. | Removed "(to ...m)", "(intended)" and "(up)/(down)" (1,273 rows). The remaining parentheses are route variant names. |
| 3. Dataset | Audit of implausible values in the features: `SEX = "X"` (1 row), `HOST = 0` "Unknown" (2 expeditions, 8 rows), `TOTMEMBERS = 0` although the expedition has member records (1 expedition, 4 rows), ages from 13 to 86. | `SEX` other than M/F and `HOST = 0` -> NaN in `dataset_v1`. `TOTMEMBERS = 0` kept; `HIRED_PER_MEMBER` is NaN there. Ages kept: they are real record ascents (13-year-olds Jordan Romero and Malavath Poorna on Everest, 86-year-old Carlos Soria on Manaslu). |
| 3. Dataset | Hundreds of rare peaks; a ">= 100 records" threshold over all years uses test information (`JANU`, `NUPT`, `RATC` reach 100 only with the test years). | `PEAK_GROUP`: threshold on the training years 1990-2017 only -> 19 peaks + `OTHER` (5,413 rows). |
| 3. Dataset | Anomalous seasons (avalanche, earthquake, COVID) distort training, but nobody knew about the avalanche and the earthquake before the start, so as a feature it would be leakage. | Column `ANOMALOUS_SEASON` (1,327 rows): spring 2014 Everest and Lhotse; spring 2015 all peaks (2.9% success); all of 2020; spring 2021 (Everest success 47% instead of 75-77%). Not a feature (in `FORBIDDEN`): use it to exclude these seasons from training or for robustness checks. |
| 4. Features | The model needs a ready list of features shared by everyone. | `features_v1.parquet`: 52,125 rows, 35 features (10 categorical); lists `FEATURES` and `CATEGORICAL` in `src/features.py`. `PEAK_GROUP` replaces `PEAKID` as a feature. |
| 4. Features | The database has no person ID, and experience must not use the future. | ID = normalized `FNAME`, `LNAME` + `YOB`; only strictly earlier (year, season) records count, and an unknown season is treated as the last one in its year. Without `YOB` -> NaN (1.7%). The maximum of 65 previous expeditions belongs to real professional leaders, not merged IDs. |
| 4. Features | A peak's success rate is unstable with few attempts and must not use the current year. | Only years before the current one; smoothing (successes + 20 * overall rate) / (attempts + 20), with the overall rate also from past years only. |
| 4. Features | Crowding must be known before the start. | Computed from permits (all 11,854 expeditions) in the same year and season, excluding the expedition itself; routes use the cleaned `ROUTE1`. |
| 5. Death risk | Hired staff appear in `members` only after significant events, so their death rate cannot come from member records. | Unit: one participation in one expedition above base camp. Clients: members on the permit above base camp (not `BCONLY`, not `NOTTOBC`) from member records; hired staff: expedition counters `HDEATHS / TOTHIRED` (`TOTHIRED` counts hired staff above base camp). Counters match member records in every expedition (845 member and 342 hired deaths). Rates per 1,000 participations with 95% Wilson intervals. |
| 5. Death risk | A rate per participation is not a risk per person per year: hired staff work several expeditions per season and pass dangerous zones more often. | Reported as risk per participation, with this caveat stated in the report. |
| 5. Death risk | Clients and hired staff climb different peaks in different years. | Comparison only on the 7,707 expeditions (1950-2025) that have both clients and hired staff above base camp; risk ratio with a 500-sample bootstrap over expeditions. Hired / clients: 0.68 (95% CI 0.56-0.81). 14 expeditions have `HDEATHS > TOTHIRED` and 17 hired deaths fall in expeditions with `TOTHIRED = 0`; both are kept as recorded. |
| 5. Death risk | Descriptive trends need older years and are not a model evaluation. | Trends use 1950-2025, including 2022-2025; group rates use the `features_v1` population (1990-2025). No death-risk model is trained, so the test set stays closed. |

## Experiment log

| Date | Who | Model / change | ROC AUC (val) | Brier (val) | Conclusion |
|---|---|---|---|---|---|
