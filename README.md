# Who will reach the summit?

Final machine learning project. Using the Himalayan Database, we predict whether a member of an expedition reaches the summit from what is known before the start, look at what the model relies on, and describe the risk of death.

## Data

We use the `exped`, `members` and `peaks` tables of Himalayan Database 2.75, autumn-winter 2025 update (https://www.himalayandatabase.com). Each table is a `.DBF` + `.FPT` pair placed in `data/raw/HIMDATA/`, which is not in git.

`docs/himalayan_fields_guide.txt` is a plain-text copy of Appendix B "Database Structure" (pages 60-66) of the Himalayan Database Guide, June 2026 edition, the official manual of the database. It describes every field and code of the three tables. The code reads it to build `docs/data_dictionary.csv` and cannot recreate it, so it should not be deleted.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt

# put the database tables into data/raw/HIMDATA/ (exped, members, peaks: .DBF + .FPT)
python -m src.load               # -> data/processed/{exped,members,peaks}

python -m src.data_dictionary    # -> docs/data_dictionary.csv

python -m src.dataset            # -> data/processed/dataset_v1.parquet

python -m src.features           # -> data/processed/features_v1.parquet

python -m pytest tests           # data, leakage, features, models, SHAP and error analysis
```

Then run the notebooks in number order, with the project `.venv` as the kernel. For example: `jupyter nbconvert --to notebook --execute --inplace notebooks/01_baselines.ipynb`.

| Notebook | What it does |
|---|---|
| `01_baselines` | Baselines and CatBoost with default settings. |
| `02_tuning` | Tuning, model comparison and calibration. Writes `reports/final_model.json`. Finished Optuna trials are kept in `himalayan_optuna.db` in the system temp folder; delete that file to tune from zero (about 25 minutes). |
| `03_leakage_audit` | Leakage checks on the final model. |
| `04_shap` | What drives the final model. |
| `05_error_analysis` | Where and why the final model is wrong. |
| `06_death_risk` | Death risk. Independent of the model. |
| `07_model_figures` | Figures for the report and slides, drawn from saved tables. Nothing is trained here. |

## Data flow

```
raw DBF --load--> exped / members / peaks.parquet   (whole database, for death risk)
                        |
                     dataset --> dataset_v1.parquet   (filters, all allowed columns incl. names)
                        |
                     features --> features_v1.parquet  <-- the model is trained on this
                        |
                     02_tuning --> reports/final_model.json  (frozen settings of the final model)
```

## Structure

```
data/raw/HIMDATA/    original DBF files (not in git)
data/processed/      parquet tables, dataset_v1, features_v1
docs/                fields guide (input), data dictionary (generated)
notebooks/           01 baselines, 02 tuning, 03 leakage audit, 04 SHAP, 05 error analysis, 06 death risk,
                     07 model figures
src/                 config, load, data_dictionary, leakage, dataset, features, split, evaluate, models,
                     experiment_log, explain, leakage_audit, error_analysis, death_risk, style
tests/               checks for the data pipeline, models and analysis
reports/             final_model.json, experiment_log.csv
reports/tables/      result tables, named after the notebook that writes them
reports/figures/     figures for the report, named the same way
```

## Decision log

| Step | Problem | Solution |
|---|---|---|
| 1. Loading | All numbers have to be reproducible, so the data version must be fixed. | We use Himalayan Database 2.75 (autumn-winter 2025): exped 11,854, members 94,999, peaks 503, years 1905-2025. `load.py` asserts these counts. |
| 1. Loading | `AGE` and `BIRTHDATE` are empty in the public version. | Age comes from `CALCAGE`, with 0 treated as missing. |
| 1. Loading | `EXPID` is not unique, because it holds only two digits of the year. 5 IDs repeat 100 years apart (`EVER21101` is both 1921 and 2021), and joining on `EXPID` alone gives 95,118 rows instead of 94,999. | The keys are (`EXPID`, `YEAR`) in exped and (`EXPID`, `MYEAR`, `MEMBID`) in members, and joins use both columns. Within 1990-2025 `EXPID` is unique, so it can still group expeditions in `GroupKFold`. |
| 1. Loading | The DBF files store years as text and show empty values as `""`, `0` or `None`, and logical fields can be unknown. | Years and numbers become `Int64` (an empty or 0 year becomes NA), text is trimmed with `""` as NaN, and dates become `datetime64[ms]`. Logical fields become `bool`, or `boolean` with NA where unknown values exist (`COMRTE` 20%, `STDRTE` 19%, `PRIMREF` 31%, `MSPEED` 46%), so that unknown does not turn into False. "0 = Unknown" codes stay as they are. |
| 2. Data dictionary | Some fields are only filled in after the climb and would show the outcome to the model. | `FORBIDDEN` in `src/leakage.py` lists them: the fields from the plan plus those we added after checking the Guide. The dictionary has 110 after, 49 before and 10 technical fields. |
| 2. Data dictionary | `O2CLIMB` records actual oxygen use rather than a plan. Among expeditions that turned back below 7000 m it is True in only 1% (5 of 460), above 8000 m in 50%, and among successful ones in 71%. | It is excluded from the prediction model. |
| 2. Data dictionary | Another 20 fields are not known before the start: `BCDATE` (empty if base camp was not reached), `BCONLY`, `NOTTOBC`, the current peak status (`PSTATUS = Unclimbed` means there was no success), first ascent data (for 378 expeditions that is the expedition itself) and peak memos. | All of them went into `FORBIDDEN`, and `BCONLY` and `NOTTOBC` serve only as filters. `EXPID` and `MEMBID` are `TECHNICAL`. |
| 3. Dataset | A derived column could carry a forbidden field under another name, such as `highpoint_bin`. | `check_no_leakage` blocks any column whose name, ignoring case, equals a `FORBIDDEN` or `TECHNICAL` field or starts with `<FIELD>_`. |
| 3. Dataset | We need one "member x expedition" table with only the people who actually attempted the summit. | Filters: 94,999 -> 78,489 (1990-2025) -> 58,100 (`HIRED == False`) -> 54,674 (no `MSMTTERM == 17`) -> 54,252 (no `NOTTOBC`) -> 52,244 (no `BCONLY`) -> **52,125** (no `MDISPUTED`/`MCLAIMED`). Train 37,969 / val 5,042 / test 9,114 rows. |
| 3. Dataset | A forbidden field could reach the model by accident. | `dataset_v1` keeps only allowed columns (45): keys, the targets `MSUCCESS` and `DEATH`, the before fields, `PEAK_GROUP` and `ANOMALOUS_SEASON`. Dropped: the duplicates `MYEAR` and `MSEASON`, the empty `AGE` and `BIRTHDATE`, `HIRED` (constant after filtering) and `ROUTE2-4`. |
| 3. Dataset | `ROUTE1` contains notes written after the fact: "(to 6000m)" appears only in unsuccessful expeditions. | We removed "(to ...m)", "(intended)" and "(up)/(down)" from 1,273 rows. The remaining parentheses are route variant names. |
| 3. Dataset | The features contain some implausible values: `SEX = "X"` (1 row), `HOST = 0` "Unknown" (2 expeditions, 8 rows), `TOTMEMBERS = 0` although the expedition has member records (1 expedition, 4 rows), and ages from 13 to 86. | `SEX` other than M/F and `HOST = 0` become NaN in `dataset_v1`. `TOTMEMBERS = 0` stays, and `HIRED_PER_MEMBER` is NaN there. Ages stay too, since these are real record ascents (13-year-olds Jordan Romero and Malavath Poorna on Everest, 86-year-old Carlos Soria on Manaslu). |
| 3. Dataset | There are hundreds of rare peaks, and a threshold of at least 100 records over all years would use test information (`JANU`, `NUPT`, `RATC` reach 100 only with the test years). | `PEAK_GROUP` applies the threshold on the training years 1990-2017 only, which leaves 19 peaks plus `OTHER` (5,413 rows). |
| 3. Dataset | Anomalous seasons (avalanche, earthquake, COVID) distort training, but nobody knew about the avalanche or the earthquake before the start, so using them as a feature would be leakage. | Column `ANOMALOUS_SEASON` (1,327 rows) marks spring 2014 on Everest and Lhotse, spring 2015 on all peaks (2.9% success), all of 2020 and spring 2021 (Everest success 47% instead of 75-76%). It is in `FORBIDDEN` and serves only to exclude these seasons from training or for robustness checks. |
| 4. Features | The models need one shared list of features. | `features_v1.parquet` has 52,125 rows and 35 features (10 categorical), listed in `FEATURES` and `CATEGORICAL` in `src/features.py`. `PEAK_GROUP` replaces `PEAKID` as a feature. |
| 4. Features | The database has no person ID, and experience must not use the future. | A person is identified by normalized `FNAME` and `LNAME` plus `YOB`. Only strictly earlier (year, season) records count, and an unknown season counts as the last one in its year. Without `YOB` the value is NaN (1.7%). The maximum of 65 previous expeditions belongs to real professional leaders, not to merged IDs. |
| 4. Features | A peak's success rate is unstable with few attempts and must not use the current year. | We use only years before the current one, smoothed as (successes + 20 * overall rate) / (attempts + 20), with the overall rate also taken from past years only. |
| 4. Features | Crowding must be known before the start. | It is computed from permits (all 11,854 expeditions) in the same year and season, leaving out the expedition itself. Routes use the cleaned `ROUTE1`. |
| 5. Death risk | Hired staff appear in `members` only after significant events, so their death rate cannot come from member records. | The unit is one participation in one expedition above base camp. For clients (members on the permit above base camp, not `BCONLY`, not `NOTTOBC`) we count member records, and for hired staff we use the counters `HDEATHS / TOTHIRED`, since `TOTHIRED` counts hired staff above base camp. The counters match member records in every expedition (845 member and 342 hired deaths). Rates are per 1,000 participations with 95% Wilson intervals. |
| 5. Death risk | A rate per participation is not a yearly risk per person: hired staff work several expeditions per season and pass dangerous zones more often. | We report risk per participation and state this caveat in the report. |
| 5. Death risk | Clients and hired staff climb different peaks in different years. | We compare them only on the 7,707 expeditions (1950-2025) that have both above base camp, with a 500-sample bootstrap over expeditions for the risk ratio. Hired / clients: 0.68 (95% CI 0.56-0.81). 14 expeditions have `HDEATHS > TOTHIRED` and 17 hired deaths fall in expeditions with `TOTHIRED = 0`; both stay as recorded. |
| 5. Death risk | Descriptive trends need older years and are not a model evaluation. | Trends use 1950-2025, including 2022-2025, and group rates use the `features_v1` population (1990-2025). No death-risk model is trained, so the test set stays closed. |
| 6. Models | Random folds do not see the change over time: CV log loss was 0.594 on a trial run against 0.648 on validation. | Tuning folds go from past to future inside train (fit to 2005, 2009, 2013; hold 2006-2009, 2010-2013, 2014-2017). CatBoost CV log loss is now 0.642 against 0.639 on validation. |
| 6. Models | Early stopping keeps 20% of train out of the final fit, and validation AUC moves with the seed and the fold (0.691-0.702 with the same settings). | The number of trees is the mean best iteration over the time folds, the model is refitted on all fitting rows, and the prediction is the mean of 5 seeds. |
| 6. Models | CatBoost picks its learning rate from the number of trees, so a refit with fewer trees would quietly change it. | Typical settings use a fixed learning rate of 0.1 (the automatic value was 0.099). |
| 6. Models | Tuned settings could win just because of seed noise. | Tuned CatBoost replaces the typical settings only if its gain in validation log loss is larger than the spread between seeds. After 30 trials the gain was 0.0006 against a spread of 0.0038, so the typical settings stay. |
| 6. Models | The plan named CatBoost as the main model, but tuned LightGBM did better on the time folds (log loss 0.631 against 0.636-0.642) and on validation (AUC 0.712 against 0.689-0.693). The paired difference is +0.018 to +0.023, with the CI above 0 when whole expeditions are resampled; the error analysis has the wider level. LightGBM also moves less between seeds (AUC 0.710-0.712 against 0.682-0.696). | LightGBM is the final model and CatBoost stays in every comparison. The frozen settings are in `reports/final_model.json`. |
| 6. Models | We wanted to know whether the win comes from the library or from the settings. Categories are hard here: `AGENCY` has 672 levels, `ROUTE1` 599, and 18-25% of later rows have an agency not seen before. | We changed one setting at a time. Default LightGBM reaches validation AUC 0.690, about the same as CatBoost at 0.689. Adding the tuned category smoothing to the default gives 0.704, tuned settings with weak smoothing fall to 0.678, and tuned settings without sampling reach 0.707. Strong category smoothing is the main cause. The CatBoost search had no such parameter, so this compares tuned setups rather than libraries. |
| 6. Models | Dropping `ROUTE1` and `AGENCY` helps CatBoost (time folds AUC 0.689 against 0.677, validation 0.697 against 0.689). | LightGBM does not need this: time folds 0.693 against 0.692, validation 0.711 against 0.712, and the paired CI includes 0. Both features stay. |
| 6. Models | Probabilities need calibrating, and validation is the only data left for it. | Platt scaling, learned on non-anomalous validation rows, moves validation log loss from 0.622 to 0.620. Isotonic makes it worse (0.625), so we do not use it. |
| 6. Models | Validation and test years lie outside the training range, so a tree can only treat new years like the last one it saw, and `YEAR` could hide what really changes. | `YEAR` is not a model input. Adding it changes almost nothing: time folds AUC 0.692 in both cases, validation AUC 0.711 against 0.712, validation log loss 0.621 against 0.622. |
| 6. Models | Anomalous seasons (avalanche, earthquake, COVID) could teach the model rules that do not hold in normal seasons. | The 521 anomalous rows of 37,969 in train are left out of fitting, and validation metrics are shown with and without these seasons. Fitting with them gives a mixed result on validation (AUC 0.719 against 0.712, log loss 0.625 against 0.622), so we kept the plan. |
| 7. SHAP | Columns carrying the same information share credit, so one column can look weak while its group is strong. | We show importance per feature and per group, where a group effect is the sum over its columns. SHAP comes from LightGBM (`pred_contrib=True`), averaged over 5 seeds, on validation only. |
| 7. SHAP | Crowding has the largest group SHAP (0.43), but it may just stand in for other things: `CROWD_ROUTE` follows `COMRTE` (Spearman 0.68) and `YEAR` (0.38). | Within one year on one peak and season its link with success is almost zero (-0.065 to 0.092), and dropping the group costs only 0.005 of validation AUC. We report it as a proxy for popular commercial routes and not as a cause. |
| 8. Leakage audit | Unit tests check each historical feature on small examples, but a mistake could appear only when everything is built together. | We flip every summit label from a chosen year on and rebuild all features. From 2005, 0 of 18,502 past rows change and all 23,837 later rows up to 2021 do. From 2018, 0 of 38,769 past rows change and all 3,334 later rows do. The 2018 case is also an automatic test. |
| 8. Leakage audit | We need to know what a real leak would look like. | With one forbidden field added to the final model (validation AUC 0.712), the AUC becomes 0.830 with `HIGHPOINT`, 0.847 with `TERMREASON`, 0.916 with `SMTMEMBERS` and 0.978 with `MPERHIGHPT`. The guard stops all of them. The best legal feature alone reaches 0.632, while `HIGHPOINT` alone reaches only 0.627, so looking at single features would not be enough. |
| 8. Leakage audit | The pipeline could find signal where there is none. | With shuffled training labels validation AUC is 0.453-0.478, a column of random noise ranks 31 of 35 by SHAP, and no feature holds more than 11.6% of total SHAP. |
| 8. Leakage audit | `COMRTE` and `STDRTE` are missing more often when success is lower (0.265 against 0.407 on the fitting rows). | This follows the recording era: 36.5% missing in the 1990s, 9.4% in the 2000s and none from 2010. No row from 2018 on has a gap, so it cannot affect validation or test metrics. |
| 8. Leakage audit | Category text could carry notes written after the fact. | Only 3 levels with at least 30 rows almost always end the same way (140 of 43,011 rows), and all are hard routes. The outcome words we found are agency names (48 levels) and route names (9 levels, 91 rows), and `STATUS` has none. Nothing needed changing. |
| 8. Leakage audit | `TOTHIRED` and `NOHIRED` count staff above base camp, so they are not strictly known before the start. `TOTMEMBERS` and the crowding columns also use the final counts in the database. | We kept them because hired staff per member is a research question. Without `TOTHIRED`, `NOHIRED` and `HIRED_PER_MEMBER` validation AUC is 0.702 against 0.712, so any leak through them is at most 0.010. This is stated as a limitation. |
| 8. Leakage audit | 52 rows come from 12 expeditions that did not attempt the climb (`TERMREASON` 12), and none of them succeeded. | We kept them: `TERMREASON` is an after-the-climb field, and removing them would change `features_v1` under all results. They raise validation AUC only from 0.7095 to 0.7116. The final metrics will be reported with and without them. |
| 8. Leakage audit | The population filters (`NOTTOBC`, `BCONLY`, `MSMTTERM`, `MDISPUTED`, `MCLAIMED`) use fields filled in after the start. | They only define who counts as a climber and are never features. The metrics therefore describe members who intended to climb and went above base camp, which we state as a limitation. |
| 8. Leakage audit | Aggregate success rates for 2022-2025 were printed during data exploration, and `DEATH` trends cover 2022-2025 on purpose. | Features, filters and models were not changed because of them. No model was tuned or evaluated on the test set. A death model would have no blind test. |
| 9. Error analysis | Climbers on one peak in one season share weather and conditions, so resampling expeditions can make intervals too narrow (errors inside one expedition have ICC 0.37, inside one peak x year x season 0.16). | We report both levels. Validation AUC 0.712 has [0.683, 0.739] by expedition and [0.652, 0.760] by peak x year x season. LightGBM stays above the rule at both levels (+0.111, [0.055, 0.157] at the wider level). Against logreg (+0.026) and CatBoost (+0.023) the interval stays above 0 only by expedition: at the wider level it is [-0.003, 0.051] and [-0.003, 0.055], although the log loss gain holds. |
| 9. Error analysis | Pooled AUC also rewards knowing which peak is easier. | AUC inside groups is 0.677 inside one peak, 0.672 inside one peak x year x season and 0.657 inside one expedition. The model still separates climbers facing the same conditions, only less sharply. |
| 9. Error analysis | The largest miss is Manaslu in autumn 2021: predicted 52%, actual 6%. | 88% of these rows come from expeditions that ended at a foresummit (`TERMREASON` 2), which does not count as the main peak. The same happened in 2016 (59%), while 2017-2020 have none. We keep the labels as recorded. Removing these 189 rows moves validation AUC only from 0.712 to 0.711, and 2021 stays weak without them (0.659). |
| 9. Error analysis | Overall metrics could hide weak groups. | After calibration women have AUC 0.707 and men 0.711, with gaps of at most 2.4 pp, and age groups from 25 to 55+ range 0.688-0.707. The weak spots are 2020 (0.612, 136 rows), 2021 (0.675), winter (78 rows, gap +23 pp), Lhotse (0.544), Kangchenjunga (0.510) and climbers with 5 or more earlier expeditions (gap +10 pp). Small groups have wide intervals. |
| 10. Structure | Names of notebooks and outputs carried author letters and an order unrelated to how the work runs, and paths and constants were repeated across files. | Notebooks are numbered in run order (01-06). Tables and figures are named after the notebook that writes them and live in `reports/tables` and `reports/figures`. Shared paths and constants are in `src/config.py` and the figure style in `src/style.py`. We reran the outputs and compared them, and no number changed. |

## Experiment log

| Date | Model / change | ROC AUC (val) | Brier (val) | Conclusion |
|---|---|---|---|---|
| 2026-11-04 | Constant (train mean) | 0.500 [0.500, 0.500] | 0.271 [0.265, 0.277] | LL 0.736; no-anomalous val: AUC 0.500, Brier 0.274, LL 0.742. Reference. Val success rate is higher than train, so Brier/log loss show the shift. |
| 2026-11-04 | Rule: peak x season rate | 0.601 [0.567, 0.632] | 0.250 [0.243, 0.256] | LL 0.693; no-anomalous val: AUC 0.614, Brier 0.249, LL 0.692. Rule baseline, no ML; peak x season, past years only. |
| 2026-11-04 | PEAK_SUCCESS_PRIOR as probability | 0.543 [0.510, 0.575] | 0.274 [0.266, 0.283] | LL 0.746; no-anomalous val: AUC 0.546, Brier 0.278, LL 0.753. Existing feature used as probability (peak only, not season-specific). |
| 2026-11-04 | Logistic regression, age hinge at 40 | 0.686 [0.656, 0.716] | 0.228 [0.216, 0.240] | LL 0.656; no-anomalous val: AUC 0.714, Brier 0.217, LL 0.627. Linear model with age knot at 40; YEAR excluded. |
| 2026-11-04 | CatBoost default | 0.691 [0.660, 0.722] | 0.227 [0.216, 0.237] | LL 0.648; no-anomalous val: AUC 0.726, Brier 0.217, LL 0.623. Default params, early stopping on a grouped fold inside train; YEAR excluded. |
| 2026-11-04 | CatBoost typical (5 seeds) | 0.689 [0.657, 0.721] | 0.224 [0.216, 0.232] | LL 0.639; no-anomalous val: AUC 0.711, Brier 0.219, LL 0.628. learning rate 0.1, 30 trees from time folds, refit on all fitting rows; CV 0.6415. |
| 2026-11-04 | CatBoost tuned (5 seeds) | 0.693 [0.662, 0.724] | 0.224 [0.217, 0.231] | LL 0.639; no-anomalous val: AUC 0.714, Brier 0.219, LL 0.628. Optuna 30 trials on time folds, 39 trees, refit on all fitting rows; CV 0.6361. |
| 2026-11-04 | LightGBM tuned (5 seeds) | 0.712 [0.683, 0.739] | 0.216 [0.209, 0.224] | LL 0.622; no-anomalous val: AUC 0.737, Brier 0.210, LL 0.608. Final model. Optuna 12 trials on time folds, 176 trees, refit on all fitting rows; CV 0.6307. |
| 2026-11-04 | LightGBM tuned (5 seeds) / before | 0.712 [0.683, 0.739] | 0.216 [0.209, 0.224] | LL 0.622; no-anomalous val: AUC 0.737, Brier 0.210, LL 0.608. Calibrated on the other half of val (out-of-fold), anomalous rows excluded from calibrator fit. |
| 2026-11-04 | LightGBM tuned (5 seeds) / Platt | 0.712 [0.684, 0.739] | 0.216 [0.206, 0.225] | LL 0.620; no-anomalous val: AUC 0.737, Brier 0.206, LL 0.598. Calibrated on the other half of val (out-of-fold), anomalous rows excluded from calibrator fit. |
| 2026-11-04 | LightGBM tuned (5 seeds) / isotonic | 0.709 [0.682, 0.736] | 0.217 [0.207, 0.226] | LL 0.625; no-anomalous val: AUC 0.734, Brier 0.207, LL 0.603. Calibrated on the other half of val (out-of-fold), anomalous rows excluded from calibrator fit. |
| 2026-11-04 | CatBoost typical (5 seeds) / before | 0.689 [0.657, 0.721] | 0.224 [0.216, 0.232] | LL 0.639; no-anomalous val: AUC 0.711, Brier 0.219, LL 0.628. Calibrated on the other half of val (out-of-fold), anomalous rows excluded from calibrator fit. |
| 2026-11-04 | CatBoost typical (5 seeds) / Platt | 0.689 [0.658, 0.720] | 0.223 [0.213, 0.232] | LL 0.635; no-anomalous val: AUC 0.709, Brier 0.216, LL 0.620. Calibrated on the other half of val (out-of-fold), anomalous rows excluded from calibrator fit. |
| 2026-11-04 | CatBoost typical (5 seeds) / isotonic | 0.684 [0.653, 0.715] | 0.224 [0.214, 0.233] | LL 0.640; no-anomalous val: AUC 0.706, Brier 0.217, LL 0.624. Calibrated on the other half of val (out-of-fold), anomalous rows excluded from calibrator fit. |
| 2026-11-04 | LightGBM without ROUTE1/AGENCY (5 seeds) | 0.711 [0.682, 0.738] | 0.217 [0.209, 0.225] | LL 0.622; no-anomalous val: AUC 0.738, Brier 0.210, LL 0.607. Sensitivity check only; the model keeps these features. |
| 2026-11-04 | CatBoost without ROUTE1/AGENCY (5 seeds) | 0.697 [0.668, 0.726] | 0.221 [0.213, 0.229] | LL 0.632; no-anomalous val: AUC 0.718, Brier 0.215, LL 0.620. Sensitivity check only; the model keeps these features. |
