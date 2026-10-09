# Research notes

## Original study plan

The original settings are in [configs/study.json](../configs/study.json). The run
record saves a hash of this file so the exact version can be checked. The
[study plan](protocol.md) was written before the first model run. It was an
internal plan; it was not registered with an outside service.

The project started with a question about forecast errors and risk control. It
did not start with a proven trading strategy or a claim that earlier research
had missed this question.

## Data checks

Before the models were tested on market data, the data code was updated to check
for the exact first and last five-minute bars of each day. These checks are
stored as `valid_open` and `valid_close`. The daily data was then rebuilt.

All opening and closing boundaries are present for 2022–2025. Missing
five-minute records are marked and left missing. The first day has no closing
price from the previous day, so its daily realized variance (RV) is not valid.

## First complete run

The validation data selected:

- Buffer strength `gamma = 0.5`.
- Smoothing weight `alpha = 1.0`, which means no smoothing.
- A no-trade band of 3 percentage points of portfolio weight.

In the two-year final test, the buffer performed worse on the main risk-tracking
measure than the fixed rule matched to its average validation investment. The
candidate parameter values, assets, main measure, and test dates were not
changed after seeing this result.

## Code and documentation changes after the first run

- Formatted the code and made the units of the risk measures clearer in tables.
- Added hashes of the code, software environment, and output files to the checks
  on saved forecasts. These checks help detect changed or outdated files.
- Explained that parameters are chosen using validation backtests with zero
  trading costs. Also explained that the daily risk measure is an approximation.
  The separate return calculation accounts for changes in portfolio weights
  caused by asset returns.
- Added independent checks of the account calculations. Added tests that change
  future data and check that earlier forecasts and decisions stay unchanged.
- Changed the way saved forecast CSVs are read so floating-point numbers keep
  their exact values. A comparison between a fresh run and a run using saved
  forecasts had found differences in the last digits.

These changes do not change the research design. The final reports are generated
again from the final code.

Record any later correction, new model, new data split, or new parameter search
here and label it as exploratory work. The results for 2024–2025 have already
been seen. Those years cannot be treated as a new, unseen test for later ideas.
