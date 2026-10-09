# Data and derived-output terms

Market-data source: **Binance Vision / Binance Public Data**, BTCUSDT and ETHUSDT
spot five-minute klines, January 2019–December 2025. This repository has no
affiliation with, sponsorship by, or endorsement from Binance.

Sources:

- [Public-data documentation](https://github.com/binance/binance-public-data)
- [Dataset terms, version 1.0, updated August 26, 2026](https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md)
- [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

The dataset terms provide free noncommercial research access and require
attribution and CC BY-NC-SA 4.0 licensing of redistributed derived works.
The code repository's MIT label must not be read as an MIT license for the data.

**All market-derived tables, charts, forecasts and empirical reports in
`reports/generated/`, and the empirical findings reproduced in README files,
are attributed to Binance Vision and supplied under CC BY-NC-SA 4.0.**
Changes: raw five-minute prices were aggregated to UTC daily realized variance;
forecasts, hypothetical allocations and summary diagnostics were calculated.
Manifest hashes identify the source snapshot. Dataset terms may be updated;
consult the linked source terms when acquiring data for a new use.

Raw archives, processed daily data and full daily forecast/decision paths are
kept local and excluded from the release. The downloader retrieves them from
the original provider. The purely synthetic simulation and original code are
covered by the MIT license. Generic code may be reused with other appropriately
licensed data; this does not grant commercial rights to this market dataset or
its derivatives.

This project uses the data for noncommercial academic/educational research.
Commercial deployment of the provider's data/derivatives requires the separate
authorization described in the provider's terms.
