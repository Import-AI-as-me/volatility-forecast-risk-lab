# References and the limits of the project claim

Primary sources checked on 2026-10-09. These references motivate the baselines
and interpretation; the project is not a replication of every cited method.
Volatility targeting, smoothing, rebalancing bands, and evaluating forecasts
through downstream decisions are established ideas. This repository currently
provides a reproducible research prototype and an elementary controlled
diagnostic. It does not establish a novel method, a publishable contribution,
tradable alpha, or profitable live execution.

1. **Fulvio Corsi (2009), “A Simple Approximate Long-Memory Model of Realized
   Volatility.”** *Journal of Financial Econometrics*, 7(2), 174–196.
   [Publisher DOI](https://doi.org/10.1093/jjfinec/nbp001);
   [author's publication record](https://people.unipi.it/fulvio_corsi/pubblicazioni/).
   The original HAR-RV paper motivates a parsimonious baseline using volatility
   information at multiple horizons. This project's log-HAR features, crypto
   calendar, forecast timing, and train-only retransformation are implementation
   choices, not an exact reproduction of the original experiment.

2. **Alan Moreira and Tyler Muir (2017), “Volatility-Managed Portfolios.”**
   *The Journal of Finance*, 72(4), 1611–1644.
   [Published paper](https://onlinelibrary.wiley.com/doi/10.1111/jofi.12513).
   This is foundational evidence on adjusting portfolio exposure in response to
   volatility. Its reported factor-portfolio results do not transfer automatically
   to a capped, long-only crypto-and-cash rule. Here net returns are secondary
   diagnostics under specified cost assumptions; the main endpoint concerns risk
   tracking.

3. **Federico Vittorio Cortesi, Giuseppe Iannone, Giulia Crippa, Tomaso Poggio,
   and Pierfrancesco Beneventano (2026), “Same Error, Different Function: The
   Optimizer as an Implicit Prior in Financial Time Series.”**
   [arXiv:2603.02620v1, 3 March 2026](https://arxiv.org/abs/2603.02620v1).
   This preprint already studies similar aggregate forecasting errors with
   different temporal behavior and portfolio turnover. It rules out presenting
   “same error, different decisions” as this project's new idea. Our permutation
   experiment isolates an elementary distinction between average losses,
   overshoot duration, and turnover; it does not reproduce their optimizer study.

4. **Nicolas Bianco and Mauro Bernardi, “Flexible variational approximations
   for stochastic volatility-managed portfolios.”**
   [arXiv:2212.07288v2, revised 31 August 2026](https://arxiv.org/abs/2212.07288v2)
   (first submitted in 2022). This preprint studies variational Bayes smoothing of
   stochastic-volatility forecasts, leverage, turnover, and performance after
   transaction costs. Smoothing is therefore a required comparator, not a novelty
   claim. The simple exposure-smoothing comparator here is not an implementation
   of the paper's variational method.

5. **Zefeng Bai, Dessislava Pachamanova, Victoria Steblovskaya, and Kai Wallbaum,
   “Target volatility strategies: optimal rebalancing boundary for transaction
   cost minimization.”** *Financial Markets and Portfolio Management*, 40,
   245–272 (2026); first published online 29 July 2025.
   [Published paper](https://link.springer.com/article/10.1007/s11408-025-00486-5).
   This work optimizes rebalancing boundaries to reduce costs while controlling
   risk. It motivates the no-trade-band comparator. Selecting a band on a
   validation set here neither reproduces its optimization problem nor establishes
   that a band is theoretically optimal.

6. **Andrzej Tokajuk and Jarosław A. Chudziak (2026), “Loss Choice or Model Choice?
   The Role of Forecast Level in Cryptocurrency Volatility Forecasting.”**
   [arXiv:2609.27024v1, 22 September 2026](https://arxiv.org/abs/2609.27024v1).
   The authors' arXiv record reports acceptance to ADMA 2026. Its comparisons of
   losses, models, validation-based forecast-level alignment, and VaR make level
   effects a material confound. That motivates our validation-matched static
   exposure comparator. Matching mean exposure is a narrower control than their
   forecast-level alignment and does not remove every policy difference.

7. **Binance, Binance Public Data and Binance Vision Dataset Terms.**
   [Official archive documentation](https://github.com/binance/binance-public-data);
   [dataset terms, version 1.0, updated 26 August 2026](https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md).
   The documentation specifies microsecond timestamps for spot data from
   2025-01-01, per-archive SHA-256 checksum files, and possible later archive
   revisions. Preserve the retrieval manifest and checksums for reproducibility.
   The dataset terms specify CC BY-NC-SA 4.0 except where otherwise designated,
   plus additional terms covering derived outputs. The repository's MIT label
   must not be treated as the market dataset's license.
