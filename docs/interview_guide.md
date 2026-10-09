# How to explain this project in a QR interview

## A 60-second explanation

I studied how errors in volatility forecasts affect a simple rule that divides
money between crypto and cash. I used seven years of public five-minute BTC and
ETH data. I compared linear HAR regression with gradient boosting, always
training on past data and testing on later data. I also tested a buffer that
reduces the amount invested after recent forecasts have been too low. In the
two-year final test, this buffer tracked the risk target worse than a simple
fixed reduction in investment. The fixed rule was matched to the buffer's
average investment during validation. A separate simulation showed that the
order of forecast errors can change how long risk stays above its target and
how much trading is needed, even when average losses stay the same. I kept the
negative result and recorded the data checks, timing, cost assumptions, and
limits of the analysis.

Use this wording only after you have reviewed and understood the code. The first
version was developed with AI assistance. Describe your own work honestly, and
be ready to explain the methods, results, and assumptions.

## Questions to be ready for

1. **How many training examples are there?** Why do many five-minute price
   records give only a few thousand daily targets? Why should training and test
   dates stay in time order instead of being split at random?
2. **How are forecasts scored?** What does QLIKE measure? Why can it rank models
   differently from squared error in log variance?
3. **How do log forecasts become variance forecasts?** Why is
   `exp(predicted log variance)` generally different from the expected variance
   given the model's inputs? What does the correction based on training errors
   (the smearing factor) do? Why does it not guarantee accurate forecast levels
   on new data?
4. **What information is available before a trade?** Explain the `t−2` cutoff
   and the monthly model fits that use a growing set of past data. How is fitting
   a model different from choosing its hyperparameters?
5. **What does the simulation show?** With constant true variance and a rule
   that uses only the current forecast, why does changing the order of the same
   errors leave average losses unchanged? Which results depend on the order?
6. **Does the buffer help, or does it just invest less?** Explain the fixed
   investment multiplier used for comparison. Why does the risk loss penalize
   being both above and below the target? Why can average investment still
   differ between the two rules during the final test?
7. **How are trades and costs counted?** Prices change the share of wealth held
   in the asset before the next trade. How does the backtest account for this?
   Derive the equation in `rebalance_cost`, where trading costs are paid from
   the account and the target weight is measured after costs.
8. **What do the uncertainty intervals mean?** Why resample blocks of dates
   rather than single dates? What assumptions does this need? What uncertainty
   is left out when the models and selected rules stay fixed in each sample?
9. **What are the limits?** Explain the effects of using two selected
   cryptoassets, one exchange, archives that can be revised, measured variance
   as an estimate of unobserved true variance, and assumed trading costs.
10. **What would you study next?** After this negative result, what new idea
    would you test? How would you get new test data without choosing the new
    method based on the final test results already seen here?

## Possible resume wording

“Built a Python project to forecast volatility and test rules for dividing money
between crypto and cash. Used 1.47M public five-minute price records, compared
HAR and gradient boosting with tests in time order, and included trading costs
and paired block-bootstrap intervals. Found that a buffer based on past forecast
errors did not improve risk tracking against a fixed investment reduction.”

The 1.47M price records are not 1.47M independent daily training examples. This
wording also requires the same honest account of your own role described above.
Do not say that the project found a trading edge, guarantees a risk limit, was
peer-reviewed, or produced results in live trading.

See the [results](../reports/generated/RESULTS.md) for the numbers,
the [study plan](protocol.md) for the test rules, and the
[simulation explanation](theory.md) for the math.
