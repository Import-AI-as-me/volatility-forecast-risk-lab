# What temporal error ordering can and cannot change

This controlled experiment is an elementary diagnostic, not a novel theorem or
evidence of a profitable trading strategy. It separates marginal forecast error
from the temporal arrangement of those errors. It deliberately draws no market
returns and makes no empirical claim about market volatility persistence.

## Setup and timing

At observation index `t`, a variance forecast is available before an exposure is
chosen for that observation. The true daily variance is the constant

\[
v=0.04/365,
\]

so its annualized volatility is 20%. This artificial diagnostic uses 365 as its
declared annualization convention; that choice does not prescribe the convention
for trading-day market data. The target annualized volatility is
\(\sigma_\star=0.15\). A forecast has log-variance error
\(e_t=\log(\widehat v_t/v)\), hence \(\widehat v_t=v\exp(e_t)\).

The exposure rule is memoryless:

\[
w_t=g(e_t)
=\min\left\{1,\frac{\sigma_\star}{\sqrt{365v}}\exp(-e_t/2)\right\}.
\]

The annualized **conditional risk proxy** is
\(R_t=\sqrt{365v}\,w_t\). It is not a rolling estimate of realized volatility,
and it is not a realized portfolio return. With no cap binding,
\(R_t=\sigma_\star\exp(-e_t/2)\).

The experiment contains 240 observations, 120 errors equal to \(-a\) and 120
equal to \(+a\), with \(a=0.4\). All scenarios begin with a negative error and
end with a positive error. Blocks of length 1, 5, and 20 change the order only.
The default exposures are approximately 0.9161 for \(-a\) and 0.6140 for \(+a\),
so the cap is inactive. Larger negative errors can activate the cap; permutation
invariance below still holds because the cap is also memoryless.

## Marginal forecast scores are invariant

We use the nonnegative normalized QLIKE loss

\[
\ell_Q(v,\widehat v)
=\frac{v}{\widehat v}-\log\left(\frac{v}{\widehat v}\right)-1.
\]

In log-error coordinates this is
\(\ell_Q(e)=\exp(-e)+e-1\). For every permutation \(\pi\),

\[
\frac1n\sum_{t=0}^{n-1}\ell_Q(e_{\pi(t)})
=\frac1n\sum_{t=0}^{n-1}\ell_Q(e_t),
\qquad
\frac1n\sum_{t=0}^{n-1}e_{\pi(t)}^2
=\frac1n\sum_{t=0}^{n-1}e_t^2.
\]

This follows by reindexing the finite sum. In this balanced two-point experiment,
the means are exactly \(\cosh(a)-1\) and \(a^2\), respectively, up to floating-point
roundoff in the implementation.

## Pointwise risk-tracking loss is also invariant

Because the truth is constant and the policy has no memory, every pointwise loss
\(L(R_t,\sigma_\star)\) is a function of \(e_t\) alone. Therefore both its sum and
mean are invariant under permutation. In particular,

\[
\operatorname{MSE}_{\rm risk}
=\frac1n\sum_t (R_t-\sigma_\star)^2
=\frac{(R_- -\sigma_\star)^2+(R_+ -\sigma_\star)^2}{2},
\]

where \(R_-=\sqrt{365v}\,g(-a)\) and \(R_+=\sqrt{365v}\,g(a)\).
The fraction of observations above the target, and the sum of any pointwise
overshoot penalty, are likewise unchanged. Under the default parameters the
overshoot fraction is exactly one half.

Consequently, **persistence alone does not increase the mean or integrated
pointwise risk-tracking loss in this experiment**. Claiming otherwise would
confound ordering with different errors, a different true-variance path, or a
policy that uses past observations.

**Metric scale:** the simulation CSV's `risk_tracking_mse` uses squared
annualized volatility units, with volatility expressed as a decimal:
\(n^{-1}\sum_t(R_t-\sigma_\star)^2\). The empirical report's `risk_mse`
instead uses the dimensionless squared relative deviation
\(n^{-1}\sum_t(R_t/\sigma_\star-1)^2\). For the same risk path and fixed target,

\[
\operatorname{MSE}_{\rm relative}
=\operatorname{MSE}_{\rm raw}/\sigma_\star^2.
\]

At the simulation defaults, the raw value is approximately 0.0009211244 and
the corresponding relative value is 0.04093886. Neither is a mean squared
log-risk score. The empirical path uses an ex-post realized-variance proxy,
whereas this diagnostic uses a constant conditional variance; their values
should not be compared as though they came from the same experiment.

## Runs and turnover depend on ordering

With the default parameters, \(R_t>\sigma_\star\) holds exactly when \(e_t=-a\).
In a complete repeated cycle consisting of \(r\) negative errors followed by
\(r\) positive errors, each overshoot run has length \(r\). There are \(n/(2r)\)
runs. The longest and mean run lengths are thus 1, 5, and 20 across the three
scenarios, despite identical time spent above the target. Runs touching the first
or final observation are included in the definition.

Let \(w_H=g(-a)\), \(w_L=g(a)\), and let \(N_{\rm switch}\) count sign switches
between adjacent observations. Entering from initial exposure \(w_{-1}=0\),
total turnover is

\[
\operatorname{TO}
=\sum_{t=0}^{n-1}|w_t-w_{t-1}|
=|w_H-0|+N_{\rm switch}|w_H-w_L|.
\]

For a complete block pattern, \(N_{\rm switch}=n/r-1\). With \(n=240\), the
switch counts are 239, 47, and 11. Longer blocks therefore produce longer
overshoot runs **and lower turnover** in this construction. All three scenarios
include the same initial entry cost in turnover. There is no terminal liquidation.
If a constant linear fee per unit turnover were specified, its total would be that
fee times `total_turnover`; no such fee or portfolio P&L is assumed here.

## Interpretation and limits

The experiment supports a narrow conclusion: scalar average accuracy and
pointwise risk losses do not describe how long a risk breach persists, nor the
trading required to implement the forecasts. It does not show that one ordering
is universally better. Longer runs might matter under a duration-sensitive risk
constraint, while lower turnover might matter under transaction costs.

Changing true variance, using smoothed or state-dependent weights, introducing
position constraints with memory, or measuring return-path drawdowns creates a
different experiment. Those effects cannot be attributed to persistence alone
using the invariance argument above. Holding one random return path fixed while
reordering exposures can also change realized P&L; that is not a contradiction
of a statement about the constant conditional-risk proxy.

A causal bias correction may use only residuals that were observable before
the decision. A trailing correction can lag a sign reversal and briefly worsen
the error. Neither this diagnostic nor its elementary identities establish a
benefit from such a correction. Empirical comparisons need explicit information
timing, out-of-sample evaluation, and the cost of exposure changes.

## Reproducible outputs

`run_simulation(output_dir)` writes `simulation_paths.csv`,
`simulation_summary.csv`, and `simulation.png`. The CSVs expose the error, risk,
weight, score, and turnover calculations. Tests verify the common error multiset,
permutation-invariant means, overshoot runs, and the two-weight turnover formula.
There is no random seed because the entire experiment is deterministic.
