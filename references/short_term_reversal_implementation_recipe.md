# Short-term reversal — implementation recipe

**Sources (all read directly, not from memory):**
- Lehmann (1990), "Fads, Martingales, and Market Efficiency" —
  `references/Fads_Martingales_and_Market_Efficiency Short Term Reversal vol1.pdf`
  — the core cross-sectional reversal mechanics.
- Nagel (2011), "Evaporating Liquidity" —
  `references/Evaporating Liquidity Short Term Reversal vol 2.pdf` — the
  VIX-conditional sizing overlay and the individual-vs-industry tier comparison.
- Blitz, van der Grient, Honarvar (2023), "Reversing the Trend of Short-Term
  Reversal" — `references/Reversing the Trend Short Term Reversal vol3.pdf` —
  cross-check only (confirms this project's 1d/5d/10d lag choice and
  sector-demean design revive an otherwise-decayed reversal premium; did not
  change the construction below).

Derived notes only — this file is not an external source. Regenerated 2026-07-24
during repo reconstruction to match `src/signals/short_term_reversal.py` and
`src/signals/vix_overlay.py` exactly.

## Core construction (Lehmann 1990)

The first genuinely CROSS-SECTIONAL signal in this project — an asset's score
depends on its return relative to its own SECTOR peer group
(`data.sectors.SECTORS`), not the full 42-name universe (comparing AUDUSD against
Corn has no clean interpretation, unlike Lehmann's single-equity-market
cross-section) and not on its own history alone.

```
weight_i = -(R_i - R_bar)          # go long recent losers, short recent winners,
                                     # relative to the peer-group mean return
```

Implemented via `cross_sectional_demean` — a RAW (non-rank) demean, deliberately
NOT `signals.transforms.cross_sectional_rank` (that shared function was extracted
later, for carry/XSMOM/value; this module predates it and uses Lehmann's own
raw-magnitude construction directly). No sign flip inside the demean step itself —
callers apply that, since the sign convention differs by tier.

## Two tiers (Nagel 2011)

Mirrors Nagel's own individual-stock vs. industry-portfolio comparison (his
industry-level reversal strategy is unconditionally near-zero but still earns a
real, VIX-conditional premium):

- **individual** — vol-standardized `lag`-day return, demeaned WITHIN each sector,
  sign-flipped. `signals.short_term_reversal.individual_reversal_signal`.
- **sector** — sector-level equal-weighted composite of the same vol-standardized
  returns, demeaned ACROSS sectors, sign-flipped, then broadcast back to every
  member asset. `signals.short_term_reversal.sector_reversal_signal`.

## Vol standardization

```
vol_standardized_return = lag_return / (annualized_vol / sqrt(252) * sqrt(lag))
```

Puts assets with very different volatility levels on a comparable scale before
cross-sectional demeaning — the same reasoning `vol_targeted_sign_signal` gives for
scaling by vol, applied here to the raw return itself rather than to position size.

## Three parallel lags — no headline pick

`LAGS = (1, 5, 10)` trading days, same discipline as momentum/breakout/crossover's
own multi-spec reporting. Confirmed against Blitz et al. (2023): shorter lookbacks
strengthen the reversal effect, consistent with including 1d/5d as parallel specs
rather than only a slower one. 6 total Books: `{individual, sector} × {1, 5, 10}d`
via `build_all_reversal_signals`.

## VIX-conditional sizing overlay (Nagel 2011)

Nagel finds short-term reversal's expected return is strongly, positively
predictable by the VIX level — reversal profits compensate for liquidity
provision, richest exactly when funding/liquidity conditions are stressed.

```
multiplier_t = VIX_t / rolling_mean(VIX, 252)_{t-1}     # shifted 1 day (Rule 3)
```

A ratio relative to VIX's own trailing history (not the raw level) stays
comparable across VIX's very different historical regimes (the ~10-90 pre-2008
range vs. the 2020 COVID spike above 80).

**Critical ordering bug, found and fixed live while building this:** the
multiplier MUST be applied to the ALREADY gross-exposure-normalized position array
(`signals.vix_overlay.apply_size_multiplier`, called AFTER
`backtest.engine.normalized_positions`), never to the raw pre-normalization
signal. A uniform daily scalar applied to every asset's position factors out of
both the position sum and the normalization denominator identically — applying it
pre-normalization is a silent no-op. This was caught because "simple" and
"VIX-adjusted" Sharpe came back bit-for-bit identical in every cell.

## Documented result

Turnover is extremely high (109-362x annualized depending on spec) and net-of-cost
Sharpe is deeply negative across every one of the 6 specs — the first outright
unprofitable result of the signal families built to that point (train net Sharpe
-0.5 to -2.8). VIX-conditioning, re-checked with Newey-West HAC standard errors
(20 lags): individual-tier is NOT statistically distinguishable from zero (t=0.58,
p=0.56); sector-tier IS significant (t=2.33, p=0.02) despite a tiny R² (0.17%),
consistent with Nagel's own industry-level finding. Neither rescues net-of-cost
profitability. See CLAUDE.md's "Short-term reversal" row for the full history.
