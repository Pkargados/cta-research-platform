"""
portfolio/book.py — Book abstraction: one self-contained signal sleeve.

Adapted from the retired Cross Asset Stat Arb Engine's Book (engine/portfolio/book.py)
— see cleanup.md section 3 for the full evaluation. Two structural changes from the
original:

1. run() is split into composable private steps (_solve_weights, _apply_vol_target,
   _apply_constraints, _period_return_map/_compute_pnl) instead of one ~150-line
   method, so a future book type (e.g. an RV-spread book with z-score entry/exit
   instead of continuous optimizer sizing) can override one step instead of
   copy-pasting the whole thing.
2. Rebalance cadence is an explicit `periods_per_year` parameter instead of a
   hardcoded weekly assumption — this project's signals already support weekly and
   monthly rebalancing (signal_lib.py's weekly_positions/monthly_positions), and a
   future book shouldn't have to fight a baked-in weekly assumption to use either.

A Book owns one signal family's alpha, covariance, optimizer parameters, and
vol-target config end-to-end — self-contained and independently testable. The
Allocator combines Books; it does not know or care how any individual Book computes
its weights.

Not yet exercised against real signals — see WORKFLOW.md Phase 5/7 for when this
actually gets wired up (2+ signal families need to exist first for it to be worth
combining). This is scaffolding, built ahead of that per direct instruction
(cleanup.md section 3), not a claim that Phase 5 has started.
"""

import numpy as np
import pandas as pd
from collections import OrderedDict

from portfolio.optimizer import solve_weights
from backtest.costs import transaction_cost_drag
from data.garch_volatility import _get_garch_functions


def daily_mark_pnl(weights: pd.DataFrame, returns_df: pd.DataFrame, cost_bps: pd.Series = None) -> pd.Series:
    """Mark an already-solved (weekly/monthly/etc.) Book weight path against
    DAILY returns, instead of the one-return-per-rebalance-date PnL `Book.run()`
    itself reports.

    Added 2026-07-23, per direct instruction, to answer a real question:
    tuning `Book` hyperparameters needs enough observations to select against
    without overfitting, and `Book.run()`'s own period-level PnL (one point per
    rebalance date) is scarce even at weekly cadence (WORKFLOW.md Phase 7's
    tuning writeup — ~90-100 validation points, and the resulting tuned
    hyperparameters did NOT generalize to test). Re-solving the optimizer more
    often (which is what the earlier monthly->weekly cadence switch did)
    changes what the STRATEGY actually does — different turnover, different
    reactivity to Sigma_t and realized vol. Daily-marking the SAME solved
    weight path changes only how finely its resulting PnL is MEASURED — the
    weight path itself, and therefore the strategy's real economic behavior,
    is unchanged. These are different knobs; this function is the second one.

    `weights` : the `w_df` a `Book.run()` result already returns (index =
    rebalance dates, columns = assets) — the weight decided AT each rebalance
    date, held constant until superseded by the next one.

    Mechanics: forward-fill `weights` onto `returns_df`'s daily index, then
    `.shift(1)` before multiplying — CLAUDE.md Rule 3 ("always shift(1) a
    signal before using it to trade today"), same convention `backtest.engine.
    normalized_positions` already uses. Without the shift, the weight solved
    ON a rebalance date (using information THROUGH that date, including that
    date's own return) would be multiplied against that same date's own daily
    return — a real look-ahead, not a rounding issue. `.sum(axis=1, skipna=
    True)` — NOT a raw `.values.sum()` — for the same reason `Book.
    _period_return_map`'s own docstring already documents: a plain numpy sum
    propagates NaN across an entire day for every asset the moment ANY one
    scattered missing-return day occurs, which this project's genuinely
    sparser-calendar assets guarantee happens routinely. Does NOT reproduce
    `Book._period_return_map`'s `max_gap_days` stale-position flattening —
    this is a measurement tool over an already-solved weight path, not a
    second risk-management layer; a caller evaluating a `Book` with real
    multi-month covariance gaps should still consult `Book.run()`'s own
    `n_stale_gaps` diagnostic.

    Returns a daily pd.Series, same convention (and directly comparable to)
    `backtest.engine.backtest_signal`'s own daily output for every other
    signal in this project.

    `cost_bps` (added 2026-07-23, alongside `Book`'s own `cost_bps` param):
    optional, pd.Series indexed by asset, one-way cost in basis points. When
    given, deducts `backtest.costs.transaction_cost_drag(weights, cost_bps)`
    — the real per-REBALANCE-DATE cost of the trade that moved the position
    to each new `weights` row — as a lump sum charged on that rebalance
    date's own daily mark (not smeared across the holding period). This
    matters for a fair comparison ACROSS rebalance frequencies specifically:
    without it, a faster-rebalancing weight path would only ever look better
    or equal in daily-marked gross terms (more chances to react), hiding the
    extra turnover cost it actually incurs — exactly the blind spot flagged
    in WORKFLOW.md Phase 7 before wiring real costs into `Book.run()` itself.
    Default None leaves this function's prior (gross) behavior unchanged.
    """
    daily_weights = weights.reindex(returns_df.index).ffill().shift(1)
    common_cols = daily_weights.columns.intersection(returns_df.columns)
    gross = (daily_weights[common_cols] * returns_df[common_cols]).sum(axis=1, skipna=True)

    if cost_bps is None:
        return gross

    cost_bps_aligned = cost_bps.reindex(weights.columns).fillna(0.0)
    rebalance_cost = transaction_cost_drag(weights, cost_bps_aligned)
    daily_cost = rebalance_cost.reindex(gross.index).fillna(0.0)
    return gross - daily_cost


class Book:
    """
    Independent signal sleeve.

    Parameters
    ----------
    name            : str                — identifier ("momentum_126_skip", etc.)
    alpha_df        : pd.DataFrame (T×N) — alpha in whatever units this book's signal
                                            construction produces (see optimizer.py)
    cov_dict        : dict               — {date: pd.DataFrame} covariance per rebalance date
    gamma           : float              — risk-aversion coefficient
    kappa           : float              — position inertia
    lambd           : float              — L1 transaction-cost penalty
    max_weight      : float              — per-asset position limit (absolute value)
    target_vol      : float              — annualized vol target for this book's PnL
    ewma_halflife   : int                — EWMA halflife (in rebalance periods) for realized vol
    scale_min       : float              — floor on the vol-targeting scale factor
    scale_max       : float              — ceiling on the vol-targeting scale factor
    periods_per_year: int                — 52 for weekly rebalancing, 12 for monthly, etc.
    dollar_neutral  : bool               — passed through to the optimizer (default False — CTA)
    is_active       : bool               — toggled by the Allocator / regime layer
    max_gap_days    : int                — see `_period_return_map`: a real-calendar-day gap
                                            between consecutive valid rebalance dates longer
                                            than this is treated as a risk-model blackout
                                            (flattened to zero PnL for that stretch), not a
                                            normal holding period. Default 60 (~2 months —
                                            tolerates one skipped monthly rebalance from
                                            ordinary data noise; found live 2026-07-22 that
                                            `build_cov_dict`'s 70% min_frac gate can silently
                                            skip 15 CONSECUTIVE month-ends when a broad active
                                            universe's scattered per-asset NaN gaps compound,
                                            producing 400+ day gaps).
    cost_bps        : pd.Series | None   — added 2026-07-23, per direct instruction (WORKFLOW.md
                                            Phase 7's "transaction cost model" follow-on to the
                                            hyperparameter-tuning work). One-way cost in basis
                                            points, indexed by asset — same object
                                            `backtest.engine.backtest_signal`'s own `cost_bps`
                                            param and `backtest.costs.liquidity_tiered_cost_bps`
                                            already produce for the standalone-signal path; reused
                                            here, not reimplemented. Default None preserves EXACT
                                            prior behavior (no deduction) for every existing
                                            caller, same opt-in convention `backtest_signal`
                                            already established. Distinct from `lambd`: `lambd` is
                                            an L1 penalty INSIDE the optimizer's objective (shapes
                                            which weights get solved, ex-ante); `cost_bps` is a
                                            REALIZED cost deducted from PnL AFTER the weights are
                                            already solved (ex-post, via `backtest.costs.
                                            transaction_cost_drag` on the Book's own actual
                                            turnover) — the same real-cost concept as every other
                                            signal's net-of-cost number in this project, now
                                            available at the Book/Allocator level too. IMPORTANT:
                                            this must be the ONLY place transaction costs are
                                            applied on the Book/Allocator path — `alpha_df` fed
                                            into a `Book` must stay a GROSS signal (no signal-level
                                            cost deduction baked in upstream), or costs would be
                                            double-counted. `Book.run()`'s own `"pnl"` therefore
                                            stays gross when `cost_bps=None` and becomes net when
                                            it's supplied; `"gross_pnl"` is always reported
                                            alongside so both are visible, matching this project's
                                            universal gross/net reporting convention.
    vol_estimator   : str                — "ewma" (default, unchanged prior behavior) or "garch".
                                            Added 2026-07-29 after `research/
                                            book_vol_targeting_estimator.py` found GJR-GARCH cuts
                                            QLIKE loss ~60-70% vs. EWMA forecasting a Book's own
                                            realized-PnL volatility (tested on DAILY-marked PnL,
                                            not this parameter's native rebalance cadence directly
                                            — the estimator-class advantage is assumed, not
                                            separately re-proven, to carry over; disclosed, not
                                            silently claimed as re-validated). "garch" walks
                                            forward using `data.garch_volatility`'s own
                                            fit_gjr_garch/filter_gjr_garch (reused directly, not
                                            reimplemented) — refit every `garch_refit_freq`
                                            periods using ONLY realized PnL through the previous
                                            period (point-in-time-safe, same discipline as the
                                            EWMA recursion it replaces), filtered forward with
                                            fixed params between refits. Falls back to the EWMA
                                            recursion during GARCH's own warmup
                                            (`garch_min_warmup` periods) and on any per-period fit/
                                            filter failure (logged, not silently ignored) — never
                                            leaves a period without SOME variance estimate.
    garch_refit_freq: int                — rebalance periods between GARCH refits when
                                            vol_estimator="garch" (default 20, matching
                                            `data.garch_volatility.REFIT_FREQ_DAYS`'s own
                                            practice-matching default — not backtested/tuned).
    garch_min_warmup: int                — minimum realized-PnL observations before the first
                                            GARCH fit is attempted (default 104, ~2 years at
                                            weekly cadence — smaller than the asset-level module's
                                            500-observation floor since that's calibrated for
                                            daily data).
    """

    def __init__(
        self, name, alpha_df, cov_dict, gamma, kappa, lambd, max_weight,
        target_vol, ewma_halflife, scale_min, scale_max,
        periods_per_year=52, dollar_neutral=False, is_active=True, max_gap_days=60,
        cost_bps=None, vol_estimator="ewma", garch_refit_freq=20, garch_min_warmup=104,
    ):
        self.name = name
        self.alpha_df = alpha_df
        self.cov_dict = cov_dict
        self.gamma = gamma
        self.kappa = kappa
        self.lambd = lambd
        self.max_weight = max_weight
        self.target_vol = target_vol
        self.ewma_halflife = ewma_halflife
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.periods_per_year = periods_per_year
        self.dollar_neutral = dollar_neutral
        self.is_active = is_active
        self.max_gap_days = max_gap_days
        self.cost_bps = cost_bps
        self.vol_estimator = vol_estimator
        self.garch_refit_freq = garch_refit_freq
        self.garch_min_warmup = garch_min_warmup

    # ------------------------------------------------------------------
    # Composable steps — override one of these in a subclass for a book type
    # that needs different mechanics (e.g. z-score entry/exit instead of
    # continuous optimizer sizing), rather than copy-pasting run().
    # ------------------------------------------------------------------

    def _solve_weights(self, alpha_t, Sigma_t, x_prev):
        return solve_weights(
            alpha_t, Sigma_t, x_prev,
            self.gamma, self.kappa, self.lambd, self.max_weight, self.dollar_neutral,
        )

    def _apply_vol_target(self, x_t, ewma_var):
        """Cap-aware EWMA power-scaling vol target.

        rv = sqrt(ewma_var * periods_per_year) — annualized EWMA realized vol.
        scale_raw = (target_vol / rv)^2 — power scaling: deleverages harder above
        target, re-levers faster below. scale_cap = max_weight / max(|x_t|) bounds
        the scale so it can't push any position past max_weight before the final
        clip. Returns (scaled weights, scale applied, whether the cap bound).
        """
        rv = float(np.sqrt(max(ewma_var, 0.0) * self.periods_per_year))
        if rv <= 1e-8:
            return x_t, 1.0, False

        scale_raw = (self.target_vol / rv) ** 2.0
        max_abs_x = float(np.max(np.abs(x_t)))
        cap_bound = False
        if max_abs_x > 1e-10:
            scale_cap = self.max_weight / max_abs_x
            if scale_raw > scale_cap:
                scale_raw = scale_cap
                cap_bound = True
        scale_applied = float(np.clip(scale_raw, self.scale_min, self.scale_max))
        return x_t * scale_applied, scale_applied, cap_bound

    def _apply_constraints(self, x_t):
        # Clip before recentering, not after: recenter is a shift by a scalar (the
        # mean), so applying it last guarantees the result is exactly dollar-neutral.
        # Doing it in the other order (recenter, then clip) lets the clip silently
        # reintroduce net exposure — caught by the portfolio smoke test, where it
        # produced sums up to 0.12 away from zero, not float noise. The tradeoff this
        # accepts: recentering after the clip can let a position drift slightly past
        # max_weight, bounded by the recentering shift itself (typically small). For a
        # book with dollar_neutral=True, exact zero net exposure is the property that
        # actually matters — that's the whole point of turning the flag on — so a
        # small, bounded max_weight overshoot is the lesser tradeoff. A joint
        # projection onto both constraints simultaneously would avoid this tradeoff
        # entirely but is out of scope for this scaffolding.
        x_t = np.clip(x_t, -self.max_weight, self.max_weight)
        if self.dollar_neutral:
            x_t = x_t - x_t.mean()
        return x_t

    def _period_return_map(self, returns_df, assets, reb_dates):
        """Sum of daily returns in each (d, d_next] rebalance window — used both for
        the EWMA vol update (in-loop) and final PnL (post-loop). Returns
        (period_ret_map, n_stale_gaps).

        `.sum(skipna=True)` (pandas' own default), not `.values.sum()` — found live
        running this against this project's real 38-asset panel for the first time
        (Phase 7's own first validation pass): a plain numpy `.sum()` on `.values`
        propagates NaN for an asset across the ENTIRE rebalance window the moment a
        single scattered missing-return day falls inside it, which this panel's
        genuinely sparser-calendar assets guarantee happens routinely — every
        period's PnL came back NaN, not just the affected asset's. Treating a
        missing day as a zero contribution that period (skipna) matches every other
        NaN-handling convention already established in this project (e.g.
        `cross_sectional_demean`'s min-group-size gating excludes a missing member
        rather than corrupting the whole group).

        Gaps longer than `max_gap_days` between consecutive REB_DATES (not scattered
        single-day NaNs — an actual missing rebalance point) are a different problem
        and treated differently: found live 2026-07-22 that `portfolio.covariance.
        build_cov_dict`'s own 70% min_frac gate can silently skip many CONSECUTIVE
        month-ends when a broad active-asset universe's scattered per-asset NaN
        gaps compound (no single asset badly broken, but the UNION of everyone's
        small gaps pushes the joint-clean-row rate below the gate for an extended
        stretch) — a real, measured 16-month blackout (2014-04-30 to 2015-08-31)
        left several Books holding a near-max-weight position, completely
        unmanaged, straight through the 2014-2015 oil price collapse and the
        January 2015 SNB franc de-pegging shock. `Book.run()` had no way to notice
        it couldn't verify risk for that long and kept marking the stale position
        against real (large) subsequent moves — a single chart point that read as
        an instantaneous "crash" but was actually 16 months of frozen, unmonitored
        leverage. Contributing a ZERO return for gaps this long is a conservative
        stand-in for "flatten when the risk model can't be verified" — not a claim
        the position was actually flat (it wasn't), a refusal to keep pricing risk
        that was never being measured. See WORKFLOW.md Phase 7 for the full
        investigation.
        """
        period_ret_map = {}
        n_stale_gaps = 0
        for i in range(len(reb_dates) - 1):
            d_curr, d_next = reb_dates[i], reb_dates[i + 1]
            if (d_next - d_curr).days > self.max_gap_days:
                period_ret_map[d_curr] = np.zeros(len(assets))
                n_stale_gaps += 1
                continue
            mask = (returns_df.index > d_curr) & (returns_df.index <= d_next)
            wret = returns_df.loc[mask, assets]
            if len(wret) > 0:
                period_ret_map[d_curr] = wret.sum(axis=0, skipna=True).values
        return period_ret_map, n_stale_gaps

    def _compute_pnl(self, w_df, period_ret_map, assets):
        """Returns (pnl, gross_pnl, turnover_s, real_cost_s, sharpe, max_dd, asset_contributions).

        `pnl` = `gross_pnl` - `lambd`-penalty (ex-ante optimizer smoothing,
        unchanged from before) - real transaction-cost drag (ex-post, only if
        `self.cost_bps` is set — see the Book docstring's `cost_bps` entry for
        why these are two distinct things, not double-counting the same cost
        twice). `gross_pnl` and `real_cost_s` are both returned so a caller
        can see gross-vs-net explicitly, matching this project's universal
        reporting convention (CLAUDE.md's every other signal already reports
        gross AND net; this Book-level path previously couldn't).

        `asset_contributions` (added 2026-07-29, per direct instruction —
        performance attribution): a (T x N) DataFrame, `w_held * nr_held`
        BEFORE the row-sum that collapses it into `gross_pnl` — each asset's
        own exact contribution to that period's GROSS return. Not
        approximated: `asset_contributions.sum(axis=1) == gross_pnl` exactly,
        since it's the same elementwise product, just not yet summed. Scoped
        to gross (not net) — `lambd`'s turnover penalty and `cost_bps`'s real
        cost are portfolio-level quantities from `turnover_s`'s aggregate
        position change, not cleanly attributable to one asset without an
        extra modeling choice this doesn't make."""
        hold_dates = w_df.index.intersection(pd.DatetimeIndex(period_ret_map.keys()))
        ret_rows = {d: pd.Series(v, index=assets) for d, v in period_ret_map.items() if d in hold_dates}
        next_ret = pd.DataFrame(ret_rows).T
        hold_dates = w_df.index.intersection(next_ret.index)

        w_held = w_df.loc[hold_dates]
        nr_held = next_ret.loc[hold_dates]
        asset_contributions = pd.DataFrame(w_held.values * nr_held.values, index=hold_dates, columns=assets)
        gross_pnl = asset_contributions.sum(axis=1)
        turnover_s = w_held.diff().abs().sum(axis=1).fillna(0.0)
        lambd_penalty_s = self.lambd * turnover_s

        if self.cost_bps is not None:
            cost_bps_aligned = self.cost_bps.reindex(assets).fillna(0.0)
            real_cost_s = transaction_cost_drag(w_held, cost_bps_aligned)
        else:
            real_cost_s = pd.Series(0.0, index=hold_dates)

        pnl = gross_pnl - lambd_penalty_s - real_cost_s

        cumret = (1 + pnl).cumprod()
        sharpe = np.sqrt(self.periods_per_year) * pnl.mean() / pnl.std() if pnl.std() > 1e-12 else np.nan
        running_max = cumret.cummax()
        max_dd = float(((cumret - running_max) / running_max).min()) if len(cumret) else np.nan

        return pnl, gross_pnl, turnover_s, real_cost_s, sharpe, max_dd, asset_contributions

    # ------------------------------------------------------------------
    def run(self, returns_df: pd.DataFrame) -> dict:
        """Full backtest loop: optimize -> vol-target -> constrain -> compute PnL.

        Returns dict with weights, pnl, sharpe, max_dd, turnover, avg_scale, n_cap_bind.
        """
        reb_dates = sorted(self.cov_dict.keys())
        alpha_df = self.alpha_df.dropna()

        common = (
            pd.DatetimeIndex(reb_dates)
            .intersection(alpha_df.index)
            .intersection(returns_df.index)
        )
        if len(common) < 20:
            return {
                "pnl": pd.Series(dtype=float), "sharpe": np.nan,
                "n_rebalance_dates_total": len(reb_dates), "n_rebalance_dates_valid": len(common),
            }

        assets = alpha_df.columns.tolist()
        n = len(assets)
        common_sorted = sorted(common)
        period_ret_map, n_stale_gaps = self._period_return_map(returns_df, assets, common_sorted)

        ewma_alpha = 1.0 - np.exp(-np.log(2.0) / self.ewma_halflife)
        ewma_var = (self.target_vol / np.sqrt(self.periods_per_year)) ** 2  # neutral prior: scale=1 at t=0

        # GARCH walk-forward state (only touched when vol_estimator="garch") -
        # refit every garch_refit_freq periods using realized PnL through the
        # PREVIOUS period only (point-in-time-safe, same discipline as the EWMA
        # recursion above), filtered forward with fixed params between refits.
        # Reuses data.garch_volatility's own fit_gjr_garch/filter_gjr_garch
        # directly (validated primitives, not reimplemented) rather than
        # hand-deriving the one-step recursion - O(T^2) over the full walk
        # (filter_gjr_garch re-runs the cumulative recursion from t=0 each
        # call), a deliberate correctness-over-speed tradeoff consistent with
        # this project's existing "GARCH is slow by construction, offline use
        # only" convention, not something to optimize away here.
        pnl_history = []
        garch_params = None
        garch_extra_scale = None
        garch_last_refit_len = 0
        if self.vol_estimator == "garch":
            fit_gjr_garch, filter_gjr_garch = _get_garch_functions()

        weights_dict = OrderedDict()
        x_prev = np.zeros(n)
        prev_x, prev_date = None, None
        scale_history, cap_bind_hist = [], []

        for date in common_sorted:
            if prev_x is not None and prev_date in period_ret_map:
                realized_pnl = float(np.dot(prev_x, period_ret_map[prev_date]))
                ewma_var = (1.0 - ewma_alpha) * ewma_var + ewma_alpha * realized_pnl ** 2
                pnl_history.append(realized_pnl)

            current_var = ewma_var
            if self.vol_estimator == "garch" and len(pnl_history) >= self.garch_min_warmup:
                history = np.asarray(pnl_history, dtype=float)
                if garch_extra_scale is None:
                    # Per-Book dynamic rescale, computed ONCE from the warmup
                    # window only (point-in-time-safe) - same fix that resolved
                    # the US_2Y DataScaleWarning bug at the asset level, applied
                    # here since a Book's own PnL scale is just as arbitrary
                    # relative to the fitting library's assumed stable range.
                    warmup_std = np.std(history[: self.garch_min_warmup])
                    garch_extra_scale = 1.0 / warmup_std if warmup_std > 0 else 1.0
                scaled_history = history * garch_extra_scale

                if garch_params is None or (len(history) - garch_last_refit_len) >= self.garch_refit_freq:
                    try:
                        fit = fit_gjr_garch(scaled_history)
                        garch_params = fit["params"]
                        garch_last_refit_len = len(history)
                    except Exception:
                        pass  # keep the last-known params (or the EWMA fallback if none yet)

                if garch_params is not None:
                    try:
                        filtered = filter_gjr_garch(scaled_history, garch_params)
                        sigma_pct_last = float(filtered["sigmas"][-1])
                        current_var = (sigma_pct_last / garch_extra_scale / 100.0) ** 2
                    except Exception:
                        current_var = ewma_var

            alpha_t = alpha_df.loc[date, assets]
            Sigma_t = self.cov_dict[date].loc[assets, assets]

            x_t = self._solve_weights(alpha_t, Sigma_t, x_prev)
            x_t, scale_applied, cap_bound = self._apply_vol_target(x_t, current_var)
            x_t = self._apply_constraints(x_t)

            scale_history.append(scale_applied)
            cap_bind_hist.append(cap_bound)
            weights_dict[date] = x_t
            x_prev = x_t.copy()
            prev_x, prev_date = x_t.copy(), date

        w_df = pd.DataFrame(weights_dict, index=assets).T
        pnl, gross_pnl, turnover_s, real_cost_s, sharpe, max_dd, asset_contributions = self._compute_pnl(w_df, period_ret_map, assets)

        return {
            "weights": w_df.loc[pnl.index] if len(pnl) else w_df,
            "pnl": pnl,
            # gross_pnl/real_cost_series added 2026-07-23 alongside `cost_bps`
            # — gross_pnl has NO lambd penalty and NO real cost deducted;
            # "pnl" nets out both (real cost is 0 for every date when
            # cost_bps=None, so "pnl" is unchanged from before for any
            # existing caller that doesn't pass it).
            "gross_pnl": gross_pnl,
            "real_cost_series": real_cost_s,
            # Per-asset attribution (added 2026-07-29) — see _compute_pnl's own
            # docstring: exact elementwise decomposition of gross_pnl, not an
            # approximation. asset_contributions.sum(axis=1) == gross_pnl.
            "asset_contributions": asset_contributions,
            "sharpe": round(float(sharpe), 4) if pd.notna(sharpe) else np.nan,
            "max_dd": round(max_dd, 4) if pd.notna(max_dd) else np.nan,
            "turnover": round(float(turnover_s.mean()), 6) if len(turnover_s) else np.nan,
            "avg_scale": round(float(np.mean(scale_history)), 4) if scale_history else 1.0,
            "n_cap_bind": int(sum(cap_bind_hist)),
            # Per-date diagnostics (dashboard/pages/14_portfolio_optimizer_health.py)
            # — same values "turnover"/"avg_scale"/"n_cap_bind" already summarize,
            # exposed as full series instead of a single aggregate.
            "turnover_series": turnover_s,
            "scale_series": pd.Series(scale_history, index=common_sorted),
            "cap_bind_series": pd.Series(cap_bind_hist, index=common_sorted),
            "n_rebalance_dates_total": len(reb_dates),
            "n_rebalance_dates_valid": len(common_sorted),
            "n_stale_gaps": n_stale_gaps,
        }
