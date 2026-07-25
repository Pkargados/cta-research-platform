"""
backtest/engine.py — Rebalance-cadence positions + the shift(1) backtest loop.

Moved from signal_lib.py (2026-07-21) - see cleanup.md section 2. `cost_bps` param
added same day (WORKFLOW.md Phase 6, pulled forward) - see backtest/costs.py for the
turnover/cost-drag mechanics and the cost-assumption caveats. `frequency="daily"`
added same day for breakout (WORKFLOW.md Phase 2b) - momentum's monthly formation
would delay reacting to a breakout until month-end, destroying the entire reason to
build that signal; breakout needs its direction re-evaluated the day a new extreme
actually prints, not on a fixed calendar.
"""

from backtest.costs import transaction_cost_drag, turnover


def weekly_positions(signal):
    weekly_signal = signal.resample("W-FRI").last()
    return weekly_signal.reindex(signal.index).ffill()


def holding_period_positions(signal, holding_months=1):
    """Month-end formation, blended across the trailing `holding_months` still-active
    formations (Jegadeesh-Titman-style overlapping-vintage averaging, Moskowitz-Ooi-
    Pedersen 2012 Section 3.2: "the return at time t represents the average return
    across all portfolios at that time... the portfolio that was constructed last
    month, the month before that... and so on for all currently active portfolios").

    Vectorized: a rolling mean over `holding_months` trailing month-end signal values
    is exactly this average, since each month-end value already IS that month's
    formation signal. holding_months=1 reduces to a plain last-formed-signal series
    (see monthly_positions).
    """
    monthly_signal = signal.resample("ME").last()
    blended = monthly_signal.rolling(holding_months).mean()
    return blended.reindex(signal.index).ffill()


def monthly_positions(signal):
    return holding_period_positions(signal, holding_months=1)


def _resample_positions(signal, frequency, holding_months):
    if frequency == "daily":
        return signal
    elif frequency == "weekly":
        return weekly_positions(signal)
    elif frequency == "monthly":
        return holding_period_positions(signal, holding_months)
    else:
        raise ValueError("frequency must be 'daily', 'weekly', or 'monthly'")


def normalized_positions(signal, frequency, holding_months=1):
    """The exact position array backtest_signal trades: resampled to the target
    cadence, shift(1)'d for no-look-ahead, then normalized to unit gross exposure
    (sum(|weights|) = 1) each day.

    Exposed separately (not just inlined in backtest_signal) so a caller measuring
    real portfolio-level turnover - e.g. a research script's turnover/cost diagnostic
    - uses the actual traded, normalized array rather than the raw un-normalized
    signal. That distinction matters concretely: a raw vol-targeted signal
    (target_vol/vol) can be huge in magnitude for a low-vol instrument before
    normalization (the same failure mode already documented in
    signals.transforms.vol_targeted_sign_signal's docstring for a per-asset,
    un-normalized view - SwissFranc's ~11-26x implied leverage), so measuring
    turnover pre-normalization overstates it wildly, not just approximately.
    """
    positions = _resample_positions(signal, frequency, holding_months)
    positions = positions.shift(1)

    gross_exposure = positions.abs().sum(axis=1)
    positions = positions.div(gross_exposure, axis=0)
    return positions.fillna(0)


def backtest_signal(signal, returns_data, frequency="monthly", holding_months=1, cost_bps=None):
    """Backtest a signal against realized returns.

    signal and returns_data must share the same (T x N) shape/columns. Positions are
    shift(1)'d before being applied to returns to avoid look-ahead bias (CLAUDE.md
    Rule 3), then normalized to unit gross exposure (sum(|weights|) = 1) each day -
    see normalized_positions.

    holding_months only applies when frequency="monthly" - see
    holding_period_positions. frequency="daily" applies the signal directly, no
    resampling - for signals that need to react the day their own trigger fires
    (e.g. breakout), not on a fixed calendar.

    `cost_bps` (optional, pd.Series indexed by asset, one-way cost in basis points) -
    when given, nets out a turnover-based cost drag (backtest/costs.py) from the
    pooled return. Default None preserves prior (gross-of-cost) behavior exactly, so
    every existing caller is unaffected unless it opts in.
    """
    positions = normalized_positions(signal, frequency, holding_months)

    strategy_returns = (positions * returns_data).sum(axis=1)
    if cost_bps is not None:
        strategy_returns = strategy_returns - transaction_cost_drag(positions, cost_bps)
    return strategy_returns.dropna()


def backtest_signal_per_asset(signal, returns_data, frequency="monthly", holding_months=1, cost_bps=None):
    """Same position construction as backtest_signal, but returns each asset's own
    strategy return separately (T x N), not summed/gross-exposure-normalized across
    the book. For asking "does this signal work for THIS asset" in isolation -
    matching Moskowitz-Ooi-Pedersen Figure 2's per-instrument Sharpe report - before
    any cross-asset combination step (CLAUDE.md: per-asset evaluation precedes
    combination).

    `cost_bps` - see backtest_signal; here the cost drag is subtracted per asset
    (each asset's own turnover × its own cost), not summed across the book first.
    """
    positions = _resample_positions(signal, frequency, holding_months)
    positions = positions.shift(1)
    strategy_returns = positions * returns_data
    if cost_bps is not None:
        strategy_returns = strategy_returns - turnover(positions) * (cost_bps / 10_000)
    return strategy_returns
