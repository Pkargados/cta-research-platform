# CTA Portfolio Development Roadmap

You have completed the **signal-discovery and infrastructure stage**. The next stage is to turn the research platform into a defined investment product.

The correct conceptual progression is:

```text
Research universe
→ tradable universe
→ single-strategy Books
→ multi-strategy portfolio
→ execution and monitoring
```

This matches the institutional CTA stack: forecasts are standardized, translated into risk budgets, passed through a covariance- and cost-aware allocator, constrained for liquidity and concentration, and then converted into executable trades.

## Phase 1 — Define the portfolio mandate

Before selecting assets or strategies, decide what portfolio you are building.

You need to specify:

- Is it a diversified macro CTA or a commodity-focused CTA?
- Is the objective absolute return, crisis diversification, or commodity alpha?
- What is the intended volatility target: 8%, 10%, 12%, or 15%?
- What is the expected holding period?
- How much turnover is acceptable?
- What drawdown and leverage would be tolerable?
- Is the portfolio intended for your own small capital or as an institutional research prototype?

This matters because a diversified macro CTA and a commodity hedge fund may use the same signals but make very different universe and risk-allocation decisions.

For your current project, the natural mandate appears to be:

> A medium-horizon systematic macro and commodity futures portfolio combining directional trend, curve-based carry, and relative-value strategies, targeting stable risk with controlled turnover.

That should become the design anchor.

## Phase 2 — Compress the 42-market research universe

The 42 markets remain your **research universe**. You now construct a smaller **core tradable universe**.

Universe compression should use four layers.

### A. Operational eligibility

Remove or downgrade contracts with:

- Poor or unreliable data.
- Insufficient history.
- Low liquidity or open interest.
- Excessive estimated trading or rolling costs.
- Unstable contract specifications.
- Difficult access through your intended broker.
- Inadequate term-structure data where the strategy requires it.

This produces the objectively investable set.

### B. Economic redundancy

Group the contracts into clusters representing similar risks.

Examples:

- ES, NQ, YM, and RTY are partially overlapping US equity exposures.
- ZT, ZF, ZN, ZB, and UB are different points on the same US duration curve.
- WTI, Brent, heating oil, and gasoline share substantial energy-complex exposure.
- Corn, wheat, and soybeans are distinct but share agricultural and weather factors.

Do not automatically retain every representative of every cluster. Ask whether the additional contract contributes a sufficiently different risk exposure or trading opportunity.

You might retain several Treasury maturities because yield-curve movements are not perfectly parallel, but give the rates cluster a common risk limit. Likewise, you might retain ES and NQ while dropping YM unless it adds something meaningful.

### C. Economic coverage

The compressed universe must still cover distinct macro drivers:

- Equity growth and risk appetite.
- Short-, intermediate-, and long-duration rates.
- Major currencies.
- Energy supply and demand.
- Precious and industrial metals.
- Agricultural and weather risks.
- Livestock or soft commodities where investable.

Compression must not accidentally turn a 42-market CTA into a concentrated US equity–Treasury portfolio.

### D. Train-period stability and marginal value

Only after the first three screens should you use train-period performance.

Evaluate whether each market:

- Contributes positively across several train subperiods.
- Improves portfolio diversification.
- Reduces or worsens drawdowns.
- Introduces excessive turnover.
- Produces unique P&L or merely duplicates another market.
- Remains useful under nearby strategy parameters.

This is not “take the highest-Sharpe contracts.” It is portfolio-level selection.

The output should probably be something like:

- **Research universe:** 42 markets.
- **Core universe:** perhaps 18–25 liquid, representative markets.
- **Extended universe:** markets that remain available for specialist strategies.
- **Excluded universe:** contracts currently unsuitable for operational reasons.

The exact number should emerge from the analysis rather than be fixed in advance.

## Phase 3 — Define the strategy taxonomy

Do not treat every implementation you tested as an independent strategy.

Organize them into economically distinct families.

### Trend Book

This can contain:

- Time-series momentum.
- Moving-average crossover.
- Donchian breakout.

These are not necessarily three independent portfolio sleeves. They are three ways of measuring trend. If their position and return correlations are high, combine them into a **single trend ensemble** rather than allocating separate risk budgets to each.

For example:

```text
Trend forecast
= w1 × TSMOM
+ w2 × Crossover
+ w3 × Breakout
```

The weights should initially be simple—equal or conservative—not optimized aggressively.

### Carry Book

This contains genuine curve-based signals using the subset of assets with reliable term-structure data.

It may eventually include:

- Cross-sectional carry.
- Time-series carry.
- Curve slope momentum.
- Contract-selection logic.

Because carry behaves differently across physical commodities, financial futures, and seasonal markets, this Book may need its own eligibility rules.

### Relative-Value Book

This should contain economically justified synthetic trades:

- Brent–WTI.
- Gold–silver.
- Corn–wheat.
- Treasury-curve spreads.
- Eventually crack or crush spreads if the contract data support them.

Here, the traded unit is the spread, not the individual contract.

### Additional or parked families

Reversal, cross-sectional momentum, and value should not be forced into the portfolio because they have already been implemented. They remain archived research candidates unless a materially different specification or dataset gives you a reason to revisit them.

The objective is not to fill six strategy slots. It is to identify a small number of distinct, defensible return mechanisms.

## Phase 4 — Build canonical single-strategy portfolios

A **single-strategy portfolio** means one economic strategy family across its appropriate markets.

It does not mean one asset.

Examples:

- Trend across 15 selected futures.
- Carry across 18 curve-eligible markets.
- Relative value across six synthetic spreads.

Each Book can have a different universe.

For each Book, freeze:

- Eligible markets.
- Signal construction.
- Parameter ensemble.
- Forecast normalization.
- Rebalancing frequency.
- Volatility estimator.
- Position caps.
- Cost assumptions.
- Portfolio volatility target.

Begin with a simple construction:

```text
raw weight(i,t) ∝ signal(i,t) / estimated volatility(i,t)
```

Then apply asset, sector, and portfolio risk limits.

At this stage, do not rely on the optimizer to rescue a weak strategy. Each Book needs to make economic and empirical sense under a transparent baseline.

## Phase 5 — Validate the single-strategy Books

Each Book must pass several distinct tests.

### Forecast validation

Does stronger signal intensity predict stronger subsequent returns?

Look for:

- Monotonic signal buckets.
- Positive versus negative signal differentiation.
- Stability across subperiods.
- Consistency across multiple markets or spreads.
- Survival after costs.

### Portfolio validation

Evaluate:

- Net Sharpe and return.
- Drawdown and recovery period.
- Turnover.
- Tail behavior.
- Long-side versus short-side performance.
- Contribution by asset and sector.
- Concentration of profits.
- Performance in different macro environments.

### Universe validation

Compare:

- Full eligible universe.
- Structurally compressed universe.
- Train-selected compact universe.

The compact universe should not merely have a better train Sharpe. It should offer a sensible trade-off between diversification, turnover, operational simplicity, and out-of-sample stability.

### Robustness validation

Use:

- Chronological train folds.
- CPCV/PBO.
- Parameter-neighborhood tests.
- Leave-one-market-out tests.
- Leave-one-sector-out tests.
- Cost and slippage sensitivity.
- Alternative covariance and volatility estimates.

The output is a verdict for each Book:

- **Production candidate.**
- **Diversifying satellite.**
- **Research only.**
- **Rejected or parked.**

## Phase 6 — Standardize the surviving Books

Before combining Books, make their outputs comparable.

A trend score, a carry spread, and an RV z-score are in different units. Convert each to a standardized, capped forecast scale.

The hierarchy should be:

```text
raw signal
→ normalized forecast
→ asset-level risk
→ Book-level risk
```

This is the stage where the `Book` architecture becomes useful: each Book owns its signal, universe, positions, risk target, costs, and standalone return stream.

## Phase 7 — Construct the multi-strategy portfolio

Once two or more Books survive, combine them through the `Allocator`.

Start with baselines before optimization.

### Baseline 1: equal Book risk

Give each accepted Book the same ex ante volatility contribution.

For example:

- Trend: 50% of risk.
- Carry: 25%.
- Relative value: 25%.

Or equal thirds if the evidence supports it.

The exact allocation should reflect expected robustness, costs, and diversification—not just standalone train Sharpe.

### Baseline 2: diversification-adjusted risk budgets

Reduce the risk allocation to Books that:

- Are highly correlated with another Book.
- Have higher turnover.
- Have weaker evidence.
- Have more severe tail risk.
- Depend on fewer markets.

This may naturally make trend the core Book and carry/RV smaller satellites.

### Optimized allocation

Only then test your Ledoit–Wolf, turnover-penalized optimizer.

The optimizer should account for:

- Book and market covariance.
- Expected trading costs.
- Changes from current holdings.
- Book-level risk limits.
- Asset and sector limits.
- Gross and net exposure.
- Margin usage.
- Liquidity.

The optimizer must beat the simple equal-risk allocation out of sample. If it does not, the simpler allocator wins.

## Phase 8 — Build the risk-management layer

The portfolio now needs limits that exist independently of the expected-return model.

### Position and market limits

- Maximum risk per contract.
- Maximum contract count relative to liquidity.
- Maximum share of open interest or ADV.
- Maximum leverage and notional exposure.

### Cluster and factor limits

- Maximum equity-index risk.
- Maximum duration risk.
- Maximum energy risk.
- Maximum agriculture risk.
- Maximum USD or global risk-on exposure.
- Maximum contribution from one Book.

This is more important than simply counting how many contracts are held. Twenty contracts can still represent two underlying economic bets.

### Portfolio risk limits

- Target volatility.
- Maximum expected shortfall.
- Stress-loss limits.
- Drawdown escalation rules.
- Margin and collateral buffers.
- Maximum daily turnover.
- Maximum gross exposure.

### Stress testing

Replay or simulate:

- 2008.
- The 2013 taper tantrum.
- The 2014–2015 oil collapse.
- March 2020.
- Negative WTI.
- The 2021–2022 inflation and commodity shock.
- Sudden equity/rates correlation changes.
- Large volatility and liquidity shocks.

The purpose is not only to estimate historical losses. It is to identify hidden concentrations and operational failure modes.

## Phase 9 — Add contract and execution logic

A futures portfolio is not complete when it produces target weights.

You must determine:

- Which maturity to trade.
- When to roll.
- Whether signals are calculated on a continuous series but executed in individual contracts.
- How the strategy handles first-notice and last-trade dates.
- How orders are rounded to integer contracts.
- Whether small target changes should be ignored.
- How orders are scheduled.
- What slippage model is used.
- How trading differs in normal and stressed liquidity.

For carry, maturity selection may be part of the alpha. For trend, the maturity choice may primarily be an execution decision. Those two concepts should not be mixed.

The final portfolio process becomes:

```text
target risk weights
→ contract selection
→ integer positions
→ trade list
→ execution
```

Execution should be evaluated with realized rather than assumed cost once paper trading begins.

## Phase 10 — Run a frozen historical simulation

Before paper trading, produce a final research backtest in which everything is frozen:

- Mandate.
- Core universe.
- Strategy-specific universes.
- Book definitions.
- Signal ensembles.
- Risk budgets.
- Constraints.
- Cost model.
- Optimizer settings.
- Roll rules.
- Rebalancing schedule.

No more model selection after viewing the final test.

The final report should include:

- Gross and net performance.
- Book-level attribution.
- Asset and sector attribution.
- Risk contributions.
- Turnover and costs.
- Margin and leverage.
- Stress tests.
- Capacity estimates.
- Comparison with simple baselines.
- Explanation of where and why the portfolio loses money.

This is the portfolio you would defend in an interview or investment committee.

## Phase 11 — Paper trading and shadow NAV

Connect the platform to live or delayed market data and run it without capital.

Each day, the system should:

1. Update data.
2. Validate data quality.
3. Generate forecasts.
4. Produce target positions.
5. Apply constraints.
6. Generate orders.
7. Simulate or submit paper trades.
8. Reconcile positions.
9. Calculate P&L and costs.
10. Record exceptions.

Maintain a **shadow NAV** and compare:

- Predicted versus realized volatility.
- Backtest costs versus paper costs.
- Intended versus filled positions.
- Model P&L versus execution P&L.
- Forecast decay.
- Risk-limit breaches.
- Roll behavior.

A strategy that works historically but cannot be operated consistently is not a finished strategy.

## Phase 12 — Governance and ongoing research

Once the portfolio is running, research and production must be separated.

You need rules for:

- When a strategy can be changed.
- How a new Book is admitted.
- How an existing Book is retired.
- How often parameters are reviewed.
- What constitutes model deterioration.
- How data and code changes are approved.
- How live performance is attributed.
- How incidents are documented.

A new signal should enter through a formal sequence:

```text
research
→ independent validation
→ shadow Book
→ small risk allocation
→ full portfolio
```

Regime conditioning should come late. First establish stable Books and a functioning risk model. A regime model should modify a known portfolio problem—such as leverage during correlation spikes—not become another mechanism for retrospectively selecting whichever strategy worked in each historical period.

## Complete roadmap

The practical order is:

1. Define the CTA mandate.
2. Compress the 42-market universe into a core tradable universe.
3. Group existing signals into genuinely distinct strategy families.
4. Build one canonical single-strategy Book per family.
5. Validate each Book, including its strategy-specific universe.
6. Standardize forecasts and risk across Books.
7. Construct a naive multi-strategy risk allocation.
8. Test the cost-aware optimizer against that naive baseline.
9. Add portfolio constraints, stress testing, and margin controls.
10. Add contract-selection, roll, and execution logic.
11. Freeze the complete process and run the final historical simulation.
12. Begin live-data paper trading and maintain a shadow NAV.
13. Establish production monitoring and model-governance rules.
14. Only then add new strategies, alternative data, or regime overlays.

The next stage is therefore not “keep researching signals.” It is to **define the investable universe and turn the existing research into a small number of independently defensible Books**. After that, the project becomes primarily one of risk allocation, implementation, and portfolio operations—not signal backtesting.
