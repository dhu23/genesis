# options coding project
## project 1: build an options pricing library

- black-scholes
- binomial
- implied-vol
- greeks (delta, gamma, vega, theta, rho)
- interpolation
- tests

pricing 
- European call
- European put
- forward
- intrinsic value
- discounted payoff


## project 2: implied volatility solver
Given market price C-market, solve for sigma
Implement at least 
- bisection
- Newton-Raphson
- Brent

Then investigate
- deep ITM (in the money)
- ATM (at the money)
- deep OTM (out of the money)
- very short maturity
- very low volatility
- very high volatility
- prices close to intrinstic value

Don't just make it converge. Ask: When does it fail? Why?
How should the API behave when the input price is bad?

## project 3: build a mini option chain
Generate an artificial option market
For example 
|  Expiry  |  Strike  |  Call IV   |  Put IV  |
|----------|----------|------------|----------|
|  1M      |   90     |  ...       |  ...     |
|  1M      |   95     |  ...       |  ...     |
|  1M      |  100     |  ...       |  ...     |
|  1M      |  105     |  ...       |  ...     |
|  1M      |  110     |  ...       |  ...     |
|  3M      |   90     |  ...       |  ...     |

Then 
1. generate prices from a known volatility model
2. calculate implied vols
3. reconstruct the volatility surface
4. plot it
5. perturb prices with noise
6. observe what happens to implied volatility
7. examine arbitrage violations


## project 4: fix a volatility surface
This would be a flagship project.
Start with market-like synthetic data.

- Step 1: Generate an option chain from a known model
- Step 2: Convert prices -> implied vols
- Step 3: Transform (K, T, sigma) -> (k, T, w)
- Step 4: Fit an SVI surface
- Step 5: Plot: 
  - IV smile by expiry
  - total variance by log-moneyness
  - full 3D surface
- Step 6: Add noise
See how the calibration breaks
- Step 7: Add bad observations
See how the calibration breaks
- Step 8: Add constraints
This is where the project becomes much more realistic


## project 5: Delta-hedged option experiment (*)
Take a short call, then simulate
1. sell option
2. delta hedge
3. underlying moves
4. rebalance
5. repeat
Measure:
- option P&L
- hedge P&L
- total P&L
- realized volatility
- implied volatility
- gamma
- theta
Then repeat with 
- realized vol < implied vol
- realized vol > implied vol
- different hedge frequeencies
- transaction costs
You will learn more about volatility trading from this
experiment than from reading another 100 pages


## project 6: toy options market maker
Build a simplified market maker:
market data -> volatility surface -> fair value -> Greeks -> 
Inventory -> Quote ad
justment -> Bid / Ask -> Fills -> 
Position -> Hedging -> P&L
You can simulate the underlying and option order flow
For example:
```
fair value = model price
reservation price = fair value - inventory * risk aversion
```

Then 
```
bid = reservation price - spread/2
ask = reservation price + spread/2
```

Let the spread depend on:
- volatility
- liquidity
- maturity
- inventory
- expected adverse selection
Then simulate a trading day. The purpose is to understand how 
the pricing model becomes a trading decision.

# Learn these Math topics specifically
- Root finding
  - bisection
  - Newton
  - secant
  - Brent
- Optimization
  - least squares
  - constrained optimization
  - gradient-based optimization
  - parameter scaling
  - regularization
- Interpolation
  - linear
  - cubic spline
  - monotonic interpolation
  - extrapolation behavior
- Numerical differentiation, Particularly 
  - finite differences
  - stability
  - step-size selection
- Integration
  - quadrature
  - adaptive integration
  - Monte Carlo
- Automatic differentiation
Understand why a quant team might prefer analytic derivatives, 
finite differences, or AD. 