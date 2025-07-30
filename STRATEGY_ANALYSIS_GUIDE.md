# Strategy Analysis Guide

## Primary Question

**Can you prove, with statistically significant evidence, that your model's trading strategy generates positive risk-adjusted returns on out-of-sample data when compared to a simple buy-and-hold benchmark?**

## How the System Answers This Question

The system addresses this primary question by answering three sub-questions, each with specific criteria and statistical rigor:

### 1. Technical Question: Does the model have predictive power?

**Method**: Walk-forward validation with AUC-ROC analysis
**Criteria**: 
- Mean AUC > 0.55 (significantly better than random)
- Statistical significance (p < 0.05)
- Confidence interval lower bound > 0.5

**Evidence**: 
- Uses expanding training windows to prevent lookahead bias
- Tests classification metrics (precision, F1-score, accuracy)
- Proves model is better than a coin flip

### 2. Financial Question: Does predictive power translate to profit?

**Method**: Risk-adjusted performance comparison vs buy-and-hold benchmark
**Criteria**:
- Strategy Sharpe ratio > Benchmark Sharpe ratio
- Excess Sharpe ratio > 0.1 (meaningful outperformance)
- Positive excess returns

**Evidence**:
- Calculates strategy returns using model predictions
- Compares against simple buy-and-hold benchmark
- Analyzes risk-adjusted metrics (Sharpe, Sortino, max drawdown)
- Ensures predictive power creates financial value

### 3. Robustness Question: Is the result real or just luck?

**Method**: Monte Carlo simulation with statistical significance testing
**Criteria**:
- p-value < 0.05 for Sharpe ratio
- Probability of achieving results by chance < 5%
- Confidence intervals exclude random performance

**Evidence**:
- Runs 1000+ Monte Carlo simulations
- Tests null hypothesis that results are due to random chance
- Provides statistical significance with confidence intervals
- Separates real alpha from random luck

## Final Answer

The system combines all three sub-analyses to provide a definitive answer:

**YES** if ALL criteria are met:
- ✅ Technical: AUC > 0.55 with p < 0.05
- ✅ Financial: Excess Sharpe > 0.1
- ✅ Robustness: Monte Carlo p < 0.05

**Confidence Levels**:
- **HIGH**: Strong evidence across all three dimensions
- **MODERATE**: Meets all criteria but with mixed evidence strength
- **LOW**: Fails to meet one or more criteria

## Implementation Details

### Technical Analysis (`_analyze_predictive_power`)
```python
# Walk-forward validation
validator = EnhancedWalkForwardValidator()
validation_results = validator.validate_model(data, model, features)

# Statistical testing
t_stat, p_value = stats.ttest_1samp(auc_scores, 0.5)
confidence_interval = stats.t.interval(alpha=0.95, df=len(auc_scores)-1, 
                                     loc=mean_auc, scale=stats.sem(auc_scores))

has_predictive_power = (mean_auc > 0.5 and p_value < 0.05 and 
                       confidence_interval[0] > 0.5)
```

### Financial Analysis (`_analyze_financial_performance`)
```python
# Calculate strategy vs benchmark performance
evaluator = PerformanceEvaluator()
backtest_results = evaluator.backtest_strategy(data, predictions, probabilities)

# Compare risk-adjusted returns
excess_sharpe = strategy_metrics["sharpe_ratio"] - benchmark_metrics["sharpe_ratio"]
outperforms_benchmark = (strategy_sharpe > benchmark_sharpe and excess_sharpe > 0.1)
```

### Robustness Analysis (`_analyze_statistical_significance`)
```python
# Monte Carlo simulation
monte_carlo = MonteCarloSimulator(n_simulations=1000)
simulation_results = monte_carlo.simulate_strategy_performance(returns, probabilities)

# Statistical significance
p_value_sharpe = simulation_results.get("p_value_sharpe", 1.0)
is_statistically_significant = p_value_sharpe < 0.05
```

## Output Files

The analysis generates comprehensive outputs:

1. **JSON Analysis Report**: `exports/strategy_analysis_YYYYMMDD_HHMMSS.json`
   - Complete results from all three sub-analyses
   - Final answer with confidence level
   - Detailed metrics and evidence

2. **CSV Exports**:
   - `exports/performance_metrics.csv`: Backtesting results
   - `exports/monte_carlo_results.csv`: Statistical significance tests
   - `exports/feature_importance.csv`: Model interpretability

3. **Visualizations**:
   - Interactive dashboards showing performance
   - Risk-return scatter plots
   - Cumulative returns comparison

## Usage

### Quick Analysis
```bash
python run_analysis.py
```

### Comprehensive Test
```bash
python test_strategy_analysis.py
```

### Full Pipeline
```bash
python main.py
```

## Interpretation

### Reading the Results

1. **Technical Evidence**: Look for AUC > 0.55 and p-value < 0.05
2. **Financial Evidence**: Check if excess Sharpe > 0.1
3. **Robustness Evidence**: Verify Monte Carlo p-value < 0.05

### Confidence Assessment

- **HIGH**: Strong predictive power (AUC > 0.6) + strong financial outperformance + high statistical significance
- **MODERATE**: Meets all criteria but with mixed evidence strength
- **LOW**: Fails one or more criteria

### Key Metrics to Monitor

- **Mean AUC**: Should be > 0.55 for predictive power
- **Excess Sharpe**: Should be > 0.1 for meaningful outperformance
- **Monte Carlo p-value**: Should be < 0.05 for statistical significance
- **Confidence Level**: Overall assessment of evidence strength

## Conclusion

This system provides a rigorous, statistically sound answer to the primary question by:

1. **Proving predictive power** through walk-forward validation
2. **Demonstrating financial value** through benchmark comparison
3. **Establishing statistical significance** through Monte Carlo simulation
4. **Providing confidence levels** based on evidence strength

The result is a definitive YES/NO answer with comprehensive evidence and statistical rigor. 