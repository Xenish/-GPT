# Research Playbooks

This directory contains templates and guides for common research workflows.

## What are Playbooks?

Playbooks are standardized workflows for conducting trading strategy research. They provide:
- Step-by-step procedures
- Best practices and gotchas
- Code templates
- Quality checklists

## Available Playbooks

### 1. Strategy Parameter Search (`strategy_param_search.md`)
Run parameter optimization for a single strategy to find best configurations.

**Use When**:
- Testing a new strategy
- Optimizing existing strategy parameters
- Exploring parameter sensitivity

### 2. Multi-Strategy Comparison (`multi_strategy_comparison.md`)
Compare multiple strategies side-by-side on the same data.

**Use When**:
- Evaluating strategy alternatives
- Building strategy universe
- Identifying complementary strategies

### 3. Ensemble Strategy Development (`ensemble_development.md`)
Build and test ensemble/meta-strategies.

**Use When**:
- Combining multiple strategies
- Reducing single-strategy risk
- Adaptive strategy allocation

### 4. Regime Analysis (`regime_analysis.md`)
Analyze strategy performance across market regimes.

**Use When**:
- Understanding when strategies work/fail
- Building regime-aware systems
- Risk management

### 5. Robustness Testing (`robustness_testing.md`)
Validate strategy performance across different conditions.

**Use When**:
- Before live deployment
- After major strategy changes
- Periodic validation

## How to Use Playbooks

1. **Choose** the playbook that matches your research question
2. **Read** the entire playbook before starting
3. **Follow** the steps sequentially
4. **Document** your process and results
5. **Review** the checklist before concluding

## Research Best Practices

### Data Integrity
- Always use `mode=research` config
- Verify data quality before experiments
- Document data periods used
- Track git SHA for reproducibility

### Methodology
- Define hypothesis before testing
- Use train/validation/test splits
- Avoid overfitting (limit parameter tuning)
- Test on out-of-sample data

### Documentation
- Record all experiments (even failures)
- Note parameter choices and rationale
- Save reports for all major experiments
- Version control research code

### Validation
- Cross-validate on multiple symbols
- Test across different timeframes
- Check for look-ahead bias
- Verify execution assumptions

## Playbook Structure

Each playbook follows this structure:

1. **Objective**: What you're trying to achieve
2. **Prerequisites**: Data, config, tools needed
3. **Steps**: Sequential workflow
4. **Code Templates**: Ready-to-use scripts
5. **Analysis**: How to interpret results
6. **Checklist**: Quality gates
7. **Common Pitfalls**: What to avoid

## Contributing

To add a new playbook:
1. Follow the standard structure above
2. Include working code examples
3. Test the workflow end-to-end
4. Add to this README index

---

**Remember**: Research is iterative. Don't expect perfect results on the first try. Use playbooks as guides, not rigid scripts.
