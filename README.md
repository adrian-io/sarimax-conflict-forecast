# Development of An Automated Conflict Prediction System by State Space ARIMA Methods

While the urgency for early detection of crises is increasing, truly reliable conflict prediction systems are still not in place, despite the emergence of better data sources and the use of state-of-the-art machine learning algorithms in recent years. Researchers still face the rarity of conflict onset events, which makes it difficult for machine learning-based systems to detect crucial escalation or de-escalation signals. As a result, prediction models can be outperformed by naive heuristics, such as the no-change model, which leads to a lack of confidence and thus limited practical usability.
In this thesis, I address this heuristic crisis with the development of a fully-automated machine learning framework capable of optimizing arbitrary econometric state space ARIMA methods in a completely data-driven manner. With this framework, I compare the predictions of a model portfolio consisting of all 8 possible combinations of a standard ARIMA, a seasonal SARIMA, an ARIMAX model with socio-economic variables, and an ARIMAX model with conflict indicators of neighboring countries as exogenous predictors. In addition, each model is examined on a monthly and quarterly periodicity. By comparing the out-of-sample prediction errors, I find that this approach can beat the no-change heuristic in the country-level one-year ahead prediction of the log change of conflict fatalities in all metrics used, including the TADDA score.
