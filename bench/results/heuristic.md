Prompts: 50 | Classifier: heuristic x50 | Baseline: claude-opus-5-5 @ xhigh | Assumed output: 800 tokens per request

| Metric | Value |
|---|---|
| Exact tier match | 48% (24/50) |
| Within one tier | 88% (44/50) |
| Estimated Claude spend, routed | $0.4099 |
| Estimated Claude spend, baseline | $0.8068 |
| Classifier spend | $0.000000 |
| Savings vs baseline | 49.2% |
| Mean classifier latency | 0.3 ms |
| Fallbacks to the heuristic | 0 |

| expected \ routed | trivial | simple | moderate | hard | expert |
|---|---|---|---|---|---|
| trivial | 10 | 0 | 0 | 0 | 0 |
| simple | 3 | 7 | 0 | 0 | 0 |
| moderate | 0 | 4 | 6 | 0 | 0 |
| hard | 0 | 0 | 7 | 1 | 2 |
| expert | 0 | 0 | 6 | 4 | 0 |

| id | expected | routed | model | effort | task type | difficulty |
|---|---|---|---|---|---|---|
| trivial-01 | trivial | trivial | claude-haiku-5-5 | high | chitchat | 0.30 |
| trivial-02 | trivial | trivial | claude-haiku-5-5 | high | chitchat | 0.30 |
| trivial-03 | trivial | trivial | claude-haiku-5-5 | high | chitchat | 0.30 |
| trivial-04 | trivial | trivial | claude-haiku-5-5 | high | factual | 0.30 |
| trivial-05 | trivial | trivial | claude-haiku-5-5 | high | factual | 0.30 |
| trivial-06 | trivial | trivial | claude-haiku-5-5 | high | factual | 0.30 |
| trivial-07 | trivial | trivial | claude-haiku-5-5 | high | factual | 0.30 |
| trivial-08 | trivial | trivial | claude-haiku-5-5 | high | chitchat | 0.30 |
| trivial-09 | trivial | trivial | claude-haiku-5-5 | high | chitchat | 0.30 |
| trivial-10 | trivial | trivial | claude-haiku-5-5 | high | factual | 0.30 |
| simple-01 | simple | simple | claude-sonnet-5-5 | high | rewrite | 1.00 |
| simple-02 | simple | simple | claude-sonnet-5-5 | high | rewrite | 1.00 |
| simple-03 | simple | trivial (+1) | claude-haiku-5-5 | high | coding | 0.30 |
| simple-04 | simple | simple | claude-sonnet-5-5 | high | rewrite | 1.00 |
| simple-05 | simple | trivial (+1) | claude-haiku-5-5 | high | rewrite | 0.30 |
| simple-06 | simple | simple | claude-sonnet-5-5 | high | coding | 1.00 |
| simple-07 | simple | simple | claude-sonnet-5-5 | high | coding | 1.00 |
| simple-08 | simple | simple | claude-sonnet-5-5 | high | rewrite | 1.00 |
| simple-09 | simple | simple | claude-sonnet-5-5 | high | factual | 1.00 |
| simple-10 | simple | trivial (+1) | claude-haiku-5-5 | high | coding | 0.30 |
| moderate-01 | moderate | moderate | claude-sonnet-5-5 | xhigh | coding | 1.50 |
| moderate-02 | moderate | moderate | claude-sonnet-5-5 | xhigh | coding | 1.50 |
| moderate-03 | moderate | simple (+1) | claude-sonnet-5-5 | high | factual | 1.00 |
| moderate-04 | moderate | moderate | claude-sonnet-5-5 | xhigh | coding | 2.30 |
| moderate-05 | moderate | moderate | claude-sonnet-5-5 | xhigh | coding | 1.50 |
| moderate-06 | moderate | moderate | claude-sonnet-5-5 | xhigh | coding | 1.50 |
| moderate-07 | moderate | simple (+1) | claude-sonnet-5-5 | high | factual | 1.00 |
| moderate-08 | moderate | simple (+1) | claude-sonnet-5-5 | high | factual | 1.00 |
| moderate-09 | moderate | moderate | claude-sonnet-5-5 | xhigh | coding | 1.50 |
| moderate-10 | moderate | simple (+1) | claude-sonnet-5-5 | high | coding | 1.00 |
| hard-01 | hard | moderate (+1) | claude-sonnet-5-5 | xhigh | rewrite | 1.80 |
| hard-02 | hard | moderate (+1) | claude-sonnet-5-5 | xhigh | analysis | 2.20 |
| hard-03 | hard | expert (+1) | claude-fable-5-1 | xhigh | analysis | 3.50 |
| hard-04 | hard | moderate (+1) | claude-sonnet-5-5 | xhigh | analysis | 2.40 |
| hard-05 | hard | hard | claude-opus-5-5 | xhigh | analysis | 3.30 |
| hard-06 | hard | moderate (+1) | claude-sonnet-5-5 | xhigh | analysis | 1.80 |
| hard-07 | hard | moderate (+1) | claude-sonnet-5-5 | xhigh | coding | 2.30 |
| hard-08 | hard | moderate (+1) | claude-sonnet-5-5 | xhigh | analysis | 2.40 |
| hard-09 | hard | expert (+1) | claude-fable-5-1 | xhigh | analysis | 3.50 |
| hard-10 | hard | moderate (+1) | claude-sonnet-5-5 | xhigh | analysis | 2.20 |
| expert-01 | expert | hard (+1) | claude-opus-5-5 | xhigh | analysis | 3.00 |
| expert-02 | expert | hard (+1) | claude-opus-5-5 | xhigh | analysis | 3.00 |
| expert-03 | expert | hard (+1) | claude-opus-5-5 | xhigh | coding | 2.90 |
| expert-04 | expert | moderate (+2) | claude-sonnet-5-5 | xhigh | analysis | 2.40 |
| expert-05 | expert | hard (+1) | claude-opus-5-5 | xhigh | analysis | 3.30 |
| expert-06 | expert | moderate (+2) | claude-sonnet-5-5 | xhigh | analysis | 1.80 |
| expert-07 | expert | moderate (+2) | claude-sonnet-5-5 | xhigh | analysis | 2.40 |
| expert-08 | expert | moderate (+2) | claude-sonnet-5-5 | xhigh | rewrite | 2.40 |
| expert-09 | expert | moderate (+2) | claude-sonnet-5-5 | xhigh | analysis | 2.40 |
| expert-10 | expert | moderate (+2) | claude-sonnet-5-5 | xhigh | coding | 2.30 |
