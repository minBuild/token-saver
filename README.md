# token-saver

Route every prompt to the cheapest Claude model and effort level that can handle it.
The decision is made by [Jev](https://docs.typesafe.ai), a System One model that
returns typed answers with probabilities instead of generated text, so routing costs
a few hundred milliseconds and a fraction of a cent instead of a full LLM call.

Work in progress. See the commit history for the current state.
