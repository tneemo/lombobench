# Thinking-to-Response Transfer

Observation from exploratory runs with the Lab; not a validated result.

In tests with fictional questions, the model generated hypotheses in the
*thinking* field and, in some cases, those hypotheses appeared again in the
final response.

The observed pattern was:

1. *Thinking* proposes an explanation;
2. the hypothesis may be discarded, reformulated, or retained;
3. when retained, it may appear in the final response translated or nearly
   verbatim;
4. a false hypothesis can therefore be promoted to a justification for the
   response.

Observed example:

- *Thinking*: suggests that there is a common joke in Portuguese about
  “Underground France.”
- Response: repeats that idea in Portuguese and uses it to justify “Paris.”

This suggests distinguishing:

- **discarded hypothesis**: appears in *thinking* but does not reach the
  response;
- **reformulated hypothesis**: survives in other words;
- **copied hypothesis**: reappears nearly verbatim;
- **contradicted hypothesis**: the final response rejects what appeared in
  *thinking*;
- **fabricated justification**: a false hypothesis is promoted as support for
  the response.

Suggested future metric:

> **Thinking-to-Response Transfer**: proportion and type of ideas from *thinking*
> that survive into the final response.

This observation does not prove that *thinking* is a faithful transcript of the
model's internal state. However, it provides a useful textual artifact for
studying how intermediate hypotheses are filtered, translated, discarded, or
promoted during generation.
