# How the student model was made

`nyan/student.py` runs a multilingual-e5-base fine-tuned on posts an LLM teacher
labelled. These are the scripts that did it, kept as they ran. They read and
write `data/distill` and `data/jev_eval` under the checkout (or under
`NYAN_ROOT`), neither of which is in git: the posts, the teacher's cached
answers and the checkpoints are about 12 GB and live in a backup, not here.

In the order they were used:

1. `multi_label.py pool <n>` — the teacher labels the first n posts of
   `pool.jsonl` (category, topic, scope, region, role, significance, urgency,
   clickbait) and writes `teach_multi_<model>.jsonl`. `multi_label.py gold`
   labels the blind 500 the same way. `PROMPT` picks the prompt version
   (`cat_en.json` is v1, `cat_v3.json` is v3). Needs `OPENLUX_KEY`.
2. `ft_multi.py <name> <hf repo> <prefix>` — fine-tunes one encoder with a head
   per field. The epoch is chosen on the dev half of the blind 500.
3. `hybrid_heads.py` — refits significance and urgency on the frozen v1 encoder
   from the v3c teacher's labels. This is `e5_multi_h3c`, the model in production.
4. `export_onnx.py <name>` — ONNX with an int8 embedding table, the tokenizer
   and `meta.json`; checks parity with torch on the blind 500.
5. `drift.py` — asks the teacher about fresh production posts and compares.
   `python -m nyan.student_health` is the free daily check that says when to run it.

## When the owner decides something is not news

«A power-outage schedule is not news» is a decision about the feed, and the
model cannot know it: it learnt from rules that put power outages under
economy, and three other models given those rules split two to one the same
way. Such a decision goes into three places, in this order:

1. The rule, in `cat_en.json`, in words a teacher can apply. Until it is there
   every later relabelling quietly undoes the decision.
2. A pattern in `configs/annotator_config.json` (`rubric_detector`), with the
   real posts it must take and the real posts it must leave in
   `tests/test_rubrics.py`. It works the day it is deployed and needs no
   training. `routine_candidates.py <posts.jsonl> taken` shows what the
   patterns take from a week of production; read that before merging, and have
   two or three other models label a sample of what is taken and what is left
   under the new rule (the 2026-11 run used `agy -p` for Gemini and Claude and
   `codex exec` for GPT, which need no API key).
3. The model, at the next retraining, from labels made under the new rule. The
   pattern stays as the net under it.

`routine_candidates.py <posts.jsonl> repeated` is where the next such decision
usually comes from: headings one channel repeats five times a week that still
pass as news (traffic reports, daily fuel prices, a region's daily tally).

The rest are experiments along the way: `ft.py` and `heads.py` (category only),
`grok_label.py` and `label_en.py` (other teachers and prompts), `quant_variants.py`,
`fit_urgent.py`, `relabel_tech.py`, `shadow_rescore.py`, and `pick_review.py` with
`owner_eval.py` for the posts the owner labelled by hand.

## What the numbers do and do not say

The blind 500 and the training posts come from the same week (15-22 September
2026) and the same channels. 139 of the 500 repeat a training post word for word
or nearly; on the other 361 the category is right on 94.2%, against 94.8% on
all of them. 187 of the 500 allow a second category, and counting only the first
gives 86%. The test half is 256 posts, so two models within three points of each
other are not told apart by it. A retrain should split by time, drop the copies
before splitting, and test on more than 500.
