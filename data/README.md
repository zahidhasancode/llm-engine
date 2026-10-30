# Prompt set

`prompts.jsonl` holds 200 prompts sampled from
[Databricks Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k), a set of
real instructions written by people, licensed under CC BY-SA 3.0. This sample carries the same
licence. Each prompt is the instruction, followed by its context passage when it has one.

The mix is fixed by seed 42: 120 short prompts (under 200 characters), 60 medium (200 to 1,500) and
20 long (1,500 to 6,000). `source_row` is the row in the original file.
