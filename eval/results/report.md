# evalcode evaluation report
gemini-3.5-flash-lite | tasks 12 | complete runs 24 | error runs 0 | incomplete pairs 0

## Summary

| mode | runs | passed | pass rate | first-try | mean attempts | mean LLM calls | mean tokens | median wall s |
|------|------|--------|-----------|-----------|---------------|----------------|-------------|---------------|
| rag | 12 | 11 | 91.7% | 11 | 1.25 | 1.25 | 2248 | 4.3 |
| norag | 12 | 12 | 100.0% | 12 | 1.00 | 1.00 | 943 | 4.1 |

## Paired comparison (K tasks with both modes complete)

both pass: 11, rag only: 0, norag only: 1, neither: 0

mean LLM calls (rag - norag): +0.25
mean attempts (rag - norag): +0.25

## Per task

| id | libs | rag | norag |
|----|------|-----|-------|
| argparse_cli | argparse | PASS a=1 c=1 | PASS a=1 c=1 |
| csv_sum_by_category | csv | PASS a=1 c=1 | PASS a=1 c=1 |
| dataclasses_inventory | dataclasses | FAIL a=4 c=4 | PASS a=1 c=1 |
| dt_business_days | datetime | PASS a=1 c=1 | PASS a=1 c=1 |
| dt_utc_iso | datetime | PASS a=1 c=1 | PASS a=1 c=1 |
| itertools_rle | itertools | PASS a=1 c=1 | PASS a=1 c=1 |
| numpy_zscore | numpy | PASS a=1 c=1 | PASS a=1 c=1 |
| pandas_monthly_totals | pandas | PASS a=1 c=1 | PASS a=1 c=1 |
| pandas_named_agg | pandas | PASS a=1 c=1 | PASS a=1 c=1 |
| pathlib_find_files | pathlib | PASS a=1 c=1 | PASS a=1 c=1 |
| re_log_parse | re | PASS a=1 c=1 | PASS a=1 c=1 |
| sqlite_top_customers | sqlite3 | PASS a=1 c=1 | PASS a=1 c=1 |

## Caveats

- Sampling temperature > 0; results may vary between runs.
- The model sees the acceptance tests; pass rate measures "can satisfy the visible spec".
- RAG's effect shows mainly in attempts / LLM calls / first-try rate.
