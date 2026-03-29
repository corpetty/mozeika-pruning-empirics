# Experiment 15: Needle-in-a-Haystack Retrieval

- Model: Qwen3-14B-AWQ (40 layers, 8 KV heads)
- 3 needles × 5 depths × 4 ctx lengths

## Accuracy by Config × Context Length

| Config | ctx=4096 | ctx=8192 | ctx=16384 | ctx=32768 | Overall |
|--------|---|---|---|---|---|
| baseline | 93%(14/15) | 93%(14/15) | 87%(13/15) | 100%(15/15) | 93% |
| k128_4bit | 93%(14/15) | 93%(14/15) | 100%(15/15) | 100%(15/15) | 97% |
| k96_4bit | 100%(15/15) | 93%(14/15) | 100%(15/15) | 27%(4/15) | 80% |

## Accuracy by Config × Depth

| Config | 10% | 25% | 50% | 75% | 90% |
|--------|---|---|---|---|---|
| baseline | 100% | 83% | 83% | 100% | 100% |
| k128_4bit | 92% | 100% | 100% | 100% | 92% |
| k96_4bit | 75% | 100% | 75% | 75% | 75% |
