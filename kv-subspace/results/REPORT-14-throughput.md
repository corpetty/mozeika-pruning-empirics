# Experiment 14: Throughput and Memory Benchmark

- Model: Qwen3-14B-AWQ (40 layers, 8 KV heads, d_head=128)
- Decode steps per trial: 128

## Decode Throughput (tokens/sec)

| ctx_len | baseline | k128_4bit | k96_4bit |
|---------|---|---|---|
| 4096 | 11.75 | 0.92 | 1.85 |
| 8192 | 12.57 | 0.94 | 1.85 |
| 16384 | 12.51 | 0.94 | 1.86 |
| 32768 | N/A | N/A | N/A |

## Peak VRAM During Prefill (GB)

| ctx_len | baseline | k128_4bit | k96_4bit |
|---------|---|---|---|
| 4096 | 11.943 | 11.943 | 11.943 |
| 8192 | 13.901 | 13.901 | 13.901 |
| 16384 | 17.816 | 17.816 | 17.816 |
| 32768 | N/A | N/A | N/A |

## Analytical KV Cache Size (GB)

| Config | ctx=4096 | ctx=8192 | ctx=16384 | ctx=32768 |
|--------|---|---|---|---|
| baseline | 0.671 | 1.342 | 2.684 | 5.369 |
| k128_4bit | 0.168 | 0.336 | 0.671 | 1.342 |
| k96_4bit | 0.126 | 0.252 | 0.503 | 1.007 |
