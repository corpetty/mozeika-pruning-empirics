# Experiment 29: Batched Throughput & Memory

| ctx,bs | baseline MB | k128 MB | reduction |
|--------|-----------|---------|-----------|
| ctx=512,bs=1 | 84 | 52 | 1.6x |
| ctx=512,bs=4 | 336 | 210 | 1.6x |
| ctx=512,bs=8 | 671 | 419 | 1.6x |
| ctx=2048,bs=1 | 336 | 210 | 1.6x |
| ctx=2048,bs=4 | 1342 | 839 | 1.6x |
| ctx=2048,bs=8 | 2684 | 1678 | 1.6x |
| ctx=8192,bs=1 | 1342 | 839 | 1.6x |
| ctx=8192,bs=4 | 5369 | 3355 | 1.6x |
| ctx=8192,bs=8 | 10737 | 6711 | 1.6x |

## Baseline tok/s
| ctx,bs | tok/s |
|--------|-------|
| ctx=512,bs=1 | 567 |
| ctx=512,bs=4 | 1716 |
| ctx=512,bs=8 | 1789 |
| ctx=2048,bs=1 | 1696 |
| ctx=2048,bs=4 | 1731 |
| ctx=2048,bs=8 | 1856 |
| ctx=8192,bs=1 | 1610 |

## Compressed overhead
| config,ctx | overhead |
|------------|----------|
| k128_4bit_ctx512 | 0.93x |
| k128_4bit_ctx2048 | 0.99x |
| k112_4bit_ctx512 | 1.20x |
| k112_4bit_ctx2048 | 1.66x |