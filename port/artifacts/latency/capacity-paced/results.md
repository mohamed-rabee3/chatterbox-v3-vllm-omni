| Run | Users | Completed | Errors / unfinished | TTFA p95 | Playback start p95 | Buffered stall p95 | Turns with >100ms stall | Meets criteria |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| u1 | 1 | 15 | 1 / 0 | 0.296s | 0.496s | 0.000s | 0.0% | no |
| u2 | 2 | 27 | 1 / 0 | 0.350s | 0.550s | 0.003s | 0.0% | no |
| u4 | 4 | 48 | 1 / 0 | 1.097s | 1.297s | 0.192s | 12.5% | no |
| warm-repeat/u4 | 4 | 65 | 1 / 0 | 0.549s | 0.749s | 0.251s | 9.2% | no |
| u8 | 8 | 93 | 1 / 0 | 1.103s | 1.303s | 0.815s | 47.3% | no |
| warm-repeat/u30 | 30 | 279 | 3 / 0 | 0.946s | 1.146s | 12.281s | 98.2% | no |

Criteria: p95 playback start <= 1.0s, p95 cumulative buffered stall <= 0.1s, zero errors/unfinished requests. See each report for its client buffer, pacing, sample size, and workload. Passing a short sample is not production qualification.
