## Why the speed difference:

Cross-region latency: You're in ap-southeast-2 (Sydney), but async requests are routed through US. Each streaming token has to cross the Pacific Ocean twice.

Streaming overhead: For short responses (32 tokens), the per-token network overhead dominates. Each token is a separate network frame.

Timing breakdown comparison:

Async Generation: 3.31s (with 1.99s to first token)
Sync Generation: 2.59s
Difference: ~0.72s extra for cross-region + streaming
The tradeoff: Streaming gives better UX (user sees tokens appearing immediately after 1.99s), but total time is longer due to cross-region routing.