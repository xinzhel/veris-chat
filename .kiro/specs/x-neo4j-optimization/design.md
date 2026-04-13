# Neo4j Query Optimization

## Problem

`get_parcel_context()` takes ~54s (cold cache) / ~35s (warm cache) for 7 queries on t3.medium (4GB RAM, 8.7M nodes).

Bottlenecks (from benchmarks):
- Overlay: 34.5s (104K nodes)
- HistoricalBusinessListing: 17.7s (553K nodes)
- Other 5 types: <1s each

Root cause: Neo4j on t3.medium can't fit the graph in memory. Every query hits disk I/O. Standard index on `hasPFI` doesn't help because `$pfi IN p.hasPFI` scans array properties.

Warm cache effect: first query 30s, subsequent 4-5s (data cached in OS page cache).

## Current Mitigations

- App-level parcel cache in `chat_api.py` (Task 5): first message slow, follow-ups instant
- Index `parcel_pfi` created but ineffective for array `IN` queries

## Optimization Options (ordered by effort/impact)

### Option 1: Single Combined Query (code change only)

Replace 7 separate queries with 1 query that returns all assessment types:

```cypher
MATCH (p:Parcel)-[rel:hasOnsiteAssessment|hasOffsiteAssessment]->(a:Resource)
WHERE $pfi IN p.hasPFI
RETURN type(rel) AS rel_type,
       [l IN labels(a) WHERE l <> "Resource"][0] AS assessment_type,
       properties(a) AS props
```

Then group results by `assessment_type` in Python. This avoids 7 round-trips and 7 separate scans.

Expected improvement: ~54s → ~35s (one scan instead of seven).

### Option 2: Upgrade EC2 to t3.large (8GB RAM)

More RAM = more data in Neo4j page cache = fewer disk reads.

```bash
aws ec2 stop-instances --region ap-southeast-2 --instance-ids i-018c87e156b4cbd8a
aws ec2 modify-instance-attribute --instance-id i-018c87e156b4cbd8a --instance-type t3.large
aws ec2 start-instances --instance-ids i-018c87e156b4cbd8a
```

Cost: $38/mo → $76/mo. Expected improvement: 2-3x faster after warm-up.

### Option 3: Denormalize PFI to scalar property

The root cause of index inefficiency: `hasPFI` is an array (from n10s `handleMultival: ARRAY`). If we add a scalar `pfi` property:

```cypher
MATCH (p:Parcel) WHERE size(p.hasPFI) > 0
SET p.pfi = p.hasPFI[0]
```

Then create index on scalar:
```cypher
CREATE INDEX parcel_pfi_scalar FOR (p:Parcel) ON (p.pfi)
```

Query becomes: `WHERE p.pfi = $pfi` (index lookup, O(1) instead of scan).

Expected improvement: ~54s → <1s. But requires one-time data migration (~hours on t3.medium).

### Option 4: Pre-compute parcel context to JSON/S3

For read-only data, pre-compute all parcel contexts and store as JSON:

```python
# One-time batch job
for each parcel:
    context = get_parcel_context(pfi)
    s3.put_object(Key=f"parcel_context/{pfi}.json", Body=json.dumps(context))
```

Then `get_parcel_context()` just reads from S3 (~100ms).

Expected improvement: ~54s → ~100ms. But requires batch job infrastructure.

### Option 5: Switch to full-text index

Neo4j full-text indexes support array properties:

```cypher
CREATE FULLTEXT INDEX parcel_pfi_ft FOR (p:Parcel) ON EACH [p.hasPFI]
```

Query: `CALL db.index.fulltext.queryNodes("parcel_pfi_ft", $pfi)`.

Expected improvement: similar to Option 3 but without data migration.

## Recommendation

1. **Now**: Option 1 (single query) — free, easy code change ✅ **DONE** (54s → 3.9s)
2. ~~If still slow: Option 3 (scalar PFI + index)~~ — not needed after Option 1
3. ~~If budget allows: Option 2 (upgrade instance)~~ — not needed after Option 1
4. **For production**: Option 4 (pre-compute) — eliminates Neo4j from hot path entirely

## Resolution

Applied Option 1: single combined query. Benchmark: **54s → 3.9s** (14x faster).
Inspired by Oz's `parcelReports.js` middleware which uses the same approach (2s on same-machine Neo4j).
App-level parcel cache (`_parcel_cache` in `chat_api.py`) ensures subsequent messages are instant (0ms).
