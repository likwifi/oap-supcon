# Dataset audit

- Source: `/home/maslanyan/oap_supcon/data/casia_b_pose/processed/dataset.npz`
- SHA-256 over standardized arrays: `1776934f9c734c06a9f275ecbd5dd046382dc45097ba027fd95ceb0add4eb100`
- Shape: `(13637, 306, 17, 2)` (`N,T,J,D`)
- Coordinate representation: `2D`
- Identity leakage check: passed
- Finite-coordinate and visibility-range checks: passed

## Split summary

| split | identities | sequences | mean_frames | min_frames | max_frames | joints | dimensions | mean_source_visibility | zero_source_visibility_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gallery | 50 | 2197 | 95.63 | 1 | 196 | 17 | 2 | 0.85630888 | 0.0 |
| probe | 50 | 3300 | 95.01 | 43 | 171 | 17 | 2 | 0.85008824 | 0.0 |
| train | 74 | 8140 | 97.72 | 24 | 306 | 17 | 2 | 0.85180014 | 0.0 |

This structural audit does not replace manual skeleton rendering, joint-mapping review, or license review.
