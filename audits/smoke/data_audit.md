# Dataset audit

- Source: `/Users/minasaslanyan/Documents/SCI Docs/Paper 5/try_9/oap_supcon_experiments/data/smoke/processed/dataset.npz`
- SHA-256 over standardized arrays: `9cce9af124fa481bc411a5c2e52737bb9b27a415213eca15d50ea9bc5be17bfc`
- Shape: `(56, 24, 17, 2)` (`N,T,J,D`)
- Coordinate representation: `2D`
- Identity leakage check: passed
- Finite-coordinate and visibility-range checks: passed

## Split summary

| split | identities | sequences | mean_frames | joints | dimensions | missing_joint_frame_fraction |
| --- | --- | --- | --- | --- | --- | --- |
| gallery | 4 | 4 | 24 | 17 | 2 | 0.0 |
| probe | 4 | 12 | 24 | 17 | 2 | 0.0 |
| train | 8 | 32 | 24 | 17 | 2 | 0.0 |
| val | 8 | 8 | 24 | 17 | 2 | 0.0 |

This structural audit does not replace manual skeleton rendering, joint-mapping review, or license review.
