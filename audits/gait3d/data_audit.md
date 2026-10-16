# Dataset audit

- Source: `/Users/minasaslanyan/Documents/SCI Docs/Code/oap/oap_supcon/data/gait3d/processed/dataset.npz`
- SHA-256 over standardized arrays: `132636dbd68c5c9f24aa269f6ea1b32be449c78035bf324c0a002aae7369001a`
- Shape: `(24204, 500, 17, 2)` (`N,T,J,D`)
- Coordinate representation: `2D`
- Identity leakage check: passed
- Finite-coordinate and visibility-range checks: passed

## Split summary

| split | identities | sequences | mean_frames | min_frames | max_frames | joints | dimensions | mean_source_visibility | zero_source_visibility_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gallery | 947 | 5122 | 130.64 | 25 | 500 | 17 | 2 | 0.8289566 | 0.0 |
| probe | 947 | 947 | 123.34 | 25 | 482 | 17 | 2 | 0.82831562 | 0.0 |
| train | 2892 | 18135 | 129.75 | 7 | 500 | 17 | 2 | 0.82936424 | 0.0 |

This structural audit does not replace manual skeleton rendering, joint-mapping review, or license review.

## Source coverage

- Protocol: `gait3d_available_pose`
- Partition SHA-256: `886e278b25a2b9ba886fd3bd57c32d1698c1a6c3d419eb76e4aeec864e572abe`
- Missing training identities: 108
- Missing test identities: 53
- Missing designated probes: 53
- Excluded probes without gallery: 0
