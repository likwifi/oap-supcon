# External baseline code

This reference package implements the controlled same-backbone comparisons and paper ablations. Published external baselines must be reproduced in their official frameworks because their preprocessing and protocols are method-specific:

- FastPoseGait: GaitGraph, GPGait/GPGait++, and GaitHeat — `https://github.com/BNU-IVC/FastPoseGait`
- OpenGait: SkeletonGait family — `https://github.com/ShiqiYu/OpenGait`
- GaitGraph2 — `https://github.com/tteepe/GaitGraph2`

Run `scripts/fetch_external_baselines.sh` on the cloud checkout to clone these sources here. Before training, record each commit, official config, pose source, and any local changes. Keep official published results and locally reproduced results in separate columns.


## Fetched commits (2026-09-18)

| repository | commit |
|---|---|
| FastPoseGait | `d3ce631971a01d806eeb3a154774d482c0951813` |
| OpenGait | `240e8cf18ae7a06f4ecd630c4e164e98ff17ae23` |
| GaitGraph2 | `0c619f370574baf9b0fa62af583588375c896e7c` |

Record these in any table that reports a reproduced baseline number, and label
published versus locally reproduced values separately.

### Status

Repositories are fetched only. No baseline has been reproduced yet: each
framework needs its own environment and its own preprocessed data layout, which
is a separate pipeline from this project's canonical NPZ. Gate 6 of
`EMPIRICAL_VALIDATION.md` remains unmet.

GaitGraph2 is the natural first target because it consumes the same CASIA-B
HRNet pose release (ScienceDB/FastPoseGait) that `data/casia_b_pose` is built
from, so no new data agreement is required -- only a converter from the
canonical NPZ back into its expected input format.
