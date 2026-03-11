# External baseline code

This reference package implements the controlled same-backbone comparisons and paper ablations. Published external baselines must be reproduced in their official frameworks because their preprocessing and protocols are method-specific:

- FastPoseGait: GaitGraph, GPGait/GPGait++, and GaitHeat — `https://github.com/BNU-IVC/FastPoseGait`
- OpenGait: SkeletonGait family — `https://github.com/ShiqiYu/OpenGait`
- GaitGraph2 — `https://github.com/tteepe/GaitGraph2`

Run `scripts/fetch_external_baselines.sh` on the cloud checkout to clone these sources here. Before training, record each commit, official config, pose source, and any local changes. Keep official published results and locally reproduced results in separate columns.

