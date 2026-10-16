# Pose gait literature and code comparison — 3 October 2026

The linked paper is a foundational benchmark paper, not a current winning method for occluded 3D skeleton gait. The strongest pose-only real-occlusion result verified in this search is **GaitHeat++: 96.71% Rank-1 on the SUSTech1K occlusion condition**. That result uses 2D pose heatmaps. A current winner restricted to 3D joint coordinates could not be established from comparable published occlusion results.

The local code and saved evidence are currently a **2D HRNet coordinate pipeline**. Accepting a tensor with three coordinate channels does not establish a validated 3D gait system.

## What the linked paper establishes

[Gait Recognition in the Wild with Dense 3D Representations and A Benchmark](https://arxiv.org/html/2204.02569v1), CVPR 2022, introduces Gait3D and SMPLGait. Gait3D contains 4,000 people and more than 25,000 sequences from 39 cameras. SMPLGait combines a silhouette CNN with a branch that uses SMPL pose, shape and camera parameters to transform appearance features. It is a useful dataset and multimodal reference. Its overall retrieval evaluation does not establish leadership on a separately scored occluded-3D-joint endpoint.

## Stronger references, with their actual scope

| Reference | Input and evaluation | Verified result | Role in this project |
|---|---|---|---|
| [GaitHeat++](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04501.pdf), ECCV 2024 | ViTPose heatmaps; SUSTech1K, Table 2 | Occlusion **96.71%**, overall **90.00%** Rank-1 | Strongest verified pose-only result found for this real-occlusion condition |
| [HiPo-Gait](https://github.com/Nico-Cubero/HiPo-Gait), T-BIOM 2026 | ViTPose heatmaps; hierarchical limb processing | DeepGaitV2 variant: occlusion **96.2%**, overall **88.6%** Rank-1 on SUSTech1K | Recent architecture comparison; author manuscript Table V verifies the numbers |
| [GPGait++](https://github.com/BNU-IVC/FastPoseGait), TPAMI 2025 | Coordinates, bones and joint angles; adaptive part graphs | No matched occlusion number verified here | Most directly actionable stronger coordinate architecture; implementation is already under `external/FastPoseGait/` |
| [MirrorGait](https://github.com/BNU-IVC/StreamGait), ACM MM 2025 | 3D-aware self-supervised pretraining from 2D poses | Fine-tuned Rank-1: Gait3D **48.1%**, GREW **79.6%**, OUMVLP-Pose **73.9%** | Important pretraining reference; these are overall scores, not an occlusion leaderboard |
| [GaitMoE / OccGait](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01016.pdf), ECCV 2024 | Silhouette sequences and action-based experts | Dedicated occlusion experiments | Relevant temporal design ideas; modality differs from coordinate pose |

HiPo-Gait scores come from the [author manuscript, Table V](https://www.researchgate.net/publication/403942426_HiPo-Gait_Hierarchical_Pose_model_decoupling_for_Gait_Recognition). Its isolated limb encoders and local attention fusion are useful design references. A newer publication date does not guarantee a higher occlusion score.

GaitHeat uses the pose estimator's original probability maps and pose-guided alignment. The ++ variant switches from HRNet to ViTPose; an estimator improvement is therefore part of its gain. Rendering Gaussian maps from stored coordinates cannot recover the original uncertainty distribution. These numbers guide model selection, but do not isolate architecture at a fixed pose source.

MirrorGait's official repository advertises the scores above, but its visible tree contains assets and a README, with training/model release listed as a task. Treat it as published evidence with a reproduction limitation. [Official repository](https://github.com/BNU-IVC/StreamGait).

For actual 3D sensor skeletons under occlusion, [OG RGB+D / Multi-Kinect SDAS](https://arxiv.org/html/2107.08990v1) is a more directly relevant benchmark than a silhouette score. It evaluates a dual-skeleton, siamese ST-GCN approach using optimized joints. This older reference establishes an applicable protocol, not a verified current leaderboard winner. Its SkeletonGait name refers to a different method from the AAAI 2024 skeleton-map model.

OccGait's [official release](https://github.com/BNU-IVC/OccGait) provides silhouettes and says pose availability remains under consideration. The project's decision to disable it in the pose runner is appropriate.

## Current empirical position

The comparison below was recomputed from the five completed `metric_baseline` test runs in `results_v2/`, dated 1 October 2026. All use 85 epochs and seeds 11, 22, 33, 44 and 55. They share configuration and source hashes. The endpoint uses single-view galleries and excludes identical-view matches. NM, BG and CL are normal walking, bag carrying and clothing change.

| Method / descriptor | NM | BG | CL | Condition mean |
|---|---:|---:|---:|---:|
| Current global descriptor, five-seed mean | 67.64 | 49.50 | 27.91 | 48.35 |
| Current raw reliability-weighted parts | 74.14 | 55.47 | 32.52 | 54.04 |
| Current raw uniform parts | 74.02 | 55.56 | 32.54 | 54.04 |
| GaitGraph2, official FastPoseGait HRNet reproduction | 80.29 | 71.40 | 63.80 | 71.83 |
| GPGait, official FastPoseGait HRNet result | 93.60 | 80.15 | 69.29 | 81.01 |

External rows are from the [official FastPoseGait model zoo](https://github.com/BNU-IVC/FastPoseGait/blob/main/docs/model_zoo.md). They use the same named pose source and benchmark protocol, but different training recipes and implementations; they have not been reproduced by the current runner. The accuracy gap warrants a baseline reproduction before attributing failure to an individual module.

The local means, seed standard deviations and complete input artifact paths are in [current_results.json](</Users/minasaslanyan/Documents/SCI Docs/Code/oap/oap_supcon/audits/research_2026_10_03/current_results.json>). Current raw-part matching gains 5.69 percentage points over global matching. Reliability-weighted and uniform matching have essentially the same condition mean.

The development factorial independently shows a cost from part supervision: adding part SupCon reduces the corruption aggregate from 63.90% to 57.78%. The updated crossed interval is −10.29 to −2.00 percentage points. This is evidence for this particular recipe, not a general rejection of part learning. [Existing revision evidence](</Users/minasaslanyan/Documents/SCI Docs/Code/oap/oap_supcon/Paper/IJCV/REVISION_NOTES.md>).

## Architectural differences that matter

These are code observations followed by hypotheses to test, rather than proven explanations of the accuracy gap.

| Component | Current active V2 model | Difference to investigate |
|---|---|---|
| Input streams | Joint, bone and velocity at lags 1 and 2 | GPGait++ explicitly models joint angles and keeps separate stream encoders; current streams merge after shallow 1×1 stems |
| Graph relations | Anatomical graph plus bounded trainable edge offsets, shared across examples | GPGait++ derives relations from each frame's features and applies anatomical partitions |
| Part processing | Five pools and MLP heads after one shared graph trunk | GPGait++ has distinct part graph branches and semantic pyramid pooling; current heads do not supply part-specific feature extraction |
| Temporal aggregation | Dilated local convolutions and confidence-weighted means | Test masked max-plus-mean or learned segment aggregation; sequence-wide temporal attention and cycle correspondence are absent from the active V2 trunk |
| Missing evidence | Features are reset to zero at absent joints after every graph block | This suppresses invalid evidence, but also prevents inferred features from persisting at missing joints; reconstruction would need a separate latent mask and observed mask |
| Reliability | Confidence is an input channel and pooling/matching weight; internal presence is `confidence > 0` | Low-confidence positive detections still participate; there is no explicit reliability-conditioned graph edge or temporal aggregation rule |
| Retrieval training | Global CE and triplet; optional projected-part objectives | GPGait++ trains multiple part embeddings with separate FCs and BNNecks; do not copy its supervision without retaining matched controls |
| Pretraining | Dataset-specific supervised training | MirrorGait and GaitPTv2 investigate large unlabeled walking corpora; current experiments cannot measure this advantage |

Code anchors:

- [Current graph block](</Users/minasaslanyan/Documents/SCI Docs/Code/oap/oap_supcon/src/oap_supcon/model.py>), lines 230–259: adjacency is model-wide, rather than computed from the sequence.
- [Current stream construction and fusion](</Users/minasaslanyan/Documents/SCI Docs/Code/oap/oap_supcon/src/oap_supcon/model.py>), lines 307–329: streams merge before all graph blocks.
- [Current part pooling](</Users/minasaslanyan/Documents/SCI Docs/Code/oap/oap_supcon/src/oap_supcon/model.py>), lines 332–353: parts pool the common feature map.
- [Official GPGait++ model](</Users/minasaslanyan/Documents/SCI Docs/Code/oap/oap_supcon/external/FastPoseGait/fastposegait/modeling/models/gpgait++.py>), lines 20–27 and 71–194: independent streams, local/global branches, semantic pooling and separate heads.
- [Official adaptive part graph](</Users/minasaslanyan/Documents/SCI Docs/Code/oap/oap_supcon/external/FastPoseGait/fastposegait/modeling/components/units/unit_pagcn.py>), lines 98–116: relations are computed for each frame.

The optional `transformer` backend has spatial self-attention followed by temporal convolutions. It is not GaitPT's hierarchical spatiotemporal transformer, and the V2 preset selects `multistream`.

Input normalization is another experimental variable. Current code centers each frame on its pelvis and divides by sequence-average radial scale. This removes root translation and absolute scale. Synthetic masks are then applied to the normalized poses, so they do not test occlusion damage to the centering/scale estimate. For 3D data, compare a canonical articulated-pose stream with a separate root-motion stream rather than assuming all translation is irrelevant. [Normalization](</Users/minasaslanyan/Documents/SCI Docs/Code/oap/oap_supcon/src/oap_supcon/data.py>), lines 135–153.

The standard local GPGait++ code hard-codes two-coordinate slices. MMGait's use of GPGait++ for 3D does not make this local implementation a ready-made 3D backend. In [MMGait Table 4](https://arxiv.org/html/2604.15979v1), the reported normal-condition Rank-1 is 84.7% for 2D pose versus 42.8% for estimated 3D pose. This is one dataset and implementation, but it shows why adding depth alone should not be assumed to improve recognition.

## Recommended order of work

1. **Reproduce a strong coordinate reference.** Use the existing official GPGait/GPGait++ source with the same HRNet CASIA-B release, official identity split and evaluator. Reconcile the accuracy gap before interpreting architectural ablations. Count optimizer steps, examples and frame budgets; epoch labels alone do not match recipes.
2. **Transfer the successful matcher.** Compare global, raw uniform and raw reliable matching on the reproduced backbone. Keep its original descriptor as a reference and include matched random-region controls. This directly tests whether the current research finding survives a stronger encoder.
3. **Improve part feature extraction in controlled stages.** Starting from the metric baseline, test angle inputs, deeper separate stream processing, sample-dependent graph relations, hierarchical part branches, and masked temporal aggregation one at a time. Keep the tested raw matcher and compare its benefit across objectives. A learned hierarchy is a stronger hypothesis than another auxiliary part loss at this point.
4. **Obtain a real-occlusion endpoint.** Reproduce SUSTech1K with its official gallery/probe and view rules. Compare coordinates and original heatmaps at a fixed pose estimator before testing an estimator upgrade. GaitHeat++ and HiPo-Gait then become relevant measured references.
5. **Validate the 3D source before expanding the claim.** Stage true 3D sensor joints or a clearly labeled lifted/SMPL-derived export; audit coordinate axes, visibility, frame rate and root motion. Report those results separately from the established 2D experiments. Add pretraining or latent completion only with matched data and compute controls.

The research recommendation is to strengthen and reproduce the encoder/input pipeline while preserving the observed raw-part matching benefit. The available evidence does not support prioritizing more part contrastive losses or claiming state-of-the-art occluded 3D recognition. This review adds a research note and a result summary; it introduces no new recognition measurements.
