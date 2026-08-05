#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root/external"

if [[ ! -d FastPoseGait ]]; then git clone https://github.com/BNU-IVC/FastPoseGait.git; fi
if [[ ! -d OpenGait ]]; then git clone https://github.com/ShiqiYu/OpenGait.git; fi
if [[ ! -d GaitGraph2 ]]; then git clone https://github.com/tteepe/GaitGraph2.git; fi

for repository in FastPoseGait OpenGait GaitGraph2; do
  git -C "$repository" rev-parse HEAD
done

