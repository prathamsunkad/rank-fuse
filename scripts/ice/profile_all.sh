#!/bin/bash
# Submit a resumable profiling chain. Usage: bash scripts/ice/profile_all.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
DATE=${PROFILE_DATE:-$(date +%F)}
ROOT="results/h200_ice_${DATE}"
mkdir -p "$ROOT"/{tables,figures,raw,ncu}
printf '# Profiling status\n\nDate: %s\n\n' "$DATE" > "$ROOT/STATUS.md"
printf 'Resume command: bash scripts/ice/profile_all.sh\n\n' >> "$ROOT/STATUS.md"

SETUP=$(sbatch --parsable scripts/ice/setup_cpu.sbatch)
CHECK=$(sbatch --parsable --dependency="afterok:${SETUP}" --export="ALL,PROFILE_DATE=${DATE},STAGE=check" scripts/ice/profile_stage.sbatch)
LAT=$(sbatch --parsable --dependency="afterok:${CHECK}" --export="ALL,PROFILE_DATE=${DATE},STAGE=latency" scripts/ice/profile_stage.sbatch)
ABL=$(sbatch --parsable --dependency="afterok:${LAT}" --export="ALL,PROFILE_DATE=${DATE},STAGE=ablation" scripts/ice/profile_stage.sbatch)
FULL=$(sbatch --parsable --dependency="afterok:${ABL}" --export="ALL,PROFILE_DATE=${DATE},STAGE=ncu_full" scripts/ice/profile_stage.sbatch)
NOFLUSH=$(sbatch --parsable --dependency="afterok:${FULL}" --export="ALL,PROFILE_DATE=${DATE},STAGE=ncu_noflush" scripts/ice/profile_stage.sbatch)
ANALYSIS=$(sbatch --parsable --dependency="afterok:${NOFLUSH}" --export="ALL,PROFILE_DATE=${DATE}" scripts/ice/profile_cpu.sbatch)
{
  printf '| Stage | Job ID | Dependency |\n|---|---:|---|\n'
  printf '| setup | %s | none |\n' "$SETUP"
  printf '| check | %s | afterok:%s |\n' "$CHECK" "$SETUP"
  printf '| latency | %s | afterok:%s |\n' "$LAT" "$CHECK"
  printf '| ablation | %s | afterok:%s |\n' "$ABL" "$LAT"
  printf '| ncu full | %s | afterok:%s |\n' "$FULL" "$ABL"
  printf '| ncu noflush | %s | afterok:%s |\n' "$NOFLUSH" "$FULL"
  printf '| analysis | %s | afterok:%s |\n' "$ANALYSIS" "$NOFLUSH"
  printf '\nNext command: `squeue -j %s,%s,%s,%s,%s,%s,%s` or read this file.\n' "$SETUP" "$CHECK" "$LAT" "$ABL" "$FULL" "$NOFLUSH" "$ANALYSIS"
} >> "$ROOT/STATUS.md"
echo "Submitted chain in $ROOT/STATUS.md"
