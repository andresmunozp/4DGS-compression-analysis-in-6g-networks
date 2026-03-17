#!/usr/bin/env bash
set -euo pipefail

########################################
# CONFIG
########################################
SCENE="coffee_martini"
WORK="${WORK:-/mnt/c/Users/Usuario2/Documents/JhojanAndres4E/4DGaussians}"
TESTING_ITERATIONS=(2000)
SAVE_ITERATIONS=(14000)
DATA_PATH="${WORK}/data/dynerf/${SCENE}"
CONFIG="${WORK}/arguments/dynerf/${SCENE}.py"
ITERATIONS=14000
OUTPUT="${WORK}/output/dynerf/${SCENE}"
PORT=6017
OUTROOT="${WORK}/outputs_eval"
RESULTS_DIR="${WORK}/results_json"

mkdir -p "${OUTROOT}" "${RESULTS_DIR}"

RUN_ID="4dgs_${SCENE}_it${ITERATIONS}_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTROOT}/${RUN_ID}"
mkdir -p "${RUN_DIR}"

EXPNAME="${RUN_DIR}/exp_${SCENE}_it${ITERATIONS}"

# Logs principales
LOG_TRAIN="${RUN_DIR}/train.log"
LOG_RENDER="${RUN_DIR}/render.log"
LOG_METRICS="${RUN_DIR}/metrics.log"
LOG_PERFRAME="${RUN_DIR}/perframe.log"

# Logs de recursos
RESOURCE_CSV="${RUN_DIR}/resources_monitor.csv"
RESOURCE_STAGE_JSON="${RUN_DIR}/resource_stage_summary.json"
RESOURCE_MONITOR_LOG="${RUN_DIR}/resource_monitor.log"

########################################
# CHECKS
########################################
command -v python >/dev/null 2>&1 || { echo "❌ python no encontrado"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "❌ python3 no encontrado"; exit 1; }
command -v awk >/dev/null 2>&1 || { echo "❌ awk no encontrado"; exit 1; }
command -v free >/dev/null 2>&1 || { echo "❌ free no encontrado"; exit 1; }
command -v df >/dev/null 2>&1 || { echo "❌ df no encontrado"; exit 1; }
command -v top >/dev/null 2>&1 || { echo "❌ top no encontrado"; exit 1; }

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "⚠️ nvidia-smi no encontrado. Se registrarán solo CPU/RAM."
  NVIDIA_AVAILABLE=0
else
  NVIDIA_AVAILABLE=1
fi

########################################
# INIT RESOURCE CSV
########################################
echo "timestamp,stage,cpu_percent,ram_used_mb,ram_total_mb,ram_percent,swap_used_mb,swap_total_mb,gpu_util_percent,vram_used_mb,vram_total_mb,vram_percent,gpu_temp_c,gpu_power_w,disk_used_kb,disk_avail_kb" > "${RESOURCE_CSV}"

########################################
# RESOURCE MONITOR
########################################
CURRENT_STAGE="idle"
RESOURCE_PID=""

start_resource_monitor() {
  echo "📊 Iniciando monitor de recursos..."
  (
    while true; do
      TS="$(date '+%Y-%m-%d %H:%M:%S')"

      # CPU total (% usado aproximado)
      CPU_PERCENT=$(top -bn1 | awk '/^%Cpu/ {gsub(",",".",$2); gsub(",",".",$4); print $2 + $4; exit}')
      CPU_PERCENT="${CPU_PERCENT:-0}"

      # RAM / Swap
      RAM_USED_MB=$(free -m | awk '/^Mem:/ {print $3}')
      RAM_TOTAL_MB=$(free -m | awk '/^Mem:/ {print $2}')
      SWAP_USED_MB=$(free -m | awk '/^Swap:/ {print $3}')
      SWAP_TOTAL_MB=$(free -m | awk '/^Swap:/ {print $2}')

      if [[ -z "${RAM_USED_MB}" || -z "${RAM_TOTAL_MB}" || "${RAM_TOTAL_MB}" == "0" ]]; then
        RAM_USED_MB=0
        RAM_TOTAL_MB=0
        RAM_PERCENT=0
      else
        RAM_PERCENT=$(awk -v u="${RAM_USED_MB}" -v t="${RAM_TOTAL_MB}" 'BEGIN { if (t==0) print 0; else printf "%.2f", (u/t)*100 }')
      fi

      # Disco del RUN_DIR
      DISK_USED_KB=$(df -k "${RUN_DIR}" | awk 'NR==2 {print $3}')
      DISK_AVAIL_KB=$(df -k "${RUN_DIR}" | awk 'NR==2 {print $4}')
      DISK_USED_KB="${DISK_USED_KB:-0}"
      DISK_AVAIL_KB="${DISK_AVAIL_KB:-0}"

      # GPU
      if [[ "${NVIDIA_AVAILABLE}" == "1" ]]; then
        GPU_RAW=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)

        if [[ -n "${GPU_RAW}" ]]; then
          GPU_UTIL=$(echo "${GPU_RAW}" | awk -F',' '{gsub(/ /,"",$1); print $1}')
          VRAM_USED_MB=$(echo "${GPU_RAW}" | awk -F',' '{gsub(/ /,"",$2); print $2}')
          VRAM_TOTAL_MB=$(echo "${GPU_RAW}" | awk -F',' '{gsub(/ /,"",$3); print $3}')
          GPU_TEMP_C=$(echo "${GPU_RAW}" | awk -F',' '{gsub(/ /,"",$4); print $4}')
          GPU_POWER_W=$(echo "${GPU_RAW}" | awk -F',' '{gsub(/ /,"",$5); print $5}')
          VRAM_PERCENT=$(awk -v u="${VRAM_USED_MB}" -v t="${VRAM_TOTAL_MB}" 'BEGIN { if (t==0) print 0; else printf "%.2f", (u/t)*100 }')
        else
          GPU_UTIL=0
          VRAM_USED_MB=0
          VRAM_TOTAL_MB=0
          VRAM_PERCENT=0
          GPU_TEMP_C=0
          GPU_POWER_W=0
        fi
      else
        GPU_UTIL=0
        VRAM_USED_MB=0
        VRAM_TOTAL_MB=0
        VRAM_PERCENT=0
        GPU_TEMP_C=0
        GPU_POWER_W=0
      fi

      echo "${TS},${CURRENT_STAGE},${CPU_PERCENT},${RAM_USED_MB},${RAM_TOTAL_MB},${RAM_PERCENT},${SWAP_USED_MB},${SWAP_TOTAL_MB},${GPU_UTIL},${VRAM_USED_MB},${VRAM_TOTAL_MB},${VRAM_PERCENT},${GPU_TEMP_C},${GPU_POWER_W},${DISK_USED_KB},${DISK_AVAIL_KB}" >> "${RESOURCE_CSV}"

      sleep 2
    done
  ) >"${RESOURCE_MONITOR_LOG}" 2>&1 &

  RESOURCE_PID=$!
  echo "✅ Monitor iniciado con PID=${RESOURCE_PID}"
}

stop_resource_monitor() {
  if [[ -n "${RESOURCE_PID}" ]] && kill -0 "${RESOURCE_PID}" 2>/dev/null; then
    echo "🛑 Deteniendo monitor de recursos..."
    kill "${RESOURCE_PID}" 2>/dev/null || true
    wait "${RESOURCE_PID}" 2>/dev/null || true
    echo "✅ Monitor detenido"
  fi
}

cleanup() {
  stop_resource_monitor
}
trap cleanup EXIT

set_stage() {
  CURRENT_STAGE="$1"
  echo "➡️ Stage actual: ${CURRENT_STAGE}"
}

########################################
# STAGE SUMMARY FROM CSV
########################################
generate_stage_resource_summary() {
python3 - << EOF
import csv, json, math, os
from collections import defaultdict

csv_path = r"${RESOURCE_CSV}"
out_path = r"${RESOURCE_STAGE_JSON}"

metrics = [
    "cpu_percent",
    "ram_used_mb",
    "ram_percent",
    "swap_used_mb",
    "gpu_util_percent",
    "vram_used_mb",
    "vram_percent",
    "gpu_temp_c",
    "gpu_power_w",
]

def safe_float(v):
    try:
        return float(v)
    except:
        return None

data = defaultdict(lambda: defaultdict(list))

if os.path.exists(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stage = row.get("stage", "unknown")
            for m in metrics:
                val = safe_float(row.get(m, ""))
                if val is not None:
                    data[stage][m].append(val)

summary = {}
for stage, stage_metrics in data.items():
    summary[stage] = {}
    for m, vals in stage_metrics.items():
        if not vals:
            continue
        summary[stage][m] = {
            "samples": len(vals),
            "min": min(vals),
            "max": max(vals),
            "avg": sum(vals) / len(vals)
        }

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"✅ Resource stage summary saved: {out_path}")
EOF
}

########################################
# START RESOURCE MONITOR
########################################
start_resource_monitor

########################################
# TRAIN
########################################
set_stage "train"
echo "🚀 Training... (log: ${LOG_TRAIN})"
TRAIN_START=$(date +%s)

python train.py \
  -s "${DATA_PATH}" \
  --port ${PORT} \
  --expname "${EXPNAME}" \
  --configs "${CONFIG}" \
  2>&1 | tee "${LOG_TRAIN}"

TRAIN_END=$(date +%s)
TRAIN_SEC=$((TRAIN_END - TRAIN_START))

########################################
# PLY_PERFRAME
########################################
set_stage "perframe"
PERFRAME_START=$(date +%s)

echo "🚀 Ply perframe... (log: ${LOG_PERFRAME})"
python export_perframe_3DGS.py \
  --iteration "${ITERATIONS}" \
  --configs "${CONFIG}" \
  --model_path "${EXPNAME}" \
  --skip_train \
  --skip_test \
  --skip_video \
  2>&1 | tee "${LOG_PERFRAME}"

PERFRAME_END=$(date +%s)
PERFRAME_SEC=$((PERFRAME_END - PERFRAME_START))

########################################
# RENDER
########################################
set_stage "render"
echo "🖼️ Rendering... (log: ${LOG_RENDER})"
RENDER_START=$(date +%s)

python render.py \
  --model_path "${EXPNAME}" \
  --configs "${CONFIG}" \
  --skip_train \
  2>&1 | tee "${LOG_RENDER}"

RENDER_END=$(date +%s)
RENDER_SEC=$((RENDER_END - RENDER_START))

########################################
# METRICS
########################################
set_stage "metrics"
echo "📏 Computing metrics... (log: ${LOG_METRICS})"
METRICS_START=$(date +%s)

python metrics.py --model_path "${EXPNAME}" \
  2>&1 | tee "${LOG_METRICS}"

METRICS_END=$(date +%s)
METRICS_SEC=$((METRICS_END - METRICS_START))

########################################
# LOCATE METRIC FILES
########################################
set_stage "postprocess"

RESULTS_JSON=$(find "${EXPNAME}" -type f -name "results.json" | head -n 1)
PERVIEW_JSON=$(find "${EXPNAME}" -type f -name "per_view.json" | head -n 1 || true)

if [[ -z "${RESULTS_JSON}" || ! -f "${RESULTS_JSON}" ]]; then
  echo "❌ No se encontró results.json dentro de ${EXPNAME}"
  echo "🔎 Archivos encontrados (top 200):"
  find "${EXPNAME}" -maxdepth 4 -type f | head -n 200
  exit 1
fi

########################################
# COLLECT PLY PER-TIMESTAMP
########################################
PLY_PERTS_DIR="${EXPNAME}/gaussian_pertimestamp"

if [[ ! -d "${PLY_PERTS_DIR}" ]]; then
  echo "❌ No existe gaussian_pertimestamp en: ${PLY_PERTS_DIR}"
  echo "🔎 Contenido de EXPNAME (top 200):"
  find "${EXPNAME}" -maxdepth 2 \( -type d -o -type f \) | head -n 200
  exit 1
fi

PLY_PERTS_COUNT=$(find "${PLY_PERTS_DIR}" -type f -name "*.ply" | wc -l | tr -d ' ')
PLY_PERTS_TOTAL_BYTES=$(find "${PLY_PERTS_DIR}" -type f -name "*.ply" -printf "%s\n" 2>/dev/null | awk '{s+=$1} END{print s+0}')

PLY_PERTS_SAMPLE_FIRST=$(find "${PLY_PERTS_DIR}" -type f -name "*.ply" | sort | head -n 1 || true)
PLY_PERTS_SAMPLE_LAST=$(find "${PLY_PERTS_DIR}" -type f -name "*.ply" | sort | tail -n 1 || true)

echo "📦 gaussian_pertimestamp:"
echo "   dir: ${PLY_PERTS_DIR}"
echo "   ply files: ${PLY_PERTS_COUNT}"
echo "   total bytes: ${PLY_PERTS_TOTAL_BYTES}"
echo "   first: ${PLY_PERTS_SAMPLE_FIRST}"
echo "   last: ${PLY_PERTS_SAMPLE_LAST}"

########################################
# LOAD FULL METRICS.JSON
########################################
FULL_METRICS=$(python3 -c "
import json
d=json.load(open('${RESULTS_JSON}','r'))
print(json.dumps(d))
")

########################################
# FFMPEG
########################################
FFMPEG="$HOME/ffmpeg_static/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg"

if [[ ! -x "${FFMPEG}" ]]; then
  echo "❌ No existe ffmpeg estático en: ${FFMPEG}"
  exit 1
fi

echo "✅ Using ffmpeg:"
"${FFMPEG}" -version | head -n 1

if ! "${FFMPEG}" -hide_banner -h filter=libvmaf >/dev/null 2>&1; then
  echo "❌ Tu ffmpeg estático NO tiene libvmaf."
  echo "🔎 Filtros relacionados con VMAF:"
  "${FFMPEG}" -hide_banner -filters 2>&1 | grep -i vmaf || true
  exit 1
fi

echo "✅ libvmaf disponible"

########################################
# VMAF
########################################
echo "🎬 Computing VMAF (test)..."

ITER="${ITERATIONS}"
TEST_ROOT="${EXPNAME}/test/ours_${ITER}"
RENDER_DIR="${TEST_ROOT}/renders"
GT_DIR="${TEST_ROOT}/gt"

if [[ ! -d "${GT_DIR}" || ! -d "${RENDER_DIR}" ]]; then
  echo "❌ No existen carpetas para VMAF:"
  echo "   GT_DIR=${GT_DIR}"
  echo "   RENDER_DIR=${RENDER_DIR}"
  echo "🔎 Directorios encontrados en EXPNAME (top 200):"
  find "${EXPNAME}" -maxdepth 4 -type d | head -n 200
  exit 1
fi

GT_FIRST=$(ls "${GT_DIR}"/*.png 2>/dev/null | head -n 1 || true)
RD_FIRST=$(ls "${RENDER_DIR}"/*.png 2>/dev/null | head -n 1 || true)

if [[ -z "${GT_FIRST}" || -z "${RD_FIRST}" ]]; then
  echo "❌ No hay PNGs en GT o renders."
  echo "   Ejemplo GT: ${GT_FIRST}"
  echo "   Ejemplo RD: ${RD_FIRST}"
  exit 1
fi

VMAF_JSON="${RUN_DIR}/vmaf_test.json"
REF_MP4="${RUN_DIR}/ref_test.mp4"
DIST_MP4="${RUN_DIR}/dist_test.mp4"

"${FFMPEG}" -y -framerate 30 -i "${GT_DIR}/%05d.png" \
  -pix_fmt yuv420p "${REF_MP4}"

"${FFMPEG}" -y -framerate 30 -i "${RENDER_DIR}/%05d.png" \
  -pix_fmt yuv420p "${DIST_MP4}"

"${FFMPEG}" -i "${DIST_MP4}" -i "${REF_MP4}" \
  -lavfi "libvmaf=log_path=${VMAF_JSON}:log_fmt=json" \
  -f null -

VMAF_TEST=$(python3 -c "import json; d=json.load(open('${VMAF_JSON}')); print(d['pooled_metrics']['vmaf']['mean'])")
echo "✅ VMAF_TEST=${VMAF_TEST}"

########################################
# STOP MONITOR AND BUILD SUMMARY
########################################
set_stage "finished"
stop_resource_monitor
generate_stage_resource_summary

########################################
# LOAD RESOURCE SUMMARY
########################################
RESOURCE_STAGE_SUMMARY=$(python3 -c "
import json
d=json.load(open('${RESOURCE_STAGE_JSON}','r'))
print(json.dumps(d))
")

########################################
# SAVE FINAL JSON
########################################
FINAL_JSON="${RESULTS_DIR}/${RUN_ID}.json"

python3 - << EOF
import json

run = {
  "scene": "${SCENE}",
  "expname": "${EXPNAME}",
  "train_time_sec": ${TRAIN_SEC},
  "ply_perframe_time_sec": ${PERFRAME_SEC},
  "render_time_sec": ${RENDER_SEC},
  "metrics_time_sec": ${METRICS_SEC},
  "metrics_full": ${FULL_METRICS},
  "VMAF": float("${VMAF_TEST}"),
  "resource_files": {
    "resource_csv": "${RESOURCE_CSV}",
    "resource_stage_summary_json": "${RESOURCE_STAGE_JSON}",
    "resource_monitor_log": "${RESOURCE_MONITOR_LOG}"
  },
  "resource_stage_summary": ${RESOURCE_STAGE_SUMMARY},
  "ply_pertimestamp": {
    "iteration": ${ITERATIONS},
    "dir": "${PLY_PERTS_DIR}",
    "num_ply_files": int("${PLY_PERTS_COUNT}"),
    "total_size_bytes": int("${PLY_PERTS_TOTAL_BYTES}"),
    "sample_first": "${PLY_PERTS_SAMPLE_FIRST}",
    "sample_last": "${PLY_PERTS_SAMPLE_LAST}",
    "log_path": "${LOG_PERFRAME}"
  }
}

json.dump(run, open("${FINAL_JSON}", "w"), indent=2)
print("✅ Saved:", "${FINAL_JSON}")
EOF

echo "🎉 DONE"
echo "📌 RUN_DIR: ${RUN_DIR}"
echo "📌 FINAL_JSON: ${FINAL_JSON}"
echo "📌 RESOURCE_CSV: ${RESOURCE_CSV}"
echo "📌 RESOURCE_STAGE_JSON: ${RESOURCE_STAGE_JSON}"