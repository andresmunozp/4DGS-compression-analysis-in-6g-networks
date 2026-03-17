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




# ✅ expname único por corrida
EXPNAME="${RUN_DIR}/exp_${SCENE}_it${ITERATIONS}"
#EXPNAME="${WORK}/outputs_eval/4dgs_coffee_martini_it2000_20260218_173718/exp_coffee_martini_it2000"
# ✅ logs
LOG_TRAIN="${RUN_DIR}/train.log"
LOG_RENDER="${RUN_DIR}/render.log"
LOG_METRICS="${RUN_DIR}/metrics.log"


LOG_PERFRAME="${RUN_DIR}/perframe.log"
#PLY_PERFRAME_DIR="${EXPNAME}/ply_perframe_it${ITERATIONS}"
#mkdir -p "${PLY_PERFRAME_DIR}"


########################################
# TRAIN
########################################
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

# ########################################
# # PLY_PERFRAME
# ########################################
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
# ########################################
# # RENDER
# ########################################
echo "🖼️ Rendering... (log: ${LOG_RENDER})"
RENDER_START=$(date +%s)

python render.py \
  --model_path "${EXPNAME}" \
  --configs "${CONFIG}" \
  --skip_train \
  2>&1 | tee "${LOG_RENDER}"

RENDER_END=$(date +%s)
RENDER_SEC=$((RENDER_END - RENDER_START))

# ########################################
# # METRICS
# ########################################
echo "📏 Computing metrics... (log: ${LOG_METRICS})"
METRICS_START=$(date +%s)

python metrics.py --model_path "${EXPNAME}" \
  2>&1 | tee "${LOG_METRICS}"

METRICS_END=$(date +%s)
METRICS_SEC=$((METRICS_END - METRICS_START))

# ########################################
# # LOCATE METRIC FILES (solo dentro del exp)
# ########################################
RESULTS_JSON=$(find "${EXPNAME}" -type f -name "results.json" | head -n 1)
PERVIEW_JSON=$(find "${EXPNAME}" -type f -name "per_view.json" | head -n 1)

if [[ -z "${RESULTS_JSON}" || ! -f "${RESULTS_JSON}" ]]; then
  echo "❌ No se encontró results.json dentro de ${EXPNAME}"
  echo "🔎 Archivos encontrados (top 200):"
  find "${EXPNAME}" -maxdepth 4 -type f | head -n 200
  exit 1
fi

# ########################################
# # COLLECT PLY PER-TIMESTAMP (gaussian_pertimestamp)
# ########################################
PLY_PERTS_DIR="${EXPNAME}/gaussian_pertimestamp"

if [[ ! -d "${PLY_PERTS_DIR}" ]]; then
  echo "❌ No existe gaussian_pertimestamp en: ${PLY_PERTS_DIR}"
  echo "🔎 Contenido de EXPNAME (top 200):"
  find "${EXPNAME}" -maxdepth 2 -type d -o -type f | head -n 200
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

# ########################################
# # LOAD FULL METRICS.JSON
# ########################################
FULL_METRICS=$(python3 -c "
import json
d=json.load(open('${RESULTS_JSON}','r'))
print(json.dumps(d))
")


########################################
# FFMPEG (usar versión estática con libvmaf)
########################################
FFMPEG="$HOME/ffmpeg_static/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg"

if [[ ! -x "${FFMPEG}" ]]; then
  echo "❌ No existe ffmpeg estático en: ${FFMPEG}"
  exit 1
fi

echo "✅ Using ffmpeg:"
"${FFMPEG}" -version | head -n 1

# Chequeo robusto del filtro libvmaf (mejor que grep con pipefail)
if ! "${FFMPEG}" -hide_banner -h filter=libvmaf >/dev/null 2>&1; then
  echo "❌ Tu ffmpeg estático NO tiene libvmaf (filtro no disponible)."
  echo "🔎 Filtros relacionados con VMAF:"
  "${FFMPEG}" -hide_banner -filters 2>&1 | grep -i vmaf || true
  exit 1
fi

echo "✅ libvmaf disponible"

########################################
# VMAF (USAR GT QUE GENERA render.py)
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

# Verifica que haya frames
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
# SAVE FINAL JSON
########################################
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
