#!/usr/bin/env bash
set -euo pipefail

########################################
# CONFIG
########################################

SCENE="${SCENE:-coffee_martini}"
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKDIR="${WORKDIR:-${ROOT_DIR}/data/dynerf/${SCENE}}"
DATATYPE="${DATATYPE:-llff}"

COLMAP_DIR="${WORKDIR}/colmap"
DBPATH="${COLMAP_DIR}/database.db"
IMGDIR="${COLMAP_DIR}/images"
SPARSE0="${COLMAP_DIR}/sparse/0"
DENSE_WS="${COLMAP_DIR}/dense/workspace"

LOGDIR="${WORKDIR}/logs"
mkdir -p "${LOGDIR}"

PIPELINE_JSONL="${LOGDIR}/pipeline.jsonl"
touch "${PIPELINE_JSONL}"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

have_cmd() { command -v "$1" >/dev/null 2>&1; }

file_size_bytes() {
  [[ -f "$1" ]] && stat -c%s "$1" 2>/dev/null || echo 0
}

dir_size_bytes() {
  [[ -d "$1" ]] && du -sb "$1" | cut -f1 || echo 0
}

dir_file_count() {
  [[ -d "$1" ]] && find "$1" -type f | wc -l | tr -d ' ' || echo 0
}

colmap_db_metrics_json() {
  if ! have_cmd sqlite3 || [[ ! -f "$DBPATH" ]]; then
    echo "{}"
    return
  fi

  images=$(sqlite3 "$DBPATH" "SELECT COUNT(*) FROM images;" 2>/dev/null || echo 0)
  keypoints=$(sqlite3 "$DBPATH" "SELECT COUNT(*) FROM keypoints;" 2>/dev/null || echo 0)
  matches=$(sqlite3 "$DBPATH" "SELECT COUNT(*) FROM matches;" 2>/dev/null || echo 0)

  echo "{\"images\":${images},\"keypoints_rows\":${keypoints},\"matches_rows\":${matches}}"
}

ply_point_count() {
  local ply="$1"
  if [[ ! -f "$ply" ]]; then
    echo 0
    return
  fi
  grep -m1 "element vertex" "$ply" | awk '{print $3}'
}

write_metrics_snapshot() {

  local step="$1"
  local out_json="${LOGDIR}/metrics_${step}.json"

  cat > "${out_json}" <<EOF
{
  "timestamp":"$(ts)",
  "scene":"${SCENE}",

  "artifacts":{

    "images_count":$(dir_file_count "${IMGDIR}"),

    "database_size_bytes":$(file_size_bytes "${DBPATH}"),

    "sparse_file_count":$(dir_file_count "${SPARSE0}"),

    "dense_workspace_size_bytes":$(dir_size_bytes "${DENSE_WS}"),

    "fused_ply_size_bytes":$(file_size_bytes "${DENSE_WS}/fused.ply"),
    "fused_ply_points":$(ply_point_count "${DENSE_WS}/fused.ply"),

    "downsampled_ply_size_bytes":$(file_size_bytes "${WORKDIR}/points3D_downsample2.ply"),

    "poses_bounds_size_bytes":$(file_size_bytes "${WORKDIR}/poses_bounds.npy")
  },

  "colmap_db_metrics":$(colmap_db_metrics_json)

}
EOF
}

run_step() {

  local step_id="$1"
  local step_name="$2"
  local cmd="$3"

  local out_log="${LOGDIR}/${step_id}_${step_name}.out.log"
  local err_log="${LOGDIR}/${step_id}_${step_name}.err.log"
  local timing_file="${LOGDIR}/timing_summary.csv"

  # Crear header si no existe
  if [[ ! -f "${timing_file}" ]]; then
    echo "step_id,step_name,duration_sec,exit_code" > "${timing_file}"
  fi

  start_epoch=$(date +%s)
  start_ts=$(ts)

  echo "{\"event\":\"start\",\"timestamp\":\"${start_ts}\",\"step\":\"${step_name}\"}" >> "${PIPELINE_JSONL}"

  set +e
  (
    cd "$(dirname "$0")"
    eval "${cmd}"
  ) >"${out_log}" 2>"${err_log}"
  exit_code=$?
  set -e

  end_epoch=$(date +%s)
  end_ts=$(ts)
  duration=$((end_epoch - start_epoch))

  write_metrics_snapshot "${step_id}_${step_name}"

  echo "{\"event\":\"end\",\"timestamp\":\"${end_ts}\",\"step\":\"${step_name}\",\"exit_code\":${exit_code},\"duration_sec\":${duration}}" >> "${PIPELINE_JSONL}"

  # 🔥 Guardar también en CSV
  echo "${step_id},${step_name},${duration},${exit_code}" >> "${timing_file}"

  if [[ "${exit_code}" -ne 0 ]]; then
    echo "❌ Failed at ${step_name}"
    echo "See: ${err_log}"
    exit "${exit_code}"
  fi

  echo "✅ ${step_name} (${duration}s)"
}


########################################
# PIPELINE
########################################

echo "Scene: ${SCENE}"
echo "Workdir: ${WORKDIR}"
echo



prepare_workspace() {
  mkdir -p "${COLMAP_DIR}"
  mkdir -p "${IMGDIR}"
  mkdir -p "${SPARSE0}"
  mkdir -p "${DENSE_WS}"
}

prepare_workspace

run_step "00" "extract frames" \
"./extract_frames.sh ${WORKDIR}"

run_step "01" "preprocess_dynerf" \
"python scripts/preprocess_dynerf.py --datadir data/dynerf/${SCENE}"

run_step "02" "datatype2colmap" \
"python scripts/${DATATYPE}2colmap.py ${WORKDIR}"

run_step "02b" "prepare_colmap_structure" \
"rm -rf ${COLMAP_DIR} && \
 mkdir -p ${COLMAP_DIR} && \
 cp -r ${WORKDIR}/image_colmap ${COLMAP_DIR}/images && \
 cp -r ${WORKDIR}/sparse_ ${COLMAP_DIR}/sparse_custom"

run_step "03" "colmap_feature_extractor" \
"colmap feature_extractor \
 --database_path ${DBPATH} \
 --image_path ${IMGDIR} \
 --ImageReader.single_camera 1"

run_step "04" "colmap_exhaustive_matcher" \
"colmap exhaustive_matcher --database_path ${DBPATH}"

run_step "05" "colmap_mapper" \
"mkdir -p ${COLMAP_DIR}/sparse && \
 rm -rf ${COLMAP_DIR}/sparse/0 && \
 colmap mapper \
  --database_path ${DBPATH} \
  --image_path ${IMGDIR} \
  --output_path ${COLMAP_DIR}/sparse \
  --Mapper.multiple_models 0"

run_step "06" "image_undistorter" \
"mkdir -p ${DENSE_WS} && colmap image_undistorter \
 --image_path ${IMGDIR} \
 --input_path ${COLMAP_DIR}/sparse/0 \
 --output_path ${DENSE_WS}"
# run_step "03" "colmap_feature_extractor" \
# "colmap feature_extractor --database_path ${DBPATH} --image_path ${IMGDIR} \
#  --SiftExtraction.max_image_size 4096 \
#  --SiftExtraction.max_num_features 16384 \
#  --SiftExtraction.estimate_affine_shape 0 \
#  --SiftExtraction.domain_size_pooling 0"

# # run_step "03" "colmap_feature_extractor" \
# # "colmap feature_extractor \
# #  --database_path ${DBPATH} \
# #  --image_path ${IMGDIR} \
# #  --ImageReader.camera_model SIMPLE_PINHOLE \
# #  --ImageReader.single_camera 1 \
# #  --SiftExtraction.max_image_size 4096 \
# #  --SiftExtraction.max_num_features 16384"


# run_step "03b" "inject_camera_model" \
# "python database.py \
#  --database_path ${DBPATH} \
#  --txt_path ${COLMAP_DIR}/sparse_custom/cameras.txt"

# run_step "04" "colmap_exhaustive_matcher" \
# "colmap exhaustive_matcher --database_path ${DBPATH}"

# # run_step "05" "point_triangulator" \
# # "mkdir -p ${SPARSE0} && colmap point_triangulator \
# #  --database_path ${DBPATH} \
# #  --image_path ${IMGDIR} \
# #  --input_path ${COLMAP_DIR}/sparse_custom \
# #  --output_path ${SPARSE0} \
# #  --clear_points 1"

# run_step "05" "colmap_mapper" \
# "mkdir -p ${COLMAP_DIR}/sparse && \
#  rm -rf ${COLMAP_DIR}/sparse/0 && \
#  colmap mapper \
#   --database_path ${DBPATH} \
#   --image_path ${IMGDIR} \
#   --output_path ${COLMAP_DIR}/sparse \
#   --Mapper.multiple_models 0"

# run_step "05b" "point_triangulator_refine" \
# "colmap point_triangulator \
#  --database_path ${DBPATH} \
#  --image_path ${IMGDIR} \
#  --input_path ${COLMAP_DIR}/sparse/0 \
#  --output_path ${COLMAP_DIR}/sparse/0 \
#  --clear_points 1"

#  run_step "05c" "camera_sync_sqlite_fix" \
# "sqlite3 ${DBPATH} \"
# UPDATE cameras
# SET model = (SELECT model FROM cameras WHERE camera_id=2),
#     width = (SELECT width FROM cameras WHERE camera_id=2),
#     height= (SELECT height FROM cameras WHERE camera_id=2),
#     params= (SELECT params FROM cameras WHERE camera_id=2),
#     prior_focal_length=1
# WHERE camera_id=1;
# \""


# run_step "05d" "verify_cameras" \
# "sqlite3 ${DBPATH} \
# 'SELECT camera_id, model, width, height, length(params) FROM cameras ORDER BY camera_id;'"

# run_step "06" "image_undistorter" \
# "mkdir -p ${DENSE_WS} && colmap image_undistorter \
#  --image_path ${IMGDIR} \
#  --input_path ${SPARSE0} \
#  --output_path ${DENSE_WS}"

run_step "07" "patch_match_stereo" \
"colmap patch_match_stereo --workspace_path ${DENSE_WS}"

run_step "08" "stereo_fusion" \
"colmap stereo_fusion --workspace_path ${DENSE_WS} \
 --output_path ${DENSE_WS}/fused.ply"

run_step "09" "downsample_pointcloud" \
"python scripts/downsample_point.py \
 ${DENSE_WS}/fused.ply \
 ${WORKDIR}/points3D_downsample2.ply"

# run_step "10" "llff_imgs2poses" \
# "python LLFF/imgs2poses.py ${WORKDIR}"

echo
echo "🎉 4DGS pipeline completed."
echo "Logs in: ${LOGDIR}"
