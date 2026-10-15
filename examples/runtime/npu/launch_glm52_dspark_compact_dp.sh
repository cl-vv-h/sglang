#!/usr/bin/env bash
set -eo pipefail

# Ascend A3 single node: TP16 / attention-DP4 / attention-TP4 by default.
# SPS_TABLE=/home/sps_npu_tp16_adp4.json bash launch_glm52_dspark_compact_dp.sh
# GRAPH=0 OVERLAP=0 ... is supported for eager debugging.
# PRINT_COMMAND=1 only prints the launch configuration.

MODE="${MODE:-dspark}"                 # dspark / target-only
GRAPH="${GRAPH:-1}"                     # 0=eager; 1=NPU graph
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8030}"
TP_SIZE="${TP_SIZE:-16}"
ATTN_DP_SIZE="${ATTN_DP_SIZE:-4}"
OVERLAP="${OVERLAP:-1}"
DEEPEP_MODE="${DEEPEP_MODE:-auto}"
NETWORK_IFACE="${NETWORK_IFACE:-lo}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-GLM-5.2-w8a8}"

# 从双机收回为单机初始预算；实际是否可容纳取决于权重、设备内存和运行环境。
CONTEXT_LENGTH="${CONTEXT_LENGTH:-69632}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-69632}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-32}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.825}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-16384}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-280000}"
# 图档位按每组请求数配置；运行时会过滤不满足 Attention TP 对齐的档位。
GRAPH_BATCH_SIZES="${GRAPH_BATCH_SIZES:-1 2 4 8 16}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_ROOT="${SGLANG_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
TARGET_MODEL="${TARGET_MODEL:-/home/weights/GLM-5.2-w8a8/}"
DRAFT_MODEL="${DRAFT_MODEL:-/home/weights/GLM-5.2-DSpark-NPU-0805/}"
SPS_TABLE="${SPS_TABLE-/home/sps_npu_tp16_adp4.json}"
SPS_PROFILE="${SPS_PROFILE:-0}"
RAGGED_VERIFY_MODE="${RAGGED_VERIFY_MODE:-compact}"
# 调试日志会增加同步开销，默认关闭；需要验证窗口变化时设为 1。
DSPARK_DEBUG="${DSPARK_DEBUG:-0}"
APPLY_QUAROT_TO_DRAFT="${APPLY_QUAROT_TO_DRAFT:-True}"
PRINT_COMMAND="${PRINT_COMMAND:-0}"
LOG_FILE="${LOG_FILE:-${SGLANG_ROOT}/server.a3.${MODE}.${RAGGED_VERIFY_MODE}.adp${ATTN_DP_SIZE}.log}"

CANN_ENV="${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
ATB_ENV="${ATB_ENV:-/usr/local/Ascend/nnal/atb/set_env.sh}"
# 可选系统调优；默认不修改宿主机 sysctl。
SYSTEM_TUNE="${SYSTEM_TUNE:-0}"
# 已安装包含 sgl-kernel-npu #791 的版本时无需 overlay。
REGISTER_KERNEL_OVERLAY="${REGISTER_KERNEL_OVERLAY:-0}"
KERNEL_REPO="${KERNEL_REPO:-/workspace/sgl-kernel-npu}"
KERNEL_OVERLAY_SCRIPT="${KERNEL_OVERLAY_SCRIPT:-${SGLANG_ROOT}/devtools/glm52_ms1/register_glm52_dspark_kernel.py}"

fail() { echo "ERROR: $*" >&2; exit 2; }
[ "$#" -eq 0 ] || fail "Single-node script takes no node-rank argument; use TP_SIZE=N."
case "$MODE" in dspark|target-only) ;; *) fail "Invalid MODE: $MODE" ;; esac
case "$DEEPEP_MODE" in auto|normal|low_latency) ;; *) fail 'Invalid DEEPEP_MODE.' ;; esac
for setting in GRAPH OVERLAP SPS_PROFILE DSPARK_DEBUG PRINT_COMMAND SYSTEM_TUNE REGISTER_KERNEL_OVERLAY; do
  case "${!setting}" in 0|1) ;; *) fail "$setting must be 0 or 1" ;; esac
done
case "$RAGGED_VERIFY_MODE" in compact|static|cap-accept) ;; *) fail 'Invalid RAGGED_VERIFY_MODE.' ;; esac
for setting in TP_SIZE ATTN_DP_SIZE PORT CONTEXT_LENGTH MAX_TOTAL_TOKENS MAX_RUNNING_REQUESTS CHUNKED_PREFILL_SIZE MAX_PREFILL_TOKENS; do
  [[ "${!setting}" =~ ^[1-9][0-9]*$ ]] || fail "$setting must be a positive integer."
done
(( TP_SIZE % ATTN_DP_SIZE == 0 )) || fail 'TP_SIZE must be divisible by ATTN_DP_SIZE.'
(( MAX_RUNNING_REQUESTS >= ATTN_DP_SIZE )) || fail 'MAX_RUNNING_REQUESTS must be at least ATTN_DP_SIZE.'
ATTN_TP_SIZE=$((TP_SIZE / ATTN_DP_SIZE))
if [ "$SPS_PROFILE" = 1 ] && { [ "$MODE" != dspark ] || [ "$GRAPH" != 1 ]; }; then
  fail 'SPS_PROFILE=1 requires MODE=dspark and GRAPH=1.'
fi
read -r -a GRAPH_BS <<< "$GRAPH_BATCH_SIZES"
[ "${#GRAPH_BS[@]}" -gt 0 ] || fail 'GRAPH_BATCH_SIZES cannot be empty.'
for size in "${GRAPH_BS[@]}"; do
  [[ "$size" =~ ^[1-9][0-9]*$ ]] || fail 'GRAPH_BATCH_SIZES must contain positive integers.'
done

SERVER_ARGS=(
  --model-path "$TARGET_MODEL"
  --attention-backend ascend
  --device npu
  --host "$HOST"
  --port "$PORT"
  --tp-size "$TP_SIZE"
  --attn-dp-size "$ATTN_DP_SIZE"
  --enable-dp-lm-head
  --nnodes 1
  --node-rank 0
  --context-length "$CONTEXT_LENGTH"
  --max-total-tokens "$MAX_TOTAL_TOKENS"
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
  --max-prefill-tokens "$MAX_PREFILL_TOKENS"
  --mem-fraction-static "$MEM_FRACTION_STATIC"
  --max-running-requests "$MAX_RUNNING_REQUESTS"
  --trust-remote-code
  --served-model-name "$SERVED_MODEL_NAME"
  --quantization modelslim
  --moe-a2a-backend deepep
  --deepep-mode "$DEEPEP_MODE"
  --enable-cache-report
  --enable-metrics
  --log-level info
)
if [ "$MODE" = dspark ]; then
  SERVER_ARGS+=(
    --speculative-algorithm DSPARK
    --speculative-draft-model-path "$DRAFT_MODEL"
    --speculative-draft-model-quantization unquant
    --speculative-draft-attention-backend ascend
    --speculative-dspark-block-size 8
    --speculative-num-draft-tokens 9
  )
  # 标定时不加载旧成本表。二维 --fracs 标定使用 compact 模式。
  if [ "$SPS_PROFILE" = 0 ] && [ "$RAGGED_VERIFY_MODE" != static ] && [ -n "$SPS_TABLE" ]; then
    SERVER_ARGS+=(--speculative-dspark-sps-table-path "$SPS_TABLE")
  fi
fi
if [ "$OVERLAP" = 0 ]; then
  SERVER_ARGS+=(--disable-overlap-schedule)
fi
if [ "$GRAPH" = 0 ]; then
  SERVER_ARGS+=(--disable-cuda-graph)
else
  SERVER_ARGS+=(--cuda-graph-bs-decode "${GRAPH_BS[@]}")
fi

if [ "$PRINT_COMMAND" = 1 ]; then
  printf '# single node: TP=%s ATTN_DP=%s ATTN_TP=%s GRAPH=%s OVERLAP=%s\n' \
    "$TP_SIZE" "$ATTN_DP_SIZE" "$ATTN_TP_SIZE" "$GRAPH" "$OVERLAP"
  printf '# SGLANG_RAGGED_VERIFY_MODE=%s DSPARK_DEBUG=%s SPS_PROFILE=%s\n' \
    "$RAGGED_VERIFY_MODE" "$DSPARK_DEBUG" "$SPS_PROFILE"
  printf '# LOG_FILE=%q\n' "$LOG_FILE"
  printf '# DEEPEP_MODE=%s HCCL_BUFFSIZE=%s DEEPEP_HCCL_BUFFSIZE=%s\n' \
    "$DEEPEP_MODE" "${HCCL_BUFFSIZE:-1000}" "${DEEPEP_HCCL_BUFFSIZE:-inherit}"
  printf '# DeepEP: LL_capacity=%s normal_rounds=%s normal_tokens_per_round=%s\n' \
    "${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-128}" \
    "${DEEPEP_NORMAL_LONG_SEQ_ROUND:-72}" "${DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS:-1024}"
  printf 'python3 -m sglang.launch_server '
  printf '%q ' "${SERVER_ARGS[@]}"
  printf '\n'
  exit 0
fi

[ -d "$SGLANG_ROOT/python/sglang" ] || fail "Invalid SGLANG_ROOT: $SGLANG_ROOT"
[ -d "$TARGET_MODEL" ] || fail "TARGET_MODEL not found: $TARGET_MODEL"
[ -f "$CANN_ENV" ] || fail "CANN_ENV not found: $CANN_ENV"
[ -f "$ATB_ENV" ] || fail "ATB_ENV not found: $ATB_ENV"
if [ "$MODE" = dspark ]; then
  [ -d "$DRAFT_MODEL" ] || fail "DRAFT_MODEL not found: $DRAFT_MODEL"
  if [ "$SPS_PROFILE" = 0 ] && [ "$RAGGED_VERIFY_MODE" != static ]; then
    if [ -n "$SPS_TABLE" ]; then
      [ -r "$SPS_TABLE" ] || fail "SPS_TABLE not readable: $SPS_TABLE; use SPS_PROFILE=1 to collect on A3."
    else
      echo 'WARNING: no SPS table selected; compact may fall back to verify-all.' >&2
    fi
  fi
  if [ "$REGISTER_KERNEL_OVERLAY" = 1 ]; then
    [ -d "$KERNEL_REPO" ] || fail "KERNEL_REPO not found: $KERNEL_REPO"
    [ -f "$KERNEL_OVERLAY_SCRIPT" ] || fail "KERNEL_OVERLAY_SCRIPT not found: $KERNEL_OVERLAY_SCRIPT"
  fi
fi
if [ "$SYSTEM_TUNE" = 1 ]; then
  if compgen -G '/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor' >/dev/null; then
    echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null
  fi
  sysctl -w vm.swappiness=0
  sysctl -w kernel.numa_balancing=0
  sysctl -w kernel.sched_migration_cost_ns=50000
fi

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
unset SGLANG_DSPARK_DEBUG_DUMP GLM52_CONTEXT_SNAPSHOT_CONFIG GLM52_PROPOSAL_SNAPSHOT_CONFIG
unset SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_METHOD SGLANG_SIMULATE_ACC_TOKEN_MODE
unset SGLANG_DSPARK_ENABLE_SPS_RECORD
unset SGLANG_SIMULATE_UNIFORM_EXPERTS SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS
unset SGLANG_ENABLE_SPEC_V2 SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE
unset SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION SGLANG_SCHEDULER_SKIP_ALL_GATHER
unset SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES SGLANG_EXPERIMENTAL_CPP_RADIX_TREE
source "$CANN_ENV"
source "$ATB_ENV"

export ASCEND_LAUNCH_BLOCKING="${ASCEND_LAUNCH_BLOCKING:-0}"
export SGLANG_SET_CPU_AFFINITY=1
export STREAMS_PER_DEVICE=32
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM="${SGLANG_ENABLE_OVERLAP_PLAN_STREAM:-1}"
export SGLANG_NPU_USE_MULTI_STREAM="${SGLANG_NPU_USE_MULTI_STREAM:-0}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-1000}"
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_SOCKET_IFNAME="$NETWORK_IFACE"
export GLOO_SOCKET_IFNAME="$NETWORK_IFACE"
export TRANSFORMERS_VERBOSITY=error
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export DEEPEP_NORMAL_LONG_SEQ_ROUND="${DEEPEP_NORMAL_LONG_SEQ_ROUND:-72}"
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS="${DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS:-1024}"
export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1
export PYTHONPATH="${SGLANG_ROOT}/python${PYTHONPATH:+:$PYTHONPATH}"
if [ "$MODE" = dspark ]; then
  export SGLANG_RAGGED_VERIFY_MODE="$RAGGED_VERIFY_MODE"
  export SGLANG_DSPARK_DEBUG_CONFIDENCE_PREFIX_SCHEDULER="$DSPARK_DEBUG"
  export SGLANG_NPU_GLM_DSPARK_APPLY_QUAROT_TO_DRAFT="$APPLY_QUAROT_TO_DRAFT"
  if [ "$SPS_PROFILE" = 1 ]; then
    export SGLANG_DSPARK_ENABLE_SPS_RECORD=1
    export SGLANG_SIMULATE_ACC_LEN=1
  fi
  if [ "$REGISTER_KERNEL_OVERLAY" = 1 ]; then
    python3 "$KERNEL_OVERLAY_SCRIPT" --kernel-repo "$KERNEL_REPO"
  fi
else
  unset SGLANG_RAGGED_VERIFY_MODE SGLANG_DSPARK_DEBUG_CONFIDENCE_PREFIX_SCHEDULER
  unset SGLANG_NPU_GLM_DSPARK_APPLY_QUAROT_TO_DRAFT SGLANG_NPU_GLM_DSPARK_QUAROT
fi

mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1
printf 'Starting A3: MODE=%s TP=%s ATTN_DP=%s ATTN_TP=%s GRAPH=%s OVERLAP=%s SPS_PROFILE=%s LOG_FILE=%s\n' \
  "$MODE" "$TP_SIZE" "$ATTN_DP_SIZE" "$ATTN_TP_SIZE" "$GRAPH" "$OVERLAP" "$SPS_PROFILE" "$LOG_FILE"
printf 'Command: python3 -m sglang.launch_server '
printf '%q ' "${SERVER_ARGS[@]}"
printf '\n'
printf 'Communication: DEEPEP_MODE=%s HCCL_BUFFSIZE=%s DEEPEP_HCCL_BUFFSIZE=%s\n' \
  "$DEEPEP_MODE" "$HCCL_BUFFSIZE" "${DEEPEP_HCCL_BUFFSIZE:-inherit}"
printf 'DeepEP: LL_capacity=%s normal_rounds=%s normal_tokens_per_round=%s\n' \
  "${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-128}" \
  "$DEEPEP_NORMAL_LONG_SEQ_ROUND" "$DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS"
exec python3 -m sglang.launch_server "${SERVER_ARGS[@]}"
