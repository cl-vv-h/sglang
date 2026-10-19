#!/usr/bin/env bash
set -eo pipefail

# Ascend A5 two nodes, 8 NPUs each. Global TP16 / Attention DP4 / Attention TP4.
# Node 0 (141.61.45.118): bash launch_glm52_dspark_compact_dp_a5_2nodes.sh 0
# Node 1 (141.61.45.124): bash launch_glm52_dspark_compact_dp_a5_2nodes.sh 1
# Use the same code, weights, SPS table and parallel settings on both nodes.
# PRINT_COMMAND=1 prints the launch configuration without starting the server.

NODE_RANK="${1:-${NODE_RANK:-}}"
NODE0_IP="${NODE0_IP:-141.61.45.118}"
NODE1_IP="${NODE1_IP:-141.61.45.124}"
case "$NODE_RANK" in
  0) NODE_IP="$NODE0_IP" ;;
  1) NODE_IP="$NODE1_IP" ;;
  *) echo "Usage: bash $0 0|1 (0=$NODE0_IP, 1=$NODE1_IP)" >&2; exit 2 ;;
esac
MASTER_ADDR="${MASTER_ADDR:-$NODE0_IP}"
DIST_PORT="${DIST_PORT:-50000}"
NNODES=2

MODE="${MODE:-dspark}"                 # dspark / target-only
GRAPH="${GRAPH:-1}"                     # 0=eager; 1=NPU graph
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8030}"
TP_SIZE="${TP_SIZE:-16}"
ATTN_DP_SIZE="${ATTN_DP_SIZE:-4}"
OVERLAP="${OVERLAP:-1}"
DEEPEP_MODE="${DEEPEP_MODE:-auto}"
NETWORK_IFACE="${NETWORK_IFACE:-data0.3001}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-GLM-5.2-w8a8}"

# 双机初始预算。先验证启动和短请求，再按实际剩余 HBM 扩大长度与并发。
CONTEXT_LENGTH="${CONTEXT_LENGTH:-8192}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-32768}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-32}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-2048}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-8192}"
# 图档位按每组请求数配置；运行时会过滤不满足 Attention TP 对齐的档位。
GRAPH_BATCH_SIZES="${GRAPH_BATCH_SIZES:-8}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_ROOT="${SGLANG_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
TARGET_MODEL="${TARGET_MODEL:-/home/weights/GLM-5.2-w8a8/}"
DRAFT_MODEL="${DRAFT_MODEL:-/home/weights/GLM-5.2-DSpark-NPU-0805/}"
SPS_TABLE="${SPS_TABLE-/home/sps_npu_a5_tp16_adp4.json}"
SPS_PROFILE="${SPS_PROFILE:-0}"
RAGGED_VERIFY_MODE="${RAGGED_VERIFY_MODE:-compact}"
# 调试日志会增加同步开销，默认关闭；需要验证窗口变化时设为 1。
DSPARK_DEBUG="${DSPARK_DEBUG:-0}"
APPLY_QUAROT_TO_DRAFT="${APPLY_QUAROT_TO_DRAFT:-True}"
PRINT_COMMAND="${PRINT_COMMAND:-0}"
LOG_FILE="${LOG_FILE:-${SGLANG_ROOT}/server.a5.node${NODE_RANK}.${MODE}.${RAGGED_VERIFY_MODE}.tp${TP_SIZE}.adp${ATTN_DP_SIZE}.log}"

CANN_ENV="${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
ATB_ENV="${ATB_ENV:-/usr/local/Ascend/nnal/atb/set_env.sh}"
# 可选系统调优；默认不修改宿主机 sysctl。
SYSTEM_TUNE="${SYSTEM_TUNE:-0}"
# 已安装包含 sgl-kernel-npu #791 的版本时无需 overlay。
REGISTER_KERNEL_OVERLAY="${REGISTER_KERNEL_OVERLAY:-0}"
KERNEL_REPO="${KERNEL_REPO:-/workspace/sgl-kernel-npu}"
KERNEL_OVERLAY_SCRIPT="${KERNEL_OVERLAY_SCRIPT:-${SGLANG_ROOT}/devtools/glm52_ms1/register_glm52_dspark_kernel.py}"

fail() { echo "ERROR: $*" >&2; exit 2; }
[ "$#" -le 1 ] || fail "Pass only the node rank (0 or 1)."
case "$MODE" in dspark|target-only) ;; *) fail "Invalid MODE: $MODE" ;; esac
case "$DEEPEP_MODE" in auto|normal|low_latency) ;; *) fail 'Invalid DEEPEP_MODE.' ;; esac
for setting in GRAPH OVERLAP SPS_PROFILE DSPARK_DEBUG PRINT_COMMAND SYSTEM_TUNE REGISTER_KERNEL_OVERLAY; do
  case "${!setting}" in 0|1) ;; *) fail "$setting must be 0 or 1" ;; esac
done
case "$RAGGED_VERIFY_MODE" in compact|static|cap-accept) ;; *) fail 'Invalid RAGGED_VERIFY_MODE.' ;; esac
for setting in TP_SIZE ATTN_DP_SIZE PORT DIST_PORT CONTEXT_LENGTH MAX_TOTAL_TOKENS MAX_RUNNING_REQUESTS CHUNKED_PREFILL_SIZE MAX_PREFILL_TOKENS; do
  [[ "${!setting}" =~ ^[1-9][0-9]*$ ]] || fail "$setting must be a positive integer."
done
(( TP_SIZE % NNODES == 0 )) || fail 'TP_SIZE must be divisible by 2.'
(( PORT <= 65535 && DIST_PORT <= 65535 )) || fail 'Ports must be <= 65535.'
(( PORT != DIST_PORT )) || fail 'HTTP PORT and DIST_PORT must differ.'
(( TP_SIZE % ATTN_DP_SIZE == 0 )) || fail 'TP_SIZE must be divisible by ATTN_DP_SIZE.'
(( MAX_RUNNING_REQUESTS >= ATTN_DP_SIZE )) || fail 'MAX_RUNNING_REQUESTS must be at least ATTN_DP_SIZE.'
ATTN_TP_SIZE=$((TP_SIZE / ATTN_DP_SIZE))
LOCAL_TP_SIZE=$((TP_SIZE / NNODES))
if [ "$GRAPH" = 1 ] && [ "$DEEPEP_MODE" = normal ]; then
  fail 'DEEPEP_MODE=normal disables graphs in this baseline; use auto for GRAPH=1.'
fi
if [ "$SPS_PROFILE" = 1 ] && { [ "$MODE" != dspark ] || [ "$GRAPH" != 1 ]; }; then
  fail 'SPS_PROFILE=1 requires MODE=dspark and GRAPH=1.'
fi
read -r -a GRAPH_BS <<< "$GRAPH_BATCH_SIZES"
[ "${#GRAPH_BS[@]}" -gt 0 ] || fail 'GRAPH_BATCH_SIZES cannot be empty.'
for size in "${GRAPH_BS[@]}"; do
  [[ "$size" =~ ^[1-9][0-9]*$ ]] || fail 'GRAPH_BATCH_SIZES must contain positive integers.'
done

if [ "$MODE" = dspark ] && [ "$GRAPH" = 1 ] && [ "$DEEPEP_MODE" != normal ]; then
  LL_CAPACITY="${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-128}"
  [[ "$LL_CAPACITY" =~ ^[1-9][0-9]*$ ]] || fail 'DeepEP LL capacity must be a positive integer.'
  CAPTURE_GRID_VALID=0
  for size in "${GRAPH_BS[@]}"; do
    width=$((size * 9))
    if (( width % ATTN_TP_SIZE == 0 )); then
      CAPTURE_GRID_VALID=1
      (( width <= LL_CAPACITY )) || fail "Graph bs=$size needs $width verify tokens; increase DeepEP LL capacity above $LL_CAPACITY."
    fi
  done
  (( CAPTURE_GRID_VALID == 1 )) || fail 'No graph batch size satisfies Attention TP alignment.'
fi

SERVER_ARGS=(
  --model-path "$TARGET_MODEL"
  --attention-backend ascend
  --device npu
  --host "$HOST"
  --port "$PORT"
  --tp-size "$TP_SIZE"
  --attn-dp-size "$ATTN_DP_SIZE"
  --enable-dp-lm-head
  --nnodes "$NNODES"
  --node-rank "$NODE_RANK"
  --dist-init-addr "${MASTER_ADDR}:${DIST_PORT}"
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
  printf '# node_rank=%s node_ip=%s nnodes=%s TP=%s local_TP=%s ATTN_DP=%s ATTN_TP=%s GRAPH=%s OVERLAP=%s\n' \
    "$NODE_RANK" "$NODE_IP" "$NNODES" "$TP_SIZE" "$LOCAL_TP_SIZE" "$ATTN_DP_SIZE" "$ATTN_TP_SIZE" "$GRAPH" "$OVERLAP"
  printf '# rendezvous=%s:%s HCCL_SOCKET_IFNAME=%s GLOO_SOCKET_IFNAME=%s\n' \
    "$MASTER_ADDR" "$DIST_PORT" "$NETWORK_IFACE" "$NETWORK_IFACE"
  printf '# SGLANG_RAGGED_VERIFY_MODE=%s DSPARK_DEBUG=%s SPS_PROFILE=%s\n' \
    "$RAGGED_VERIFY_MODE" "$DSPARK_DEBUG" "$SPS_PROFILE"
  printf '# LOG_FILE=%q\n' "$LOG_FILE"
  printf '# DEEPEP_MODE=%s HCCL_BUFFSIZE=%s DEEPEP_HCCL_BUFFSIZE=%s\n' \
    "$DEEPEP_MODE" "${HCCL_BUFFSIZE:-1000}" "${DEEPEP_HCCL_BUFFSIZE:-inherit}"
  printf '# DeepEP: LL_capacity=%s normal_rounds=%s normal_tokens_per_round=%s\n' \
    "${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-128}" \
    "${DEEPEP_NORMAL_LONG_SEQ_ROUND:-72}" "${DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS:-1024}"
  printf '# DEEP_NORMAL_MODE_USE_INT8_QUANT=%s\n' "${DEEP_NORMAL_MODE_USE_INT8_QUANT:-0}"
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
      [ -r "$SPS_TABLE" ] || fail "SPS_TABLE not readable: $SPS_TABLE; use SPS_PROFILE=1 to collect on these A5 nodes."
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
# MXFP8 experts quantize BF16 dispatch output into block-scaled activations.
# The legacy INT8 fallback would instead produce incompatible per-token scales.
export DEEP_NORMAL_MODE_USE_INT8_QUANT="${DEEP_NORMAL_MODE_USE_INT8_QUANT:-0}"
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
printf 'Starting node=%s ip=%s: MODE=%s TP=%s local_TP=%s ATTN_DP=%s ATTN_TP=%s GRAPH=%s OVERLAP=%s SPS_PROFILE=%s LOG_FILE=%s\n' \
  "$NODE_RANK" "$NODE_IP" "$MODE" "$TP_SIZE" "$LOCAL_TP_SIZE" "$ATTN_DP_SIZE" "$ATTN_TP_SIZE" "$GRAPH" "$OVERLAP" "$SPS_PROFILE" "$LOG_FILE"
printf 'Rendezvous=%s:%s interface=%s\n' "$MASTER_ADDR" "$DIST_PORT" "$NETWORK_IFACE"
printf 'DEEP_NORMAL_MODE_USE_INT8_QUANT=%s\n' "$DEEP_NORMAL_MODE_USE_INT8_QUANT"
printf 'Command: python3 -m sglang.launch_server '
printf '%q ' "${SERVER_ARGS[@]}"
printf '\n'
printf 'Communication: DEEPEP_MODE=%s HCCL_BUFFSIZE=%s DEEPEP_HCCL_BUFFSIZE=%s\n' \
  "$DEEPEP_MODE" "$HCCL_BUFFSIZE" "${DEEPEP_HCCL_BUFFSIZE:-inherit}"
printf 'DeepEP: LL_capacity=%s normal_rounds=%s normal_tokens_per_round=%s\n' \
  "${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-128}" \
  "$DEEPEP_NORMAL_LONG_SEQ_ROUND" "$DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS"
# Keep bash as the parent of both Python and tee. SGLang cleans up Python's
# descendants in its finally block; exec would make tee a Python child and
# could kill the logger before an uncaught exception prints its traceback.
SERVER_STATUS=0
python3 -u -m sglang.launch_server "${SERVER_ARGS[@]}" || SERVER_STATUS=$?
printf 'Server exited: status=%s LOG_FILE=%s\n' "$SERVER_STATUS" "$LOG_FILE" >&2
exit "$SERVER_STATUS"
