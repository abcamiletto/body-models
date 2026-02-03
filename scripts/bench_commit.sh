#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/bench_commit.sh <commit> [options]

Runs benchmark at <commit> and its baseline (default: <commit>^) and prints
speedup values suitable for the README tables.

Options:
  -b, --baseline <commit>       Baseline commit (default: <commit>^)
  -m, --model <name>            Model name (default: smplx)
      --device <cpu|mps|cuda>   Single device (overrides default: mps)
      --devices <list>          Comma-separated devices (overrides default: mps)
      --dtype <float16|bfloat16|float32>
                                Single dtype (overrides default: bfloat16)
      --dtypes <list>           Comma-separated dtypes (overrides default: bfloat16)
      --batch-sizes <list>      Comma-separated batch sizes (default: 256)
      --min-run-time <sec>      Min benchmark time per op (default: 5)
      --baseline-mode <mode>    Mode in baseline JSON to compare (default: default)
      --compare-mode <mode>     Mode in target JSON to summarize (default: default)
      --compile                 Also run torch.compile (default mode; enabled by default)
      --compile-autotune        Also run torch.compile(mode='max-autotune')
      --compile-reduce-overhead Also run torch.compile(mode='reduce-overhead')
  -h, --help                    Show this help
EOF
}

commit=""
baseline=""
model="smplx"
batch_sizes="256"
min_run_time="5"
baseline_mode="default"
compare_mode="default"
compile_flags=("--compile")
device_list=()
dtype_list=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -b|--baseline) baseline="$2"; shift 2;;
    -m|--model) model="$2"; shift 2;;
    --device) device_list=("$2"); shift 2;;
    --devices) IFS=',' read -r -a device_list <<< "$2"; shift 2;;
    --dtype) dtype_list=("$2"); shift 2;;
    --dtypes) IFS=',' read -r -a dtype_list <<< "$2"; shift 2;;
    --batch-sizes) batch_sizes="$2"; shift 2;;
    --min-run-time) min_run_time="$2"; shift 2;;
    --baseline-mode) baseline_mode="$2"; shift 2;;
    --compare-mode) compare_mode="$2"; shift 2;;
    --compile) compile_flags+=("--compile"); shift;;
    --compile-autotune) compile_flags+=("--compile-autotune"); shift;;
    --compile-reduce-overhead) compile_flags+=("--compile-reduce-overhead"); shift;;
    -h|--help) usage; exit 0;;
    *)
      if [[ -z "$commit" ]]; then
        commit="$1"; shift
      else
        echo "Unknown argument: $1" >&2
        usage
        exit 1
      fi
      ;;
  esac
done

if [[ -z "$commit" ]]; then
  usage
  exit 1
fi

if [[ -z "$baseline" ]]; then
  baseline="${commit}^"
fi

root="$(git rev-parse --show-toplevel)"
git -C "$root" rev-parse --verify "${commit}^{commit}" >/dev/null
git -C "$root" rev-parse --verify "${baseline}^{commit}" >/dev/null

if (( ${#device_list[@]} == 0 )); then
  devices_csv="$(uv run scripts/bench_devices.py)"
  if [[ ",${devices_csv}," != *",mps,"* ]]; then
    echo "mps not available on this machine" >&2
    exit 1
  fi
  device_list=("mps")
fi

if (( ${#dtype_list[@]} == 0 )); then
  dtype_list=("bfloat16")
fi

tmp_root="$(mktemp -d "${TMPDIR:-/tmp}/body_models_bench.XXXXXX")"
baseline_dir="$tmp_root/baseline"
target_dir="$tmp_root/target"

cleanup() {
  git -C "$root" worktree remove -f "$baseline_dir" >/dev/null 2>&1 || true
  git -C "$root" worktree remove -f "$target_dir" >/dev/null 2>&1 || true
  rm -rf "$tmp_root"
}
trap cleanup EXIT

git -C "$root" worktree add -d "$baseline_dir" "$baseline" >/dev/null
git -C "$root" worktree add -d "$target_dir" "$commit" >/dev/null

extra_flags=()
if (( ${#compile_flags[@]} )); then
  extra_flags=("${compile_flags[@]}")
fi

cp "$root/scripts/benchmark.py" "$baseline_dir/scripts/benchmark.py"
cp "$root/scripts/benchmark.py" "$target_dir/scripts/benchmark.py"

echo "Baseline: $baseline"
echo "Target: $commit"

for device in "${device_list[@]}"; do
  for dtype in "${dtype_list[@]}"; do
    echo
    echo "=== ${device} / ${dtype} ==="
    baseline_json="$tmp_root/baseline_${device}_${dtype}.json"
    target_json="$tmp_root/target_${device}_${dtype}.json"

    (
      cd "$baseline_dir"
      args=(scripts/benchmark.py \
        --model "$model" \
        --device "$device" \
        --dtype "$dtype" \
        --batch-sizes "$batch_sizes" \
        --min-run-time "$min_run_time" \
        --json-out "$baseline_json")
      if (( ${#extra_flags[@]} )); then
        args+=("${extra_flags[@]}")
      fi
      uv run "${args[@]}"
    )

    (
      cd "$target_dir"
      args=(scripts/benchmark.py \
        --model "$model" \
        --device "$device" \
        --dtype "$dtype" \
        --batch-sizes "$batch_sizes" \
        --min-run-time "$min_run_time" \
        --json-out "$target_json" \
        --baseline-json "$baseline_json" \
        --baseline-mode "$baseline_mode")
      if (( ${#extra_flags[@]} )); then
        args+=("${extra_flags[@]}")
      fi
      uv run "${args[@]}"
    )

    uv run scripts/bench_summary.py \
      --baseline "$baseline_json" \
      --target "$target_json" \
      --baseline-mode "$baseline_mode" \
      --compare-mode "$compare_mode" \
      --batch-sizes "$batch_sizes"
  done
done
