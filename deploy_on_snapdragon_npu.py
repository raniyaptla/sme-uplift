"""
Score SME customers for promotional-rate uplift on-device.

On a Snapdragon X PC (Windows on ARM) with `onnxruntime-qnn` installed, this
runs the INT8 model on the Hexagon NPU via the QNN execution provider.
Anywhere else it falls back to CPU and SAYS SO - it never claims NPU
execution unless the QNN provider actually loaded.

Usage:
    python deploy_on_snapdragon_npu.py                 # auto: NPU if available, else CPU
    python deploy_on_snapdragon_npu.py --provider cpu  # force CPU
    python deploy_on_snapdragon_npu.py --runs 500      # more benchmark iterations
"""
import argparse
import platform
import statistics
import time

import numpy as np
import onnxruntime as ort

MODEL_INT8 = "artifacts/uplift_int8.onnx"
MODEL_FP32 = "artifacts/uplift_fp32.onnx"

# Decision rules (see README). Tuned so the split on the synthetic portfolio is
# roughly 25% offer / 16% borrows-anyway / 59% do-not-extend.
UPLIFT_OFFER = 0.04       # >= 4 percentage-point lift  -> worth a discount
BASELINE_REGARDLESS = 0.30  # >= 30% chance of borrowing with no offer

# Feature order must match FEATURES in train_and_export_uplift_model.py:
# business_age_years, monthly_revenue_lakh, credit_utilization_pct,
# past_repayment_score, industry_risk_tier (0/1/2), existing_relationship_months
EXAMPLE_CUSTOMERS = {
    "Young mid-size retailer, new to bank":   [3.0, 10.0, 50.0, 75.0, 1, 6.0],
    "Established high-revenue manufacturer":  [18.0, 120.0, 40.0, 88.0, 0, 180.0],
    "High-risk, weak repayment history":      [2.0, 4.0, 85.0, 35.0, 2, 8.0],
    "Mid-size services firm, loyal 3 yrs":    [8.0, 12.0, 45.0, 80.0, 1, 36.0],
    "Small trader, moderate profile":         [5.0, 3.0, 60.0, 65.0, 1, 24.0],
}


def build_session(model_path, provider_choice):
    """Return (session, provider_actually_used)."""
    available = ort.get_available_providers()
    want_npu = provider_choice in ("auto", "npu")

    if want_npu and "QNNExecutionProvider" in available:
        try:
            sess = ort.InferenceSession(
                model_path,
                providers=[
                    ("QNNExecutionProvider", {
                        "backend_path": "QnnHtp.dll",      # Hexagon NPU backend
                        "htp_performance_mode": "burst",
                    }),
                    "CPUExecutionProvider",
                ],
            )
            active = sess.get_providers()[0]
            return sess, active
        except Exception as e:  # driver/SDK problem -> fall back honestly
            print(f"[warn] QNN provider failed to initialise ({e}); using CPU.")

    if provider_choice == "npu":
        raise SystemExit("--provider npu requested but QNNExecutionProvider is not available. "
                         "Install onnxruntime-qnn on a Snapdragon X PC.")
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return sess, sess.get_providers()[0]


def recommend(uplift, baseline):
    if uplift >= UPLIFT_OFFER:
        return "OFFER promotional rate (1.5% reduction)"
    if baseline >= BASELINE_REGARDLESS:
        return "NO OFFER NEEDED - likely to borrow regardless"
    return "DO NOT EXTEND - offer unlikely to convert"


def score(sess, X):
    """Score customers. The INT8 model has a fixed batch of 1 (required by the NPU
    backend and AI Hub), so customers are scored one at a time - which is also how a
    loan officer would use it."""
    X = X.astype(np.float32)
    if sess.get_inputs()[0].shape[0] == 1:
        outs = [sess.run(["uplift", "baseline_prob"], {"features": X[i:i + 1]}) for i in range(len(X))]
        return (np.array([o[0][0, 0] for o in outs]), np.array([o[1][0, 0] for o in outs]))
    uplift, baseline = sess.run(["uplift", "baseline_prob"], {"features": X})
    return uplift.ravel(), baseline.ravel()


def benchmark(sess, runs, warmup=20):
    x = np.array([EXAMPLE_CUSTOMERS["Young mid-size retailer, new to bank"]], dtype=np.float32)
    for _ in range(warmup):
        sess.run(None, {"features": x})
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {"features": x})
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return {
        "median_ms": statistics.median(times),
        "p95_ms": times[int(0.95 * len(times)) - 1],
        "min_ms": times[0],
        "runs": runs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["auto", "npu", "cpu"], default="auto")
    ap.add_argument("--runs", type=int, default=200)
    args = ap.parse_args()

    print(f"Machine : {platform.machine()} | {platform.system()} | onnxruntime {ort.__version__}")
    print(f"Available providers: {ort.get_available_providers()}")

    sess, active = build_session(MODEL_INT8, args.provider)
    on_npu = active == "QNNExecutionProvider"
    print(f"Running INT8 model on: {active}  ->  {'NPU (Hexagon)' if on_npu else 'CPU fallback'}\n")

    names = list(EXAMPLE_CUSTOMERS)
    X = np.array([EXAMPLE_CUSTOMERS[n] for n in names], dtype=np.float32)
    uplift, baseline = score(sess, X)

    print(f"{'Customer':42s} {'Uplift':>8s} {'Baseline':>9s}  Recommendation")
    print("-" * 110)
    for n, u, b in zip(names, uplift, baseline):
        print(f"{n:42s} {u*100:+7.1f}% {b*100:8.1f}%  {recommend(u, b)}")

    # Agreement between FP32 reference and the INT8 model actually deployed
    ref, _ = build_session(MODEL_FP32, "cpu")
    ref_up, _ = score(ref, X)
    print(f"\nFP32 vs INT8 max abs uplift difference on these customers: "
          f"{np.abs(ref_up - uplift).max()*100:.2f} percentage points")

    stats = benchmark(sess, args.runs)
    print(f"\nLatency, single customer, {stats['runs']} runs on {active}:")
    print(f"  median {stats['median_ms']:.3f} ms | p95 {stats['p95_ms']:.3f} ms | min {stats['min_ms']:.3f} ms")
    if not on_npu:
        print("  (CPU numbers - NOT representative of Snapdragon NPU. "
              "Use the AI Hub profile job for real device figures.)")


if __name__ == "__main__":
    main()
