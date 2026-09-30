"""
SME Loan/Credit Offer Uplift - train, export to ONNX, quantize to INT8.

Question the model answers for every SME customer:
    "How much does a promotional interest rate change the probability
     that this business takes the loan?"      uplift = P(loan | offer) - P(loan | no offer)

Method: two-model ("T-learner") uplift estimator
    - control model : trained on customers who were NOT offered the promo rate
    - treated model : trained on customers who WERE offered the promo rate
Both models are exported into ONE ONNX graph that outputs the uplift directly,
so on-device scoring is a single inference call.

Outputs (in ./artifacts):
    uplift_fp32.onnx   full-precision reference graph, dynamic batch (outputs: uplift, baseline_prob)
    uplift_int8.onnx   static batch-1, statically quantized (QDQ INT8) graph - the NPU/AI Hub one
    model_weights.js   trained weights for demo.html (the browser demo runs the real model)
    sample_inputs.npy  a few raw feature rows for the deploy script / AI Hub
"""
import json
import os
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_static, CalibrationDataReader, QuantType, QuantFormat,
)

SEED = 7
OUT_DIR = "artifacts"
FEATURES = [
    "business_age_years",            # 0.5 - 30
    "monthly_revenue_lakh",          # INR lakh per month
    "credit_utilization_pct",        # 0 - 100
    "past_repayment_score",          # 0 - 100
    "industry_risk_tier",            # 0 low, 1 medium, 2 high
    "existing_relationship_months",  # months banked with this lender
]
N_FEATURES = len(FEATURES)


# Synthetic SME lending data with a known causal structure baked in, so we can
# check the uplift model actually recovers it instead of just eyeballing AUC.
def generate_data(n=40000, seed=SEED):
    rng = np.random.default_rng(seed)
    age = np.clip(rng.gamma(2.2, 3.0, n), 0.5, 30)
    revenue = np.clip(rng.lognormal(mean=2.3, sigma=0.8, size=n), 0.5, 200)
    util = np.clip(rng.normal(55, 22, n), 0, 100)
    repay = np.clip(rng.normal(68, 16, n), 0, 100)
    risk = rng.choice([0, 1, 2], size=n, p=[0.35, 0.45, 0.20])
    rel = np.clip(rng.gamma(2.0, 24.0, n), 1, 240)

    # Baseline propensity to borrow WITHOUT any offer
    # Loyal, high-revenue, well-performing businesses tend to borrow anyway.
    logit0 = (
        -1.6
        + 0.9 * (np.log(revenue) - 2.3)
        + 0.012 * (rel - 48)
        + 0.010 * (repay - 68)
        + 0.010 * (util - 55)
        - 0.25 * (risk - 1)
    )

    # TRUE causal effect of the promo rate on the logit (heterogeneous):
    #  - biggest for mid-revenue, medium-risk, short-relationship businesses
    #  - near zero for already-loyal high-revenue businesses (would borrow anyway)
    #  - near zero / slightly negative for very risky, low-repayment businesses
    mid_rev = np.exp(-((np.log(revenue) - 2.3) ** 2) / (2 * 0.55 ** 2))
    short_rel = np.exp(-rel / 36.0)
    med_risk = np.where(risk == 1, 1.0, np.where(risk == 0, 0.6, 0.25))
    good_repay = np.clip((repay - 40) / 60, 0, 1)
    tau = 1.9 * mid_rev * (0.35 + 0.65 * short_rel) * med_risk * (0.4 + 0.6 * good_repay) - 0.15

    # Randomised treatment (as in an A/B test of promo offers)
    treat = rng.binomial(1, 0.5, n)

    p = 1 / (1 + np.exp(-(logit0 + treat * tau)))
    y = rng.binomial(1, p)

    X = np.column_stack([age, revenue, util, repay, risk, rel]).astype(np.float32)
    p0 = 1 / (1 + np.exp(-logit0))
    p1 = 1 / (1 + np.exp(-(logit0 + tau)))
    true_uplift = (p1 - p0).astype(np.float32)
    return X, treat, y, true_uplift


# Two-model (T-learner) uplift estimator: separate control/treated classifiers.
def make_model():
    return Pipeline([
        ("scale", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu",
                              max_iter=400, early_stopping=True,
                              n_iter_no_change=15, random_state=SEED)),
    ])


def train_two_models(X, treat, y):
    m_ctrl = make_model().fit(X[treat == 0], y[treat == 0])
    m_treat = make_model().fit(X[treat == 1], y[treat == 1])
    return m_ctrl, m_treat


def sklearn_uplift(m_ctrl, m_treat, X):
    return m_treat.predict_proba(X)[:, 1] - m_ctrl.predict_proba(X)[:, 1]


# ONNX export - build the graph by hand from the trained weights rather than going
# through skl2onnx. skl2onnx pulls in ai.onnx.ml ops (Scaler, ArrayFeatureExtractor,
# ZipMap...) that the QNN/Hexagon backend doesn't like, plus a label-output branch we
# don't need. Doing it manually keeps the graph to Sub/Mul/Gemm/Relu/Sigmoid only.
# Two outputs: uplift = P(loan|offer) - P(loan|no offer), baseline_prob = P(loan|no offer).
def folded_layers(pipe):
    """[(W, b), ...] with the scaler absorbed into the first layer (used for the browser demo)."""
    mean, scale = pipe.named_steps["scale"].mean_, pipe.named_steps["scale"].scale_
    mlp = pipe.named_steps["mlp"]
    Ws = [w.astype(np.float64) for w in mlp.coefs_]
    bs = [b.astype(np.float64) for b in mlp.intercepts_]
    Ws[0], bs[0] = Ws[0] / scale[:, None], bs[0] - (mean / scale) @ mlp.coefs_[0]
    return [(w.astype(np.float32), b.astype(np.float32)) for w, b in zip(Ws, bs)]


def build_uplift_graph(m_ctrl, m_treat, path, batch=None):
    """batch=None -> dynamic batch (reference / CPU). batch=1 -> static shape
    (required by Qualcomm AI Hub and the QNN NPU backend)."""
    nodes, inits = [], []

    def tower(pipe, tag):
        # normalisation stays a separate FLOAT step (excluded from quantisation): raw features
        # span very different ranges, so quantising them directly would lose precision.
        sc = pipe.named_steps["scale"]
        inits.append(helper.make_tensor(f"{tag}_mean", TensorProto.FLOAT, [N_FEATURES], sc.mean_.astype(np.float32).tolist()))
        inits.append(helper.make_tensor(f"{tag}_inv_scale", TensorProto.FLOAT, [N_FEATURES], (1.0 / sc.scale_).astype(np.float32).tolist()))
        nodes.append(helper.make_node("Sub", ["features", f"{tag}_mean"], [f"{tag}_centered"], name=f"{tag}_norm_sub"))
        nodes.append(helper.make_node("Mul", [f"{tag}_centered", f"{tag}_inv_scale"], [f"{tag}_normed"], name=f"{tag}_norm_mul"))
        cur = f"{tag}_normed"
        mlp = pipe.named_steps["mlp"]
        layers = [(w.astype(np.float32), b.astype(np.float32)) for w, b in zip(mlp.coefs_, mlp.intercepts_)]
        for i, (W, b) in enumerate(layers):
            inits.append(helper.make_tensor(f"{tag}_W{i}", TensorProto.FLOAT, W.shape, W.flatten().tolist()))
            inits.append(helper.make_tensor(f"{tag}_b{i}", TensorProto.FLOAT, b.shape, b.tolist()))
            out = f"{tag}_z{i}"
            nodes.append(helper.make_node("Gemm", [cur, f"{tag}_W{i}", f"{tag}_b{i}"], [out], name=f"{tag}_gemm{i}"))
            cur = out
            last = i == len(layers) - 1
            act = f"{tag}_a{i}"
            nodes.append(helper.make_node("Sigmoid" if last else "Relu", [cur], [act], name=f"{tag}_act{i}"))
            cur = act
        return cur

    p0 = tower(m_ctrl, "ctrl")
    p1 = tower(m_treat, "treat")
    nodes.append(helper.make_node("Sub", [p1, p0], ["uplift"], name="uplift_sub"))
    nodes.append(helper.make_node("Identity", [p0], ["baseline_prob"], name="baseline_out"))

    B = batch if batch is not None else "N"
    graph = helper.make_graph(
        nodes, "sme_loan_uplift",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [B, N_FEATURES])],
        [helper.make_tensor_value_info("uplift", TensorProto.FLOAT, [B, 1]),
         helper.make_tensor_value_info("baseline_prob", TensorProto.FLOAT, [B, 1])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path


def export_demo_weights(m_ctrl, m_treat, path):
    """Write the trained weights as a JS file so demo.html runs the REAL model in the browser."""
    r = lambda a: np.round(a, 5).tolist()
    payload = {
        "features": FEATURES,
        "control": [{"W": r(W), "b": r(b)} for W, b in folded_layers(m_ctrl)],
        "treated": [{"W": r(W), "b": r(b)} for W, b in folded_layers(m_treat)],
    }
    with open(path, "w") as f:
        f.write("// Auto-generated by train_and_export_uplift_model.py - do not edit\n")
        f.write("const MODEL = " + json.dumps(payload, separators=(",", ":")) + ";\n")
    return payload


def numpy_forward(layers, X):
    h = X.astype(np.float64)
    for i, l in enumerate(layers):
        h = h @ np.array(l["W"]) + np.array(l["b"])
        h = 1 / (1 + np.exp(-h)) if i == len(layers) - 1 else np.maximum(h, 0)
    return h.ravel()


# Static QDQ INT8 quantization - dynamic quantization doesn't run on the Hexagon
# NPU, it just falls back to CPU, so static is the only one worth doing here.
class CalibReader(CalibrationDataReader):
    def __init__(self, X, n_rows=500):
        self.batches = [X[i:i + 1] for i in range(n_rows)]
        self.it = iter(self.batches)

    def get_next(self):
        b = next(self.it, None)
        return None if b is None else {"features": b.astype(np.float32)}


NORM_NODES = ["ctrl_norm_sub", "ctrl_norm_mul", "treat_norm_sub", "treat_norm_mul"]


def quantize_int8(fp32_path, int8_path, X_calib):
    quantize_static(
        fp32_path, int8_path, CalibReader(X_calib, n_rows=2000),
        nodes_to_exclude=NORM_NODES,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
    )
    return int8_path


# --------------------------------------------------------------------------
def run_onnx(path, X):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    if sess.get_inputs()[0].shape[0] == 1:   # static batch-1 model: score row by row
        return np.array([sess.run(["uplift"], {"features": X[i:i + 1].astype(np.float32)})[0][0, 0]
                         for i in range(len(X))])
    return sess.run(["uplift"], {"features": X.astype(np.float32)})[0].ravel()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    X, treat, y, true_uplift = generate_data()
    Xtr, Xte, ttr, tte, ytr, yte, utr, ute = train_test_split(
        X, treat, y, true_uplift, test_size=0.25, random_state=SEED)

    print(f"Rows: {len(X):,}  | promo offered: {treat.mean():.1%}  | loan taken: {y.mean():.1%}")
    m_ctrl, m_treat = train_two_models(Xtr, ttr, ytr)

    # sanity: each arm's model is a decent loan-taken predictor
    for name, m, arm in [("control", m_ctrl, 0), ("treated", m_treat, 1)]:
        mask = tte == arm
        print(f"  {name:8s} AUC on held-out: {roc_auc_score(yte[mask], m.predict_proba(Xte[mask])[:, 1]):.3f}")

    up_sk = sklearn_uplift(m_ctrl, m_treat, Xte)
    corr = np.corrcoef(up_sk, ute)[0, 1]
    print(f"Estimated vs TRUE uplift correlation (synthetic ground truth): {corr:.3f}")

    # top-decile targeting check: does ranking by predicted uplift find real responders?
    order = np.argsort(-up_sk)
    top = order[: len(order) // 10]
    print(f"Mean TRUE uplift: top decile by model = {ute[top].mean():.3f} | everyone = {ute.mean():.3f}")

    fp32 = os.path.join(OUT_DIR, "uplift_fp32.onnx")          # dynamic batch, reference
    fp32_b1 = os.path.join(OUT_DIR, "uplift_fp32_b1.onnx")    # static batch=1
    int8 = os.path.join(OUT_DIR, "uplift_int8.onnx")          # static batch=1, QDQ INT8 (deploy this)
    build_uplift_graph(m_ctrl, m_treat, fp32)
    build_uplift_graph(m_ctrl, m_treat, fp32_b1, batch=1)

    # --- verify sklearn == ONNX FP32 ------------------------------------
    up_fp32 = run_onnx(fp32, Xte)
    print(f"\nsklearn vs ONNX fp32  max abs diff: {np.abs(up_sk - up_fp32).max():.2e}")

    # --- browser-demo weights + self-check of the exported numbers ------
    payload = export_demo_weights(m_ctrl, m_treat, os.path.join(OUT_DIR, "model_weights.js"))
    up_js = numpy_forward(payload["treated"], Xte) - numpy_forward(payload["control"], Xte)
    print(f"demo weights (rounded) vs sklearn max abs diff: {np.abs(up_sk - up_js).max():.2e}")

    # --- quantize (from the static batch-1 graph) + verify ---------------
    quantize_int8(fp32_b1, int8, Xtr)
    n_eval = 5000
    up_i8 = run_onnx(int8, Xte[:n_eval])
    up_ref = up_fp32[:n_eval]
    print(f"fp32 vs INT8          max abs diff: {np.abs(up_ref - up_i8).max():.4f}  "
          f"mean abs diff: {np.abs(up_ref - up_i8).mean():.4f}   (first {n_eval} test rows)")
    print(f"INT8 vs TRUE uplift correlation: {np.corrcoef(up_i8, ute[:n_eval])[0, 1]:.3f}")
    k = n_eval // 10
    top_ref = set(np.argsort(-up_ref)[:k].tolist()); top_i8 = set(np.argsort(-up_i8)[:k].tolist())
    print(f"Top-decile customer overlap fp32 vs INT8: {len(top_ref & top_i8) / k:.1%}")
    print(f"Mean TRUE uplift of INT8 top decile: {ute[:n_eval][list(top_i8)].mean():.3f}")

    np.save(os.path.join(OUT_DIR, "sample_inputs.npy"), Xte[:8])
    os.remove(fp32_b1)
    print(f"\nSizes: fp32 {os.path.getsize(fp32)/1024:.1f} KB | int8 {os.path.getsize(int8)/1024:.1f} KB")
    print("Saved: uplift_fp32.onnx, uplift_int8.onnx, model_weights.js, sample_inputs.npy in ./" + OUT_DIR)


if __name__ == "__main__":
    main()
