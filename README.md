# Who needs the offer to borrow? On-device SME loan uplift scoring

Snapdragon® AI Lab Build & Present Challenge entry.

Banks either mass-offer promotional loan rates (margin lost on SMEs who would borrow anyway) or under-target
(missed loans). This scores each SME by **uplift**, `P(loan | offer) − P(loan | no offer)`, and recommends
one of three actions: offer the rate / no offer needed / do not extend. Scoring runs on the loan officer's
PC through ONNX Runtime with the Qualcomm QNN provider (Hexagon NPU), so customer financials never leave
the device.

> **Data:** trained on *synthetic* SME data with a planted causal effect. Results show the pipeline
> recovers that effect; they're not real-world lending performance.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

python train_and_export_uplift_model.py   # trains, exports + INT8-quantises, writes artifacts/
python deploy_on_snapdragon_npu.py        # scores example customers; NPU if available, else CPU (says which)
```

Open `demo.html` in a browser (no server needed) to try the model interactively - it runs the real trained
weights, not a mock-up.

### Real on-device numbers (Qualcomm AI Hub)

```bash
qai-hub configure --api_token <YOUR_TOKEN>   # stays on your machine, never commit it
python profile_on_ai_hub.py                  # compile + profile + on-device check on a hosted Snapdragon device
```

Writes `aihub_results.json` / `aihub_results.js` (already populated here from a real run on a Snapdragon X
Elite CRD - `demo.html`'s telemetry panel reads these). On a Snapdragon X PC, install
`requirements-snapdragon.txt` instead of plain `onnxruntime` to actually use the NPU locally.

## Files

| File | Purpose |
|---|---|
| `train_and_export_uplift_model.py` | Synthetic data, two-model uplift estimator, ONNX export, static INT8 (QDQ) quantisation, self-checks |
| `deploy_on_snapdragon_npu.py` | Scoring + latency benchmark; QNN/NPU with honest CPU fallback |
| `profile_on_ai_hub.py` | Compile/profile/validate on a hosted Snapdragon device via Qualcomm AI Hub |
| `demo.html` | Interactive demo running the real trained weights in the browser |
| `artifacts/` | `uplift_fp32.onnx` (reference), `uplift_int8.onnx` (static batch-1, what actually gets deployed), `model_weights.js`, sample inputs |
| `aihub_results.json` / `.js` | Real profiling results from Qualcomm AI Hub |

## Model and quantisation notes

- The ONNX graph is built by hand from the trained weights (not via skl2onnx) so it only contains ops the
  Hexagon backend supports: Sub, Mul, Gemm, Relu, Sigmoid.
- Input normalisation stays a float step outside quantisation - the raw features span very different
  ranges (revenue vs. months banked), and quantising them directly hurts accuracy.
- The deployed INT8 model has a fixed batch of 1, since that's what the NPU backend and AI Hub want.
  Customers get scored one at a time, which is also how a loan officer would actually use this.
- INT8 vs FP32: ~92% agreement on the top-decile customers, mean uplift difference ≈ 0.4 percentage points.

## How this would ship

Package the scoring code and the ~10 KB ONNX model as a lightweight Windows-on-ARM desktop app, or expose
the model to an existing loan-origination tool as a local library call. Retrain centrally on the bank's
randomised promotional-rate history and push only the updated ONNX file to devices.

## Limitations

Synthetic data only. Thresholds (`UPLIFT_OFFER = 0.04`, `BASELINE_REGARDLESS = 0.30`) are illustrative and
should be set from real margin economics. Needs an explainability and fairness review before any live use.
