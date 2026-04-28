# Testing & Verification Guide

## Quick Start: Verify Everything Works

### Test 1: Synthetic Test (No Camera Needed) — 5 mins
```powershell
$env:PYTHONPATH = "$pwd"
python scripts/test_edge_inference.py --frames 60
```

**Expected output:**
- ✅ Model loads
- ✅ Compliance engine runs 4 test scenarios (all pass)
- ✅ Zero network connections
- ✅ Detection latency ~100-300ms (depends on CPU/GPU)

**What it tests:**
- Worker/PPE model loading
- Compliance rule engine
- Temporal smoothing
- Edge-only (offline) operation

---

### Test 2: Live Webcam Test — 10 mins

**Terminal 1 — Local Model:**
```powershell
cd C:\GitHub\Yolov8-Worker-Safety-System
$env:PYTHONPATH = "$pwd"
python scripts/run_live_inference.py --camera 0 --worker-model models/yolo/ppe_detector.pt --ppe-model models/yolo/ppe_detector.pt --fp16 --input-size 480 --output results/local_live/
```

**Expected behavior:**
- Live video opens with overlay
- Bounding boxes around workers/PPE
- Compliance status: ✓ (green) or ✗ (red)
- Press Q to quit after 30 seconds

**Check results:**
```powershell
cat results/local_live/session_report_*.json | findstr /C:"hard_hat" /C:"hi_vis_vest" /C:"safety_glasses"
```

Should show detection counts > 0 for each class.

---

### Test 3: Validation Report

After running Test 2, examine the output JSON:

```powershell
python -c "
import json
import glob

report = json.load(open(glob.glob('results/local_live/session_report_*.json')[0]))
print('Frames processed:', report.get('frames_processed'))
print('Avg latency:', report.get('latency', {}).get('mean_ms'), 'ms')
print('FPS:', report.get('fps'))
print('Detections:')
for cls, count in report.get('detections', {}).items():
    print(f'  {cls}: {count}')
"
```

**Expected values:**
| Metric | CPU | GPU |
|--------|-----|-----|
| Latency | 200-400ms | 40-60ms |
| FPS | 3-5 | 15-25 |
| Detections | hard_hat, vest, glasses detected |

---

## Detailed Testing Scenarios

### Scenario A: Test PPE Detection Accuracy
**Objective:** Verify model correctly identifies PPE items

**What to do:**
1. Wear different combinations of PPE in front of camera
   - ✓ With hard hat + vest + glasses
   - ✓ Without hat (should flag violation)
   - ✓ Without vest
   - ✓ Mixed compliance

2. Watch overlay:
   - Green bounding boxes = detected
   - Compliance status should change: ✓ → ✗

3. Check report:
   - `hard_hat`, `hi_vis_vest`, `safety_glasses` counts increase
   - When missing, `missing_equipment` list grows

---

### Scenario B: Test Multi-Worker Detection
**Objective:** Verify system handles multiple people

**What to do:**
1. Get 2+ people in frame
2. Each should get separate detection
3. Report should show:
   ```json
   {
     "worker_0": { "is_compliant": true, ... },
     "worker_1": { "is_compliant": false, "missing": ["hard_hat"], ... }
   }
   ```

---

### Scenario C: Test Temporal Smoothing
**Objective:** Verify false alerts don't trigger on single frame

**What to do:**
1. Run with temporal smoothing:
   ```powershell
   python scripts/run_live_inference.py --camera 0 --worker-model models/yolo/ppe_detector.pt --ppe-model models/yolo/ppe_detector.pt --temporal-smoothing 5 --output results/smooth_test/
   ```

2. Briefly remove hard hat (1-2 frames)
3. Compliance status should NOT change
4. Alert should NOT fire (checked via `--temporal-smoothing 5`)

Without smoothing:
```powershell
python scripts/run_live_inference.py --camera 0 --worker-model models/yolo/ppe_detector.pt --ppe-model models/yolo/ppe_detector.pt --temporal-smoothing 0 --output results/no_smooth_test/
```

Brief removal WILL trigger alert.

---

## Verification Checklist

- [ ] **Phase 2A Model**: `test_edge_inference.py` passes all 4 compliance scenarios
- [ ] **Local Inference**: Live camera shows detections (hard_hat, vest, glasses visible)
- [ ] **Latency**: <50ms on GPU or <150ms on CPU
- [ ] **FPS**: >15 on GPU or >5 on CPU
- [ ] **Multi-worker**: 2+ people detected correctly
- [ ] **Compliance verdicts**: ✓ when all PPE present, ✗ when missing
- [ ] **Temporal smoothing**: Single-frame noise doesn't trigger false alerts
- [ ] **Zero network calls**: (local model only, no Roboflow)
- [ ] **JSON reports**: Contain all expected metrics

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Camera not detected | Try `--camera 1` or `--camera 2` |
| No detections | Verify model file: `models/yolo/ppe_detector.pt` (should be ~52 MB) |
| Slow inference | Use `--fp16 --input-size 320` for speed vs accuracy trade-off |
| Import errors | Set `$env:PYTHONPATH = "$pwd"` in PowerShell |
| No JSON report | Check `results/local_live/` folder exists |

---

## Success Criteria

✅ **System is working when:**
1. `test_edge_inference.py` completes without errors
2. Live inference detects PPE classes (hard_hat, vest, glasses, boots)
3. Compliance verdicts change based on equipment presence
4. Latency is acceptable for your hardware (GPU: <50ms, CPU: <150ms)
5. FPS is acceptable (GPU: >15, CPU: >5)
6. JSON reports contain all expected fields

Once all ✅ pass, the computer vision pipeline is validated and ready for:
- Phase 2C: Fine-tuning on real data
- Phase 3: Edge deployment (ONNX → QNN)
- Roboflow integration (once project has trained models)

