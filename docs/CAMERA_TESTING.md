# Live Camera Testing Guide

**Real-time PPE compliance monitoring from webcam or network streams**

This guide walks you through setting up and running live camera inference with real-time compliance overlay.

---

## Quick Start (< 5 minutes)

### Step 1: Set up camera

**Webcam (built-in or USB):**
```bash
# Verify camera is available
ls /dev/video*  # Linux
# Should show /dev/video0, /dev/video1, etc.
```

**RTSP Network Camera:**
```bash
# Test connectivity
ffplay rtsp://192.168.1.100:554/stream  # Replace with your IP
# Press 'q' to quit
```

### Step 2: Run live inference

**Minimal command (Roboflow backend, no optimization):**
```bash
ROBOFLOW_API_KEY=rf-YOUR_KEY python scripts/run_live_inference.py --camera 0
```

**Optimized command (local models, FP16 + smaller input):**
```bash
python scripts/run_live_inference.py \
    --camera 0 \
    --worker-model models/worker.pt \
    --ppe-model models/ppe.pt \
    --fp16 \
    --input-size 480
```

**RTSP Stream (IP camera):**
```bash
python scripts/run_live_inference.py \
    --camera "rtsp://192.168.1.100:554/stream" \
    --use-roboflow \
    --output results/
```

### Step 3: Interact with the window

While the live inference window is open:
- **Q** or **ESC** — quit
- **S** — save snapshot of current frame
- **R** — reset statistics counter

---

## Camera Sources

### Webcam (Local USB/Built-in)

**Linux/Mac:**
```bash
# Find available devices
ls -la /dev/video*

# Run inference on device 0 (first camera)
python scripts/run_live_inference.py --camera 0

# Run on device 1 (second camera, if available)
python scripts/run_live_inference.py --camera 1
```

**Windows:**
```bash
# Device indices are numeric (same as Linux)
python scripts/run_live_inference.py --camera 0
```

### RTSP Network Cameras (IP Cameras)

**Basic RTSP URL:**
```bash
python scripts/run_live_inference.py \
    --camera "rtsp://192.168.1.100:554/stream"
```

**RTSP with Authentication:**
```bash
python scripts/run_live_inference.py \
    --camera "rtsp://username:password@192.168.1.100:554/stream"
```

**Common IP Camera Formats:**

| Vendor | URL Format | Example |
|--------|-----------|---------|
| Hikvision | `rtsp://IP:554/Streaming/Channels/101` | rtsp://192.168.1.10:554/Streaming/Channels/101 |
| Axis | `rtsp://IP/axis-media/mp4/default` | rtsp://192.168.1.20/axis-media/mp4/default |
| Dahua | `rtsp://IP:554/rtsp/` | rtsp://192.168.1.30:554/rtsp/ |
| Generic | `rtsp://IP:554/stream` | rtsp://192.168.1.100:554/stream |

### HTTP/MJPEG Streams

```bash
# Some cameras use HTTP instead of RTSP
python scripts/run_live_inference.py \
    --camera "http://192.168.1.100:8080/stream.mjpg"
```

---

## Configuration Options

### Frame Rate Control

**Default:** 30 FPS target (will skip frames if camera is faster)

```bash
# Reduce FPS for lower latency
python scripts/run_live_inference.py --camera 0 --fps 15

# Increase for smoother video (if GPU can handle it)
python scripts/run_live_inference.py --camera 0 --fps 30
```

**Typical values:**
- `--fps 10` — Lightweight, high latency
- `--fps 15` — Balanced (recommended for live monitoring)
- `--fps 30` — Smooth (requires powerful GPU)

### Performance Optimization

**Slow camera (need speed):**
```bash
python scripts/run_live_inference.py \
    --camera 0 \
    --fp16 \
    --input-size 480 \
    --temporal-smoothing 5
# Expected: ~25 FPS
```

**Fast/powerful GPU (can trade quality):**
```bash
python scripts/run_live_inference.py \
    --camera 0 \
    --fp16 \
    --input-size 320 \
    --fps 30
# Expected: ~50 FPS
```

See [OPTIMIZATION.md](OPTIMIZATION.md) for detailed tuning guide.

### Output & Logging

**Save video + session report:**
```bash
python scripts/run_live_inference.py \
    --camera 0 \
    --output results/
```

**Outputs created:**
- `results/annotated_TIMESTAMP.mp4` — Full video with overlay
- `results/session_report_TIMESTAMP.json` — Compliance statistics
- `results/snapshot_TIMESTAMP_XXXXXX.jpg` — Snapshots (when S pressed)

**Headless mode (no display window):**
```bash
python scripts/run_live_inference.py \
    --camera "rtsp://..." \
    --no-display \
    --output results/
# Useful for server deployment
```

---

## Live Display Interpretation

### Compliance Overlay

**Green boxes + "COMPLIANT":**
- Worker is wearing all required PPE
- No alert generated

**Red boxes + "NON-COMPLIANT":**
- Worker is missing one or more required items
- Alert severity depends on what's missing
- Example: "Missing hard_hat, safety_glasses"

**Orange boxes + "UNCERTAIN":**
- Detection confidence below threshold
- Could be occlusion, poor lighting, or genuine detection failure

### Bottom-Right FPS Counter

```
FPS 24.5  frame 156
```

- **FPS:** Current frame rate (aim for 15+ in live mode)
- **frame:** Total frames processed since start

### Worker ID Labels

```
W0 COMPLIANT  85%
W1 NON-COMPLIANT
```

- **W0, W1, ...** — Worker identifier (assigned by frame order)
- **% score** — Compliance percentage (how many required items detected)

---

## Common Scenarios

### Scenario 1: Monitor a Single Camera Feed

```bash
python scripts/run_live_inference.py \
    --camera "rtsp://192.168.1.100:554/stream" \
    --use-roboflow \
    --temporal-smoothing 5 \
    --output warehouse_monitoring/
```

**What happens:**
1. Opens RTSP stream
2. Displays live worker + PPE detections
3. Overlay shows compliance status (green/red)
4. Saves session report + video to `warehouse_monitoring/`

### Scenario 2: Webcam + Local Models (Optimized)

```bash
python scripts/run_live_inference.py \
    --camera 0 \
    --worker-model models/worker.pt \
    --ppe-model models/ppe.pt \
    --fp16 \
    --input-size 480 \
    --fps 15
```

**What happens:**
1. Opens local webcam (device 0)
2. Runs inference at target 15 FPS
3. Uses FP16 + 480px for speed (~25 FPS achieved)
4. Displays live with compliance overlay
5. No output file saved (use `--output` to record)

### Scenario 3: Multi-Camera Setup

For multiple cameras, run separate instances in different terminals:

```bash
# Terminal 1 — Camera 0 (main warehouse)
python scripts/run_live_inference.py \
    --camera 0 \
    --output results/camera_0/

# Terminal 2 — Camera 1 (safety area)
python scripts/run_live_inference.py \
    --camera 1 \
    --output results/camera_1/

# Terminal 3 — RTSP stream (maintenance area)
python scripts/run_live_inference.py \
    --camera "rtsp://192.168.1.50:554/stream" \
    --output results/camera_rtsp/
```

---

## Output Files Explained

### `annotated_TIMESTAMP.mp4`

Full video with:
- Detection bounding boxes (red for PPE, blue for workers)
- Compliance status overlay
- FPS/frame counter
- Worker ID and compliance score

**View:**
```bash
ffplay results/annotated_20260418_143022.mp4
mpv results/annotated_*.mp4
```

### `session_report_TIMESTAMP.json`

Structured compliance report:
```json
{
  "session": {
    "timestamp": "20260418_143022",
    "camera_source": "0",
    "duration_s": 120.5,
    "frames_processed": 1805,
    "backend": "local_yolov8"
  },
  "compliance": {
    "avg_compliance_rate": 0.87,
    "total_alerts": 23
  },
  "performance": {
    "total_ms": {
      "mean": 40.2,
      "median": 38.5,
      "max": 105.3,
      "min": 25.1
    },
    "fps": 24.9
  }
}
```

### `snapshot_TIMESTAMP_XXXXXX.jpg`

Individual frames saved when you press **S** during monitoring:
```
snapshot_20260418_143022_000042.jpg   (frame 42)
snapshot_20260418_143022_000103.jpg   (frame 103)
```

Use these for compliance documentation or incident investigation.

---

## Troubleshooting

### Issue: Camera Not Detected

**Error:** `Cannot open camera source: 0`

**Solutions:**
1. Check camera exists:
   ```bash
   ls /dev/video*  # Linux
   # Should show /dev/video0 or /dev/video1
   ```

2. Try different device index:
   ```bash
   python scripts/run_live_inference.py --camera 1
   python scripts/run_live_inference.py --camera 2
   ```

3. Check permissions (Linux):
   ```bash
   sudo usermod -a -G video $USER
   # Log out and back in
   ```

4. Test with alternative tool:
   ```bash
   ffplay /dev/video0  # Linux
   # or mpv av://dshow?video="Camera Name"  # Windows
   ```

### Issue: High Latency (>500ms)

**Causes & Solutions:**

| Symptom | Cause | Fix |
|---------|-------|-----|
| Occasional spikes | Thermal throttling | Reduce `--input-size` to 320 |
| Consistent 100+ms | CPU-bound | Enable `--fp16` (GPU) |
| Video stuttering | Network (RTSP) | Reduce `--fps` to 10 |
| Dropped frames | Memory pressure | Close other apps, use `--input-size 320` |

**Optimize:**
```bash
python scripts/run_live_inference.py \
    --camera 0 \
    --fp16 \
    --input-size 320 \
    --fps 10
# Should achieve <30ms latency
```

### Issue: Memory Leak (increasing RAM over time)

**Diagnosis:**
```bash
# Monitor memory while running
watch -n 1 'ps aux | grep run_live_inference'
```

**Causes & Fixes:**
1. **Long session (hours)** → Normal; restart periodically
2. **Rapidly growing** → Memory leak bug
   - Try: `--temporal-smoothing 0` (disable rolling window)
   - If persists: Report with session duration + GPU model

### Issue: RTSP Connection Fails

**Error:** `Cannot open camera source: rtsp://...`

**Solutions:**

1. **Verify RTSP URL:**
   ```bash
   ffplay rtsp://192.168.1.100:554/stream
   # If this works, URL is correct
   ```

2. **Check authentication:**
   ```bash
   # Try with credentials in URL
   python scripts/run_live_inference.py \
       --camera "rtsp://user:pass@192.168.1.100:554/stream"
   ```

3. **Firewall/network:**
   ```bash
   # Test connectivity
   nc -zv 192.168.1.100 554
   # Should show "Connection successful"
   ```

4. **Try HTTP instead:**
   ```bash
   python scripts/run_live_inference.py \
       --camera "http://192.168.1.100:8080/stream.mjpg"
   ```

### Issue: Poor Detection Quality

**Symptoms:** Workers/PPE not detected, false positives

**Causes & Fixes:**

| Issue | Cause | Fix |
|-------|-------|-----|
| "No detections" | Low light | Increase `--temporal-smoothing` |
| Flickering boxes | High FPS noise | Use `--temporal-smoothing 5` |
| False PPE detections | Small input size | Try `--input-size 640` (slower) |
| Misses hard hats | Model accuracy | Use `--use-roboflow` (production model) |

**Quality check:**
```bash
python scripts/run_live_inference.py \
    --camera 0 \
    --use-roboflow \  # Production accuracy
    --temporal-smoothing 5 \
    --output debug/
# Check output/frame_results.json for detections per frame
```

---

## Performance Targets

| Metric | Target | Typical |
|--------|--------|---------|
| **Latency** | <100ms | 40–80ms |
| **FPS** | 15+ | 15–30 |
| **Memory** | <1GB | 400–800MB |
| **CPU** | <50% | 20–40% |
| **GPU** | 60–80% | 70–85% |

If you're not hitting these targets, check [OPTIMIZATION.md](OPTIMIZATION.md) for tuning guidance.

---

## Production Deployment (Headless)

For 24/7 monitoring without display:

```bash
# Run in background with log
python scripts/run_live_inference.py \
    --camera "rtsp://192.168.1.50:554/stream" \
    --use-roboflow \
    --fps 10 \
    --no-display \
    --output /var/log/ppe_monitoring/ \
    > /var/log/ppe_monitoring/debug.log 2>&1 &

# Monitor log
tail -f /var/log/ppe_monitoring/debug.log

# Find PID and kill if needed
ps aux | grep run_live_inference
kill <PID>
```

**Recommended settings for 24/7:**
```bash
--camera "rtsp://..."      # Network camera (reliable)
--use-roboflow             # Cloud models (no local GPU wear)
--fps 5                    # Reduce load
--temporal-smoothing 10    # Smooth alerts
--no-display               # Save resources
--output /var/log/ppe_monitoring/
```

---

## References

- **Main README:** [README.md](../README.md)
- **Optimization Guide:** [OPTIMIZATION.md](OPTIMIZATION.md)
- **Testing Plan:** [TESTING.md](../TESTING.md)
- **Architecture:** [CLAUDE.md](../CLAUDE.md)

---

**Last Updated:** April 2026  
**Maintainer:** A-Kuo
