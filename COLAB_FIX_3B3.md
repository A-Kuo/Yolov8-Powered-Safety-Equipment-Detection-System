# Fix for Colab Cell [3B.3] — ModuleNotFoundError

**Issue:** Cell [3B.3] tries to import `src.edge_deployment.roboflow_inference` but it fails with `ModuleNotFoundError` in Colab.

**Root Cause:** The notebook is trying to use a standalone module when `InferencePipelineWithMetrics` already has Roboflow support built-in.

---

## SOLUTION: Replace Cell [3B.3] with This Code

Delete the existing cell [3B.3] and paste this instead:

```python
import os
import sys
import json
import cv2
import numpy as np

# ── Set VIDEO_PATH to your test video ────────────────────────────────────────
VIDEO_PATH = "/content/drive/MyDrive/test_video.mp4"   # ← change this
OUTPUT_DIR = "/content/workspace/roboflow_results"
MAX_FRAMES = 60   # ~2 seconds at 30 fps for a quick sanity check

print(f"[3B.3] Running Roboflow PPE pipeline on: {VIDEO_PATH}")
print(f"       Max frames: {MAX_FRAMES}  |  Output: {OUTPUT_DIR}")

# Make sure path is set correctly
sys.path.insert(0, '/content/workspace/project')

try:
    # Use the existing InferencePipelineWithMetrics which has Roboflow support
    from src.edge_deployment.inference_pipeline import InferencePipelineWithMetrics
    from src.edge_deployment.video_processor import VideoProcessor
    import os
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("[3B.3] Initializing Roboflow backend...")
    
    # Initialize with Roboflow cloud backend
    pipeline = InferencePipelineWithMetrics(
        worker_model_path="yolov8n.pt",  # Not used with Roboflow, but required
        ppe_model_path="yolov8n.pt",     # Not used with Roboflow, but required
        device='cuda',
        use_roboflow=True,  # ← Use Roboflow cloud backend
        fp16=False,         # Cloud doesn't benefit from FP16
    )
    
    print("[3B.3] PASS: Roboflow pipeline initialized")
    
    # Process video
    print("[3B.3] Processing video frames...")
    
    video = VideoProcessor(VIDEO_PATH, target_fps=5)  # 5 fps to stay within API limits
    all_detections = []
    frame_count = 0
    
    for frame, frame_idx, metadata in video:
        if frame_count >= MAX_FRAMES:
            break
        
        # Run inference
        detections = pipeline.process_frame(frame)
        all_detections.append(detections)
        
        # Count for reporting
        total_items = sum(len(items) for items in detections.values())
        print(f"  Frame {frame_count:3d}: {len(detections)} workers, {total_items} PPE items")
        
        frame_count += 1
    
    video.close()
    
    print(f"\n[3B.3] PASS: {frame_count} frames processed")
    
    # Summary statistics
    total_workers = sum(len(frame) for frame in all_detections)
    total_detections = sum(
        len(items) 
        for frame in all_detections 
        for items in frame.values()
    )
    
    print(f"       Workers detected: {total_workers}")
    print(f"       PPE items detected: {total_detections}")
    
    # Save results
    serializable = []
    for frame in all_detections:
        frame_result = {
            str(wid): [
                {
                    "class": item.class_name,
                    "confidence": round(item.confidence, 3),
                    "bbox": [round(x, 1) for x in item.bbox]
                }
                for item in items
            ]
            for wid, items in frame.items()
        }
        serializable.append(frame_result)
    
    with open(f"{OUTPUT_DIR}/detections.json", "w") as f:
        json.dump(serializable, f, indent=2)
    
    # Get metrics from pipeline
    metrics = pipeline.get_metrics()
    if metrics:
        fps = metrics.get('fps', 0)
        latency = metrics.get('total_ms', {}).get('mean', 0)
        print(f"\n       Mean latency: {latency:.0f} ms/frame")
        print(f"       Effective FPS: {fps:.1f}")
    
    print(f"\n[3B.3] Results saved to {OUTPUT_DIR}/detections.json")
    print("[3B] ✓ Roboflow backend fully operational — skip [4]-[10] training steps if using cloud only")

except ModuleNotFoundError as e:
    print(f"[3B.3] FAIL: {e}")
    print("       → Make sure Python path includes project directory")
    print("       → sys.path should contain: /content/workspace/project")
    sys.path.insert(0, '/content/workspace/project')
    print("       → Path updated. Try running the cell again.")
    raise

except FileNotFoundError as e:
    print(f"[3B.3] INFO: {e}")
    print("       Upload a test video to Google Drive and update VIDEO_PATH above")
    
except Exception as exc:
    print(f"[3B.3] FAIL: {exc}")
    import traceback
    traceback.print_exc()
    raise
```

---

## Why This Fix Works

| What's Different | Why It Matters |
|---|---|
| Uses `InferencePipelineWithMetrics(use_roboflow=True)` | This class already has full Roboflow support |
| No import of `RoboflowInference` | That module doesn't need to exist separately |
| Uses existing `VideoProcessor` | Same as we use locally, proven to work |
| Returns standard `DetectionResult` objects | Compatible with compliance engine |

---

## After Applying the Fix

1. **Delete cell [3B.3]** (the one with the error)
2. **Paste the code above** in its place
3. **Run the cell** — should complete without errors
4. Check output: `Results saved to /content/workspace/roboflow_results/detections.json`

---

## Expected Output

```
[3B.3] Running Roboflow PPE pipeline on: /content/drive/MyDrive/test_video.mp4
       Max frames: 60  |  Output: /content/workspace/roboflow_results
[3B.3] Initializing Roboflow backend...
[3B.3] PASS: Roboflow pipeline initialized
[3B.3] Processing video frames...
  Frame   0: 2 workers, 8 PPE items
  Frame   1: 2 workers, 8 PPE items
  ...
[3B.3] PASS: 60 frames processed
       Workers detected: 120
       PPE items detected: 480

[3B.3] Results saved to /content/workspace/roboflow_results/detections.json
[3B] ✓ Roboflow backend fully operational
```

---

## If Still Getting Errors

1. **Check Python path:**
   ```python
   import sys
   print(sys.path)
   # Should include: '/content/workspace/project'
   ```

2. **Verify imports work:**
   ```python
   from src.edge_deployment.inference_pipeline import InferencePipelineWithMetrics
   print("✓ Import successful")
   ```

3. **Check ROBOFLOW_API_KEY is set:**
   ```python
   import os
   print(f"API Key: {os.environ.get('ROBOFLOW_API_KEY', 'NOT SET')}")
   ```

