# Face AI Studio

A local web demo of [InsightFace](https://github.com/deepinsight/insightface) 2.1 features, with bundled public-domain test media.

Unofficial project; not affiliated with or endorsed by the InsightFace / deepinsight team.

| Tab | InsightFace feature |
| --- | --- |
| Face analysis | SCRFD detection, 5 keypoints, 106 / 3D-68 landmarks, head pose, age, gender, 512-d ArcFace embedding (`buffalo_l` or compact `raccoon_s`) |
| Verify 1:1 | Cosine similarity of aligned 112x112 embeddings with adjustable threshold |
| Identify 1:N | Gallery search; enroll your own people from uploads (saved to `photos/people/`) |
| Liveness | RGB liveness addon, with an optional simulated screen-replay attack |
| Album clustering | Unsupervised grouping of every face in the sample album |
| Video privacy | PrivateFrame: blur everyone, blur only selected people, or keep selected people visible (Gaussian or mosaic) |
| Person tracking | PersonAnalysis (`cheetah_s`): face plus body ReID matching, annotated video and appearance timeline |

## Run

```bash
./run.sh            # creates .venv on first run, then serves http://127.0.0.1:7860
```

Requires Python 3.10+ and `ffmpeg` on `PATH` (used to make browser-playable MP4s). The first start downloads the model packs
(~400 MB) to `~/.insightface/models`. On Apple Silicon ONNX Runtime uses CoreML with CPU fallback; elsewhere it uses CPU
(install `onnxruntime-gpu` for CUDA).

## Media

- `photos/people/<name>/`: four NASA portraits each of the Artemis II crew, used as the identification gallery.
- `photos/groups/`: group photos for detection, identification and clustering.
- `video/`: two short Artemis II clips for PrivateFrame and PersonAnalysis.

All photos and videos are public domain (NASA) from Wikimedia Commons; sources are in `photos/CREDITS.json`.
`fetch_photos.py` re-downloads the photos.

## Notes

- InsightFace pretrained models are licensed for non-commercial research use only.
- Thresholds are adjustable because similarity drops with large pose, age and lighting differences.
- The screen-replay simulation is a synthetic approximation, not a real presentation attack.
- People enrolled in the app are saved to `photos/people/<name>/` and reloaded on start; removing one moves the folder to `work/deleted/`.
- Uploads and outputs go to `work/`.
