# samples

Audio is not committed (size + third-party rights). The transcripts and RTTMs
produced by livediar are, so you can see what the pipeline outputs.

Fetch the clips used in the README measurements (needs `yt-dlp` and the
`imageio-ffmpeg` binary, both in `requirements.txt`):

```bash
FF=$(python -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
fetch() { yt-dlp -q -f "bestaudio[ext=m4a]/bestaudio" -o "samples/$2.%(ext)s" "https://www.youtube.com/watch?v=$1" \
          && "$FF" -y -loglevel error -i samples/$2.m4a -ac 1 -ar 16000 -sample_fmt s16 samples/$2.wav && rm samples/$2.m4a; }
fetch 85BlwjC4Jwg obama_trump_cspan          # C-SPAN Oval Office pool feed, 2016-11-10 (public-affairs pool footage)
fetch 9TzZNdqSohU fob_mandarin_codeswitch    # sitcom scene, English/Mandarin code-switching
fetch _J2EkW4bkz0 interview_mandarin         # Mandarin job-interview skit, two speakers
```

`synth_overlap.wav` (used for the overlap measurements) is stitched from the
C-SPAN clip: Obama solo 0–12 s, Trump over Obama 12–18 s, Trump solo 18–30 s,
Obama over Trump 30–35 s, Obama solo 35–45 s. Rebuild it with:

```bash
python - <<'PY'
import numpy as np, wave
w=wave.open("samples/obama_trump_cspan.wav"); pcm=np.frombuffer(w.readframes(w.getnframes()),np.int16).astype(np.float32)/32768
seg=lambda a,b: pcm[int(a*16000):int(b*16000)]; rms=lambda x: float(np.sqrt(np.mean(x**2))+1e-9)
O2,T1=seg(32,38),seg(125,131); T3,O3=seg(143,148),seg(38,43)
out=np.clip(np.concatenate([seg(20,32), O2+T1*(0.8*rms(O2)/rms(T1)), seg(131,143), T3+O3*(0.8*rms(T3)/rms(O3)), seg(43,53)]),-1,1)
with wave.open("samples/synth_overlap.wav","wb") as f:
    f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000); f.writeframes((out*32767).astype(np.int16).tobytes())
PY
```
