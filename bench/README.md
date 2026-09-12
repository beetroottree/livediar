# bench

`ami_bench.py` scores livediar on the AMI Meeting Corpus test set (pyannote
split, 16 meetings, ~9 h) with ground truth: DER against the pyannote
references and speaker-attributed WER (cpWER; idWER with enrolled voices)
against the AMI word-level transcripts. See the docstring for stages and
`ami/results/RESULTS.md` for numbers.

Fetch the data (CC BY 4.0; ~1.1 GB audio + 23 MB annotations):

```bash
cd bench/ami
git clone --depth 1 https://github.com/pyannote/AMI-diarization-setup.git setup
curl -sL -o ami_manual.zip https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip && unzip -q ami_manual.zip -d manual
mkdir -p wav && for m in $(ls setup/only_words/rttms/test | sed 's/.rttm//'); do
  curl -sL -o wav/$m.wav "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/$m/audio/$m.Mix-Headset.wav"; done
cd ../..
sh bench/run_diarize.sh 5      # Sortformer once per meeting (parallel), cached
sh bench/run_eval.sh           # none / words / enrolled caption conditions
python bench/ami_bench.py report
```
