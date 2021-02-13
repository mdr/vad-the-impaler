#!/usr/bin/env python3
from pathlib import Path
from random import Random
import subprocess
import os

random = Random(42)

output_dir = Path('data')
output_dir.mkdir(exist_ok=True)
(output_dir / 'speech').mkdir(exist_ok=True)
(output_dir / 'not-speech').mkdir(exist_ok=True)

all_speech_paths = Path('vox1_test_wav').rglob('*.wav')
selected_speech_paths = random.sample(list(all_speech_paths), 3)
for speech_path in selected_speech_paths:
    print("-".join(speech_path.parts[-3:]))
    output_file = (output_dir / 'speech' / speech_path.name)
    if not output_file.exists():
        subprocess.run(['ffmpeg', '-i', speech_path, '-ar', '44100', output_file]).check_returncode()
