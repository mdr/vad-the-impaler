#!/usr/bin/env python3
from pathlib import Path
from random import Random
import subprocess
import json
import shutil

random = Random(42)

output_dir = Path('audio.2k')
speech_dir = (output_dir / 'speech')
not_speech_dir = (output_dir / 'not-speech')
output_dir.mkdir(exist_ok=True)
speech_dir.mkdir(exist_ok=True)
not_speech_dir.mkdir(exist_ok=True)

NUMBER_OF_SPEECH_FILES = 1000
NUMBER_OF_NOT_SPEECH_FILES = 1000

# Copy and resample "speech" files
all_speech_paths = Path('vox1_test_wav').rglob('*.wav')
selected_speech_paths = random.sample(list(all_speech_paths), NUMBER_OF_SPEECH_FILES)
for speech_path in selected_speech_paths:
    output_file = speech_dir / "-".join(speech_path.parts[-3:])
    if not output_file.exists():
        subprocess.run(['ffmpeg', '-i', speech_path, '-ar', '44100', output_file]).check_returncode()

def load_json(path):
    with open(path) as f:
        return json.load(f)

# Copy "not-speech" files
samples = load_json('fsd50k/FSD50K.ground_truth/eval.json')
def source_file(name):
    return Path(f"fsd50k/FSD50K.eval_audio/{name}.wav")
all_not_speech_files = [ source_file(sample['fname']) for sample in samples if 'Human_voice' not in sample['labels'].split(',')]
selected_not_speech_files = random.sample(all_not_speech_files, NUMBER_OF_NOT_SPEECH_FILES)
for file in selected_not_speech_files:
    output_file = not_speech_dir / file.name
    if not output_file.exists():
        shutil.copyfile(file, output_file)
