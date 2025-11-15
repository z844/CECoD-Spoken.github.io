import argparse
import os
import json
import torch
import soundfile as sf
import torchaudio
import numpy as np
from SECap.model2 import MotionAudio
from glob import glob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda",
        type=int,
        default=-1,
        help="CUDA device index (use -1 for CPU, default). Example: --cuda 0",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ---------------------
    if args.cuda >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.cuda}")
    else:
        device = torch.device("cpu")

    folder = "data/input"   
    wav_files = glob(os.path.join(folder, "*.wav"))
    if not wav_files:
        raise FileNotFoundError("文件夹中没有 wav 文件")
    latest_wav = max(wav_files, key=os.path.getmtime)
    file = os.path.abspath(latest_wav)

    model=MotionAudio()
    state_dict = torch.load("SECap/model.ckpt",map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model=model.to(torch.device(device))
    

    if file.endswith(".wav"):
        wav, sr = sf.read(file)
        wav = np.array(wav).flatten()
        if sr != 16000:
            wav = torchaudio.transforms.Resample(sr, 16000)(torch.tensor(wav).unsqueeze(0).to(torch.float32)).squeeze(0).numpy()                
        wavform=[wav]
        output = model.inference(wavform)
        with open(f"secap_caption.json", "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    
    main()
