from secap.model2 import MotionAudio
import torch
from lightning.pytorch import Trainer, LightningDataModule, LightningModule, Callback, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import lightning.pytorch as pl
import soundfile as sf
import torchaudio
import os,json
import numpy as np

file = ""
cuda_num = ""

model=MotionAudio()
state_dict = torch.load("../model.ckpt",map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model=model.to(torch.device(cuda_num))

if file.endswith(".wav"):
    wav, sr = sf.read(file)
    wav = np.array(wav).flatten()
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(torch.tensor(wav).unsqueeze(0).to(torch.float32)).squeeze(0).numpy()                
    wavform=[wav]
    output = model.inference(wavform)
    with open(f"secap_caption.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
