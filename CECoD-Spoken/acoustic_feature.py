import librosa
import numpy as np
import pyworld as pw

def get_mean_pitch_label(wav_path: str) -> str:
    x, fs = librosa.load(wav_path, sr=None)
    x = x.astype(np.float64)
    _f0, t = pw.dio(x, fs)
    f0 = pw.stonemask(x, _f0, t, fs)
    data = f0
    mask = data != 0
    segments = np.split(data, np.where(np.diff(mask))[0] + 1)
    non_zero_num = 0
    value_sum = 0.0
    for seg in segments:
        if len(seg) == 0 or seg[0] == 0:
            continue
        non_zero_num += len(seg)
        value_sum += seg.sum()
    if non_zero_num == 0:
        return "音高适中"
    mean_f0 = value_sum / non_zero_num
    if mean_f0 < 136.6:
        return "音高较低"
    elif mean_f0 > 196.1:
        return "音高较强"
    else:
        return "音高适中"

def get_energy_label(wav_path: str) -> str:
    y, sr = librosa.load(wav_path, sr=None)
    energy = librosa.feature.rms(y=y).mean()
    if energy < 0.0333:
        return "音频能量较低"
    elif energy > 0.0505:
        return "音频能量较强"
    else:
        return "音频能量适中"

def get_speed_label(wav_path: str, text: str) -> str:
    text_len = max(len(text), 1)
    y, sr = librosa.load(wav_path, sr=None)
    duration = len(y) / sr 
    tempo = duration / text_len
    if tempo < 0.252:
        return "音速快"
    elif tempo > 0.38645446:
        return "音速慢"
    else:
        return "音速适中"
