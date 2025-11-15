import torch
import os
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from acoustic_feature import get_mean_pitch_label, get_energy_label, get_speed_label
from gender.gender_inf import predict_gender
import argparse
from glob import glob
from funasr import AutoModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda_num",
        "-c",
        type=str,
        default="",
        help="Device / device_map for LLM (e.g. 'cuda:0', 'auto', or '').",
    )
    parser.add_argument(
        "--model_id",
        "-m",
        type=str,
        required=True,
        help="HuggingFace model id for the base LLM.",
    )
    parser.add_argument(
        "--lora_weight",
        "-l",
        type=str,
        required=True,
        help="Path or id of LoRA weights.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cuda_num = args.cuda_num
    model_id = args.model_id
    lora_weight = args.lora_weight

    folder = "data/input"   
    wav_files = glob(os.path.join(folder, "*.wav"))
    if not wav_files:
        raise FileNotFoundError("文件夹中没有 wav 文件")
    latest_wav = max(wav_files, key=os.path.getmtime)
    wav_file = os.path.abspath(latest_wav)

    secap_path = "data/output_secap_caption/secap_caption.json"
    prompt_path = "prompt.json"
    out_dir = "data/output_audio/"

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=cuda_num,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, lora_weight)

    with open(secap_path, "r", encoding="utf-8") as f:
        secap_caption = json.load(f)

    def whisper(wav_file):
        model = AutoModel(
            model="paraformer-zh",
            model_revision="v2.0.4",
            vad_model="fsmn-vad",
            vad_model_revision="v2.0.4",
            punc_model="ct-punc-c",
            punc_model_revision="v2.0.4",
            # spk_model="cam++", spk_model_revision="v2.0.2",
        )
        res = model.generate(
            input=wav_file,
            batch_size_s=300,
            hotword="魔搭",
        )
        return res[0]["text"]

    def Acoustic_Feature(wav_file, text):
        af_list = []
        af_list.append(
            [
                predict_gender(wav_file),
                get_mean_pitch_label(wav_file),
                get_energy_label(wav_file),
                get_speed_label(wav_file, text),
            ]
        )
        return af_list

    def load_prompts(path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def llm_generate(system_prompt, user_content, head):
        input = (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>"
            f"{system_prompt}"
            "<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>"
            f"{user_content}"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>"
            f"{head}"
        )
        model_inputs = tokenizer([input], return_tensors="pt").to("cuda:1")
        outputs = model.generate(
            **model_inputs,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.8,
        )
        output = tokenizer.batch_decode(outputs)[0]
        return output

    current_text = whisper(wav_file)
    history = []
    caption_history = []
    prompts = load_prompts(prompt_path)
    history.append(current_text)

    # ------------------------------------------------------------------------------------
    for json_num in range(len(history)):
        text = history[json_num]
        choose_caption_prompt = prompts["choose_caption_prompt"]
        sum_cap = ""
        for each_cap in secap_caption:
            sum_cap += "[" + each_cap + "]"
        F_caption = llm_generate(
            choose_caption_prompt,
            f"当前对话:[{text}]可选情感描述:{sum_cap}",
            "",
        )

    # ------------------------------------------------------------------------------------
    acoustic_feature = Acoustic_Feature(wav_file, text)[0]
    grain_caption_prompt = prompts["grain_caption_prompt"]
    grain_user_prompt = (
        "声学特征:"
        + acoustic_feature[0]
        + ","
        + acoustic_feature[1]
        + ","
        + acoustic_feature[2]
        + ","
        + acoustic_feature[3]
        + "\n情感描述:"
        + F_caption
    )
    current_caption = llm_generate(grain_caption_prompt, grain_user_prompt, "")
    caption_history.append(current_caption)

    # ------------------------------------------------------------------------------------

    clue_prompt = prompts["clue_prompt"]
    respones_prompt = prompts["respones_prompt"]
    caption_prompt = prompts["caption_prompt"]
    label_prompt = prompts["label_prompt"]

    conversations = "对话内容:\n"
    s = ""
    for i in range(len(history)):
        s += history[i] + "<caption>" + caption_history[i] + "</caption>\n"

    clues = llm_generate(clue_prompt, conversations, "情感线索:")

    response = llm_generate(respones_prompt, conversations + clues, "")

    generate_caption = llm_generate(
        caption_prompt, conversations + clues + "\n" + response, "<caption>"
    )
    if generate_caption.endswith("</caption>"):
        generate_caption = generate_caption[: -len("</caption>")].strip()

    label_response = llm_generate(
        label_prompt, conversations + clues + "\n" + response + generate_caption, ""
    )

    history.append(response)
    caption_history.append(generate_caption)

    # ------------------------------------------------------------------------------------

    import sys

    sys.path.append("CosyVoice/third_party/Matcha-TTS")

    from CosyVoice.cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
    from CosyVoice.cosyvoice.utils.file_utils import load_wav
    import torchaudio

    cosyvoice = CosyVoice2(
        "pretrained_models/CosyVoice2-0.5B",
        load_jit=False,
        load_trt=False,
        load_vllm=False,
        fp16=False,
    )

    def TTS_generate(label_response, generate_caption):
        prompt_speech_16k = load_wav("./asset/zero_shot_prompt.wav", 16000)
        # instruct usage
        for i, j in enumerate(
            cosyvoice.inference_instruct2(
                label_response, generate_caption, prompt_speech_16k, stream=False
            )
        ):
            out_path = os.path.join(out_dir, f"zero_shot_{i}.wav")
            print("保存到：", out_path)
            torchaudio.save(out_path, j["tts_speech"], cosyvoice.sample_rate)


    TTS_generate(label_response, generate_caption)


if __name__ == "__main__":
    main()

