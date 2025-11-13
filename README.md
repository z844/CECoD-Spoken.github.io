# CECoD-Spoken

Our code and data are being continuously updated. You can view the demo page through the following link:

[Demo Page](https://z844.github.io/CECoD-Spoken.github.io/)

## Running

### Step 1 : Download Required Models

Download the following models into:

```
CECoD-Spoken/model/
```

| Component | Model Name                | Source                                                                                                                                                                                                                                   |
| --------- | ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| LLM       | Llama3-8B-Chinese-Chat    | [https://huggingface.co/shenzhi-wang/Llama3-8B-Chinese-Chat](https://huggingface.co/shenzhi-wang/Llama3-8B-Chinese-Chat)                                                                                                                 |
| LLM LoRA  | CECoD-Spoken LoRA Weights | [https://z844.github.io/CECoD-Spoken.github.io/](https://z844.github.io/CECoD-Spoken.github.io/)                                                                                                                                         |
| TTS       | CosyVoice2-0.5B           | [https://www.modelscope.cn/studios/iic/CosyVoice2-0.5B](https://www.modelscope.cn/studios/iic/CosyVoice2-0.5B)                                                                                                                           |
| ASR       | Paraformer-zh             | [https://www.modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/](https://www.modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/) |

---

### Step 2 : Environment Setup

1. Prepare SECap from [here](https://github.com/thuhcsi/SECap) (for Chinese emotion captioning).

2. Create conda environment
    ```
    conda env create -f ./requirements.yaml
    mv ./CECoD-Spoken/model2.py $your_SECap_dir
    ```

3. Prepare Cosyvoice env
    ```
    conda create -n cosyvoice -y python=3.10
    conda activate cosyvoice
    pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
    ```


### Step 3 : Running
Please place the input audio file in the CECoD-Spoken/data/input folder.The output results are located in the CECoD-Spoken/data/output_audio folder.

    ```
    cd CECoD-Spoken
    sh run_pipeline.sh
    ```
