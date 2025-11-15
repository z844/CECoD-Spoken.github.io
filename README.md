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

---

### Step 2 : Environment Setup
1. git clone
    ```
    git clone https://github.com/z844/CECoD-Spoken.github.io.git
    cd CECoD-Spoken
    ```
2. Prepare Cosyvoice env
    ```
    git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
    cd CosyVoice
    git submodule update --init --recursive
    conda create -n cosyvoice -y python=3.10
    conda activate cosyvoice
    pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
    cd ..
    ```

3. Prepare SECap from [here](https://github.com/thuhcsi/SECap) (for Chinese emotion captioning).

    Create SECap conda environment

    ```
    conda env create -f ./requirements.yaml
    mv ./CECoD-Spoken/model2.py $your_SECap_dir
    ```




### Step 3 : Running
Please place the input audio file in the CECoD-Spoken/data/input folder.The output results are located in the CECoD-Spoken/data/output_audio folder.

    ```
    cd ..
    sh run_pipeline.sh
    ```
