# Retriever for HotpotQA / 2WikiMultiHopQA

The agent's `web_search` tool queries a local dense retriever (e5-base-v2 + FAISS flat index over the
2018 Wikipedia dump). The server comes from [Search-R1](https://github.com/PeterGriffinJin/Search-R1) via
[agent-distillation](https://github.com/Nardien/agent-distillation) and runs in its own environment.

```bash
conda create -n retriever python=3.10 -y && conda activate retriever
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini uvicorn fastapi
conda install -c pytorch -c nvidia faiss-gpu=1.8.0

# index (61GB) + corpus (14GB) from the Search-R1 HF datasets; the download parts take another 62GB until removed
save_path=retriever/database/wikipedia
mkdir -p $save_path
python retriever/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz && rm $save_path/part_*

# serve on 127.0.0.1:8005 (run from the repository root; uses two GPUs for the index)
CUDA_VISIBLE_DEVICES=0,1 nohup python -u retriever/retriever_server.py > retriever.log 2>&1 &

# check (in the ProD environment)
python retriever/test.py
```

The agent connects to `http://127.0.0.1:8005/retrieve`; pass `--retriever_port` to `prod.rollout` if you
serve it on another port.
