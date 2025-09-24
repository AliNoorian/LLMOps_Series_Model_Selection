---
jupyter:
  colab:
    name: LLMOps_Series_Part1_Model_Selection.ipynb
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  nbformat: 4
  nbformat_minor: 5
---

::: {#18ae5f31 .cell .markdown}
# LLMOps Series --- Part 1: Model Selection (Colab/Jupyter Notebook) {#llmops-series--part-1-model-selection-colabjupyter-notebook}

> **Use this notebook to choose, test, and cost out LLMs for your use
> case.**\
> It includes setup cells, side‑by‑side comparisons (proprietary vs
> open‑source), latency/throughput tests, context‑window experiments,
> quantization notes, and lightweight benchmarking utilities.

**Contents**

1.  [Environment Check & Setup](#env)
2.  [Your Use Case Checklist](#checklist)
3.  [Proprietary vs Open-Source: Decision Guide](#decision)
4.  [Open-Source: Try a Small Model (Transformers)](#transformers)
5.  [Open-Source: Try a GGUF Model (llama.cpp /
    llama-cpp-python)](#gguf)
6.  [Latency & Throughput Testing](#perf)
7.  [Context Window Experiments](#ctx)
8.  [Prompt Engineering vs Fine‑Tuning (Overview + Demo)](#tuning)
9.  [Cost Estimation --- API & Self-Hosting Calculators](#cost)
10. [Minimal RAG Harness (Optional)](#rag)
11. [Production Inference (vLLM/TGI) --- Optional Installs](#prod)
12. [Quick Benchmarking Utilities](#bench)
13. [References & Next Steps](#refs)

------------------------------------------------------------------------

**Two Main Model Types**\
**Proprietary** (e.g., GPT-5, Claude, Gemini) → *Plug-and-play*,
top-tier performance, pay‑per‑use, limited data control.\
**Open-Source** (e.g., LLaMA, Mistral, Falcon, Zephyr) → *Full control*,
lower long‑term cost, infra/DevOps required.

> **General rule:** Start fast with proprietary APIs, then migrate to
> open‑source for cost/privacy control.
:::

::: {#ea9846e2 .cell .markdown}

------------------------------------------------------------------------

`<a id="env">`{=html}`</a>`{=html}

## 1) Environment Check & Setup {#1-environment-check--setup}

This section verifies Python version, GPU availability, and installs
commonly used libraries.\
Run each cell once per runtime.
:::

::: {#c4bfa9cd .cell .code}
``` python
import sys, platform, os, subprocess, json, textwrap, math, time, random
from datetime import datetime

print("Python:", sys.version.split()[0])
print("Platform:", platform.platform())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
!nvidia-smi || echo "No NVIDIA GPU detected."
```
:::

::: {#31489f29 .cell .code}
``` python
# Core libraries used across the notebook
!pip -q install transformers accelerate sentencepiece bitsandbytes --upgrade
!pip -q install llama-cpp-python --upgrade
# Optional helpers
!pip -q install einops datasets evaluate tiktoken --upgrade
```
:::

::: {#b88a9844 .cell .markdown}

------------------------------------------------------------------------

`<a id="checklist">`{=html}`</a>`{=html}

## 2) Your Use Case Checklist {#2-your-use-case-checklist}

Before choosing a model, clarify:

-   **Use case**: chatbot, summarizer, code assistant, RAG, agent, etc.
-   **Privacy**: healthcare/finance/government constraints?
-   **Budget**: API pay‑per‑use vs GPU hosting?
-   **Scale**: daily active users (DAU), peak RPS, latency targets?
-   **Fit**: prompt‑only vs fine‑tune; domain‑specific data?

Run the next cell to record your choices. You can re-run and modify
anytime.
:::

::: {#02b13e51 .cell .code}
``` python
from dataclasses import dataclass, asdict

@dataclass
class UseCaseConfig:
    name: str = "My Assistant"
    use_case: str = "chatbot"
    privacy_level: str = "standard"  # options: standard, high, extreme
    budget_mode: str = "api"         # options: api, self-host, hybrid
    target_latency_ms: int = 800
    target_rps: float = 2.0
    need_finetune: bool = False
    context_window_tokens: int = 8000
    notes: str = "add any constraints here"

cfg = UseCaseConfig()
print(cfg)
```
:::

::: {#24dcadab .cell .markdown}

------------------------------------------------------------------------

`<a id="decision">`{=html}`</a>`{=html}

## 3) Proprietary vs Open‑Source --- Decision Guide {#3-proprietary-vs-opensource--decision-guide}

  -------------------------------------------------------------------------------
  Factor                  Proprietary             Open-Source
                          (GPT/Claude/Gemini)     (LLaMA/Mistral/Falcon/Zephyr)
  ----------------------- ----------------------- -------------------------------
  **Speed to MVP**        ◎ Fast                  ○ Medium

  **Peak Quality**        ◎ Very high             ○ High (varies by model/size)

  **Cost at Scale**       △ Increases with usage  ◎ Can be cheaper long‑term

  **Data Control**        △ Limited               ◎ Full

  **Customization**       ○ Prompting & fine‑tune ◎ Full (fine‑tune/quantize)
                          (sometimes)             

  **Ops Overhead**        ◎ Low                   △ Requires MLOps/DevOps
  -------------------------------------------------------------------------------

**Rule of thumb:** Prototype on proprietary APIs → baseline quality/perf
→ evaluate open‑source (quantized) for cost/privacy.
:::

::: {#9ec34d07 .cell .markdown}

------------------------------------------------------------------------

`<a id="transformers">`{=html}`</a>`{=html}

## 4) Open‑Source: Try a Small Model (🤗 Transformers) {#4-opensource-try-a-small-model--transformers}

Below we load a small model to keep downloads quick in Colab. You can
swap to any compatible causal LM on Hugging Face.
:::

::: {#3b9ca279 .cell .code}
``` python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch, os

# Choose a lightweight model for demo
model_id = os.environ.get("DEMO_MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

print("Loading:", model_id)
tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

pipe = pipeline("text-generation", model=model, tokenizer=tok, device_map="auto")
res = pipe("You are a helpful assistant. Q: What's a good first step for LLM model selection?
A:", max_new_tokens=120, do_sample=False)
print(res[0]['generated_text'])
```
:::

::: {#4764f050 .cell .markdown}

------------------------------------------------------------------------

`<a id="gguf">`{=html}`</a>`{=html}

## 5) Open‑Source: Try a GGUF Model (llama.cpp via `llama-cpp-python`) {#5-opensource-try-a-gguf-model-llamacpp-via-llama-cpp-python}

**GGUF** enables running quantized models on CPU/GPU with low memory.
Below is a small demo using a tiny GGUF model.\
Swap `gguf_url` to another model if desired (check model license/terms).
:::

::: {#fcab95f7 .cell .code}
``` python
import os, urllib.request, pathlib, shutil
from llama_cpp import Llama

base_dir = pathlib.Path("/content") if pathlib.Path("/content").exists() else pathlib.Path(".")
gguf_dir = base_dir / "gguf_models"
gguf_dir.mkdir(parents=True, exist_ok=True)

# A tiny GGUF for quick demo. Replace with another GGUF URL if you prefer.
# Example sources: TheBloke/*-GGUF on Hugging Face (respect licenses).
gguf_url = os.environ.get("DEMO_GGUF_URL",
    "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin?download=true"  # tiny toy
)
gguf_path = gguf_dir / "tiny-stories.gguf"

if not gguf_path.exists():
    print("Downloading tiny GGUF...")
    urllib.request.urlretrieve(gguf_url, gguf_path)
else:
    print("GGUF already present:", gguf_path)

llm = Llama(model_path=str(gguf_path), n_ctx=2048, n_threads=os.cpu_count())
out = llm("Q: Give me one sentence about why GGUF can be useful.
A:", max_tokens=64, stop=["
"])
print(out["choices"][0]["text"])
```
:::

::: {#a84df1b1 .cell .markdown}

------------------------------------------------------------------------

`<a id="perf">`{=html}`</a>`{=html}

## 6) Latency & Throughput Testing {#6-latency--throughput-testing}

**Latency:** time to first token / full response.\
**Throughput:** requests per second (RPS) or tokens/sec under load.

Below: simple utilities to measure both on the current pipeline.
:::

::: {#d3e3f4c3 .cell .code}
``` python
import time, statistics, asyncio
from concurrent.futures import ThreadPoolExecutor

def time_single_inference(prompt, max_new_tokens=64):
    t0 = time.perf_counter()
    _ = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000  # ms

# Warmup
_ = pipe("Warmup.", max_new_tokens=8, do_sample=False)

prompts = [f"Prompt {i}: Summarize LLMOps model selection in 1 sentence." for i in range(5)]
latencies = [time_single_inference(p, 64) for p in prompts]
print("Latency (ms) per request:", [round(x,1) for x in latencies])
print("Avg:", round(statistics.mean(latencies),1), "ms | p95:", round(statistics.quantiles(latencies, n=20)[-1],1), "ms")

# Simple concurrent throughput test
def run_one(p):
    return pipe(p, max_new_tokens=32, do_sample=False)

async def concurrent_test(n=5):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=n) as ex:
        t0 = time.perf_counter()
        futs = [loop.run_in_executor(ex, run_one, f"Concurrent {i}: Say 'ok'.") for i in range(n)]
        res = await asyncio.gather(*futs)
        t1 = time.perf_counter()
    total_time = t1 - t0
    print(f"Completed {n} requests in {total_time:.2f}s → {n/total_time:.2f} RPS")

await concurrent_test(4)
```
:::

::: {#000e9d5b .cell .markdown}

------------------------------------------------------------------------

`<a id="ctx">`{=html}`</a>`{=html}

## 7) Context Window Experiments {#7-context-window-experiments}

Test how performance changes as you increase input length. Use this to
pick the right **context window** for your use case.
:::

::: {#359af1eb .cell .code}
``` python
def synth_context(n_words=1000):
    # Generate a synthetic passage ~n_words
    words = ["llmops","scaling","latency","throughput","quantization","context","window","benchmark","tokens","inference"]
    return " ".join(random.choice(words) for _ in range(n_words))

for words in [200, 1000, 3000]:
    ctx = synth_context(words)
    t0 = time.perf_counter()
    _ = pipe(f"Read this and answer in 1 sentence: {ctx}\nQuestion: What are two performance levers?", max_new_tokens=64, do_sample=False)
    t1 = time.perf_counter()
    print(f"Input ~{words} words → time {t1 - t0:.2f}s")
```
:::

::: {#abb42fc9 .cell .markdown}

------------------------------------------------------------------------

`<a id="tuning">`{=html}`</a>`{=html}

## 8) Prompt Engineering vs Fine‑Tuning {#8-prompt-engineering-vs-finetuning}

-   **Prompting**: Fast to iterate, zero training cost.
-   **Fine‑tuning**: Best for domain/format adherence & compliance. For
    small tasks, use **LoRA/QLoRA** to reduce cost.

Below is a *minimal* LoRA fine‑tune sketch (pseudo‑small dataset) you
can adapt. For real training, increase data/epochs and enable GPU.
:::

::: {#d5ce1c80 .cell .code}
``` python
# Minimal LoRA sketch using PEFT (optional)
!pip -q install peft datasets --upgrade

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model

# Tiny toy dataset (replace with your domain data)
train_texts = [
    "### Instruction: In one sentence, define LLMOps.\n### Response: LLMOps is the practice of operating, monitoring, and optimizing large language model systems in production.",
    "### Instruction: List two ways to reduce latency.\n### Response: Use quantization and faster inference backends like vLLM or TGI."
]
dataset = Dataset.from_dict({"text": train_texts})

tok2 = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", use_fast=True)
base = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
dc = DataCollatorForLanguageModeling(tok2, mlm=False)

lora_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
peft_model = get_peft_model(base, lora_cfg)

def tok_fn(batch):
    return tok2(batch["text"], truncation=True, max_length=512)

tok_ds = dataset.map(tok_fn, batched=True)
args = TrainingArguments(
    output_dir="./lora-out",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    learning_rate=2e-4,
    logging_steps=1,
    save_steps=5,
    max_steps=10
)
trainer = Trainer(model=peft_model, args=args, data_collator=dc, train_dataset=tok_ds)
trainer.train()

# Inference with adapted model
pipe_lora = pipeline("text-generation", model=peft_model, tokenizer=tok2, device_map="auto")
print(pipe_lora("### Instruction: In one sentence, define LLMOps.\n### Response:", max_new_tokens=60, do_sample=False)[0]['generated_text'])
```
:::

::: {#61edc3a2 .cell .markdown}

------------------------------------------------------------------------

`<a id="cost">`{=html}`</a>`{=html}

## 9) Cost Estimation --- API & Self‑Hosting Calculators {#9-cost-estimation--api--selfhosting-calculators}

Use these helpers to compare **API per‑token pricing** vs **GPU
hosting**. Adjust numbers below for your scenario.
:::

::: {#0cd869ae .cell .code}
``` python
from math import ceil

def estimate_api_cost(req_per_day=10000, in_tokens=600, out_tokens=300, price_in_per_1k=0.0005, price_out_per_1k=0.0015):
    daily_tokens_in = req_per_day * in_tokens
    daily_tokens_out = req_per_day * out_tokens
    cost_in = (daily_tokens_in/1000) * price_in_per_1k
    cost_out = (daily_tokens_out/1000) * price_out_per_1k
    return {"daily_usd": cost_in + cost_out, "monthly_usd": 30*(cost_in+cost_out)}

def estimate_gpu_hosting(num_gpus=1, hourly_gpu_cost=1.2, monthly_fixed=300):
    # hourly_gpu_cost: e.g., on-demand A10/A100 instance cost; adjust for your cloud
    gpu_month = 24*30*hourly_gpu_cost*num_gpus
    return {"monthly_usd": gpu_month + monthly_fixed}

api = estimate_api_cost()
gpu = estimate_gpu_hosting(num_gpus=2, hourly_gpu_cost=1.8, monthly_fixed=200)

print("API cost (example):", api)
print("GPU hosting (example):", gpu)

def break_even(api_monthly, gpu_monthly):
    if gpu_monthly <= 0: return "n/a"
    return api_monthly / gpu_monthly

print("Break‑even (API_monthly / GPU_monthly):", break_even(api["monthly_usd"], gpu["monthly_usd"]))
```
:::

::: {#5de4b28b .cell .markdown}

------------------------------------------------------------------------

`<a id="rag">`{=html}`</a>`{=html}

## 10) Minimal RAG Harness (Optional) {#10-minimal-rag-harness-optional}

A tiny example using `tiktoken` for chunking and naive retrieval. For
production, consider tools like LlamaIndex or LangChain.
:::

::: {#c9c79d5f .cell .code}
``` python
import re, math
import tiktoken

encoder = tiktoken.get_encoding("cl100k_base")

def chunk_text(text, tokens_per_chunk=300):
    toks = encoder.encode(text)
    chunks = []
    for i in range(0, len(toks), tokens_per_chunk):
        sub = encoder.decode(toks[i:i+tokens_per_chunk])
        chunks.append(sub)
    return chunks

# Naive embedding stand‑in using hashing (demo only)
def embed(text):
    random.seed(hash(text) % (2**32))
    return [random.random() for _ in range(64)]

def cosine(a,b):
    num = sum(x*y for x,y in zip(a,b))
    da = math.sqrt(sum(x*x for x in a))
    db = math.sqrt(sum(x*x for x in b))
    return num/(da*db + 1e-9)

# Build a toy index
docs = [
    "LLMOps involves monitoring, cost control, and performance optimization.",
    "Quantization reduces model size and improves latency at some accuracy cost.",
    "vLLM and TGI are high‑throughput inference backends."
]
chunks = [c for d in docs for c in chunk_text(d, 80)]
vecs = [embed(c) for c in chunks]

def retrieve(query, k=2):
    qv = embed(query)
    sims = [(cosine(qv, v), i) for i,v in enumerate(vecs)]
    sims.sort(reverse=True)
    return [chunks[i] for _, i in sims[:k]]

q = "How to reduce LLM latency?"
ctx = "\n\n".join(retrieve(q, k=3))
print("Retrieved context:\n", ctx)

print("\nAnswer:")
print(pipe(f"Answer using context only.\nContext:\n{ctx}\n\nQ: {q}\nA:", max_new_tokens=120, do_sample=False)[0]['generated_text'])
```
:::

::: {#f6c49b3e .cell .markdown}

------------------------------------------------------------------------

`<a id="prod">`{=html}`</a>`{=html}

## 11) Production Inference (vLLM/TGI) --- Optional {#11-production-inference-vllmtgi--optional}

These backends boost throughput significantly. Installs may take time
and require GPUs with sufficient memory.

**vLLM (example):**

``` bash
pip install vllm
python -m vllm.entrypoints.api_server --model meta-llama/Meta-Llama-3-8B-Instruct
# Then query via OpenAI-compatible endpoint: POST /v1/completions
```

**Text Generation Inference (TGI):**

``` bash
pip install text-generation
text-generation-launcher --model meta-llama/Meta-Llama-3-8B-Instruct
```

> In Colab you can try these, but for production use managed endpoints
> or your cloud GPU VMs.
:::

::: {#38bd4078 .cell .markdown}

------------------------------------------------------------------------

`<a id="bench">`{=html}`</a>`{=html}

## 12) Quick Benchmarking Utilities {#12-quick-benchmarking-utilities}

Micro‑benchmarks to compare prompts, decoding params, or small model
swaps. For comprehensive evals use `lm-eval-harness`.
:::

::: {#af1a3692 .cell .code}
``` python
tests = [
    ("Closed‑book QA", "Q: What is LLMOps in one sentence? A:"),
    ("Instruction Following", "Follow exactly: Reply with 'YES'."),
    ("Reasoning (Toy)", "I have 3 apples and buy 2 more, then eat 1. How many left?"),
]
for name, prompt in tests:
    t0 = time.perf_counter()
    out = pipe(prompt, max_new_tokens=64, do_sample=False)[0]['generated_text']
    dt = time.perf_counter() - t0
    print(f"=== {name} ===")
    print(out.strip())
    print(f"Time: {dt:.2f}s\n")
```
:::

::: {#92eee342 .cell .markdown}

------------------------------------------------------------------------

`<a id="refs">`{=html}`</a>`{=html}

## 13) References & Next Steps {#13-references--next-steps}

-   **Model hubs**: Hugging Face Hub --- compare models/sizes/context
    windows.
-   **Quantization**: GGUF (llama.cpp), bitsandbytes (int8/4), AWQ,
    GPTQ.
-   **Inference**: vLLM, Text Generation Inference (TGI),
    DeepSpeed-Inference.
-   **App frameworks**: LangChain, LlamaIndex, Guidance, Haystack.
-   **Eval**: HELM, lm-eval-harness, Ragas (for RAG).

**Suggested path**\
1) Prototype on a proprietary API to get baseline quality.\
2) Replicate with an open‑source model (quantized if needed).\
3) Measure latency/throughput & cost at realistic loads.\
4) Decide on prompt‑only or fine‑tuning.\
5) Plan production (vLLM/TGI), observability, and guardrails.
:::
