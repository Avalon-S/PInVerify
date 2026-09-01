# Installation

This branch uses up to **three conda environments**. They disagree on Python,
CUDA and numpy majors, so they cannot be merged:

| | `pv_bench` | `pv_train` | `sensenova` |
|---|---|---|---|
| Used for | Qwen3-VL agents, Grounding DINO, CLIP/SigLIP2, all evaluation | LoRA fine-tuning with ms-swift | SenseNova-SI baseline only |
| Python | 3.10 | 3.10 | 3.11 |
| torch | 2.1.2 + cu118 | 2.2.0 + cu118 | 2.5.1 + cu121 |
| transformers | 4.52.4 | 4.57.0 | 4.57.3 |
| numpy | 1.26 | 1.26 | 2.2 |
| Section | [1](#1-pv_bench) | [4](#4-pv_train-optional) | [3](#3-sensenova-optional) |

`pv_bench` alone reproduces every number in the paper, including the trained
agents of Table 6: evaluating a LoRA adapter only needs the evaluation stack.
Build `pv_train` only if you intend to re-train, and `sensenova` only if you
want the SenseNova-SI row of Table 5.

The capture pipeline on the `data-collection` branch is yet another stack
(Python 3.9, torch 1.13.1+cu117, habitat 0.2.3). It lives on its own branch for
exactly this reason.

---

## 1. `pv_bench`

```bash
conda create -n pv_bench python=3.10 -y
conda activate pv_bench

pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

The versions in `requirements.txt` are pinned to what produced the paper's
numbers. `torch==2.1.2+cu118` needs the extra index URL; without it pip resolves
a build that will not match the Grounding DINO extension compiled below.

Verify:

```bash
python -c "import torch, transformers; print(torch.__version__, torch.version.cuda, transformers.__version__)"
# 2.1.2+cu118 11.8 4.52.4
```

## 2. Grounding DINO

Needed for `method.bbox_mode=dino`, which is what the main tables use. It
compiles a CUDA extension, so **the CUDA toolkit version has to match torch's**
(11.8 here). Check `nvcc --version` before starting; a mismatch produces
confusing compile errors.

```bash
conda activate pv_bench

git clone https://github.com/IDEA-Research/GroundingDINO
cd GroundingDINO
pip install -e .

mkdir -p weights && cd weights
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..

# The server wrapper reads its config and checkpoint by relative path,
# so it runs from inside this checkout.
cp /path/to/PInVerify/servers/run_groundingdino_server.py .
python run_groundingdino_server.py --port 12183
```

If you only want to reproduce the GT-box ablations (`method.bbox_mode=gt`), you
can skip this entirely.

**Alternative: a separate environment.** Grounding DINO also runs isolated,
which avoids any dependency pressure on `pv_bench`. The reference setup used
Python 3.8 with torch 2.0.1+cu118 for exactly this. Since PInVerify talks to it
over HTTP, the two environments never have to agree on anything:

```bash
conda create -n gdino python=3.8 -y && conda activate gdino
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
cd GroundingDINO && pip install -e .
```

The only hard requirement either way is that the compiled `_C` extension matches
the CUDA build of the torch in the same environment.

## 3. `sensenova` (optional)

Only for the SenseNova-SI row of Table 5.

```bash
conda create -n sensenova python=3.11 -y
conda activate sensenova

git clone https://github.com/OpenSenseNova/SenseNova-SI
cd SenseNova-SI
pip install -r requirements.txt   # follow their instructions

pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

`flash-attn` is the awkward part: building it from source takes a long time.
The reference environment used a prebuilt wheel matching the exact stack
(`flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl`).
Get the matching wheel from the
[flash-attention releases](https://github.com/Dao-AILab/flash-attention/releases)
rather than compiling.

Point `PYTHONPATH` at the checkout when running the server:

```bash
export PYTHONPATH=./SenseNova-SI:$PYTHONPATH
python servers/run_sensenova_si_server.py --port 12182
```

## 4. `pv_train` (optional)

Only for re-training. `ms-swift` 3.12.6 pulls a transformers newer than the one
`requirements.txt` pins, so it gets its own environment rather than disturbing
the evaluation one:

```bash
conda create -n pv_train python=3.10 -y && conda activate pv_train
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118
pip install ms-swift==3.12.6
```

How the two stacks differ:

| | `pv_bench` (what `requirements.txt` pins) | `pv_train` |
|---|---|---|
| torch | 2.1.2+cu118 | 2.2.0+cu118 |
| transformers | 4.52.4 | 4.57.0 |
| ms-swift | not installed | 3.12.6 |
| peft | not installed | 0.18.1 |

Train in `pv_train`, then evaluate the resulting adapter in `pv_bench`: the
pinned versions are the ones that reproduce the tables.

See [TRAINING.md](TRAINING.md).

## 5. Model weights

```
./models/
├── Qwen3-VL-4B-Instruct/                 primary backbone
├── Qwen3-VL-8B-Instruct/                 cross-model comparison
├── clip-vit-large-patch14/               embedding baseline
├── siglip2-so400m-patch14-384/           embedding baseline
└── SenseNova-SI-1.2-InternVL3-8B/        optional, ~15 GB
```

Paths are set per config in `method.clip_model` and in the server scripts, and
can be overridden on the command line.

## 6. Dataset

See [DATASET.md](DATASET.md). Expected at `./data/pv_dataset`, overridable with
`dataset.root=`.

---

## If huggingface.co is unreachable

Downloading weights and the dataset works through whatever mirror or proxy your
site provides; that setup is yours, not the project's. Two things are worth
knowing before you reach for one:

- **Read-only mirrors cannot accept uploads.** `scripts/prepare_hf_release.py`
  needs real access to the Hub, and probes it before transferring anything.
- **A non-interactive SSH session does not read `.bashrc`.** If a proxy is set
  there, put it in the same command as the upload rather than assuming it is
  already in the environment.

## Hardware

The paper's numbers were produced on **NVIDIA RTX 3090** (24 GB), 8 of them for
the training-free sweeps and 4 for training. That is what the reported runtimes
assume, and it is also the main reason those runtimes are large:

| Workload | GPUs | Wall time | GPU time |
|---|---|---|---|
| One training-free config, multi-view + LLM-NBV | 8 | 3 h 53 m | 31 h |
| The same config with GT boxes | 8 | 5 h 22 m | 43 h |
| The full 9-config main table (DINO) | 8 | ~17 h | ~153 h |
| Evaluating one trained adapter | 4 | ~2 h 40 m | ~11 h |
| SFT (3 epochs, 21.9 k samples) | 4 | ~3 h | ~12 h |
| GSPO | 4 | ~6 h | ~24 h |

**If you have anything newer than a 3090, use it.** This workload is
MLLM-inference bound, running many short generations per episode, and the 3090
is the slow part rather than the method. Newer datacenter or Ada-generation
cards cut these times substantially. We have not measured the speedup, so no
factor is quoted here.

24 GB per card is enough throughout: the 4B backbone with LoRA fits comfortably,
and the 8B cross-model runs fit as well. Less than 24 GB has not been tested.

The evaluator shards per episode across whatever GPUs you give it, so fewer or
more cards changes wall time but not the result.

## Multi-GPU serving

Once the environment works, the launchers bring up one model server plus one
detector per GPU in tmux, at ports `12182 + 100i` and `12183 + 100i`:

```bash
bash scripts/start_multigpu_servers.sh 4
bash scripts/manage_servers_multigpu.sh stop
```

They call `conda activate $CONDA_ENV`, defaulting to `pv_bench`. Override with
`CONDA_ENV=<name>`.
