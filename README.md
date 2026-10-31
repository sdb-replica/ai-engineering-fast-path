# AI Engineering Fast Path

Sai's 100-day journey from distributed systems to AI engineering — one byte-size
lesson per day, about an hour each evening. Theory bite, visuals, runnable code,
one hands-on win.

Each day has its own folder:

- `lesson.html` — the visual lesson (diagrams, analogies, code with expected
  output). Open it in any browser; it works offline.
- `lesson.md` — the same lesson as Markdown (renders natively here on GitHub).
- `diagram.svg` — the day's diagram as a standalone image.
- `code.py` — the day's runnable code. Run it with the day's environment.

## Setup (once)

```bash
python3 -m venv ~/ai-lab
source ~/ai-lab/bin/activate
pip install --upgrade pip
pip install torch numpy
```

## Lessons
| Day | Lesson (visual) | Lesson (markdown) | Code |
|-----|-----------------|-------------------|------|
| 1 | [Build Your AI Lab](day-001-build-your-ai-lab/lesson.html) | [lesson.md](day-001-build-your-ai-lab/lesson.md) | [code.py](day-001-build-your-ai-lab/code.py) |
| 2 | [Tensors: The Universal Container](day-002-tensors-the-universal-container/lesson.html) | [lesson.md](day-002-tensors-the-universal-container/lesson.md) | [code.py](day-002-tensors-the-universal-container/code.py) |
| 3 | [Tensor Ops & Broadcasting](day-003-tensor-ops-broadcasting/lesson.html) | [lesson.md](day-003-tensor-ops-broadcasting/lesson.md) | [code.py](day-003-tensor-ops-broadcasting/code.py) |
| 4 | [Autograd: The Tape Recorder](day-004-autograd-the-tape-recorder/lesson.html) | [lesson.md](day-004-autograd-the-tape-recorder/lesson.md) | [code.py](day-004-autograd-the-tape-recorder/code.py) |
| 5 | [Loss Functions: How a Model Knows It's Wrong](day-005-loss-functions/lesson.html) | [lesson.md](day-005-loss-functions/lesson.md) | [code.py](day-005-loss-functions/code.py) |
| 6 | [Gradient Descent From Scratch](day-006-gradient-descent-from-scratch/lesson.html) | [lesson.md](day-006-gradient-descent-from-scratch/lesson.md) | [code.py](day-006-gradient-descent-from-scratch/code.py) |
| 7 | [The Training Loop Anatomy](day-007-training-loop-anatomy/lesson.html) | [lesson.md](day-007-training-loop-anatomy/lesson.md) | [code.py](day-007-training-loop-anatomy/code.py) |
| 8 | [Train, Val, Test & Overfitting](day-008-train-val-test-and-overfitting/lesson.html) | [lesson.md](day-008-train-val-test-and-overfitting/lesson.md) | [code.py](day-008-train-val-test-and-overfitting/code.py) |
| 9 | [Fit a Line to Real Data](day-009-fit-a-line-to-real-data/lesson.html) | [lesson.md](day-009-fit-a-line-to-real-data/lesson.md) | [code.py](day-009-fit-a-line-to-real-data/code.py) |
| 10 | [Distributions: The Vocabulary of Uncertainty](day-010-distributions-the-vocab-of-uncertainty/lesson.html) | [lesson.md](day-010-distributions-the-vocab-of-uncertainty/lesson.md) | [code.py](day-010-distributions-the-vocab-of-uncertainty/code.py) |
| 11 | [Expectation, Variance & Maximum Likelihood](day-011-expectation-variance-maximum-likelihood/lesson.html) | [lesson.md](day-011-expectation-variance-maximum-likelihood/lesson.md) | [code.py](day-011-expectation-variance-maximum-likelihood/code.py) |
| 12 | [Vectors, Dot Products & the Core Op](day-012-vectors-dot-products-matmul/lesson.html) | [lesson.md](day-012-vectors-dot-products-matmul/lesson.md) | [code.py](day-012-vectors-dot-products-matmul/code.py) |
| 13 | [Why GPUs Exist](day-013-why-gpus-exist/lesson.html) | [lesson.md](day-013-why-gpus-exist/lesson.md) | [code.py](day-013-why-gpus-exist/code.py) |
| 14 | [Softmax & Cross-Entropy](day-014-softmax-and-cross-entropy/lesson.html) | [lesson.md](day-014-softmax-and-cross-entropy/lesson.md) | [code.py](day-014-softmax-and-cross-entropy/code.py) |
| 15 | [Embeddings: Categories → Vectors](day-015-embeddings-categories-to-vectors/lesson.html) | [lesson.md](day-015-embeddings-categories-to-vectors/lesson.md) | [code.py](day-015-embeddings-categories-to-vectors/code.py) |
| 16 | [Tiny Classifier on Real Data](day-016-tiny-classifier-real-data/lesson.html) | [lesson.md](day-016-tiny-classifier-real-data/lesson.md) | [code.py](day-016-tiny-classifier-real-data/code.py) |
| 17 | [Regularization: L2 & Dropout](day-017-regularization-l2-dropout/lesson.html) | [lesson.md](day-017-regularization-l2-dropout/lesson.md) | [code.py](day-017-regularization-l2-dropout/code.py) |
| 18 | [Optimizers: SGD vs Adam](day-018-optimizers-sgd-vs-adam/lesson.html) | [lesson.md](day-018-optimizers-sgd-vs-adam/lesson.md) | [code.py](day-018-optimizers-sgd-vs-adam/code.py) |
| 19 | [Data Loaders & Batching](day-019-data-loaders-and-batching/lesson.html) | [lesson.md](day-019-data-loaders-and-batching/lesson.md) | [code.py](day-019-data-loaders-and-batching/code.py) |
| 20 | [Phase A Recap: Learning = Differentiable Programming](day-020-phase-a-recap-differentiable-programming/lesson.html) | [lesson.md](day-020-phase-a-recap-differentiable-programming/lesson.md) | [code.py](day-020-phase-a-recap-differentiable-programming/code.py) |
| 21 | [Perceptron → MLP: Stacking Layers With Non-Linearities](day-021-perceptron-to-mlp/lesson.html) | [lesson.md](day-021-perceptron-to-mlp/lesson.md) | [code.py](day-021-perceptron-to-mlp/code.py) |
| 22 | [Activation Functions: ReLU, Sigmoid & Tanh](day-022-activation-functions/lesson.html) | [lesson.md](day-022-activation-functions/lesson.md) | [code.py](day-022-activation-functions/code.py) |
| 23 | [Backprop Intuition: Credit Assignment Through Layers](day-023-backprop-credit-assignment/lesson.html) | [lesson.md](day-023-backprop-credit-assignment/lesson.md) | [code.py](day-023-backprop-credit-assignment/code.py) |
| 24 | [PyTorch nn.Module: Writing a Clean Model Class](day-024-nn-module-clean-model-class/lesson.html) | [lesson.md](day-024-nn-module-clean-model-class/lesson.md) | [code.py](day-024-nn-module-clean-model-class/code.py) |
| 25 | [Training Loop, Done Right: Device, Eval Mode & no_grad](day-025-training-loop-done-right/lesson.html) | [lesson.md](day-025-training-loop-done-right/lesson.md) | [code.py](day-025-training-loop-done-right/code.py) |
| 26 | [CNNs 1: Convolution — The Sliding Pattern Scanner](day-026-cnns-1-convolution-intuition/lesson.html) | [lesson.md](day-026-cnns-1-convolution-intuition/lesson.md) | [code.py](day-026-cnns-1-convolution-intuition/code.py) |
| 27 | [CNNs 2: Build a Small Image Classifier](day-027-cnns-2-build-a-small-image-classifier/lesson.html) | [lesson.md](day-027-cnns-2-build-a-small-image-classifier/lesson.md) | [code.py](day-027-cnns-2-build-a-small-image-classifier/code.py) |
| 28 | [RNNs & Sequences: Recurrence and the Fading Memory](day-028-rnns-and-sequences/lesson.html) | [lesson.md](day-028-rnns-and-sequences/lesson.md) | [code.py](day-028-rnns-and-sequences/code.py) |
| 29 | [Batch Norm & Layer Norm: Stabilizing Training](day-029-batch-norm-layer-norm/lesson.html) | [lesson.md](day-029-batch-norm-layer-norm/lesson.md) | [code.py](day-029-batch-norm-layer-norm/code.py) |
| 30 | [Hyperparameters That Actually Matter: LR, Batch Size, Depth vs Width](day-030-hyperparameters-that-matter/lesson.html) | [lesson.md](day-030-hyperparameters-that-matter/lesson.md) | [code.py](day-030-hyperparameters-that-matter/code.py) |
| 31 | [Debugging Training: Loss Curves as Observability](day-031-debugging-training-loss-curves/lesson.html) | [lesson.md](day-031-debugging-training-loss-curves/lesson.md) | [code.py](day-031-debugging-training-loss-curves/code.py) |
| 32 | [Build: MLP on Tabular Data — Predict Next-Day Direction](day-032-mlp-tabular-next-day-direction/lesson.html) | [lesson.md](day-032-mlp-tabular-next-day-direction/lesson.md) | [code.py](day-032-mlp-tabular-next-day-direction/code.py) |
| 33 | [Weight Init & Why It Matters](day-033-weight-init-and-why-it-matters/lesson.html) | [lesson.md](day-033-weight-init-and-why-it-matters/lesson.md) | [code.py](day-033-weight-init-and-why-it-matters/code.py) |
| 34 | [What a Neural Net Is: A Composed Function + an Optimizer](day-034-what-a-neural-net-is/lesson.html) | [lesson.md](day-034-what-a-neural-net-is/lesson.md) | [code.py](day-034-what-a-neural-net-is/code.py) |
| 35 | [Phase B Project: The End-to-End Mini-Project](day-035-phase-b-project-end-to-end/lesson.html) | [lesson.md](day-035-phase-b-project-end-to-end/lesson.md) | [code.py](day-035-phase-b-project-end-to-end/code.py) |
| 36 | [Sequence Modeling Recap: Why Attention Replaced Recurrence](day-036-why-attention-replaced-recurrence/lesson.html) | [lesson.md](day-036-why-attention-replaced-recurrence/lesson.md) | [code.py](day-036-why-attention-replaced-recurrence/code.py) |
| 37 | [Attention Intuition: Queries, Keys, Values](day-037-attention-intuition-qkv/lesson.html) | [lesson.md](day-037-attention-intuition-qkv/lesson.md) | [code.py](day-037-attention-intuition-qkv/code.py) |
| 38 | [Scaled Dot-Product Attention, By Hand in Torch](day-038-scaled-dot-product-attention-by-hand/lesson.html) | [lesson.md](day-038-scaled-dot-product-attention-by-hand/lesson.md) | [code.py](day-038-scaled-dot-product-attention-by-hand/code.py) |
| 39 | [Multi-Head Attention: Why Multiple Heads](day-039-multi-head-attention/lesson.html) | [lesson.md](day-039-multi-head-attention/lesson.md) | [code.py](day-039-multi-head-attention/code.py) |
| 40 | [The Transformer Block: Attention + MLP + Residuals + Norms](day-040-the-transformer-block/lesson.html) | [lesson.md](day-040-the-transformer-block/lesson.md) | [code.py](day-040-the-transformer-block/code.py) |
| 41 | [Positional Encodings: Giving the Model a Sense of Order](day-041-positional-encodings/lesson.html) | [lesson.md](day-041-positional-encodings/lesson.md) | [code.py](day-041-positional-encodings/code.py) |
| 42 | [Build a Tiny GPT: A nanoGPT-Style Transformer on Text](day-042-build-tiny-transformer-nanogpt/lesson.html) | [lesson.md](day-042-build-tiny-transformer-nanogpt/lesson.md) | [code.py](day-042-build-tiny-transformer-nanogpt/code.py) |
| 43 | [Tokenization: BPE Intuition](day-043-tokenization-bpe-intuition/lesson.html) | [lesson.md](day-043-tokenization-bpe-intuition/lesson.md) | [code.py](day-043-tokenization-bpe-intuition/code.py) |
| 44 | [Training Dynamics of Transformers: Reading the Curve, Knowing When to Stop](day-044-transformer-training-dynamics/lesson.html) | [lesson.md](day-044-transformer-training-dynamics/lesson.md) | [code.py](day-044-transformer-training-dynamics/code.py) |
| 45 | [Encoder vs Decoder vs Encoder-Decoder: Same Block, Three Wirings](day-045-encoder-vs-decoder/lesson.html) | [lesson.md](day-045-encoder-vs-decoder/lesson.md) | [code.py](day-045-encoder-vs-decoder/code.py) |
| 46 | ["Attention Is All You Need": A Guided Read](day-046-attention-is-all-you-need-guided-read/lesson.html) | [lesson.md](day-046-attention-is-all-you-need-guided-read/lesson.md) | [code.py](day-046-attention-is-all-you-need-guided-read/code.py) |
| 47 | [Inference 1: Autoregressive Generation & Sampling](day-047-autoregressive-generation-sampling/lesson.html) | [lesson.md](day-047-autoregressive-generation-sampling/lesson.md) | [code.py](day-047-autoregressive-generation-sampling/code.py) |
| 48 | [The KV Cache: Never Recompute the Past](day-048-kv-cache/lesson.html) | [lesson.md](day-048-kv-cache/lesson.md) | [code.py](day-048-kv-cache/code.py) |
| 49 | [Context Windows & Why Long Context Is Expensive](day-049-context-windows-cost/lesson.html) | [lesson.md](day-049-context-windows-cost/lesson.md) | [code.py](day-049-context-windows-cost/code.py) |
| 50 | [Recap: The Transformer Mental Model, End to End](day-050-transformer-mental-model-recap/lesson.html) | [lesson.md](day-050-transformer-mental-model-recap/lesson.md) | [code.py](day-050-transformer-mental-model-recap/code.py) |
| 51 | [Pretraining: What It Actually Is](day-051-pretraining-what-it-actually-is/lesson.html) | [lesson.md](day-051-pretraining-what-it-actually-is/lesson.md) | [code.py](day-051-pretraining-what-it-actually-is/code.py) |
| 52 | [Fine-Tuning 1: Supervised Fine-Tuning (SFT)](day-052-fine-tuning-1-sft/lesson.html) | [lesson.md](day-052-fine-tuning-1-sft/lesson.md) | [code.py](day-052-fine-tuning-1-sft/code.py) |
| 53 | [Fine-Tuning 2: LoRA / QLoRA](day-053-fine-tuning-2-lora/lesson.html) | [lesson.md](day-053-fine-tuning-2-lora/lesson.md) | [code.py](day-053-fine-tuning-2-lora/code.py) |
| 54 | [RLHF 1: Why SFT Isn't Enough — Preferences & Reward Models](day-054-rlhf-1-reward-models/lesson.html) | [lesson.md](day-054-rlhf-1-reward-models/lesson.md) | [code.py](day-054-rlhf-1-reward-models/code.py) |
| 55 | [DPO: Skip the Reward Model, Train on the Pairs](day-055-rlhf-2-dpo/lesson.html) | [lesson.md](day-055-rlhf-2-dpo/lesson.md) | [code.py](day-055-rlhf-2-dpo/code.py) |
| 56 | [RLVR: Reinforcement Learning With Verifiable Rewards](day-056-rlvr-verifiable-rewards/lesson.html) | [lesson.md](day-056-rlvr-verifiable-rewards/lesson.md) | [code.py](day-056-rlvr-verifiable-rewards/code.py) |
| 57 | [Evals: How Anyone Knows a Model Is Good](day-057-evals-how-models-are-judged/lesson.html) | [lesson.md](day-057-evals-how-models-are-judged/lesson.md) | [code.py](day-057-evals-how-models-are-judged/code.py) |
| 58 | [RAG 1: Retrieval-Augmented Generation, the Architecture](day-058-rag-1-architecture/lesson.html) | [lesson.md](day-058-rag-1-architecture/lesson.md) | [code.py](day-058-rag-1-architecture/code.py) |
| 59 | [RAG 2: Build a Tiny RAG Over Your Own Docs](day-059-rag-2-build-tiny-rag/lesson.html) | [lesson.md](day-059-rag-2-build-tiny-rag/lesson.md) | [code.py](day-059-rag-2-build-tiny-rag/code.py) |
| 60 | [Agents 1: Tool Use — Schemas In, Tool Calls Out](day-060-agents-1-tool-use/lesson.html) | [lesson.md](day-060-agents-1-tool-use/lesson.md) | [code.py](day-060-agents-1-tool-use/code.py) |
| 61 | [Agents 2: The Agentic Loop — Plan, Act, Observe](day-061-agents-2-agentic-loop/lesson.html) | [lesson.md](day-061-agents-2-agentic-loop/lesson.md) | [code.py](day-061-agents-2-agentic-loop/code.py) |
| 62 | [Prompt Engineering, the Engineer's Way: Structure, Not Spells](day-062-prompt-engineering-engineers-way/lesson.html) | [lesson.md](day-062-prompt-engineering-engineers-way/lesson.md) | [code.py](day-062-prompt-engineering-engineers-way/code.py) |
| 63 | [Recap: The Full Modern Stack on One Page](day-063-recap-full-modern-stack/lesson.html) | [lesson.md](day-063-recap-full-modern-stack/lesson.md) | [code.py](day-063-recap-full-modern-stack/code.py) |
| 64 | [Inference Serving 1: vLLM & Continuous Batching](day-064-inference-serving-1-continuous-batching/lesson.html) | [lesson.md](day-064-inference-serving-1-continuous-batching/lesson.md) | [code.py](day-064-inference-serving-1-continuous-batching/code.py) |
| 65 | [Inference Serving 2: Quantization, INT8/INT4 Intuition](day-065-inference-serving-2-quantization/lesson.html) | [lesson.md](day-065-inference-serving-2-quantization/lesson.md) | [code.py](day-065-inference-serving-2-quantization/code.py) |
| 66 | [Latency vs Throughput: The Serving Tradeoff](day-066-latency-vs-throughput-serving/lesson.html) | [lesson.md](day-066-latency-vs-throughput-serving/lesson.md) | [code.py](day-066-latency-vs-throughput-serving/code.py) |
| 67 | [Experiment Tracking: The 10-Minute Setup](day-067-experiment-tracking-local/lesson.html) | [lesson.md](day-067-experiment-tracking-local/lesson.md) | [code.py](day-067-experiment-tracking-local/code.py) |
| 68 | [Model Registries & Versioning: The Model Is a Build Artifact](day-068-model-registries-versioning/lesson.html) | [lesson.md](day-068-model-registries-versioning/lesson.md) | [code.py](day-068-model-registries-versioning/code.py) |
| 69 | [Data Pipelines for ML: What 'MLOps' Actually Means Day-to-Day](day-069-data-pipelines-for-ml/lesson.html) | [lesson.md](day-069-data-pipelines-for-ml/lesson.md) | [code.py](day-069-data-pipelines-for-ml/code.py) |
| 70 | [Monitoring Models in Prod: Drift](day-070-monitoring-models-in-prod-drift/lesson.html) | [lesson.md](day-070-monitoring-models-in-prod-drift/lesson.md) | [code.py](day-070-monitoring-models-in-prod-drift/code.py) |
| 71 | [Build: Serve a Model Behind a Tiny API With Metrics](day-071-build-serve-model-api-metrics/lesson.html) | [lesson.md](day-071-build-serve-model-api-metrics/lesson.md) | [code.py](day-071-build-serve-model-api-metrics/code.py) |
| 72 | [Cost Math: $/1M Tokens & GPU-Hour Thinking](day-072-cost-math-token-economics/lesson.html) | [lesson.md](day-072-cost-math-token-economics/lesson.md) | [code.py](day-072-cost-math-token-economics/code.py) |
| 73 | [The Memory Wall: Why Training Doesn't Fit on One GPU](day-073-why-training-doesnt-fit-memory-wall/lesson.html) | [lesson.md](day-073-why-training-doesnt-fit-memory-wall/lesson.md) | [code.py](day-073-why-training-doesnt-fit-memory-wall/code.py) |
| 74 | [Data Parallelism: The Straightforward Scale-Out](day-074-data-parallelism-all-reduce/lesson.html) | [lesson.md](day-074-data-parallelism-all-reduce/lesson.md) | [code.py](day-074-data-parallelism-all-reduce/code.py) |
| 75 | [Tensor & Pipeline Parallelism: Splitting the Model Itself](day-075-tensor-pipeline-parallelism/lesson.html) | [lesson.md](day-075-tensor-pipeline-parallelism/lesson.md) | [code.py](day-075-tensor-pipeline-parallelism/code.py) |
| 76 | [FSDP & DeepSpeed: Shard the State, Gather Just in Time](day-076-fsdp-deepspeed-frameworks/lesson.html) | [lesson.md](day-076-fsdp-deepspeed-frameworks/lesson.md) | [code.py](day-076-fsdp-deepspeed-frameworks/code.py) |
| 77 | [Mixed Precision: FP16/BF16, and Why It (Mostly) Just Works](day-077-mixed-precision-fp16-bf16/lesson.html) | [lesson.md](day-077-mixed-precision-fp16-bf16/lesson.md) | [code.py](day-077-mixed-precision-fp16-bf16/code.py) |
| 78 | [Checkpointing & Fault Tolerance at Scale](day-078-checkpointing-fault-tolerance-scale/lesson.html) | [lesson.md](day-078-checkpointing-fault-tolerance-scale/lesson.md) | [code.py](day-078-checkpointing-fault-tolerance-scale/code.py) |
| 79 | [The GPU: Just Enough CUDA Mental Model](day-079-gpu-cuda-mental-model/lesson.html) | [lesson.md](day-079-gpu-cuda-mental-model/lesson.md) | [code.py](day-079-gpu-cuda-mental-model/code.py) |
| 80 | [Inference at Scale: Disaggregated Serving & Speculative Decoding](day-080-inference-at-scale-disagg-speculative/lesson.html) | [lesson.md](day-080-inference-at-scale-disagg-speculative/lesson.md) | [code.py](day-080-inference-at-scale-disagg-speculative/code.py) |
| 81 | [Profile a Training Step: Find the Bottleneck](day-081-build-profile-training-step/lesson.html) | [lesson.md](day-081-build-profile-training-step/lesson.md) | [code.py](day-081-build-profile-training-step/code.py) |
| 82 | [Recap: The ML Systems Map — Where Your Infra Career Plugs In](day-082-recap-ml-systems-map/lesson.html) | [lesson.md](day-082-recap-ml-systems-map/lesson.md) | [code.py](day-082-recap-ml-systems-map/code.py) |
| 83 | [Quant ML 1: Why Finance Is the Hardest ML Domain](day-083-quant-ml-1-why-finance-hardest/lesson.html) | [lesson.md](day-083-quant-ml-1-why-finance-hardest/lesson.md) | [code.py](day-083-quant-ml-1-why-finance-hardest/code.py) |
| 84 | [Quant ML 2: The #1 Killer — Overfitting Backtests](day-084-quant-ml-2-overfitting-backtests/lesson.html) | [lesson.md](day-084-quant-ml-2-overfitting-backtests/lesson.md) | [code.py](day-084-quant-ml-2-overfitting-backtests/code.py) |
| 85 | [Features for Market Data: Returns, Volatility, Microstructure Basics](day-085-features-for-market-data/lesson.html) | [lesson.md](day-085-features-for-market-data/lesson.md) | [code.py](day-085-features-for-market-data/code.py) |
| 86 | [Labeling: The Triple-Barrier Method](day-086-labeling-triple-barrier/lesson.html) | [lesson.md](day-086-labeling-triple-barrier/lesson.md) | [code.py](day-086-labeling-triple-barrier/code.py) |
| 87 | [Walk-Forward Validation: The Only Honest Backtest](day-087-walk-forward-validation/lesson.html) | [lesson.md](day-087-walk-forward-validation/lesson.md) | [code.py](day-087-walk-forward-validation/code.py) |
| 88 | [Build 1: A Baseline Model on Equity Data](day-088-build-1-baseline-equity-model/lesson.html) | [lesson.md](day-088-build-1-baseline-equity-model/lesson.md) | [code.py](day-088-build-1-baseline-equity-model/code.py) |
| 89 | [Build 2: Crypto Data — Same Pipeline, Different Beast](day-089-build-2-crypto-pipeline/lesson.html) | [lesson.md](day-089-build-2-crypto-pipeline/lesson.md) | [code.py](day-089-build-2-crypto-pipeline/code.py) |
| 90 | [Position Sizing & Risk: Kelly Intuition, Volatility Targeting](day-090-position-sizing-risk-kelly/lesson.html) | [lesson.md](day-090-position-sizing-risk-kelly/lesson.md) | [code.py](day-090-position-sizing-risk-kelly/code.py) |
| 91 | [From Notebook to Bot: The Paper-Trading Harness](day-091-notebook-to-bot-paper-trading-harness/lesson.html) | [lesson.md](day-091-notebook-to-bot-paper-trading-harness/lesson.md) | [code.py](day-091-notebook-to-bot-paper-trading-harness/code.py) |
| 92 | [Recap: The Capstone System on One Page](day-092-recap-capstone-system/lesson.html) | [lesson.md](day-092-recap-capstone-system/lesson.md) | [code.py](day-092-recap-capstone-system/code.py) |
| 93 | [Deep Dive: Attention Over Market Sequences](day-093-deep-dive-attention-market-sequences/lesson.html) | [lesson.md](day-093-deep-dive-attention-market-sequences/lesson.md) | [code.py](day-093-deep-dive-attention-market-sequences/code.py) |
| 94 | [Deep Dive: LoRA Hands-On at Runnable Scale](day-094-deep-dive-lora-hands-on/lesson.html) | [lesson.md](day-094-deep-dive-lora-hands-on/lesson.md) | [code.py](day-094-deep-dive-lora-hands-on/code.py) |
| 95 | [Deep Dive: RAG Over Your Trading Journal](day-095-deep-dive-rag-trading-journal/lesson.html) | [lesson.md](day-095-deep-dive-rag-trading-journal/lesson.md) | [code.py](day-095-deep-dive-rag-trading-journal/code.py) |
| 96 | [Deep Dive: A Runnable Evals Harness](day-096-deep-dive-runnable-evals-harness/lesson.html) | [lesson.md](day-096-deep-dive-runnable-evals-harness/lesson.md) | [code.py](day-096-deep-dive-runnable-evals-harness/code.py) |
| 97 | [Deep Dive: Quantization Hands-On](day-097-deep-dive-quantization-hands-on/lesson.html) | [lesson.md](day-097-deep-dive-quantization-hands-on/lesson.md) | [code.py](day-097-deep-dive-quantization-hands-on/code.py) |
| 98 | [Data Parallelism for Real: All-Reduce Across Processes](day-098-deep-dive-all-reduce-cpu-multiprocess/lesson.html) | [lesson.md](day-098-deep-dive-all-reduce-cpu-multiprocess/lesson.md) | [code.py](day-098-deep-dive-all-reduce-cpu-multiprocess/code.py) |
| 99 | [Capstone: The Finance-ML System, Assembled End to End](day-099-capstone-finance-ml-system/lesson.html) | [lesson.md](day-099-capstone-finance-ml-system/lesson.md) | [code.py](day-099-capstone-finance-ml-system/code.py) |
| 100 | [The AI Engineer Map: What You Have, What's Next](day-100-ai-engineer-map/lesson.html) | [lesson.md](day-100-ai-engineer-map/lesson.md) | [code.py](day-100-ai-engineer-map/code.py) |
