## MHMLM 脚本目录说明

- `preprocess/`：数据预处理脚本（格式转换、清洗等）。
- `postprocess/`：评测结果与日志的汇总、可视化等后处理脚本。
- `eval/`：各类评测入口脚本（ChemBench、MMLU、SmolInstruct 等）。
- `train/`：训练入口脚本（如 `gvp_mlp_pretrain.py`, `train_ldmol.py` 等）。
- `run/`：一键式 pipeline / 批量评测 Shell 脚本（例如 `run_chembench_all_tasks.sh`, `run_eval_layer2_testset.sh`）。
- `dev/`：开发与调试用脚本（例如 `debug_sft_tester.py`）。
- `ckpt/`：模型 checkpoint 拆分、转换相关工具脚本。
- `layer2/`, `layer2_llm/`：与 Layer2 / LLM 训练与数据生成相关的专用脚本。
- `utils/`：通用辅助脚本。

