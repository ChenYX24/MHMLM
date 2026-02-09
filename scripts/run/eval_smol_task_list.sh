#!/bin/bash
# 自动并行评测脚本 - 通过配置列表自动调度
# 支持：模型路径、输出名称、fewshot开关、GPU分配、任务选择、其他可选参数
#
# 使用方法：
# 1. 编辑 TASK_LIST 数组，添加你的评测任务
# 2. 运行脚本: bash eval_smol_task_list.sh
#
# ==================== 可选任务列表 ====================
# 所有可选的评测任务（通过 include_tasks 参数指定，逗号分隔）：
# - molecule_generation          # 分子生成
# - molecule_captioning          # 分子描述
# - name_conversion-i2f          # IUPAC名称转分子式
# - name_conversion-i2s          # IUPAC名称转SMILES
# - name_conversion-s2f          # SMILES转分子式
# - name_conversion-s2i          # SMILES转IUPAC名称
# - forward_synthesis            # 正向合成
# - retrosynthesis               # 逆合成
# - property_prediction-bbbp     # BBBP性质预测
# - property_prediction-clintox  # ClinTox性质预测
# - property_prediction-esol     # ESOL性质预测
# - property_prediction-hiv      # HIV性质预测
# - property_prediction-lipo     # Lipo性质预测
# - property_prediction-sider    # SIDER性质预测
#
# 如果不指定 include_tasks，则运行所有任务。
#
# ==================== 任务配置格式 ====================
# 每个任务用 | 分隔，格式为：
# model_path|output_name|fewshot|gpu|tasks|extra_args
#
# 字段说明：
# - model_path: 模型路径（必需）
# - output_name: 输出名称（可选，为空则从模型路径自动生成）
# - fewshot: true/false（必需）
# - gpu: GPU ID（必需，单个数字，如 6）
# - tasks: 要评测的任务列表（可选，逗号分隔，如 molecule_generation,forward_synthesis，为空则运行所有任务）
# - extra_args: 其他可选参数（可选，格式：key1=value1,key2=value2）
#
# ==================== 使用示例 ====================
# 示例1：只评测 molecule_generation 任务
# declare -a TASK_LIST=(
#     "/data1/chenyuxuan/checkpoint/model1||true|6|molecule_generation|"
# )
#
# 示例2：评测多个任务
# declare -a TASK_LIST=(
#     "/data1/chenyuxuan/checkpoint/model1||true|6|molecule_generation,forward_synthesis|"
# )
#
# 示例3：四种设置组合（fewshot/no fewshot × n-gram默认/开启）
# declare -a TASK_LIST=(
#     # fewshot + n-gram默认3
#     "/data1/chenyuxuan/checkpoint/qwen3_8b_cpt_sft/epoch2/LLM_nofreeze/checkpoint-4200||true|6|molecule_generation|"
#     # fewshot + n-gram开启（设置为0）
#     "/data1/chenyuxuan/checkpoint/qwen3_8b_cpt_sft/epoch2/LLM_nofreeze/checkpoint-4200||true|7|molecule_generation|no_repeat_ngram_size=0"
#     # no fewshot + n-gram默认3
#     "/data1/chenyuxuan/checkpoint/qwen3_8b_cpt_sft/epoch2/LLM_nofreeze/checkpoint-4200||false|6|molecule_generation|"
#     # no fewshot + n-gram开启（设置为0）
#     "/data1/chenyuxuan/checkpoint/qwen3_8b_cpt_sft/epoch2/LLM_nofreeze/checkpoint-4200||false|7|molecule_generation|no_repeat_ngram_size=0"
# )
#
# 示例4：带额外参数
# declare -a TASK_LIST=(
#     "/data1/chenyuxuan/checkpoint/model1||true|6|molecule_generation|batch_size=8,data_limit=50"
# )
#
# ==================== 支持的额外参数（extra_args） ====================
# - batch_size: 批次大小（默认: 16）
# - data_limit: 数据限制（默认: 100）
# - max_new_tokens: 最大生成token数（默认: 512）
# - temperature: 温度（默认: 0.2）
# - top_p: top_p采样（默认: 0.9）
# - repetition_penalty: 重复惩罚（默认: 1.06）
# - no_repeat_ngram_size: n-gram重复限制（默认: 3，设置为0表示开启n-gram）
# - realtime_mol: 实时分子处理（默认: 0）
# - few_shot: fewshot数量（默认: 2，仅在fewshot=true时有效）
# - prompt_style: 提示风格（默认: strict）

cd /data1/chenyuxuan/MHMLM

# ==================== 环境检查 ====================
# 检查bash版本（wait -n 需要 bash 4.3+）
BASH_VERSION_CHECK=$(bash --version | head -n1 | grep -oE '[0-9]+\.[0-9]+' | head -n1)
BASH_MAJOR=$(echo "$BASH_VERSION_CHECK" | cut -d. -f1)
BASH_MINOR=$(echo "$BASH_VERSION_CHECK" | cut -d. -f2)

if [ "$BASH_MAJOR" -lt 4 ] || ([ "$BASH_MAJOR" -eq 4 ] && [ "$BASH_MINOR" -lt 3 ]); then
    echo "⚠️  警告: bash 版本 $BASH_VERSION_CHECK 可能不支持 wait -n（需要 4.3+）"
    echo "   如果并行调度失败，请升级bash或使用兼容模式"
fi

# ==================== 配置区域 ====================

# 使用 SMolInstruct 的测试数据
SMOLINSTRUCT_DIR="/data1/lvchangwei/LLM/SMolInstruct"
RAW_DATA_DIR="${SMOLINSTRUCT_DIR}/constructed_test"
TEMPLATE_DIR="${SMOLINSTRUCT_DIR}/data/template/instruction_tuning"
DEV_DATA_DIR="${SMOLINSTRUCT_DIR}/data/constructed_dev"
TOKEN_CLS_PATH="/data1/lvchangwei/LLM/Lora/qwen3_mlp_token_head.pt"
MODEL_DIR="/data1/chenyuxuan/base_model"

# 输出目录（可通过环境变量覆盖）
# 统一放到 MHMLM_ROOT/eval_results/results 下，避免根目录堆满 results
MHMLM_ROOT="${MHMLM_ROOT:-/data1/chenyuxuan/MHMLM}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-${MHMLM_ROOT}/eval_results/results/smol_eval_$(date +%Y%m%d_%H%M%S)}"

# 默认评估参数（可以被任务配置覆盖）
DEFAULT_MAX_NEW_TOKENS=512
DEFAULT_TEMPERATURE=0.2
DEFAULT_TOP_P=0.9
DEFAULT_REPETITION_PENALTY=1.06
DEFAULT_NO_REPEAT_NGRAM_SIZE=3
DEFAULT_DATA_LIMIT=100
DEFAULT_FEW_SHOT=2
DEFAULT_FEW_SHOT_SEED=42
DEFAULT_PROMPT_STYLE="strict"
DEFAULT_BATCH_SIZE=16
DEFAULT_REALTIME_MOL=1

# ==================== 任务配置列表 ====================
# 格式：model_path|output_name|fewshot|gpu|tasks|extra_args
# 注意：tasks 字段为空表示运行所有任务

declare -a TASK_LIST=(
    # 示例：四种设置组合（fewshot/no fewshot × n-gram默认/开启），只评测 molecule_generation
    "/data1/chenyuxuan/checkpoint/qwen3_8b_cpt_sft/epoch2/LLM_nofreeze/name_conversion/checkpoint-268|qwen3_8b_cpt_sft_gvp_name_conversion_fewshot_ngram0|true|0||no_repeat_ngram_size=0"
    "/data1/chenyuxuan/checkpoint/qwen3_8b_cpt_sft/epoch2/LLM_nofreeze/name_conversion/checkpoint-268|qwen3_8b_cpt_sft_gvp_name_conversion_nofewshot_ngram0|false|1||no_repeat_ngram_size=0"
    "/data1/chenyuxuan/checkpoint/qwen3_8b_cpt_sft/epoch2/LLM_nofreeze/name_conversion/checkpoint-268|qwen3_8b_cpt_sft_gvp_name_conversion_fewshot_ngram3|true|2||"
    "/data1/chenyuxuan/checkpoint/qwen3_8b_cpt_sft/epoch2/LLM_nofreeze/name_conversion/checkpoint-268|qwen3_8b_cpt_sft_gvp_name_conversion_nofewshot_ngram3|false|3||"
)

# ==================== 工具函数 ====================

# 从模型路径生成输出名称
generate_output_name() {
    local model_path=$1
    local fewshot=$2
    
    # 移除路径前缀，保留关键部分
    local name=$(echo "$model_path" | sed 's|.*/checkpoint/||' | sed 's|.*/model/||' | sed 's|/|_|g')
    
    # 清理特殊字符
    name=$(echo "$name" | sed 's/[^a-zA-Z0-9_-]/_/g')
    
    # 添加fewshot后缀
    if [ "$fewshot" = "true" ]; then
        name="${name}_fewshot"
    else
        name="${name}_nofewshot"
    fi
    
    echo "$name"
}

# 解析任务配置
parse_task() {
    local task=$1
    IFS='|' read -r model_path output_name fewshot gpu tasks extra_args <<< "$task"
    
    # 如果输出名称为空，自动生成
    if [ -z "$output_name" ]; then
        output_name=$(generate_output_name "$model_path" "$fewshot")
    fi
    
    # 如果 tasks 为空，设置为空字符串（表示运行所有任务）
    if [ -z "$tasks" ]; then
        tasks=""
    fi
    
    echo "$model_path|$output_name|$fewshot|$gpu|$tasks|$extra_args"
}

# 解析额外参数
parse_extra_args() {
    local extra_args=$1
    local args=""
    
    if [ -n "$extra_args" ]; then
        IFS=',' read -ra PARAMS <<< "$extra_args"
        for param in "${PARAMS[@]}"; do
            if [[ "$param" == *"="* ]]; then
                IFS='=' read -r key value <<< "$param"
                args="${args} --${key} ${value}"
            fi
        done
    fi
    
    echo "$args"
}

# 运行评估任务
run_evaluation() {
    local model_path=$1
    local output_name=$2
    local fewshot=$3
    local gpu=$4
    local tasks=$5
    local extra_args=$6
    
    local model_output="${OUTPUT_BASE_DIR}/${output_name}"
    mkdir -p "${model_output}"
    
    # 构建基础命令
    local cmd="CUDA_VISIBLE_DEVICES=${gpu} uv run --preview-features extra-build-dependencies python eval/eval_smolinstruct.py"
    cmd="${cmd} --raw_data_dir \"${RAW_DATA_DIR}\""
    cmd="${cmd} --template_dir \"${TEMPLATE_DIR}\""
    cmd="${cmd} --output_dir \"${model_output}\""
    cmd="${cmd} --molaware_ckpt \"${model_path}\""
    cmd="${cmd} --token_classifier_path \"${TOKEN_CLS_PATH}\""
    cmd="${cmd} --realtime_mol ${DEFAULT_REALTIME_MOL}"
    cmd="${cmd} --max_new_tokens ${DEFAULT_MAX_NEW_TOKENS}"
    cmd="${cmd} --temperature ${DEFAULT_TEMPERATURE}"
    cmd="${cmd} --top_p ${DEFAULT_TOP_P}"
    cmd="${cmd} --repetition_penalty ${DEFAULT_REPETITION_PENALTY}"
    cmd="${cmd} --no_repeat_ngram_size ${DEFAULT_NO_REPEAT_NGRAM_SIZE}"
    cmd="${cmd} --data_limit ${DEFAULT_DATA_LIMIT}"
    
    # 添加fewshot参数
    if [ "$fewshot" = "true" ]; then
        cmd="${cmd} --few_shot ${DEFAULT_FEW_SHOT}"
        cmd="${cmd} --few_shot_dir \"${DEV_DATA_DIR}\""
        cmd="${cmd} --few_shot_seed ${DEFAULT_FEW_SHOT_SEED}"
    fi
    
    # 添加任务选择参数
    if [ -n "$tasks" ]; then
        cmd="${cmd} --include_tasks \"${tasks}\""
    fi
    
    cmd="${cmd} --prompt_style ${DEFAULT_PROMPT_STYLE}"
    cmd="${cmd} --batch_size ${DEFAULT_BATCH_SIZE}"
    cmd="${cmd} --disable_verbose_logging"
    # cmd="${cmd} --verbose_gnn"
    cmd="${cmd} --save_json \"${model_output}/metrics.json\""
    cmd="${cmd} --use_flash_attention"
    
    # 添加额外参数（会覆盖默认值）
    local parsed_extra=$(parse_extra_args "$extra_args")
    if [ -n "$parsed_extra" ]; then
        cmd="${cmd} ${parsed_extra}"
    fi
    
    # 执行命令并记录日志
    echo "[GPU ${gpu}] ============================================================"
    echo "[GPU ${gpu}] 评估模型: ${model_path}"
    echo "[GPU ${gpu}] 输出目录: ${model_output}"
    echo "[GPU ${gpu}] Fewshot: ${fewshot}"
    if [ -n "$tasks" ]; then
        echo "[GPU ${gpu}] 评测任务: ${tasks}"
    else
        echo "[GPU ${gpu}] 评测任务: 所有任务"
    fi
    echo "[GPU ${gpu}] ============================================================"
    
    # 设置UTF-8编码环境变量，确保日志文件正确保存中文
    export PYTHONIOENCODING=utf-8
    export LC_ALL=C.UTF-8
    export LANG=C.UTF-8
    
    # 使用tee命令并确保UTF-8编码，同时将输出写入文件
    eval "${cmd}" 2>&1 | tee -a "${model_output}/evaluation.log"
    
    # 如果tee失败，尝试直接重定向（作为备选方案）
    # eval "${cmd}" 2>&1 | python3 -c "import sys; [sys.stdout.buffer.write(line.encode('utf-8', errors='replace') + b'\n') for line in sys.stdin]" | tee "${model_output}/evaluation.log"
    
    local exit_code=${PIPESTATUS[0]}
    if [ $exit_code -eq 0 ]; then
        echo "[GPU ${gpu}] ✅ ${output_name} 评估完成"
    else
        echo "[GPU ${gpu}] ❌ ${output_name} 评估失败 (退出码: $exit_code)"
    fi
    
    return $exit_code
}

# ==================== 智能调度系统 ====================

# 创建输出目录
mkdir -p "${OUTPUT_BASE_DIR}"

# 检查任务列表是否为空
if [ ${#TASK_LIST[@]} -eq 0 ]; then
    echo "⚠️  警告: 任务列表为空，请先配置 TASK_LIST"
    echo ""
    echo "配置示例："
    echo "declare -a TASK_LIST=("
    echo "    \"/path/to/model1||true|6|\""
    echo "    \"/path/to/model1||false|7|\""
    echo "    \"/path/to/model2|custom_name|true|6|batch_size=8,data_limit=50\""
    echo ")"
    exit 1
fi

# 解析所有任务并构建任务队列
declare -a PARSED_TASKS=()
for task in "${TASK_LIST[@]}"; do
    parsed=$(parse_task "$task")
    PARSED_TASKS+=("$parsed")
done

# 提取所有使用的GPU
declare -A GPU_SET
for task in "${PARSED_TASKS[@]}"; do
    IFS='|' read -r model_path output_name fewshot gpu tasks extra_args <<< "$task"
    # 处理多个GPU（逗号分隔）
    IFS=',' read -ra GPUS <<< "$gpu"
    for g in "${GPUS[@]}"; do
        GPU_SET[$g]=1
    done
done

# 获取GPU列表
GPU_LIST=($(printf '%s\n' "${!GPU_SET[@]}" | sort -n))

if [ ${#GPU_LIST[@]} -eq 0 ]; then
    echo "❌ 错误: 未找到有效的GPU配置"
    exit 1
fi

echo "============================================================"
echo "🚀 智能调度评估系统"
echo "============================================================"
echo "总任务数: ${#PARSED_TASKS[@]}"
echo "使用GPU: ${GPU_LIST[*]}"
echo "输出目录: ${OUTPUT_BASE_DIR}"
echo ""

# 关联数组：跟踪每个GPU上的进程PID
declare -A GPU_PIDS
declare -A GPU_TASK_NAMES

# 初始化GPU状态
for gpu in "${GPU_LIST[@]}"; do
    GPU_PIDS[$gpu]=""
    GPU_TASK_NAMES[$gpu]=""
done

FAILED=0
TASK_INDEX=0
TOTAL_TASKS=${#PARSED_TASKS[@]}
step=0  # 用于定期显示并行状态

# 函数：启动下一个分配给指定GPU的任务
start_next_task_for_gpu() {
    local gpu=$1
    local start_idx=$2
    
    for ((i=start_idx; i<TOTAL_TASKS; i++)); do
        local task="${PARSED_TASKS[$i]}"
        IFS='|' read -r model_path output_name fewshot task_gpu tasks extra_args <<< "$task"
        
        # 检查GPU是否匹配（支持逗号分隔的多个GPU）
        IFS=',' read -ra TASK_GPUS <<< "$task_gpu"
        for tgpu in "${TASK_GPUS[@]}"; do
            if [ "$tgpu" == "$gpu" ]; then
                echo "[SCHEDULER] 在 GPU ${gpu} 上启动任务: ${output_name}"
                
                # 在后台运行任务
                run_evaluation "$model_path" "$output_name" "$fewshot" "$gpu" "$tasks" "$extra_args" &
                
                GPU_PIDS[$gpu]=$!
                GPU_TASK_NAMES[$gpu]="$output_name"
                return $i  # 返回任务索引
            fi
        done
    done
    return 255  # 没有找到任务
}

# 启动初始任务（填充所有GPU）
CURRENT_INDEX=0
for gpu in "${GPU_LIST[@]}"; do
    start_next_task_for_gpu $gpu $CURRENT_INDEX
    idx=$?
    if [ $idx -ge 0 ] && [ $idx -lt 255 ]; then
        CURRENT_INDEX=$((idx + 1))
        sleep 2  # 避免同时启动导致资源竞争
    fi
done
TASK_INDEX=$CURRENT_INDEX

# 主调度循环
while [ $TASK_INDEX -lt $TOTAL_TASKS ] || [ -n "$(printf '%s\n' "${GPU_PIDS[@]}" | grep -v '^$')" ]; do
    # 收集所有活动的PID
    ACTIVE_PIDS=()
    for gpu in "${GPU_LIST[@]}"; do
        if [ -n "${GPU_PIDS[$gpu]}" ]; then
            ACTIVE_PIDS+=("${GPU_PIDS[$gpu]}")
    fi
    done
    
    if [ ${#ACTIVE_PIDS[@]} -gt 0 ]; then
        # 显示当前并行运行的任务数
        if [ $((step % 10)) -eq 0 ]; then
            echo "[SCHEDULER] 当前并行运行: ${#ACTIVE_PIDS[@]} 个任务 (GPU: $(printf '%s ' "${!GPU_PIDS[@]}"))"
        fi
        step=$((step + 1))
        
        # 等待任意一个任务完成
        wait -n "${ACTIVE_PIDS[@]}" 2>/dev/null
        EXIT_CODE=$?
        
        # 找出哪个GPU的任务完成了
        for gpu in "${GPU_LIST[@]}"; do
            if [ -n "${GPU_PIDS[$gpu]}" ]; then
                # 检查进程是否已经结束
                if ! kill -0 "${GPU_PIDS[$gpu]}" 2>/dev/null; then
                    COMPLETED_GPU=$gpu
                    COMPLETED_PID="${GPU_PIDS[$gpu]}"
                    COMPLETED_TASK="${GPU_TASK_NAMES[$gpu]}"
                    
                    # 等待进程完全结束并获取退出码
                    wait "$COMPLETED_PID" 2>/dev/null
                    EXIT_CODE=$?
                    
                    if [ $EXIT_CODE -eq 0 ]; then
                        echo "[SCHEDULER] ✅ GPU ${COMPLETED_GPU} 任务完成: ${COMPLETED_TASK}"
                    else
                        echo "[SCHEDULER] ❌ GPU ${COMPLETED_GPU} 任务失败: ${COMPLETED_TASK} (退出码: $EXIT_CODE)"
                        FAILED=$((FAILED + 1))
                    fi
                    
                    # 清空该GPU的状态
                    GPU_PIDS[$COMPLETED_GPU]=""
                    GPU_TASK_NAMES[$COMPLETED_GPU]=""
                    
                    # 如果还有待运行的任务，启动新任务到该GPU
                    if [ $TASK_INDEX -lt $TOTAL_TASKS ]; then
                        start_next_task_for_gpu $COMPLETED_GPU $TASK_INDEX
                        new_index=$?
                        if [ $new_index -ge 0 ] && [ $new_index -lt 255 ]; then
                            TASK_INDEX=$((new_index + 1))
                            sleep 1
                        else
                            # 当前GPU没有更多任务，但还有其他任务，继续循环
                            TASK_INDEX=$((TASK_INDEX + 1))
                        fi
                    fi
                    break
                fi
            fi
        done
    else
        # 如果没有运行的任务但还有待运行的任务，启动下一个
        if [ $TASK_INDEX -lt $TOTAL_TASKS ]; then
            local task="${PARSED_TASKS[$TASK_INDEX]}"
            IFS='|' read -r model_path output_name fewshot task_gpu tasks extra_args <<< "$task"
            
            # 选择第一个可用的GPU（如果任务指定了多个GPU）
            IFS=',' read -ra TASK_GPUS <<< "$task_gpu"
            local selected_gpu="${TASK_GPUS[0]}"
            
            echo "[SCHEDULER] 在 GPU ${selected_gpu} 上启动任务: ${output_name}"
            run_evaluation "$model_path" "$output_name" "$fewshot" "$selected_gpu" "$tasks" "$extra_args" &
            GPU_PIDS[$selected_gpu]=$!
            GPU_TASK_NAMES[$selected_gpu]="$output_name"
            TASK_INDEX=$((TASK_INDEX + 1))
            sleep 1
        fi
    fi
    
    # 避免CPU占用过高
    sleep 1
done

# 等待所有剩余任务完成
for gpu in "${GPU_LIST[@]}"; do
    if [ -n "${GPU_PIDS[$gpu]}" ]; then
        echo "[SCHEDULER] 等待 GPU ${gpu} 的最后一个任务完成: ${GPU_TASK_NAMES[$gpu]}"
        wait "${GPU_PIDS[$gpu]}"
        EXIT_CODE=$?
        if [ $EXIT_CODE -ne 0 ]; then
            echo "[SCHEDULER] ❌ GPU ${gpu} 任务失败: ${GPU_TASK_NAMES[$gpu]} (退出码: $EXIT_CODE)"
            FAILED=$((FAILED + 1))
        else
            echo "[SCHEDULER] ✅ GPU ${gpu} 任务完成: ${GPU_TASK_NAMES[$gpu]}"
        fi
    fi
done

echo ""
echo "============================================================"
echo "✅ 所有任务完成"
echo "============================================================"

if [ $FAILED -eq 0 ]; then
    echo "✅ 所有评估任务成功完成！"
    echo "输出目录: ${OUTPUT_BASE_DIR}"
    exit 0
else
    echo "⚠️  有 $FAILED 个任务失败，请检查日志文件"
    exit 1
fi
