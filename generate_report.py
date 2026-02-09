#!/usr/bin/env python3
"""
生成评估报告：提取所有子文件夹的 metrics 指标，生成表格和图表
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import seaborn as sns
from pathlib import Path
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

# 定义任务和指标的映射
TASK_METRICS_MAPPING = {
    "name_conversion": {
        "I2F EM": ("name_conversion-i2f", "num_t1_ele_match", lambda x, total: (x / total * 100) if total > 0 else 0),
        "I2S fts": ("name_conversion-i2s", "t1_rdk_fps", lambda x, total: x * 100 if x is not None else 0),
        "I2S Valid": ("name_conversion-i2s", "num_t1_invalid", lambda x, total: ((total - x) / total * 100) if total > 0 else 0),
        "S2F EM": ("name_conversion-s2f", "num_t1_ele_match", lambda x, total: (x / total * 100) if total > 0 else 0),
        "S2I Em": ("name_conversion-s2i", "num_t1_split_match", lambda x, total: (x / total * 100) if total > 0 else 0),
    },
    "property_prediction": {
        "ESOL RMSE↓": ("property_prediction-esol", "RMSE", lambda x, total: x if x is not None else None),
        "Lipo RMSE↓": ("property_prediction-lipo", "RMSE", lambda x, total: x if x is not None else None),
        "BBBP Acc": ("property_prediction-bbbp", "num_correct", lambda x, total: (x / total * 100) if total > 0 else 0),
        "Clintox Acc": ("property_prediction-clintox", "num_correct", lambda x, total: (x / total * 100) if total > 0 else 0),
        "HIV Acc": ("property_prediction-hiv", "num_correct", lambda x, total: (x / total * 100) if total > 0 else 0),
    },
    "molecule_generation": {
        "RDK-FTS(%)": ("molecule_generation", "t1_rdk_fps", lambda x, total: x * 100 if x is not None else 0),
        "Valid(%)": ("molecule_generation", "num_t1_invalid", lambda x, total: ((total - x) / total * 100) if total > 0 else 0),
    },
    "forward_synthesis": {
        "RDK-FTS(%)": ("forward_synthesis", "t1_rdk_fps", lambda x, total: x * 100 if x is not None else 0),
        "Valid(%)": ("forward_synthesis", "num_t1_invalid", lambda x, total: ((total - x) / total * 100) if total > 0 else 0),
    },
    "retrosynthesis": {
        "RDK-FTS(%)": ("retrosynthesis", "t1_rdk_fps", lambda x, total: x * 100 if x is not None else 0),
        "Valid(%)": ("retrosynthesis", "num_t1_invalid", lambda x, total: ((total - x) / total * 100) if total > 0 else 0),
    },
    "molecule_captioning": {
        "METEOR": ("molecule_captioning", "meteor_score", lambda x, total: x * 100 if x is not None else 0),
    }
}

# 指标显示顺序
METRIC_ORDER = [
    # name conversion
    "I2F EM", "I2S fts", "I2S Valid", "S2F EM", "S2I Em",
    # property prediction
    "ESOL RMSE↓", "Lipo RMSE↓", "BBBP Acc", "Clintox Acc", "HIV Acc",
    # Mol generation
    "RDK-FTS(%)", "Valid(%)", "Rdk-fts", "Rdk FTS(%)",
    # forward_synthesis
    "RDK-FTS(%)", "Valid(%)",
    # Retrosynthesis
    "RDK-FTS(%)", "Valid(%)",
    # Molecule_captioning
    "METEOR"
]

# 任务顺序
TASK_ORDER = [
    "name_conversion",
    "property_prediction",
    "molecule_generation",
    "forward_synthesis",
    "retrosynthesis",
    "molecule_captioning"
]


def load_metrics(base_dir):
    """加载所有子文件夹的 metrics.json 文件"""
    metrics_data = {}
    base_path = Path(base_dir)
    
    for subdir in sorted(base_path.iterdir()):
        if subdir.is_dir():
            metrics_file = subdir / "metrics.json"
            if metrics_file.exists():
                try:
                    with open(metrics_file, 'r', encoding='utf-8') as f:
                        metrics_data[subdir.name] = json.load(f)
                except Exception as e:
                    print(f"Warning: Failed to load {metrics_file}: {e}")
    
    return metrics_data


def extract_metric_value(metrics, task_key, metric_key, transform_func):
    """从 metrics 字典中提取指标值"""
    if task_key not in metrics:
        return None
    
    task_data = metrics[task_key]
    
    # 获取总数
    total = task_data.get("num_all", 100)
    
    # 获取指标值
    value = task_data.get(metric_key)
    
    if value is None:
        return None
    
    # 应用转换函数
    try:
        return transform_func(value, total)
    except Exception as e:
        print(f"Warning: Failed to transform metric {task_key}.{metric_key}: {e}")
        return None


def build_dataframe(metrics_data):
    """构建包含所有指标的数据框"""
    rows = []
    
    for model_name, metrics in metrics_data.items():
        row = {"Model": model_name}
        
        # 遍历所有任务和指标
        for task_name, task_metrics in TASK_METRICS_MAPPING.items():
            for metric_name, (task_key, metric_key, transform_func) in task_metrics.items():
                # 为指标添加任务前缀以避免重复列名
                # 但保持原始指标名用于显示
                col_name = f"{task_name}_{metric_name}" if metric_name in ["RDK-FTS(%)", "Valid(%)"] else metric_name
                value = extract_metric_value(metrics, task_key, metric_key, transform_func)
                row[col_name] = value
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # 重命名列以显示任务信息（仅对重复的指标）
    rename_dict = {}
    for task_name, task_metrics in TASK_METRICS_MAPPING.items():
        for metric_name in task_metrics.keys():
            if metric_name in ["RDK-FTS(%)", "Valid(%)"]:
                col_name = f"{task_name}_{metric_name}"
                display_name = f"{task_name.replace('_', ' ').title()} {metric_name}"
                if col_name in df.columns:
                    rename_dict[col_name] = display_name
    
    df = df.rename(columns=rename_dict)
    return df


def save_table(df, output_path):
    """保存表格为 CSV 和 Markdown"""
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存为 CSV
    csv_path = output_path.with_suffix('.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"Saved table to {csv_path}")
    
    # 保存为 Markdown
    md_path = output_path.with_suffix('.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Metrics Report\n\n")
        f.write(df.to_markdown(index=False))
    print(f"Saved table to {md_path}")


def plot_metrics(df, output_dir):
    """根据指定顺序绘制图表"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 为每个任务创建图表
    for task_name in TASK_ORDER:
        if task_name not in TASK_METRICS_MAPPING:
            continue
        
        task_metrics = TASK_METRICS_MAPPING[task_name]
        metric_names = list(task_metrics.keys())
        
        # 确定这个任务在 METRIC_ORDER 中的位置
        task_metric_indices = []
        for metric_name in metric_names:
            # 查找在 METRIC_ORDER 中的位置（考虑任务上下文）
            # 由于可能有重复，我们需要找到属于当前任务的那个
            found_idx = None
            current_task_start = None
            for i, t in enumerate(TASK_ORDER):
                if t == task_name:
                    current_task_start = i
                    break
            
            # 在 METRIC_ORDER 中查找，考虑任务顺序
            for idx, m in enumerate(METRIC_ORDER):
                if m == metric_name:
                    # 简单检查：如果这是第一个匹配，或者我们找到了属于当前任务的指标
                    if found_idx is None:
                        found_idx = idx
                    break
            
            if found_idx is not None:
                task_metric_indices.append(found_idx)
        
        if not task_metric_indices:
            continue
        
        # 按顺序排序
        sorted_metrics = sorted(zip(metric_names, task_metric_indices), key=lambda x: x[1])
        metric_names = [m[0] for m in sorted_metrics]
        
        # 创建图表
        n_metrics = len(metric_names)
        if n_metrics == 0:
            continue
        
        # 根据指标数量决定布局
        if n_metrics <= 2:
            fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 6))
            if n_metrics == 1:
                axes = [axes]
        elif n_metrics <= 4:
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            axes = axes.flatten()
        else:
            cols = 3
            rows = (n_metrics + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows))
            axes = axes.flatten() if rows > 1 else [axes]
        
        fig.suptitle(f'{task_name.replace("_", " ").title()} Metrics', fontsize=16, fontweight='bold')
        
        for idx, metric_name in enumerate(metric_names):
            ax = axes[idx]
            
            # 确定 DataFrame 中的列名
            if metric_name in ["RDK-FTS(%)", "Valid(%)"]:
                col_name = f"{task_name.replace('_', ' ').title()} {metric_name}"
            else:
                col_name = metric_name
            
            # 如果列不存在，尝试原始名称
            if col_name not in df.columns:
                col_name = metric_name
            
            # 准备数据
            if col_name not in df.columns:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(metric_name)
                continue
            
            plot_data = df[['Model', col_name]].copy()
            plot_data = plot_data.dropna(subset=[col_name])
            
            if len(plot_data) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(metric_name)
                continue
            
            # 排序（RMSE 越小越好，其他越大越好）
            ascending = "RMSE" in metric_name
            plot_data = plot_data.sort_values(by=col_name, ascending=ascending)
            
            # 绘制条形图
            bars = ax.barh(range(len(plot_data)), plot_data[col_name].values, 
                          color=sns.color_palette("husl", len(plot_data)))
            
            # 设置标签
            ax.set_yticks(range(len(plot_data)))
            ax.set_yticklabels(plot_data['Model'].values, fontsize=8)
            ax.set_xlabel(metric_name, fontsize=10)
            ax.set_title(metric_name, fontsize=12, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # 添加数值标签
            for i, (bar, val) in enumerate(zip(bars, plot_data[col_name].values)):
                if not np.isnan(val):
                    ax.text(val, i, f'{val:.2f}', va='center', ha='left' if val >= 0 else 'right', 
                           fontsize=7)
        
        # 隐藏多余的子图
        for idx in range(len(metric_names), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        # 保存图表
        task_file_name = task_name.replace("_", "-")
        plot_path = output_path / f'{task_file_name}_metrics.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {plot_path}")
        plt.close()


def main():
    import os
    mhmlm_root = os.environ.get("MHMLM_ROOT", "/data1/chenyuxuan/MHMLM")
    base_dir = os.path.join(mhmlm_root, "artifacts", "results")
    output_dir = os.path.join(base_dir, "report")
    
    print("Loading metrics...")
    metrics_data = load_metrics(base_dir)
    print(f"Loaded metrics from {len(metrics_data)} models")
    
    print("Building dataframe...")
    df = build_dataframe(metrics_data)
    
    # 保存表格
    output_path = Path(output_dir) / "metrics_table"
    save_table(df, output_path)
    
    # 绘制图表
    print("Generating plots...")
    plot_metrics(df, output_dir)
    
    print("\nReport generation completed!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

