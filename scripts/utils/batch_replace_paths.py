#!/usr/bin/env python3
"""
批量替换文件中的路径
输出所有路径到文件供确认
"""
import os
import re
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
from collections import defaultdict

# 路径映射规则 (original_path, target_path)
PATH_MAPPINGS = [
    # 注意：顺序很重要，先替换长的路径，避免部分匹配
    ("/data2/zengzheni/chenyuxuan/Project/MHMLM", "/data1/chenyuxuan/MHMLM"),
    ("/data2/zengzheni/chenyuxuan/Project/MSMLM", "/data1/chenyuxuan/MSMLM"),
    ("/data2/zengzheni/xiaoyunduo/Layer2", "/data1/chenyuxuan/Layer2"),
    ("/data2/zengzheni/cpt_data", "/data1/chenyuxuan/train_data/cpt_data"),
    ("/data2/zengzheni/SFT_DATA", "/data1/chenyuxuan/train_data/SFT_DATA"),
    ("/data1/zengzheni/base_model", "/data1/chenyuxuan/base_model"),
    ("/data1/zengzheni/checkpoint", "/data1/chenyuxuan/checkpoint"),
    # 新增：从 valid_unreplaced_paths.json 中确认的映射（suggested_path 存在）
    ("/data2/zengzheni/checkpoint", "/data1/chenyuxuan/checkpoint"),
    ("/data2/zengzheni/model", "/data1/chenyuxuan/base_model"),
    ("/data1/zengzheni/model", "/data1/chenyuxuan/base_model"),
    ("/data2/zengzheni/lvchangwei", "/data1/lvchangwei"),
    # Conda 路径映射
    ("/data2/zengzheni/chenyuxuan/miniconda3", "/home/chenyuxuan/miniconda3"),
]

# 需要处理的文件扩展名
INCLUDE_EXTENSIONS = {'.py', '.yaml', '.yml', '.sh', '.json', '.md', '.txt', '.jsonl'}

# 排除的目录
EXCLUDE_DIRS = {'.git', '__pycache__', '.venv', 'node_modules', '.pytest_cache', 'artifacts', 'scripts/utils'}

# 排除的文件（通常是生成的或不需要修改的）
EXCLUDE_FILES = {
    'batch_replace_paths.py',  # 自己
}

# 创建路径到新路径的映射字典
PATH_MAP_DICT = {old: new for old, new in PATH_MAPPINGS}


def should_process_file(file_path: Path) -> bool:
    """判断是否应该处理该文件"""
    # 检查扩展名
    if file_path.suffix not in INCLUDE_EXTENSIONS:
        return False
    
    # 检查是否在排除列表中
    if file_path.name in EXCLUDE_FILES:
        return False
    
    # 检查是否在排除目录中
    for part in file_path.parts:
        if part in EXCLUDE_DIRS:
            return False
    
    # 排除 scripts/utils 目录及其子目录
    try:
        parts = file_path.parts
        if 'scripts' in parts and 'utils' in parts:
            scripts_idx = parts.index('scripts')
            if scripts_idx + 1 < len(parts) and parts[scripts_idx + 1] == 'utils':
                return False
    except (ValueError, IndexError):
        pass
    
    return True


def extract_variable_name(line: str, path: str) -> str:
    """尝试从行中提取变量名"""
    # 尝试匹配常见的变量赋值模式
    patterns = [
        r'(\w+)\s*[:=]\s*["\']?' + re.escape(path),  # var = "path" 或 var: "path"
        r'["\']?' + re.escape(path) + r'["\']?\s*[:=]\s*(\w+)',  # "path" = var
        r'(\w+)\s*=\s*["\']?' + re.escape(path),  # var = "path"
        r'--(\w+)\s+["\']?' + re.escape(path),  # --arg "path"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, line, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return ""


def find_paths_in_line(line: str, line_num: int, file_path: Path) -> List[Dict[str, Any]]:
    """在行中查找所有需要映射的路径"""
    found_paths = []
    
    for old_path, new_path in PATH_MAPPINGS:
        # 转义特殊字符
        escaped_old = re.escape(old_path)
        # 匹配路径（可能后面跟着 / 或其他字符，或在引号中）
        pattern = escaped_old + r'(?=/|"|\'| |\n|$|,|\)|]|})'
        
        matches = list(re.finditer(pattern, line))
        for match in matches:
            start_pos = match.start()
            end_pos = match.end()
            
            # 获取上下文（前后各30个字符）
            context_start = max(0, start_pos - 30)
            context_end = min(len(line), end_pos + 30)
            context = line[context_start:context_end]
            
            # 尝试提取变量名
            var_name = extract_variable_name(line, old_path)
            
            found_paths.append({
                "file_path": str(file_path),
                "line_number": line_num,
                "variable_name": var_name,
                "context": context.strip(),
                "original_path": old_path,
                "replaced_path": new_path,
                "position": (start_pos, end_pos)
            })
    
    return found_paths


def scan_file_for_paths(file_path: Path, root_dir: Path) -> Tuple[List[Dict], List[Dict]]:
    """扫描文件，找出所有需要替换和不需要替换的路径"""
    replaced_paths = []  # 已找到并可以替换的路径
    all_old_paths = set()  # 所有旧路径的集合
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"⚠️ 无法读取文件 {file_path}: {e}")
            return [], []
    except Exception as e:
        print(f"⚠️ 读取文件失败 {file_path}: {e}")
        return [], []
    
    # 扫描每一行
    for line_num, line in enumerate(lines, 1):
        # 查找所有需要映射的路径
        found = find_paths_in_line(line, line_num, file_path.relative_to(root_dir))
        replaced_paths.extend(found)
        
        # 记录所有旧路径（用于后续检查未替换的）
        for old_path in PATH_MAP_DICT.keys():
            if old_path in line:
                all_old_paths.add(old_path)
    
    return replaced_paths, []


def find_unreplaced_paths(file_path: Path, root_dir: Path) -> List[Dict[str, Any]]:
    """查找文件中未映射的旧路径（包含 /data2/zengzheni 或 /data1/zengzheni 但不在映射规则中）"""
    unreplaced = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception:
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                lines = f.readlines()
        except:
            return []
    except:
        return []
    
    # 定义需要检测的旧路径前缀
    old_prefixes = ['/data2/zengzheni', '/data1/zengzheni']
    
    # 获取所有已映射的路径前缀（用于排除）
    mapped_prefixes = set()
    for old_path, _ in PATH_MAPPINGS:
        # 提取前缀（到第一个子目录）
        parts = old_path.split('/')
        if len(parts) >= 4:
            mapped_prefixes.add('/'.join(parts[:4]))  # 例如 /data2/zengzheni/chenyuxuan
    
    # 匹配完整路径的正则表达式
    # 匹配 /data2/zengzheni/... 或 /data1/zengzheni/... 格式的路径
    path_pattern = re.compile(r'(/data[12]/zengzheni/[^"\' \n,\)\]}]+)')
    
    for line_num, line in enumerate(lines, 1):
        # 查找所有匹配的路径
        matches = path_pattern.finditer(line)
        for match in matches:
            found_path = match.group(1)
            
            # 检查这个路径是否已经在映射规则中
            is_mapped = False
            for old_path, new_path in PATH_MAPPINGS:
                # 检查是否是已映射路径的前缀或完全匹配
                if found_path.startswith(old_path + '/') or found_path == old_path:
                    is_mapped = True
                    break
            
            # 如果不在映射规则中，添加到未替换列表
            if not is_mapped:
                # 获取上下文
                start_pos = match.start()
                end_pos = match.end()
                context_start = max(0, start_pos - 30)
                context_end = min(len(line), end_pos + 30)
                context = line[context_start:context_end].strip()
                
                # 尝试提取变量名
                var_name = extract_variable_name(line, found_path)
                
                # 尝试推断建议的新路径
                suggested_path = ""
                if found_path.startswith('/data2/zengzheni/checkpoint'):
                    suggested_path = found_path.replace('/data2/zengzheni/checkpoint', '/data1/chenyuxuan/checkpoint')
                elif found_path.startswith('/data2/zengzheni/lvchangwei'):
                    # 这个路径需要用户确认，暂时保持原样或提供建议
                    suggested_path = found_path.replace('/data2/zengzheni/lvchangwei', '/data1/lvchangwei')
                elif found_path.startswith('/data1/zengzheni/checkpoint'):
                    suggested_path = found_path.replace('/data1/zengzheni/checkpoint', '/data1/chenyuxuan/checkpoint')
                elif found_path.startswith('/data1/zengzheni/base_model'):
                    suggested_path = found_path.replace('/data1/zengzheni/base_model', '/data1/chenyuxuan/base_model')
                else:
                    # 其他情况，尝试通用替换
                    if '/data2/zengzheni' in found_path:
                        suggested_path = found_path.replace('/data2/zengzheni', '/data1/chenyuxuan')
                    elif '/data1/zengzheni' in found_path:
                        suggested_path = found_path.replace('/data1/zengzheni', '/data1/chenyuxuan')
                
                unreplaced.append({
                    "file_path": str(file_path.relative_to(root_dir)),
                    "line_number": line_num,
                    "variable_name": var_name,
                    "context": context,
                    "original_path": found_path,
                    "suggested_path": suggested_path,
                    "note": "路径不在映射规则中，需要手动确认"
                })
    
    return unreplaced


def process_directory(root_dir: Path, mappings: List[Tuple[str, str]], dry_run: bool = False):
    """处理目录中的所有文件"""
    root_dir = Path(root_dir)
    
    print(f"🔍 扫描目录: {root_dir}")
    print(f"📋 路径映射规则:")
    for old, new in mappings:
        print(f"   {old}")
        print(f"   → {new}")
    print()
    
    all_replaced = []
    all_unreplaced = []
    total_files = 0
    
    # 遍历所有文件
    for file_path in root_dir.rglob('*'):
        if not file_path.is_file():
            continue
        
        if not should_process_file(file_path):
            continue
        
        total_files += 1
        
        # 扫描文件
        replaced, _ = scan_file_for_paths(file_path, root_dir)
        unreplaced = find_unreplaced_paths(file_path, root_dir)
        
        all_replaced.extend(replaced)
        all_unreplaced.extend(unreplaced)
        
        if replaced:
            print(f"✅ {file_path.relative_to(root_dir)}: 找到 {len(replaced)} 个可替换路径")
    
    print()
    print("=" * 60)
    print(f"📊 统计:")
    print(f"   扫描文件数: {total_files}")
    print(f"   找到可替换路径: {len(all_replaced)} 个")
    print(f"   需要确认的路径: {len(all_unreplaced)} 个")
    print("=" * 60)
    
    # 保存结果
    output_dir = root_dir / "scripts" / "utils" / "path_replacement_logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    replaced_file = output_dir / "replaced_paths.json"
    unreplaced_file = output_dir / "unreplaced_paths.json"
    
    # 保存已替换路径
    with open(replaced_file, 'w', encoding='utf-8') as f:
        json.dump(all_replaced, f, indent=2, ensure_ascii=False)
    print(f"\n💾 已保存可替换路径到: {replaced_file}")
    print(f"   共 {len(all_replaced)} 条记录")
    
    # 保存未替换路径
    with open(unreplaced_file, 'w', encoding='utf-8') as f:
        json.dump(all_unreplaced, f, indent=2, ensure_ascii=False)
    print(f"💾 已保存待确认路径到: {unreplaced_file}")
    print(f"   共 {len(all_unreplaced)} 条记录")
    
    # 如果不在dry_run模式，执行实际替换
    if not dry_run and all_replaced:
        print("\n🔄 开始执行替换...")
        replace_paths_in_files(root_dir, all_replaced)
    
    return all_replaced, all_unreplaced


def replace_paths_in_files(root_dir: Path, replacements: List[Dict[str, Any]]):
    """根据记录执行实际替换"""
    # 按文件分组
    files_to_modify = defaultdict(list)
    for rep in replacements:
        file_path = root_dir / rep["file_path"]
        files_to_modify[file_path].append(rep)
    
    total_replacements = 0
    
    for file_path, reps in files_to_modify.items():
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except:
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
            except Exception as e:
                print(f"⚠️ 无法读取文件 {file_path}: {e}")
                continue
        
        original_content = content
        
        # 按顺序应用所有替换（从后往前替换，避免位置偏移）
        for rep in sorted(reps, key=lambda x: x["position"][0], reverse=True):
            old_path = rep["original_path"]
            new_path = rep["replaced_path"]
            
            # 转义并替换
            escaped_old = re.escape(old_path)
            pattern = escaped_old + r'(?=/|"|\'| |\n|$|,|\)|]|})'
            content = re.sub(pattern, new_path, content, count=1)
        
        # 如果有修改，写回文件
        if content != original_content:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                total_replacements += len(reps)
                print(f"✅ {file_path.relative_to(root_dir)}: 替换了 {len(reps)} 处")
            except Exception as e:
                print(f"⚠️ 写入文件失败 {file_path}: {e}")
    
    print(f"\n✅ 总共替换了 {total_replacements} 处路径")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='批量替换文件中的路径')
    parser.add_argument('--root', type=str, default='/data1/chenyuxuan/MHMLM',
                       help='要处理的根目录')
    parser.add_argument('--dry-run', action='store_true', default=True,
                       help='只检查，不实际修改文件（默认）')
    parser.add_argument('--execute', action='store_true',
                       help='执行实际替换（覆盖--dry-run）')
    parser.add_argument('--mapping', type=str, nargs=2, action='append',
                       help='自定义映射规则: --mapping old_path new_path')
    
    args = parser.parse_args()
    
    root_dir = Path(args.root)
    if not root_dir.exists():
        print(f"❌ 目录不存在: {root_dir}")
        return
    
    # 使用自定义映射或默认映射
    mappings = PATH_MAPPINGS
    if args.mapping:
        mappings = [(old, new) for old, new in args.mapping]
    
    dry_run = not args.execute
    
    if dry_run:
        print("🔍 扫描模式（不会实际修改文件）")
        print("   使用 --execute 来实际执行替换")
        print()
    else:
        print("⚠️  执行模式（将实际修改文件）")
        print()
    
    replaced, unreplaced = process_directory(root_dir, mappings, dry_run=dry_run)
    
    if dry_run and replaced:
        print("\n💡 使用 --execute 来实际执行替换")


if __name__ == '__main__':
    main()
