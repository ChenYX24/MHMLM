from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

# 修改导入路径：使用相对导入
from .io_utils import iter_jsonl, open_text
from .masking import MaskingConfig, apply_dynamic_mask

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    from torch.utils.data import IterableDataset, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    # 如果没有 torch，定义占位符类
    class IterableDataset:  # type: ignore
        pass

    class Dataset:  # type: ignore
        pass

    TORCH_AVAILABLE = False


class Layer2JsonlIterable(IterableDataset):
    """
    最小可用的 Iterable Dataset（纯 Python）。
    - 支持读取 .jsonl / .jsonl.gz
    - 可选：在线生成动态 mask view
    """

    def __init__(
        self,
        path: str | Path,
        *,
        masking: bool = True,
        masking_cfg: Optional[MaskingConfig] = None,
        filter_has_yield: Optional[bool] = None,
    ) -> None:
        """
        Args:
            path: JSONL文件路径
            masking: 是否应用动态masking
            masking_cfg: Masking配置
            filter_has_yield: 如果为True，只返回has_yield=True的数据；如果为False，只返回has_yield=False的数据；如果为None，不过滤
        """
        self.path = str(path)
        self.masking = bool(masking)
        self.masking_cfg = masking_cfg or MaskingConfig()
        self.filter_has_yield = filter_has_yield

    def __iter__(self) -> Iterator[dict[str, Any]]:
        # 支持 DDP：每个进程只处理部分数据
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                rank = dist.get_rank()
                world_size = dist.get_world_size()
            else:
                rank = 0
                world_size = 1
        except Exception:
            rank = 0
            world_size = 1
        
        view_id = 0
        for idx, ex in enumerate(iter_jsonl(self.path)):
            # DDP: 每个进程只处理 rank 对应的数据
            if idx % world_size != rank:
                continue
            
            # 按has_yield过滤
            if self.filter_has_yield is not None:
                has_yield = ex.get("has_yield", False)
                if has_yield != self.filter_has_yield:
                    continue
            
            if not self.masking:
                yield ex
                continue
            view_id += 1
            yield apply_dynamic_mask(ex, self.masking_cfg, view_id=view_id)


def _build_offsets_with_filter(path: str | Path, filter_has_yield: bool) -> List[int]:
    """
    为 .jsonl 文件构建 byte offset 索引，同时按has_yield过滤（不支持 .gz）。
    """
    p = Path(path)
    if p.suffix == ".gz":
        raise ValueError("Indexed 模式不支持 .gz，请改用 Iterable 或先解压。")
    
    is_main = True
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            is_main = (dist.get_rank() == 0)
    except Exception:
        pass
    
    if is_main:
        file_size_gb = p.stat().st_size / 1024 / 1024 / 1024
        print(f"[INFO] 正在构建索引（过滤has_yield={filter_has_yield}）: {p.name} (文件大小: {file_size_gb:.2f} GB)...")
    
    offsets: List[int] = []
    off = 0
    import json
    
    with p.open("rb") as f:
        if tqdm is not None and is_main:
            iterator = tqdm(f, desc="构建索引并过滤", unit="行", unit_scale=True)
        else:
            iterator = f
        
        for line in iterator:
            try:
                ex = json.loads(line.decode("utf-8").strip())
                has_yield = ex.get("has_yield", False)
                if has_yield == filter_has_yield:
                    offsets.append(off)
            except Exception:
                pass  # 跳过无效行
            off += len(line)
    
    if is_main:
        print(f"[INFO] 索引构建完成: {len(offsets):,} 行（过滤has_yield={filter_has_yield}）")
    
    return offsets


def _build_offsets(path: str | Path) -> List[int]:
    """
    为 .jsonl 文件构建 byte offset 索引（不支持 .gz）。
    """
    p = Path(path)
    if p.suffix == ".gz":
        raise ValueError("Indexed 模式不支持 .gz，请改用 Iterable 或先解压。")

    # 检查是否是主进程（避免DDP时每个进程都打印）
    is_main = True
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            is_main = (dist.get_rank() == 0)
    except Exception:
        pass
    
    if is_main:
        file_size_gb = p.stat().st_size / 1024 / 1024 / 1024
        print(f"[INFO] 正在构建索引: {p.name} (文件大小: {file_size_gb:.2f} GB)...")
    
    offsets: List[int] = []
    off = 0
    
    with p.open("rb") as f:
        # 使用 tqdm 显示进度
        if tqdm is not None and is_main:
            iterator = tqdm(f, desc="构建索引", unit="行", unit_scale=True)
        else:
            iterator = f
        
        for line in iterator:
            offsets.append(off)
            off += len(line)
    
    if is_main:
        print(f"[INFO] 索引构建完成: {len(offsets):,} 行")
    
    return offsets


class Layer2JsonlIndexed(Dataset):
    """
    简单的随机访问 Dataset：预先扫描 .jsonl 的行偏移。
    说明：
    - 适用于需要 shuffle 的训练（通常更稳定）
    - 不支持 .gz（可用外部解压或改用 IterableDataset + 预 shuffle）
    """

    def __init__(
        self,
        path: str | Path,
        *,
        masking: bool = True,
        masking_cfg: Optional[MaskingConfig] = None,
        filter_has_yield: Optional[bool] = None,
    ) -> None:
        """
        Args:
            path: JSONL文件路径
            masking: 是否应用动态masking
            masking_cfg: Masking配置
            filter_has_yield: 如果为True，只返回has_yield=True的数据；如果为False，只返回has_yield=False的数据；如果为None，不过滤
        """
        self.path = str(path)
        self.masking = bool(masking)
        self.masking_cfg = masking_cfg or MaskingConfig()
        self.filter_has_yield = filter_has_yield
        # 如果有过滤条件，需要先扫描文件构建有效索引
        if self.filter_has_yield is not None:
            self._offsets = _build_offsets_with_filter(self.path, filter_has_yield=self.filter_has_yield)
        else:
            self._offsets = _build_offsets(self.path)

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        import json

        off = self._offsets[idx]
        try:
            with open(self.path, "rb") as f:
                f.seek(off)
                line = f.readline().decode("utf-8").strip()
            ex = json.loads(line)
        except Exception as e:
            raise RuntimeError(f"读取数据失败 (idx={idx}, offset={off}): {e}")
        
        # 按has_yield过滤（Indexed模式在构建索引时已过滤，这里不需要再过滤，但保留检查）
        if self.filter_has_yield is not None:
            has_yield = ex.get("has_yield", False)
            if has_yield != self.filter_has_yield:
                # 这种情况不应该发生（索引已过滤），但为了安全保留检查
                raise RuntimeError(f"数据与过滤条件不匹配: has_yield={has_yield}, filter={self.filter_has_yield}")
        
        if not self.masking:
            return ex
        return apply_dynamic_mask(ex, self.masking_cfg, view_id=idx)

