import torch
import numpy as np
from typing import List, Optional, Dict, Tuple

def _to_long_tensor(x, device):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.to(device=device, dtype=torch.long)

def make_frame_labels_equal_blocks(allweather_data: torch.Tensor,
                                   num_classes: int,
                                   dtype=torch.long) -> torch.Tensor:
    """
    为等长类别块构造 1D 帧标签，长度为 T（与 dim=2 对齐）。
    类别 id 从 0...(num_classes-1)，顺序按块拼接。
    例如：T=27, num_classes=3 => 每类 9 帧 => [0*9, 1*9, 2*9]
    """
    assert allweather_data.dim() >= 3, "allweather_data 需要至少有 3 个维度 [N, C, T, ...]"
    T = int(allweather_data.shape[2])
    device = allweather_data.device

    if T % num_classes != 0:
        raise ValueError(f"T={T} 无法被 num_classes={num_classes} 整除，请改用 make_frame_labels_by_blocks")

    block_len = T // num_classes
    labels = []
    for cid in range(num_classes):
        labels.append(torch.full((block_len,), cid, dtype=dtype, device=device))
    return torch.cat(labels, dim=0)  # [T]

def make_frame_labels_by_blocks(allweather_data: torch.Tensor,
                                block_lengths: List[int],
                                dtype=torch.long) -> torch.Tensor:
    """
    为不等长类别块构造 1D 帧标签，长度为 T。block_lengths 按类别顺序给出每类帧数。
    例如：block_lengths=[12, 8, 7] => 0*12, 1*8, 2*7
    """
    # assert allweather_data.dim() >= 3, "allweather_data 需要至少有 3 个维度 [N, C, T, ...]"
    # T = int(allweather_data.shape[2])
    T = int(allweather_data.shape[0])
    # device = allweather_data.device

    if sum(block_lengths) != T:
        raise ValueError(f"block_lengths 之和为 {sum(block_lengths)} 与 T={T} 不一致")

    labels = []
    for cid, L in enumerate(block_lengths):
        # labels.append(torch.full((int(L),), int(cid), dtype=dtype, device=device))
        labels.append(torch.full((int(L),), int(cid), dtype=dtype))
    return torch.cat(labels, dim=0)  # [T]

def apply_same_shuffle_1d(labels_1d: torch.Tensor, new_order) -> torch.Tensor:
    """
    对 1D 标签应用与数据相同的打乱顺序。new_order 可以是 numpy 或 torch。
    返回与 labels_1d 同设备、同 dtype 的标签（已打乱）。
    """
    new_order = _to_long_tensor(new_order, device=labels_1d.device)
    return labels_1d[new_order]  # [T]

class FrameCategoryTracker:
    """
    轻量级的帧类别查询器：传入已打乱的 1D 标签张量，即可用 get 查询帧类别。
    """
    def __init__(self, labels_shuffled: torch.Tensor, id2name: Optional[Dict[int, str]] = None):
        self.labels_shuffled = labels_shuffled.to(dtype=torch.long)
        # 默认把类别 id 转成字符串；建议传入 id2name={0:"雨滴",1:"雪花",2:"雨雾"} 之类的映射
        uniq = torch.unique(self.labels_shuffled).tolist()
        self.id2name = id2name or {int(i): str(int(i)) for i in uniq}

    def get(self, frame_idx: int) -> Tuple[int, str]:
        """
        返回 (category_id, category_name)
        """
        cid = int(self.labels_shuffled[frame_idx].item())
        return cid, self.id2name.get(cid, str(cid))