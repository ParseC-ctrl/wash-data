#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 说明：
# - 将多个目录/文件中的 CSV 合并为单一文件，只保留第一个文件的表头。
# - 为速度，采用二进制拼接；支持多进程并行读取，其它均保持顺序写入。
# - 注意：read_* 函数当前一次性读取整个文件到内存，适合单文件较大的批量处理场景；若内存紧张，可改为分块流式读取。

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Tuple

def list_csv_files(inputs: List[str], recursive: bool = True) -> List[str]:
    """
    列出输入路径中的所有 CSV 文件。
    - 支持传入：单个字符串路径 或 路径列表（目录/文件均可）。
    - 目录将按 recursive 控制是否递归搜索 *.csv。
    - 返回值：按路径名排序且去重后的 CSV 文件列表。
    """
    # 允许传入单个字符串路径（避免把字符串当作可迭代逐字符遍历）
    if isinstance(inputs, (str, os.PathLike)):
        inputs = [str(inputs)]
    files = []
    for p in inputs:
        path = Path(p).expanduser()
        if path.is_dir():
            if recursive:
                files.extend([str(fp) for fp in path.rglob("*.csv") if fp.is_file()])
            else:
                files.extend([str(fp) for fp in path.glob("*.csv") if fp.is_file()])
        elif path.is_file() and path.suffix.lower() == ".csv":
            files.append(str(path))
    # 去重 + 排序（按路径稳定输出）
    files = sorted(set(files))
    return files

def read_tail_bytes_skip_header(file_path: str) -> bytes:
    """
    读取文件的“非表头部分”（首行之后的所有字节）。
    - 二进制方式读取，找到第一个换行符后返回其后的全部内容。
    - 注意：一次性读入整个文件到内存；如需更省内存，可改为分块查找首行换行符并流式写出。
    """
    # 纯二进制读取，定位首个换行符后直接返回剩余字节
    with open(file_path, "rb") as f:
        data = f.read()
    nl = data.find(b"\n")
    if nl == -1:
        return b""
    return data[nl + 1 :]

def read_header_and_tail(file_path: str) -> Tuple[bytes, bytes]:
    """
    同时读取“表头行（含换行）”与“非表头部分”。
    - 返回 (header_bytes, tail_bytes)。
    - 若文件没有换行，视为只有表头或空。
    """
    with open(file_path, "rb") as f:
        data = f.read()
    nl = data.find(b"\n")
    if nl == -1:
        # 无换行，视作只有表头或空
        return data, b""
    return data[: nl + 1], data[nl + 1 :]

def concat_csv(
    inputs: List[str],
    output: str,
    workers: int = max(1, os.cpu_count() // 2),
    preserve_order: bool = True,
) -> None:
    """
    合并 CSV 主流程：
    - inputs: 目录或文件路径（可混合/可多个），支持字符串或列表。
    - output: 输出文件路径（若上级目录不存在会自动创建）。
    - workers: 并行进程数（>1 启用多进程），建议为 CPU 的一半或略少。
    - preserve_order: 是否保持输入顺序（当前实现为顺序写出）。
    流程：
      1) 收集并排序所有 CSV。
      2) 读取第一个文件的表头并写入输出，再写入其余内容。
      3) 其他文件仅写入去表头后的内容。
      4) 可并行读取，加速 I/O，主进程顺序写出，降低内存峰值。
    """
    csv_files = list_csv_files(inputs)
    if not csv_files:
        raise SystemExit("未找到任何 CSV 文件。请检查输入路径。")

    # 取首文件的表头与内容（只保留首个表头）
    first = csv_files[0]
    header, first_tail = read_header_and_tail(first)

    # 输出文件准备（保证目录存在）
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 开始写入
    with open(out_path, "wb") as out_f:
        # 写入表头
        out_f.write(header)
        # 写入首文件的非表头内容
        if first_tail:
            out_f.write(first_tail)

        # 其余文件的内容（跳过表头）
        rest = csv_files[1:]
        if not rest:
            return

        # 限制并发数量，避免一次性占用过多内存
        workers = max(1, int(workers))
        if workers == 1:
            # 串行（单盘顺序 I/O 时也很快）
            for fp in rest:
                out_f.write(read_tail_bytes_skip_header(fp))
        else:
            # 多进程并行读取，主进程顺序写入
            # 为减少峰值内存，保持输入顺序（map 而不是 unordered）
            with ProcessPoolExecutor(max_workers=workers) as ex:
                for chunk in ex.map(read_tail_bytes_skip_header, rest, chunksize=1):
                    if chunk:
                        out_f.write(chunk)

def main():
    """
    直接在代码中指定输入/输出与并发度的入口。
    - 这里示例：合并 `./输出结果/2021年数据` 目录下全部 CSV，输出到同级目录。
    """
    inputs = [os.path.join(os.path.dirname(__file__), './输出结果/2017年数据')]
    output = os.path.join(os.path.dirname(__file__), './输出结果/2017年裁判文书数据.csv')
    # 并发：CPU 核心数减 1（至少为 1）
    workers = max(1, (os.cpu_count() or 4) - 1)
    concat_csv(
        inputs=inputs,
        output=output,
        workers=workers,
        preserve_order=True,
    )

if __name__ == "__main__":
    main()