import os
import heapq
from typing import BinaryIO
import regex as re
import collections
import multiprocessing as mp
import time
import pickle
from functools import reduce

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

class ReverseLexOrderPair:
    def __init__(self, pair):
        self.pair = pair

    def __lt__(self, other: 'ReverseLexOrderPair') -> bool:
        return self.pair > other.pair
    
    def __eq__(self, other: 'ReverseLexOrderPair') -> bool:
        return self.pair == other.pair
    

def find_chunk_boundaries(
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes
) -> list[int]:
    assert isinstance(split_special_token, bytes), "特殊token必须是bytes类型"
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        pos = chunk_boundaries[bi]
        file.seek(pos)

        while True:
            mini_chunk = file.read(mini_chunk_size)

            if not mini_chunk:
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = pos + found_at
                break
            pos += mini_chunk_size
    
    return sorted(set(chunk_boundaries))

def pre_tokenize_chunk(
    chunk: str,
    special_pattern: re.Pattern | None
) -> dict[tuple[bytes], int]:
    
    freqs: dict[tuple[bytes], int] = {}

    sub_chunks = special_pattern.split(chunk) if special_pattern else [chunk]
    for sub_chunk in sub_chunks:
        for match in PAT.finditer(sub_chunk):
            word = match.group()

            match_bytes = tuple(bytes([b]) for b in word.encode("UTF-8"))

            freqs[match_bytes] = freqs.get(match_bytes, 0) + 1
    return freqs

def merge_freq_dicts(
    dict1: dict[tuple[bytes], int],
    dict2: dict[tuple[bytes], int]
) -> dict[tuple[bytes], int]:
    result = dict1.copy()
    for key, value in dict2.items():
        result[key] = result.get(key, 0) + value
    return result

def pre_tokenize(
    input_path: str,
    special_tokens: list[str]
) -> dict[tuple[bytes], int]:
    num_processes = mp.cpu_count()
    ctx = mp.get_context('spawn')
    pool = ctx.Pool(processes=num_processes)

    chunk_freqs = []
    special_pattern = None
    if special_tokens:
        pattern_str = "|".join(re.escape(tok) for tok in special_tokens)
        special_pattern = re.compile(pattern_str)

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk_bytes = f.read(end - start)
            chunk_str = chunk_bytes.decode("utf-8", errors="ignore")
            result = pool.apply_async(
                pre_tokenize_chunk,
                    (chunk_str, special_pattern)
                
            )
            chunk_freqs.append(result)

    pool.close()
    pool.join()
    freq_dicts = [res.get() for res in chunk_freqs]
    combined_freqs = reduce(merge_freq_dicts, freq_dicts, {})
    
    return combined_freqs

def get_pair_freqs(
    freqs: dict[tuple[bytes], int]
) -> tuple[dict[tuple[bytes, bytes], int], dict[tuple[bytes, bytes], set[tuple[bytes]]]]:
    pair_freqs: dict[tuple[bytes, bytes], int] = collections.defaultdict(int)
    pairs_to_keys: dict[tuple[bytes, bytes], set[tuple[bytes]]] = collections.defaultdict(set)
    for symbols, freq in freqs.items():
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])

            pair_freqs[pair] += freq

            pairs_to_keys[pair].add(symbols)
    return pair_freqs, pairs_to_keys

def build_new_repr(
    old_repr: tuple[bytes],
    pair: tuple[bytes, bytes]
) -> tuple[bytes]:
    new_symbols = []
    i = 0
    while i < len(old_repr):
        if(i < len(old_repr) - 1) and old_repr[i] == pair[0] and old_repr[i + 1] == pair[1]:
            new_symbols.append(old_repr[i] + old_repr[i + 1])
            i += 2
        else:
            new_symbols.append(old_repr[i])
            i += 1
            
    return tuple(new_symbols)

def merge(
    freqs: dict[tuple[bytes], int],
    pair_freqs: dict[tuple[bytes, bytes], int],
    pairs_to_keys: dict[tuple[bytes, bytes], set[tuple[bytes]]],
    pair: tuple[bytes, bytes]
) -> set[tuple[bytes, bytes]]:
    
    changed_pairs = set()
    key_to_modify = pairs_to_keys[pair].copy()
    for old_key in key_to_modify:
        old_freq = freqs.pop(old_key)

        new_key = build_new_repr(old_key, pair)

        for i in range(len(old_key) - 1):
            left, right = old_key[i], old_key[i + 1]

            pair_freqs[left, right] -= old_freq
            changed_pairs.add((left, right))
            if pair_freqs[left, right] <= 0:
                del pair_freqs[left, right]
            pairs_to_keys[left, right].discard(old_key)

        for i in range(len(new_key) - 1):
            left, right = new_key[i], new_key[i + 1]
            
            pair_freqs[left, right] += old_freq
            changed_pairs.add((left, right)) 
            pairs_to_keys[left, right].add(new_key)

        freqs[new_key] = freqs.get(new_key, 0) + old_freq
    pairs_to_keys[pair] = set()

    return changed_pairs

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    merges_outpath: str = None,
    vocab_outpath: str = None,
    **kwargs  # 接受额外参数（兼容测试框架）
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    train_start_time = time.time()

    # ========================================================================
    # 阶段1: 初始化词汇表
    # ========================================================================
    print("=" * 70)
    print("🎓 BPE Tokenizer Training")
    print("=" * 70)
    
    # Special tokens放在最前面，然后是256个字节
    initial_tokens = [tok.encode("UTF-8") for tok in special_tokens]
    initial_tokens += [bytes([i]) for i in range(256)]
    
    # 创建词汇表: {0: b'<|endoftext|>', 1: b'\x00', 2: b'\x01', ...}
    vocab = {i: token for i, token in enumerate(initial_tokens)}
    merges = []
    
    print(f"✅ 初始词汇表大小: {len(vocab)}")
    print(f"   - Special tokens: {len(special_tokens)}")
    print(f"   - Byte tokens: 256")
    
    # ========================================================================
    # 阶段2: 预分词
    # ========================================================================
    print("\n📖 阶段2: 预分词（多进程并行）")
    start_time = time.time()
    
    freqs = pre_tokenize(input_path, special_tokens)
    
    elapsed = time.time() - start_time
    print(f"✅ 预分词完成！耗时: {elapsed:.2f}秒")
    print(f"   - Unique words: {len(freqs):,}")
    print(f"   - Total tokens: {sum(freqs.values()):,}")
    
    # ========================================================================
    # 阶段3: 构建pair频率表
    # ========================================================================
    print("\n🔢 阶段3: 构建pair频率表")
    start_time = time.time()
    
    pair_freqs, pairs_to_keys = get_pair_freqs(freqs)
    
    # 使用最小堆来高效管理pairs
    # 堆中存储: (-frequency, ReverseLexOrderPair, actual_pair)
    # 为什么用负频率？因为Python的heapq是最小堆，负频率让最大频率的排在前面
    pair_heap = []
    for p, f in pair_freqs.items():
        if f > 0:
            heapq.heappush(pair_heap, (-f, ReverseLexOrderPair(p), p))
    
    elapsed = time.time() - start_time
    print(f"✅ 完成！耗时: {elapsed:.2f}秒")
    print(f"   - Unique pairs: {len(pair_freqs):,}")
    
    # ========================================================================
    # 阶段4: BPE合并循环
    # ========================================================================
    print("\n🔄 阶段4: BPE合并循环")
    
    n_initial_tokens = len(initial_tokens)
    n_merges = vocab_size - n_initial_tokens
    
    print(f"   目标: 执行 {n_merges:,} 次合并")
    
    start_time = time.time()
    
    for i in range(n_initial_tokens, n_initial_tokens + n_merges):
        # 检查堆是否为空
        if not pair_heap:
            print(f"⚠️  堆已空，提前停止")
            break
        
        # 从堆中找到真正的最高频pair
        # 使用lazy deletion: 堆中可能有过时的entries
        while pair_heap:
            neg_freq, _, top_pair = heapq.heappop(pair_heap)
            freq = -neg_freq
            
            # 验证这个pair的频率是否仍然正确
            if pair_freqs.get(top_pair, 0) == freq:
                pair = top_pair
                break
            
            # 频率变了但仍然>0，重新加入堆
            if top_pair in pair_freqs and pair_freqs[top_pair] > 0:
                heapq.heappush(
                    pair_heap,
                    (-pair_freqs[top_pair], ReverseLexOrderPair(top_pair), top_pair)
                )
        else:
            # while循环正常结束（没有break），说明堆空了
            print(f"⚠️  没有更多可合并的pairs")
            break
        
        # 双重验证
        if pair_freqs.get(pair, 0) <= 0:
            print(f"⚠️  Pair频率<=0，停止")
            break
        
        # 将合并后的token添加到词汇表
        vocab[i] = pair[0] + pair[1]
        merges.append(pair)
        
        # 执行合并，更新所有数据结构
        changed_pairs = merge(freqs, pair_freqs, pairs_to_keys, pair)
        
        # 将频率变化的pairs重新加入堆
        for cp in changed_pairs:
            if cp in pair_freqs and pair_freqs[cp] > 0:
                heapq.heappush(
                    pair_heap,
                    (-pair_freqs[cp], ReverseLexOrderPair(cp), cp)
                )
        
        # 打印进度（每100次或最后一次）
        if ((i > n_initial_tokens) and ((i - n_initial_tokens + 1) % 100 == 0)) or \
           (i == n_initial_tokens + n_merges - 1):
            elapsed = time.time() - start_time
            progress = (i - n_initial_tokens + 1) / n_merges * 100
            print(f"   📈 {i - n_initial_tokens + 1}/{n_merges} "
                  f"({progress:.1f}%) - {elapsed:.2f}s")
    
    elapsed = time.time() - start_time
    print(f"✅ 合并完成！耗时: {elapsed:.2f}秒")
    
    # ========================================================================
    # 完成
    # ========================================================================
    total_time = time.time() - train_start_time
    print("\n" + "=" * 70)
    print("✨ 训练完成！")
    print("=" * 70)
    print(f"⏱️  总时间: {total_time:.2f}秒")
    print(f"📚 最终vocab大小: {len(vocab)}")
    print(f"🔀 合并次数: {len(merges)}")
    print("=" * 70)
    
    # 可选: 保存到文件
    if merges_outpath:
        write_merges(merges, merges_outpath)
    if vocab_outpath:
        write_vocab(vocab, vocab_outpath)
    
    return vocab, merges


# ============================================================================
# 辅助函数: 保存
# ============================================================================

def write_merges(merges, outpath):
    """将merges保存为pickle文件"""
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "wb") as f:
        pickle.dump(merges, f)
    print(f"💾 Saved {len(merges)} merges to {outpath}")


def write_vocab(vocab, outpath):
    """将vocab保存为pickle文件"""
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "wb") as f:
        pickle.dump(vocab, f)  # ✅ 添加 .dump(vocab, f)
    print(f"💾 Saved vocabulary with {len(vocab)} tokens to {outpath}")

# ============================================================================
# 主程序入口（用于测试）
# ============================================================================

if __name__ == "__main__":
    # 示例：训练一个小的tokenizer
    vocab, merges = train_bpe(
        input_path="./data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
        merges_outpath="./out/ts-valid-merges.pkl",
        vocab_outpath="./out/ts-valid-vocab.pkl",
    )
    print(f"\n✅ 训练完成！vocab size = {len(vocab)}, merges = {len(merges)}")