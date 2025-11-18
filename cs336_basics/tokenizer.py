from collections.abc import Iterable, Iterator
import regex as re
import pickle

from cs336_basics import train_bpe


class Tokenizer:
    
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        
        self.vocab_inv = {v: k for k, v in vocab.items()}
        
        self.merges = merges
        
        self.merges_dict = {merge: i for i, merge in enumerate(merges)}
        
        self.encode_cache = {}
        self.cache_hits = 0 
        
        self.pretokenize_pattern = re.compile(train_bpe.PAT)
        
        if special_tokens:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
            
            self.special_pattern = "(" + "|".join(
                re.escape(k) for k in self.special_tokens
            ) + ")"
            
            next_id = max(self.vocab.keys()) + 1
            for token in special_tokens:
                token_bytes = token.encode("UTF-8")
                if token_bytes not in self.vocab_inv:
                    self.vocab[next_id] = token_bytes
                    self.vocab_inv[token_bytes] = next_id
                    next_id += 1
        else:
            self.special_tokens = None
            self.special_pattern = None
    
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None
    ):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        
        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]:
        if not self.special_tokens:
            return self._encode_chunk(text)
        
        special_chunks = re.split(self.special_pattern, text)
        
        ids = []
        for part in special_chunks:
            if part in self.special_tokens:
                ids.append(self.vocab_inv[part.encode("UTF-8")])
            else:
                ids.extend(self._encode_chunk(part))
        
        return ids
    
    def _encode_chunk(self, text: str) -> list[int]:
        pretokens = self._pretokenize(text)
        
        pretoken_reprs: dict[str, list[bytes]] = {}
        
        ids = []
        
        for p in pretokens:
            if p in self.encode_cache:
                ids.extend(self.encode_cache[p])
                self.cache_hits += 1
            else:
                if p not in pretoken_reprs:
                    match_bytes = list(bytes([b]) for b in p.encode("UTF-8"))
                    pretoken_reprs[p] = match_bytes
                
                merged = self._merge_subword(pretoken_reprs[p])
                
                token_ids = [self.vocab_inv[subword] for subword in merged]
                
                self.encode_cache[p] = token_ids
                
                ids.extend(token_ids)
        
        return ids
    
    def _merge_subword(self, rep: list[bytes]) -> list[bytes]:
        while True:
            best_rank = float("inf")
            best_idx = None
            
            for i in range(len(rep) - 1):
                pair = (rep[i], rep[i + 1])
                rank = self.merges_dict.get(pair)
                
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_idx = i
            
            if best_idx is None:
                return rep
            
            merged = rep[best_idx] + rep[best_idx + 1]
            rep = rep[:best_idx] + [merged] + rep[best_idx + 2:]
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id
    
    def decode(self, ids: list[int]) -> str:
        text_bytes = b"".join(self.vocab[id] for id in ids)
        
        return text_bytes.decode("UTF-8", errors="replace")
    
    def _pretokenize(self, text: str) -> list[str]:
        pretokens: list[str] = []
        
        for match in self.pretokenize_pattern.finditer(text):
            match_str = match.group()
            pretokens.append(match_str)
        
        return pretokens


if __name__ == "__main__":
    print("=" * 70)
    print("🧪 测试Tokenizer")
    print("=" * 70)
    
    tokenizer = Tokenizer.from_files(
        vocab_filepath="./out/ts-valid-vocab.pkl",
        merges_filepath="./out/ts-valid-merges.pkl",
        special_tokens=["<|endoftext|>"]
    )
    
    test_text = "Once upon a time, there was a little girl."
    print(f"\n原始文本: {test_text}")
    
    token_ids = tokenizer.encode(test_text)
    print(f"Token IDs: {token_ids}")
    print(f"Token数量: {len(token_ids)}")
    
    decoded = tokenizer.decode(token_ids)
    print(f"解码文本: {decoded}")
    print(f"匹配: {test_text == decoded}")
    
    num_bytes = len(test_text.encode('utf-8'))
    compression_ratio = num_bytes / len(token_ids)
    print(f"压缩比: {compression_ratio:.2f} bytes/token")
    
    print("\n" + "=" * 70)