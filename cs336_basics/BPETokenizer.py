import regex as re
from collections import Counter
from typing import List, Tuple, Dict
from multiprocessing import Process, Queue
from collections import defaultdict
import os
from typing import BinaryIO
from typing import Iterable, Iterator

def find_chunk_boundaries(
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def kill_special(input_file, special_token):
    delimeter = "|"
    escaped_special_tokens = [re.escape(token) for token in special_token]  # 获得特殊符号的转义字符
    pattern = delimeter.join(escaped_special_tokens)
    split_parts = re.split(pattern, input_file)
    return split_parts


def pre_tokenizer(texts: str, special_token):
    parts = kill_special(texts, special_token)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    tokens_list = []
    for part in parts:
        string_list = re.findall(PAT, part)  # 分出来的都是字 some , text , that
        part_list = [s.encode('utf-8') for s in string_list]
        tokens_list.append(
            part_list)  # [[b'This', b' is', b' some', b' text', b'. '],      [b' Here', b' is', b' another', b'!']]
    # 展平成一个 token 序列
    tokens = [token for tokens in tokens_list for token in tokens]
    return tokens

def replace_seq_to_new_id(seq, pair, new_id):
    a, b = pair
    out = []
    i = 0
    while i < len(seq):
        if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
            out.append(new_id)  # 把原先的 (104,106) 给替换为(256)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


def worker(chunk, q, special_token):
    pretokens = pre_tokenizer(chunk, special_token)
    q.put(pretokens)


def train_bpe(input_path, vocab_size, special_token):
    vocab = {x: bytes([x]) for x in range(256)}
    merges = []
    chunk_list = []
    special_token = special_token or []
    num_merges = max(vocab_size - len(special_token) - 256, 0)
    for i, token in enumerate(special_token):
        vocab[256 + i] = token.encode('utf-8')

    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunk_list.append(chunk)
        # Run pre-tokenization on your chunk and store the counts for each pre-token

    # Parallelizing pretokenization
    pre_token_list = []
    process = []
    q = Queue()
    for chunk in chunk_list:
        p = Process(target=worker, args=(chunk, q, special_token))
        p.start()
        process.append(p)
    pre_token_list = [q.get() for _ in process]

    for p in process:
        p.join()

    pretokens = [token for tokens in pre_token_list for token in
                 tokens]  # 里面的是每一个的单词 pretokens = [[97, 98, 256, 99]  # 'a','b','<|endoftext|>','c'] 特别长的句子
    counts = defaultdict(int)  # count 的值是这个pair 出现的次数
    index = defaultdict(set)
    for j, token in enumerate(pretokens):  # token 一个单词
        for index1, index2 in zip(token, token[1:]):
            counts[index1, index2] += 1
            index[index1, index2].add(j)


    for i in range(num_merges):
        # Prefer lexicographically greater pair
        # Example: max([("A", "B"), ("A", "C"), ("B", "ZZ"), ("BA", "A")]) = ('BA', 'A')
        max_pair = max(
            counts.items(),
            key=lambda x: (
                x[1],  # 先按照索引来查找
                vocab[x[0][0]].decode("utf-8", errors="ignore"),
                vocab[x[0][1]].decode("utf-8", errors="ignore")
            )
        )[0]
        index1, index2 = max_pair
        new_index = 256 + i + len(special_token)
        vocab[new_index] = vocab[index1] + vocab[index2]
        merges.append((vocab[index1], vocab[index2]))
        merge(counts, index, pretokens, max_pair, new_index)
    return (vocab, merges)


def merge(counts: dict[tuple[int, int], int], index_dict: dict[tuple[int, int], set[int]], pretokens: list[list[int]],
          max_pair: (int, int), new_index: int):
    """Merge the pairs with highest frequency and update counts, index_dict"""
    index_set = index_dict[max_pair]

    for i in index_set:
        pretoken = pretokens[i]
        new_pretoken = []

        pos_list = []  # Store positions of max_pair for each new pretoken after merge
        pos = 0
        j = 0

        # Replace max_pair with new_index in each pretoken
        while j < len(pretoken):
            if (j < len(pretoken) - 1) and ((pretoken[j], pretoken[j + 1]) == max_pair):
                new_pretoken.append(new_index)
                pos_list.append(pos)
                j += 2
            else:
                new_pretoken.append(pretoken[j])
                j += 1
            pos += 1

        # Update counts and index_dict
        for pos in pos_list:
            counts[max_pair] -= 1

            if pos > 0:
                if new_pretoken[pos - 1] == new_index:
                    counts[(max_pair[1], max_pair[0])] -= 1
                else:
                    counts[(new_pretoken[pos - 1], max_pair[0])] -= 1

                counts[(new_pretoken[pos - 1], new_pretoken[pos])] += 1
                index_dict[(new_pretoken[pos - 1], new_pretoken[pos])].add(i)

            if pos < len(new_pretoken) - 1:
                if new_pretoken[pos + 1] == new_index:
                    counts[(max_pair[1], max_pair[0])] -= 1
                else:
                    counts[(max_pair[1], new_pretoken[pos + 1])] -= 1

                counts[(new_pretoken[pos], new_pretoken[pos + 1])] += 1
                index_dict[(new_pretoken[pos], new_pretoken[pos + 1])].add(i)

        pretokens[i] = new_pretoken


class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str]| None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges"""
        raise NotImplementedError

    def encode(self, text:str) -> list[int]:
        """Encode an input text into a sequence of token IDs."""

        vocab_reversed = {v: k for k, v in self.vocab.items()}  # bytes: int
        byte_pretokens = pre_tokenizer(text, self.special_tokens, drop_special_token=False)   # list[bytes]
        byte_special_tokens = [token.encode('utf-8') for token in self.special_tokens]
        pretokens = []  # list[list[int]]

        # Convert pretokens from bytes to list[int] by vocab
        for i, pretoken in enumerate(byte_pretokens):

            new_pretoken = []

            if pretoken in byte_special_tokens:
                index = vocab_reversed[pretoken]
                new_pretoken.append(index)
            else:
                for b in pretoken:
                    index = vocab_reversed[bytes([b])]
                    new_pretoken.append(index)

            pretokens.append(new_pretoken)

        # Merge
        for i, pretoken in enumerate(pretokens):
            for merge in self.merges:
                new_pretoken = []
                new_index = vocab_reversed[merge[0] + merge[1]]
                j = 0
                while j < len(pretoken):
                    if (j < len(pretoken)-1) and ((self.vocab[pretoken[j]], self.vocab[pretoken[j+1]]) == merge):
                        new_pretoken.append(new_index)
                        j += 2
                    else:
                        new_pretoken.append(pretoken[j])
                        j += 1

                pretoken = new_pretoken

            pretokens[i] = pretoken

        tokens = [token for pretoken in pretokens for token in pretoken]
        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle),
        return a generator that lazily yields token IDs.
        This is required for memory-eﬀicient tokenization of large files
        that we cannot directly load into memory.
        """
        for line in iterable:
            for idx in self.encode(line):
                yield idx


    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text."""
        tokens = bytes()
        vocab_size = len(self.vocab)
        replacement_char = "\uFFFD"

        for token_id in ids:
            if token_id < vocab_size:
                token = self.vocab[token_id]    # bytes
            else:
                token = bytes(replacement_char, encoding='utf-8')   # Replace tokens with Unicode replacement characters if index out of bounds

            tokens += token
        decoded = tokens.decode(encoding='utf-8', errors='replace')

        return decoded


if __name__ == "__main__":
    current_path =os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(current_path,"..","data","TinyStoriesV2-GPT4-valid.txt")
    special_tokens = ["<|endoftext|>"]
    symbol_table, _ = train_bpe(input_path,vocab_size=280,special_token=special_tokens)
    for i in range(256, 280):
        if i in symbol_table:
            print(i, "→", symbol_table[i])