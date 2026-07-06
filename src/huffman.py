"""
huffman.py — Pure-Python Huffman coding compressor.

This is the "from scratch" implementation so you can see exactly how
Huffman compression works. It's used as the fallback when the C
extension (fast_huffman) isn't built, and as the reference implementation
for learning/tweaking.

HOW HUFFMAN CODING WORKS (quick primer):
1. Count how often each byte (0-255) appears in the file.
2. Build a binary tree: repeatedly take the two least-frequent nodes and
   merge them into a new node (frequency = sum of the two). Do this until
   one node (the root) remains.
3. Walk the tree: going left = bit 0, going right = bit 1. The path from
   root to each byte's leaf is that byte's compressed code.
4. Frequent bytes end up near the root -> short codes.
   Rare bytes end up deep -> long codes.
5. Replace every byte in the file with its code -> smaller output
   (as long as the data isn't already random/uniform).
6. Store the tree (or frequency table) in the output so we can decode later.
"""

import heapq
import pickle
import struct
from collections import Counter


class Node:
    """A node in the Huffman tree. Leaves hold a byte value; internal
    nodes just hold combined frequency and two children."""

    def __init__(self, freq, byte=None, left=None, right=None):
        self.freq = freq
        self.byte = byte      # only set on leaf nodes
        self.left = left
        self.right = right

    # heapq needs a way to compare nodes — compare by frequency only.
    def __lt__(self, other):
        return self.freq < other.freq


def build_tree(freq_table: dict) -> Node:
    """Build the Huffman tree from a {byte: frequency} table."""
    heap = [Node(freq, byte=b) for b, freq in freq_table.items()]
    heapq.heapify(heap)

    # Edge case: file with only one distinct byte value.
    if len(heap) == 1:
        only = heap[0]
        return Node(only.freq, left=only, right=None)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(left.freq + right.freq, left=left, right=right)
        heapq.heappush(heap, merged)

    return heap[0]


def build_codes(node: Node, prefix: str = "", codebook: dict = None) -> dict:
    """Walk the tree and record the bit-string code for every byte."""
    if codebook is None:
        codebook = {}
    if node is None:
        return codebook
    if node.byte is not None:  # leaf
        codebook[node.byte] = prefix or "0"  # handle single-symbol edge case
        return codebook
    build_codes(node.left, prefix + "0", codebook)
    build_codes(node.right, prefix + "1", codebook)
    return codebook


def _bits_to_bytes(bitstring: str) -> bytes:
    """Pack a string of '0'/'1' characters into real bytes.
    Pads the last byte with zeros if needed and records how much padding
    was added so decoding can strip it off exactly."""
    padding = (8 - len(bitstring) % 8) % 8
    bitstring += "0" * padding
    out = bytearray()
    for i in range(0, len(bitstring), 8):
        out.append(int(bitstring[i:i + 8], 2))
    return bytes([padding]) + bytes(out)


def _bytes_to_bits(data: bytes) -> str:
    """Reverse of _bits_to_bytes: unpack bytes back into a '0'/'1' string,
    stripping the padding that was added at compress time."""
    padding = data[0]
    body = data[1:]
    bits = "".join(f"{byte:08b}" for byte in body)
    if padding:
        bits = bits[:-padding]
    return bits


MAGIC = b"HUFF"  # simple file-format signature so we can sanity-check on decompress


def compress(data: bytes) -> bytes:
    """Compress raw bytes -> Huffman-coded bytes with header."""
    if not data:
        freq_table = {}
        codebook = {}
        encoded_body = b"\x00"
    else:
        freq_table = Counter(data)
        tree = build_tree(freq_table)
        codebook = build_codes(tree)
        bitstring = "".join(codebook[b] for b in data)
        encoded_body = _bits_to_bytes(bitstring)

    # Header: MAGIC + pickled freq table (so decoder can rebuild the same tree)
    header = pickle.dumps(freq_table)
    return MAGIC + struct.pack(">I", len(header)) + header + encoded_body


def decompress(blob: bytes) -> bytes:
    """Reverse of compress(): Huffman-coded bytes -> original raw bytes."""
    if blob[:4] != MAGIC:
        raise ValueError("Not a valid huffman-compressed blob (bad magic header)")

    header_len = struct.unpack(">I", blob[4:8])[0]
    freq_table = pickle.loads(blob[8:8 + header_len])
    encoded_body = blob[8 + header_len:]

    if not freq_table:
        return b""

    tree = build_tree(freq_table)
    bits = _bytes_to_bits(encoded_body)

    out = bytearray()
    node = tree
    # Special case: only one distinct byte in original data
    if tree.right is None:
        return bytes([tree.left.byte]) * tree.freq

    for bit in bits:
        node = node.left if bit == "0" else node.right
        if node.byte is not None:
            out.append(node.byte)
            node = tree
    return bytes(out)
