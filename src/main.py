#!/usr/bin/env python3
"""
compress_cli.py — Command-line file compressor using Huffman coding.

USAGE:
    python3 compress_cli.py compress   input.txt  output.huff
    python3 compress_cli.py decompress output.huff restored.txt
    python3 compress_cli.py stats      input.txt

If the C extension `fast_huffman` has been built (see setup.py), byte
counting and bit-packing run in C for a big speed boost on large files.
Otherwise this transparently falls back to the pure-Python versions in
huffman.py — same output format either way.
"""

import sys
import os
import time
import struct
import pickle
from collections import Counter

import huffman

# --- Try to use the compiled C extension; fall back to pure Python. ---
try:
    import fast_huffman
    HAVE_C_EXT = True
except ImportError:
    HAVE_C_EXT = False


def count_bytes(data: bytes) -> dict:
    if HAVE_C_EXT:
        return fast_huffman.count_bytes(data)
    return dict(Counter(data))


def bits_to_bytes(bitstring: str) -> bytes:
    if HAVE_C_EXT:
        return fast_huffman.pack_bits(bitstring)
    return huffman._bits_to_bytes(bitstring)


def bytes_to_bits(data: bytes) -> str:
    if HAVE_C_EXT:
        return fast_huffman.unpack_bits(data)
    return huffman._bytes_to_bits(data)


def compress_file(in_path: str, out_path: str):
    with open(in_path, "rb") as f:
        data = f.read()

    t0 = time.time()
    freq_table = count_bytes(data) if data else {}

    if data:
        tree = huffman.build_tree(freq_table)
        codebook = huffman.build_codes(tree)
        bitstring = "".join(codebook[b] for b in data)
        encoded_body = bits_to_bytes(bitstring)
    else:
        encoded_body = b"\x00"

    header = pickle.dumps(freq_table)
    blob = huffman.MAGIC + struct.pack(">I", len(header)) + header + encoded_body

    with open(out_path, "wb") as f:
        f.write(blob)
    elapsed = time.time() - t0

    orig_size = len(data)
    comp_size = len(blob)
    ratio = (1 - comp_size / orig_size) * 100 if orig_size else 0
    backend = "C extension" if HAVE_C_EXT else "pure Python"

    print(f"Compressed using {backend}")
    print(f"  {in_path} ({orig_size:,} bytes) -> {out_path} ({comp_size:,} bytes)")
    print(f"  Reduction: {ratio:.1f}%   Time: {elapsed:.3f}s")


def decompress_file(in_path: str, out_path: str):
    with open(in_path, "rb") as f:
        blob = f.read()

    t0 = time.time()
    if blob[:4] != huffman.MAGIC:
        raise ValueError("Not a valid .huff file (bad magic header)")

    header_len = struct.unpack(">I", blob[4:8])[0]
    freq_table = pickle.loads(blob[8:8 + header_len])
    encoded_body = blob[8 + header_len:]

    if not freq_table:
        data = b""
    else:
        tree = huffman.build_tree(freq_table)
        if tree.right is None:  # single distinct byte edge case
            data = bytes([tree.left.byte]) * tree.freq
        else:
            bits = bytes_to_bits(encoded_body)
            out = bytearray()
            node = tree
            for bit in bits:
                node = node.left if bit == "0" else node.right
                if node.byte is not None:
                    out.append(node.byte)
                    node = tree
            data = bytes(out)

    with open(out_path, "wb") as f:
        f.write(data)
    elapsed = time.time() - t0

    backend = "C extension" if HAVE_C_EXT else "pure Python"
    print(f"Decompressed using {backend}")
    print(f"  {in_path} -> {out_path} ({len(data):,} bytes)   Time: {elapsed:.3f}s")


def show_stats(path: str):
    with open(path, "rb") as f:
        data = f.read()
    freq = count_bytes(data)
    tree = huffman.build_tree(freq) if data else None
    codebook = huffman.build_codes(tree) if tree else {}

    print(f"File: {path}  ({len(data):,} bytes)")
    print(f"Distinct byte values: {len(freq)}")
    print(f"Backend available: {'C extension' if HAVE_C_EXT else 'pure Python only'}")
    if codebook:
        avg_len = sum(len(codebook[b]) * c for b, c in freq.items()) / len(data)
        print(f"Average code length: {avg_len:.2f} bits/byte  (raw = 8 bits/byte)")
        top5 = sorted(freq.items(), key=lambda x: -x[1])[:5]
        print("Most frequent bytes:")
        for b, c in top5:
            char = chr(b) if 32 <= b < 127 else f"\\x{b:02x}"
            print(f"  byte {b:3d} ('{char}')  count={c:<8} code={codebook[b]}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "compress" and len(sys.argv) == 4:
        compress_file(sys.argv[2], sys.argv[3])
    elif cmd == "decompress" and len(sys.argv) == 4:
        decompress_file(sys.argv[2], sys.argv[3])
    elif cmd == "stats" and len(sys.argv) == 3:
        show_stats(sys.argv[2])
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
