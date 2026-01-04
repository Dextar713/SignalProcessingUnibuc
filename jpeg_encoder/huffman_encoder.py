import heapq
import struct
from collections import Counter


class Node:
    def __init__(self, data: tuple[int | None, int], left=None, right=None):
        self.data = data  # (symbol, frequency)
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.data[1] < other.data[1]


class HuffmanEncoder:
    def __init__(self) -> None:
        self.root: Node | None = None
        self.heap = []
        self.codes: dict[int, str] = {}

    def build_heap(self, freq_map: dict[int, int]) -> None:
        self.heap = [Node((symbol, freq)) for symbol, freq in freq_map.items()]
        heapq.heapify(self.heap)

        if not self.heap:
            return

        if len(self.heap) == 1:
            node = heapq.heappop(self.heap)
            self.root = Node((None, node.data[1]), left=node)
            return

        while len(self.heap) > 1:
            node1 = heapq.heappop(self.heap)
            node2 = heapq.heappop(self.heap)
            merged_freq = node1.data[1] + node2.data[1]
            node_sum = Node(data=(None, merged_freq), left=node1, right=node2)
            heapq.heappush(self.heap, node_sum)

        self.root = heapq.heappop(self.heap)

    def dfs_encode(self, node=None, current_code="") -> None:
        if node is None:
            node = self.root

        if node.left is None and node.right is None:
            if node.data[0] is not None:
                self.codes[node.data[0]] = current_code if current_code else "0"
            return

        if node.left:
            self.dfs_encode(node.left, current_code + "0")
        if node.right:
            self.dfs_encode(node.right, current_code + "1")

    def encode_data(self, data: list[int]) -> tuple[bytes, int]:
        # FIX 2: Encode the actual data stream, not the dictionary
        bit_parts = []
        for symbol in data:
            if symbol not in self.codes:
                raise ValueError(f"Symbol {symbol} found in data but not in tree!")
            bit_parts.append(self.codes[symbol])

        full_bitstring = "".join(bit_parts)
        return bits_to_bytes(full_bitstring)

    def decode_stream(self, bitstring: str) -> list[int]:
        # FIX 3: Decode using Tree Traversal (Much faster than string slicing)
        vals = []
        node = self.root
        if not node:
            return []

        # Optimization: Local variable access is faster
        root = self.root

        for bit in bitstring:
            if bit == "0":
                node = node.left
            else:
                node = node.right
            if node.left is None and node.right is None:
                vals.append(node.data[0])
                node = root

        return vals

    def build_from_config(self, code_map: dict[int, str]) -> None:
        self.root = Node((None, 0))
        for symbol, code in code_map.items():
            node = self.root
            for bit in code:
                if bit == "0":
                    if node.left is None:
                        node.left = Node((None, 0))
                    node = node.left
                else:
                    if node.right is None:
                        node.right = Node((None, 0))
                    node = node.right
            node.data = (symbol, 0)


def bits_to_bytes(bitstring: str) -> tuple[bytearray, int]:
    padding = (8 - len(bitstring) % 8) % 8
    bitstring += "0" * padding
    byte_array = bytearray()
    # int(..., 2) is slow for huge strings.
    # This loop is acceptable for Python, but could be optimized.
    for i in range(0, len(bitstring), 8):
        byte = bitstring[i:i + 8]
        byte_array.append(int(byte, 2))
    return byte_array, padding


def bytes_to_bits(byte_array: bytes, padding: int = 0) -> str:
    bits = [f'{b:08b}' for b in byte_array]
    bitstring = "".join(bits)
    if padding > 0:
        return bitstring[:-padding]
    return bitstring


def pipeline_save(data: list[int], H:int, W:int, output_path='huffman.bin') -> None:
    freq_map = Counter(data)

    encoder = HuffmanEncoder()
    encoder.build_heap(freq_map)
    encoder.dfs_encode()

    encoded_bytes, padding = encoder.encode_data(data)
    with open(output_path, 'wb') as f:
        f.write(b'HUFF')
        f.write(struct.pack('>H', H))
        f.write(struct.pack('>H', W))
        f.write(struct.pack('>H', len(encoder.codes)))  # Num symbols
        for symbol, code in encoder.codes.items():
            f.write(struct.pack('>h', symbol))  # Symbol (4 bytes signed)
            f.write(struct.pack('>B', len(code)))  # Code Len (1 byte)

            # Variable length code bits packed into bytes
            code_bytes, code_pad = bits_to_bytes(code)
            f.write(code_bytes)

        f.write(struct.pack('>B', padding))  # Padding for the main body
        f.write(encoded_bytes)


def pipeline_read(input_path='huffman.bin') -> tuple[list[int], int, int]:
    with open(input_path, 'rb') as f:
        if f.read(4) != b'HUFF':
            raise ValueError("Invalid Header")
        H = struct.unpack('>H', f.read(2))[0]
        W = struct.unpack('>H', f.read(2))[0]
        (num_symbols,) = struct.unpack('>H', f.read(2))
        code_map = {}

        for _ in range(num_symbols):
            symbol = struct.unpack('>h', f.read(2))[0]
            code_len = struct.unpack('>B', f.read(1))[0]

            num_bytes_for_code = (code_len + 7) // 8
            code_raw = f.read(num_bytes_for_code)

            code_bits_full = bytes_to_bits(code_raw)
            code_map[symbol] = code_bits_full[:code_len]

        padding = struct.unpack('>B', f.read(1))[0]
        encoded_data = f.read()

    bitstring = bytes_to_bits(encoded_data, padding)

    encoder = HuffmanEncoder()
    encoder.build_from_config(code_map)
    return encoder.decode_stream(bitstring), H, W