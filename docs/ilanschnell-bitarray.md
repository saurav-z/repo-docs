# Bitarray: Efficient Arrays of Booleans in Python

**Bitarray provides a high-performance, memory-efficient way to represent and manipulate arrays of booleans, offering significant advantages over standard Python lists for boolean data.** ([Original Repo](https://github.com/ilanschnell/bitarray))

## Key Features

*   **Endianness Control:** Specify little- or big-endian representation for each bitarray.
*   **Sequence Operations:** Supports slicing (including assignment and deletion), `+`, `*`, `+=`, `*=`, `in`, and `len()`.
*   **Bitwise Operations:** Includes `~`, `&`, `|`, `^`, `<<`, `>>` and in-place equivalents.
*   **Variable-Length Prefix Codes:** Fast encoding and decoding methods.
*   **Buffer Protocol Support:** Allows import/export of buffers, enabling interaction with other objects like memory-mapped files and NumPy ndarrays.
*   **Data Conversion:** Packing and unpacking to other binary data formats.
*   **Serialization:** Pickling and unpickling of bitarray objects.
*   **Immutable Frozen Bitarrays:** Hashable `frozenbitarray` objects for use as dictionary keys.
*   **Search and Analysis:** Sequential search, counting functions, interval analysis and more.
*   **Type Hinting:** Supports type hinting for improved code clarity.
*   **Extensive Testing:** Comprehensive test suite with over 500 unit tests.
*   **Utility Module (`bitarray.util`):**
    *   Hexadecimal string conversion
    *   Random bitarray generation
    *   Pretty printing
    *   Conversion to/from integers
    *   Huffman code generation
    *   Sparse bitarray compression/decompression
    *   Serialization/deserialization
    *   Count functions
    *   Various utilities

## Installation

Install using pip:

```bash
pip install bitarray
```

Verify installation:

```bash
python -c 'import bitarray; bitarray.test()'
```

## Usage

Bitarrays behave like lists, but are optimized for boolean data.
```python
from bitarray import bitarray

# Create a bitarray
a = bitarray()         # Empty bitarray
a.append(1)           # Append a bit
a.extend([1, 0])      # Extend with a list
print(a)              # bitarray('110')
```

## Detailed Reference
For a comprehensive overview of methods and functions, please refer to the extensive documentation within the original README, which includes information about bitwise operations, endianness, buffer protocol support, and the utility module.
```