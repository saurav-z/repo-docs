# bitarray: Efficient Arrays of Booleans for Python

**bitarray is a Python library that provides a highly efficient and flexible way to represent and manipulate arrays of booleans, offering performance advantages over standard Python lists.**

[View the bitarray Repository](https://github.com/ilanschnell/bitarray)

## Key Features

*   **Bit-Endianness Control:** Specify little- or big-endian representation for each bitarray object.
*   **Sequence-Like Operations:** Supports slicing (including assignment and deletion), `+`, `*`, `+=`, `*=`, `in` operator, and `len()`.
*   **Bitwise Operations:**  Includes `~`, `&`, `|`, `^`, `<<`, `>>`, and their in-place versions (`&=`, `|=`, `^=`, `<<=`, `>>=`).
*   **Variable Bit Length Prefix Codes:** Fast methods for encoding and decoding.
*   **Buffer Protocol Support:** Enables interaction with other buffer-compatible objects, including memory-mapped files.
*   **Data Conversion:** Packing/unpacking to binary data formats like `numpy.ndarray`.
*   **Serialization:** Supports pickling and unpickling for persistent storage.
*   **Immutable Frozen Bitarrays:** Provides `frozenbitarray` objects for hashable, immutable bit arrays suitable as dictionary keys.
*   **Efficient Algorithms:** Implements sequential search, etc.
*   **Type Hinting:** Provides type hints for improved code clarity and maintainability.
*   **Extensive Testing:** Includes a comprehensive test suite with over 500 unit tests.
*   **Utility Module (`bitarray.util`):**
    *   Hexadecimal string conversions
    *   Random bitarray generation
    *   Pretty printing
    *   Integer conversions
    *   Huffman code generation
    *   Sparse bitarray compression and decompression
    *   Serialization and deserialization
    *   Various count functions
    *   And more!

## Installation

Install `bitarray` easily using `pip`:

```bash
pip install bitarray
```

Or, use `conda`:

```bash
conda install bitarray
```

Verify the installation and view some basic information by running the test suite:

```bash
python -c 'import bitarray; bitarray.test()'
```

## Usage

`bitarray` objects closely resemble Python lists, but are optimized for storing boolean values efficiently.

```python
from bitarray import bitarray

# Create a bitarray
a = bitarray()

# Append bits
a.append(1)
a.extend([1, 0])
print(a) # Output: bitarray('110')

# Initialize from a string
b = bitarray('1001011')
print(b) # Output: bitarray('1001011')

# Indexing
print(b[0]) # Output: 1
print(b[1:4]) # Output: bitarray('001')

# Slice assignment
c = bitarray(50)
c.setall(0)
c[11:37:3] = 9 * bitarray('1')
print(c)
```

## Bit-Endianness

`bitarray` supports both big-endian and little-endian bit representations. By default, it uses big-endian.

```python
a = bitarray(b'A') # big-endian by default
print(a) # Output: bitarray('01000001')
b = bitarray(b'A', endian='little')
print(b) # Output: bitarray('10000010')
```

## Further Information

*   **Buffer Protocol:** Learn more about the [buffer protocol](https://github.com/ilanschnell/bitarray/blob/master/doc/buffer.rst).
*   **Examples:** See the [mmapped-file example](https://github.com/ilanschnell/bitarray/blob/master/examples/mmapped-file.py).
*   **Change Log:** Review the [change log](https://github.com/ilanschnell/bitarray/blob/master/doc/changelog.rst).
*   **Bitarray Representations:** Explore [bitarray representations](https://github.com/ilanschnell/bitarray/blob/master/doc/represent.rst).
*   **Canonical Huffman Coding:** Learn about [Canonical Huffman Coding](https://github.com/ilanschnell/bitarray/blob/master/doc/canonical.rst).
*   **Random Bitarrays:** See [Random Bitarrays](https://github.com/ilanschnell/bitarray/blob/master/doc/random_p.rst).
*   **Sparse Bitarray Compression:** Understand [Sparse Bitarray Compression](https://github.com/ilanschnell/bitarray/blob/master/doc/sparse_compression.rst).
*   **Variable length bitarray format:** See [Variable length bitarray format](https://github.com/ilanschnell/bitarray/blob/master/doc/variable_length.rst).