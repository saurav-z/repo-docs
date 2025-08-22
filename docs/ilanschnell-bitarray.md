# bitarray: Efficient Arrays of Booleans in Python

**The `bitarray` library provides an efficient way to store and manipulate arrays of booleans, offering performance advantages over standard Python lists.**

[![PyPI](https://img.shields.io/pypi/v/bitarray.svg)](https://pypi.org/project/bitarray/)
[![Downloads](https://img.shields.io/pypi/dm/bitarray.svg)](https://pypi.org/project/bitarray/)
[![License](https://img.shields.io/pypi/l/bitarray.svg)](https://github.com/ilanschnell/bitarray/blob/master/LICENSE)
[![Python Versions](https://img.shields.io/pypi/pyversions/bitarray.svg)](https://pypi.org/project/bitarray/)

Key Features:

*   **Bit-Endianness Control:** Specify the bit-endianness (big-endian or little-endian) for each bitarray object, allowing flexible data representation.
*   **Sequence Operations:** Full support for sequence methods, including slicing (with assignment and deletion), `+`, `*`, `+=`, `*=`, `in` operator, and `len()`.
*   **Bitwise Operations:** Efficient bitwise operations: `~`, `&`, `|`, `^`, `<<`, `>>` and their in-place counterparts.
*   **Variable-Length Prefix Codes:** Fast methods for encoding and decoding variable bit-length prefix codes (e.g., Huffman coding).
*   **Buffer Protocol Support:**  Seamless integration with the Python buffer protocol for importing and exporting data.
*   **Data Conversion:**  Packing and unpacking to and from other binary data formats (e.g., `numpy.ndarray`).
*   **Pickling and Unpickling:** Serialization and deserialization of bitarray objects.
*   **Immutable Frozen Bitarrays:**  `frozenbitarray` objects offer immutability and hashability, suitable for use as dictionary keys.
*   **Additional Functionality:** Sequential search, type hinting, and an extensive test suite with over 500 unit tests.
*   **Utility Module:** `bitarray.util` provides a range of helpful functions for:
    *   Hexadecimal string conversions.
    *   Generating random bitarrays.
    *   Pretty printing.
    *   Integer conversions.
    *   Huffman code generation.
    *   Compression of sparse bitarrays.
    *   Serialization/Deserialization.
    *   Various count functions and other useful utilities.

## Installation

Install the package using `pip`:

```bash
pip install bitarray
```

To verify the installation, run the following:

```bash
python -c 'import bitarray; bitarray.test()'
```

The `test()` function is part of the API and provides detailed information about the installation.

## Usage

`bitarray` objects behave similarly to Python lists but are optimized for storing boolean data. The library allows you to access and manipulate the underlying machine representation of bits, including specifying bit-endianness.

```python
from bitarray import bitarray

# Create a bitarray
a = bitarray()
a.append(1)
a.extend([1, 0])
print(a)  # Output: bitarray('110')

# Initialize from a string
b = bitarray('1001011')
print(b) # Output: bitarray('1001011')

# Slice assignment
a[0:2] = [0, 1]
print(a) # Output: bitarray('010')
```

For detailed usage, including bitwise operations, slice assignment, and bit-endianness handling, refer to the [original README](https://github.com/ilanschnell/bitarray).

## Detailed Documentation

*   [Original GitHub Repository](https://github.com/ilanschnell/bitarray)
*   [Change Log](https://github.com/ilanschnell/bitarray/blob/master/doc/changelog.rst)
*   [Indexing Documentation](https://github.com/ilanschnell/bitarray/blob/master/doc/indexing.rst)
*   [Buffer Protocol Documentation](https://github.com/ilanschnell/bitarray/blob/master/doc/buffer.rst)
*   [Variable length bitarray format documentation](https://github.com/ilanschnell/bitarray/blob/master/doc/variable_length.rst)
*   [Sparse Compression documentation](https://github.com/ilanschnell/bitarray/blob/master/doc/sparse_compression.rst)
*   [Canonical Huffman Coding documentation](https://github.com/ilanschnell/bitarray/blob/master/doc/canonical.rst)
*   [Random Bitarrays documentation](https://github.com/ilanschnell/bitarray/blob/master/doc/random_p.rst)
*   [Bitarray representations documentation](https://github.com/ilanschnell/bitarray/blob/master/doc/represent.rst)