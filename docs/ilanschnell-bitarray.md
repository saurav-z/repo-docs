# bitarray: Efficient Arrays of Booleans in Python

**Need an efficient and versatile way to represent and manipulate arrays of booleans?** The `bitarray` library provides a high-performance, C-implemented solution for working with bit-level data in Python.  [Visit the original repository](https://github.com/ilanschnell/bitarray) for more information.

## Key Features

*   **Endianness Control:** Specify bit-endianness (big or little) for each bitarray object.
*   **Sequence-like Operations:** Supports slicing, concatenation (`+`), repetition (`*`), in-place operations (`+=`, `*=`), `in` operator, and `len()`.
*   **Bitwise Operations:** Efficient bitwise operations including: `~`, `&`, `|`, `^`, `<<`, `>>` and their in-place counterparts.
*   **Variable-Length Prefix Codes:**  Fast encoding and decoding of variable bit length prefix codes, essential for data compression and communication protocols.
*   **Buffer Protocol Support:**  Allows import and export of buffers for seamless integration with other Python libraries like NumPy, and memory mapping.
*   **Data Serialization & Integration:** Supports packing/unpacking to binary formats (e.g., NumPy arrays) and pickling for persistence.
*   **Immutable Frozen Bitarrays:**  Includes `frozenbitarray` objects for use as dictionary keys.
*   **Search and Indexing:**  Provides methods for sequential search and advanced indexing capabilities.
*   **Type Hinting:**  Includes type hints for improved code readability and maintainability.
*   **Comprehensive Testing:**  Extensive test suite with over 500 unit tests.
*   **Utility Module (bitarray.util):** Offers tools for:
    *   Hexadecimal conversions
    *   Random bitarray generation
    *   Pretty printing
    *   Integer conversions
    *   Huffman code generation
    *   Sparse bitarray compression
    *   Serialization/Deserialization
    *   Various counting functions and more.

## Installation

Install `bitarray` using pip:

```bash
pip install bitarray
```

To verify the installation and run tests:

```bash
python -c 'import bitarray; bitarray.test()'
```

## Usage

`bitarray` objects closely mimic Python lists, making them easy to learn. The key advantage is efficient bit-level access and manipulation.

```python
from bitarray import bitarray

# Create a bitarray from a list
a = bitarray([1, 0, 1, 1])
print(a)  # Output: bitarray('1011')

# Slice assignment
a[1:3] = bitarray('01')
print(a)  # Output: bitarray('1011')

# Bitwise operations
b = bitarray('1100')
c = a & b
print(c)  # Output: bitarray('1000')
```

For more details on specific methods, data descriptors, and the `bitarray.util` module, please refer to the full documentation in the original README or the repository's documentation.