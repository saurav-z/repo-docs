# bitarray: High-Performance Boolean Arrays in Python

**Need an efficient and versatile way to handle arrays of booleans?** The `bitarray` library provides a fast and flexible solution, implemented in C, for working with bit-level data in Python. [Visit the original repo](https://github.com/ilanschnell/bitarray)

Key Features:

*   **Bit-Endianness Control:** Choose between `big` or `little` endianness for each bitarray, giving you fine-grained control over data representation.
*   **Sequence-Like Operations:** Utilize familiar list-like functionalities, including slicing (with assignment and deletion), and the `+`, `*`, `+=`, `*=`, and `in` operators, and `len()`.
*   **Bitwise Operations:** Perform efficient bitwise operations like `~`, `&`, `|`, `^`, `<<`, `>>` (and their in-place counterparts).
*   **Variable-Length Prefix Codes:** Benefit from fast encoding and decoding methods for prefix codes, ideal for data compression and related tasks.
*   **Buffer Protocol Support:** Directly interact with memory buffers, enabling seamless integration with other Python objects and memory-mapped files.
*   **Data Serialization & Integration:** Pack and unpack bitarrays to/from various binary data formats, including NumPy ndarrays, and support for pickling.
*   **Frozen Bitarrays:** Use hashable, immutable `frozenbitarray` objects for dictionary keys.
*   **Search Capabilities:** Efficient sequential search methods are included.
*   **Type Hinting:** Benefit from type hinting for improved code readability and maintainability.
*   **Extensive Testing:** A comprehensive test suite with approximately 600 unit tests ensures reliability.
*   **Utility Module:** A powerful `bitarray.util` module provides:
    *   Hexadecimal string conversions
    *   Random bitarray generation
    *   Pretty printing
    *   Integer conversions
    *   Huffman code generation
    *   Sparse bitarray compression
    *   Serialization/Deserialization
    *   Various counting functions
    *   And more helpful utilities

## Installation

Easily install `bitarray` using pip:

```bash
pip install bitarray
```

Verify the installation with a quick test:

```bash
python -c 'import bitarray; bitarray.test()'
```

## Usage and Examples

`bitarray` objects behave similarly to Python lists, with the key difference being their efficient, bit-level storage. Here are a few examples:

```python
from bitarray import bitarray

# Create an empty bitarray and append elements
a = bitarray()
a.append(1)
a.extend([1, 0])
print(a)  # Output: bitarray('110')

# Initialize from a string
b = bitarray('1001011')
print(b)  # Output: bitarray('1001011')

# Initialize from an iterable
lst = [1, 0, False, True, True]
c = bitarray(lst)
print(c)  # Output: bitarray('10011')

# Indexing and Slicing
print(c[2])    # Output: 0
print(c[2:4])  # Output: bitarray('01')

# Count occurrences
print(c.count(1)) # Output: 3
```

## Reference

For detailed information, please refer to the [original repository's README](https://github.com/ilanschnell/bitarray).