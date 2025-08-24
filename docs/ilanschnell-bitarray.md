# bitarray: Efficient Arrays of Booleans

**Store and manipulate boolean data with unparalleled efficiency using the `bitarray` library, a fast and versatile Python package for working with arrays of bits.** [View on GitHub](https://github.com/ilanschnell/bitarray)

## Key Features

*   **Bit-Endianness Control:** Specify big- or little-endian representation for each bitarray, offering flexibility in data interpretation.
*   **Sequence-Like Functionality:**  Leverage familiar sequence operations, including slicing (with assignment and deletion), `+`, `*`, `+=`, `*=`, `in`, and `len()`.
*   **Bitwise Operations:** Perform efficient bitwise calculations using `~`, `&`, `|`, `^`, `<<`, `>>`, and their in-place counterparts.
*   **Variable-Length Prefix Codec:** Encode and decode data using fast methods for variable bit length prefix codes, including Huffman coding.
*   **Buffer Protocol Support:**  Seamlessly integrate with other Python objects that utilize the buffer protocol, including memory-mapped files.
*   **Data Serialization:** Pack and unpack bitarrays to and from other binary data formats (e.g., `numpy.ndarray`).
*   **Immutability:** Utilize `frozenbitarray` objects for hashable, immutable bit arrays.
*   **Advanced Utilities:** A dedicated utility module (`bitarray.util`) provides functionality for:
    *   Hexadecimal conversion
    *   Random bitarray generation
    *   Pretty printing
    *   Integer conversion
    *   Huffman code generation
    *   Sparse bitarray compression
    *   (De-)serialization
    *   Counting and other helpful functions
*   **Extensive Testing:** Rigorously tested with a comprehensive suite of over 500 unit tests.
*   **Type Hinting:** Support for modern Python development practices.

## Installation

Install `bitarray` easily using `pip`:

```bash
pip install bitarray
```

Test your installation:

```bash
python -c 'import bitarray; bitarray.test()'
```

## Usage

`bitarray` objects mimic lists, allowing you to create, manipulate, and query arrays of bits efficiently.

```python
from bitarray import bitarray

# Create a bitarray
a = bitarray()  # Empty bitarray
a.append(1)
a.extend([1, 0])
print(a)  # Output: bitarray('110')

# Initialize from a string
b = bitarray('10101')

# Access and modify elements
print(b[0])  # Output: 1
b[1:3] = bitarray('00')  # Slice assignment
print(b) # Output: bitarray('10001')
```

Refer to the complete documentation for detailed information on [Bitarray indexing](https://github.com/ilanschnell/bitarray/blob/master/doc/indexing.rst).

## Bit-Endianness Explained

Understanding bit-endianness is crucial when interacting with the machine representation of bitarrays, especially when using methods like `.tobytes()`, `.frombytes()`, `.tofile()`, or `.fromfile()`. By default, bitarrays use big-endian representation.  You can specify either `big` or `little` endianness during object creation. Refer to the complete documentation for detailed information on [buffer protocol](https://github.com/ilanschnell/bitarray/blob/master/doc/buffer.rst).

## Reference

*   **Version:** 3.7.0
*   **Changelog:** [View Changelog](https://github.com/ilanschnell/bitarray/blob/master/doc/changelog.rst)

Complete details about methods, data descriptors, and functions are in the original README.