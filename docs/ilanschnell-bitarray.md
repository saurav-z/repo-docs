# bitarray: Efficient Arrays of Booleans in Python

**bitarray is a Python library that provides a fast and memory-efficient way to represent and manipulate arrays of booleans.**  [View the original repository](https://github.com/ilanschnell/bitarray).

## Key Features

*   **Bit-Endianness Control:** Specify big-endian or little-endian representation for your bitarrays.
*   **Sequence-like Behavior:** Supports slicing, concatenation (`+`), repetition (`*`), the `in` operator, and `len()`.
*   **Bitwise Operations:** Offers bitwise operators (`~`, `&`, `|`, `^`, `<<`, `>>`, and in-place versions) for efficient manipulation.
*   **Variable-Length Prefix Codes:** Includes fast methods for encoding and decoding variable bit length prefix codes, useful for data compression.
*   **Buffer Protocol Support:** Works seamlessly with the Python buffer protocol, allowing import/export of buffers.
*   **Integration with Binary Data Formats:** Enables packing and unpacking to formats like `numpy.ndarray`.
*   **Immutability:** Offers `frozenbitarray` objects, which are immutable and hashable, ideal for use as dictionary keys.
*   **Additional Utilities:** The `bitarray.util` module provides helpful functions like conversion to and from hexadecimal strings, random bitarray generation, Huffman code creation, and compression of sparse bitarrays.
*   **Type Hinting:** Includes type hints for improved code readability and maintainability.
*   **Extensive Testing:** Comes with a comprehensive test suite (over 500 unit tests).

## Installation

Install `bitarray` easily using pip:

```bash
pip install bitarray
```

You can verify the installation and see version and system information by running:

```bash
python -c 'import bitarray; bitarray.test()'
```

## Usage

`bitarray` objects behave like Python lists but store boolean values efficiently. Here's a quick overview:

```python
from bitarray import bitarray

# Create an empty bitarray
a = bitarray()
a.append(1)
a.extend([1, 0])
print(a)  # Output: bitarray('110')

# Initialize from an iterable
lst = [1, 0, False, True, True]
a = bitarray(lst)
print(a)  # Output: bitarray('10011')

# Access elements by index
print(a[2])  # Output: 0
print(a[2:4])  # Output: bitarray('01')

# Slice assignment
a[2:3] = True
print(a)
```

## Bit-Endianness

When working with the machine representation of bitarrays, you can specify the bit-endianness:

```python
a = bitarray(b'A', endian='little')
print(a)  # Output: bitarray('10000010')
```

## More Information

*   **[Bitarray indexing](https://github.com/ilanschnell/bitarray/blob/master/doc/indexing.rst)**
*   **[Bitarray 3 transition](https://github.com/ilanschnell/bitarray/blob/master/doc/bitarray3.rst)**
*   **[Bitarray representations](https://github.com/ilanschnell/bitarray/blob/master/doc/represent.rst)**
*   **[Canonical Huffman Coding](https://github.com/ilanschnell/bitarray/blob/master/doc/canonical.rst)**
*   **[Compression of sparse bitarrays](https://github.com/ilanschnell/bitarray/blob/master/doc/sparse_compression.rst)**
*   **[Variable length bitarray format](https://github.com/ilanschnell/bitarray/blob/master/doc/variable_length.rst)**
*   **[Random Bitarrays](https://github.com/ilanschnell/bitarray/blob/master/doc/random_p.rst)**
*   **[Examples of Bitarray usage](https://github.com/ilanschnell/bitarray/blob/master/examples/mmapped-file.py)**
*   **[Change Log](https://github.com/ilanschnell/bitarray/blob/master/doc/changelog.rst)**