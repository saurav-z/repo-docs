# bitarray: Efficient Arrays of Booleans

**bitarray is a Python library that provides a memory-efficient and feature-rich way to work with arrays of booleans, offering performance comparable to low-level languages.** ([Original Repository](https://github.com/ilanschnell/bitarray))

## Key Features

*   **Bit-Endianness Control:** Choose between big-endian and little-endian representations for each bitarray object.
*   **Sequence Operations:** Supports slicing (including assignment and deletion), `+`, `*`, `+=`, `*=`, `in` operator, and `len()`.
*   **Bitwise Operations:** Includes `~`, `&`, `|`, `^`, `<<`, `>>` and in-place versions (`&=`, `|=`, `^=`, `<<=`, `>>=`).
*   **Prefix Code Encoding/Decoding:** Fast methods for encoding and decoding variable bit length prefix codes, useful for data compression and more.
*   **Buffer Protocol Support:** Allows import and export of buffers, including interaction with memory-mapped files and other binary formats like NumPy.
*   **Data Conversion:** Offers methods for packing and unpacking to/from other binary data formats (e.g. NumPy arrays).
*   **Pickling & Freezing:** Supports pickling and unpickling of bitarray objects, as well as immutable `frozenbitarray` objects that can be used as dictionary keys.
*   **Additional Utilities:**
    *   Conversion to and from hexadecimal strings
    *   Generating random bitarrays
    *   Pretty printing of bitarrays
    *   Conversion to and from integers
    *   Creating Huffman codes
    *   Compression of sparse bitarrays
    *   Serialization/Deserialization
    *   Various count functions
    *   And other helpful functions in the `bitarray.util` module
*   **Extensive Testing:**  A comprehensive test suite with over 600 unit tests ensures reliability.
*   **Type Hinting:**  Supports type hinting for improved code readability and maintainability.
*   **Sequential search**
*   **Bitarray indexing**

## Installation

Install `bitarray` using pip:

```bash
pip install bitarray
```

Verify the installation by running the test suite:

```bash
python -c 'import bitarray; bitarray.test()'
```

## Usage

```python
from bitarray import bitarray

# Create a bitarray
a = bitarray()
a.append(1)
a.extend([1, 0])
print(a)  # Output: bitarray('110')

# Initialize from an iterable
lst = [1, 0, False, True, True]
a = bitarray(lst)
print(a)  # Output: bitarray('10011')

# Indexing
print(a[2])  # Output: 0
print(a[2:4]) # Output: bitarray('01')

# Slice assignment
a[2:4] = bitarray('11')
print(a)  # Output: bitarray('10111')

# Bitwise operations
b = bitarray('111001011')
result = a & b
print(result) # Output: bitarray('101000001')

```

## Documentation

*   [Bitarray indexing](https://github.com/ilanschnell/bitarray/blob/master/doc/indexing.rst)
*   [Bit-endianness](https://github.com/ilanschnell/bitarray/blob/master/doc/endianness.rst)
*   [Buffer protocol](https://github.com/ilanschnell/bitarray/blob/master/doc/buffer.rst)
*   [Bitarray representations](https://github.com/ilanschnell/bitarray/blob/master/doc/represent.rst)
*   [Canonical Huffman Coding](https://github.com/ilanschnell/bitarray/blob/master/doc/canonical.rst)
*   [Compression of sparse bitarrays](https://github.com/ilanschnell/bitarray/blob/master/doc/sparse_compression.rst)
*   [Variable length bitarray format](https://github.com/ilanschnell/bitarray/blob/master/doc/variable_length.rst)
*   [Random Bitarrays](https://github.com/ilanschnell/bitarray/blob/master/doc/random_p.rst)
*   [Bitarray 3 transition](https://github.com/ilanschnell/bitarray/blob/master/doc/bitarray3.rst)

## Reference

*   [Change Log](https://github.com/ilanschnell/bitarray/blob/master/doc/changelog.rst)
*   API Reference is included in the original README.