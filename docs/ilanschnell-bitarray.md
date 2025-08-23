# bitarray: Efficient Arrays of Booleans in Python

**Effortlessly manage and manipulate arrays of booleans with the `bitarray` library – your fast and flexible solution for bit-level operations in Python!** 

[View the original repository](https://github.com/ilanschnell/bitarray)

## Key Features

*   **Bit-Endianness Control:** Choose between little-endian and big-endian representations for each bitarray object.
*   **Sequence Operations:** Enjoy familiar list-like functionality including slicing, assignment, deletion, `+`, `*`, `+=`, `*=`, `in`, and `len()`.
*   **Bitwise Operations:** Perform efficient bitwise operations: `~`, `&`, `|`, `^`, `<<`, `>>` (and their in-place counterparts).
*   **Prefix Code Encoding/Decoding:**  Fast methods for encoding and decoding variable bit length prefix codes, ideal for compression and data representation.
*   **Buffer Protocol Support:** Leverage the buffer protocol for direct access and interaction with other Python objects and memory-mapped files.
*   **Data Conversion:** Seamlessly pack and unpack bitarrays to/from binary data formats like `numpy.ndarray`.
*   **Immutability:** Create immutable `frozenbitarray` objects for use as dictionary keys.
*   **Utility Module:** Utilize `bitarray.util` for:
    *   Hexadecimal string conversions
    *   Random bitarray generation
    *   Pretty printing
    *   Integer conversions
    *   Huffman code generation
    *   Sparse bitarray compression
    *   Serialization/Deserialization
    *   Various counting functions
*   **Extensive Testing:** Comprehensive test suite with over 500 unit tests.
*   **Type Hinting:** Benefit from type hints for improved code readability and maintainability.

## Installation

Install the `bitarray` package via pip:

```bash
pip install bitarray
```

Verify installation and run tests with:

```bash
python -c 'import bitarray; bitarray.test()'
```

## Usage

Bitarray objects behave similarly to lists, with a focus on efficiency and bit-level control.

```python
from bitarray import bitarray

# Create an empty bitarray
a = bitarray()

# Append and extend
a.append(1)
a.extend([1, 0])
print(a)  # Output: bitarray('110')

# Initialize from a string
b = bitarray('1001011')
print(b)

# Initialize from a list
lst = [1, 0, False, True, True]
c = bitarray(lst)
print(c)  # Output: bitarray('10011')

# Indexing and slicing
print(c[2])   # Output: 0
print(c[2:4]) # Output: bitarray('01')

# Bitwise operations
a = bitarray('101110001')
b = bitarray('111001011')
print(~a)     # Output: bitarray('010001110')
print(a ^ b)  # Output: bitarray('010111010')
```

**Bit-Endianness:**  The library supports both big-endian (default) and little-endian bit representations. Be mindful of endianness when working with buffer representations or machine-level interactions.

**Buffer Protocol:**  Use the buffer protocol for advanced interactions.
```python
#Example of the buffer protocol
from bitarray import bitarray
a = bitarray('0101')
buffer_info = a.buffer_info()
print(buffer_info)
```

## Further Information

*   [Change Log](https://github.com/ilanschnell/bitarray/blob/master/doc/changelog.rst)
*   [Documentation](https://github.com/ilanschnell/bitarray/blob/master/doc/index.rst)