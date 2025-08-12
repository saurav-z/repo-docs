# bitarray: Efficient Arrays of Booleans in Python

**bitarray is a high-performance Python library for creating and manipulating arrays of bits, offering significant speed and memory advantages over standard Python lists.**  Access the original repository here: [https://github.com/ilanschnell/bitarray](https://github.com/ilanschnell/bitarray)

## Key Features

*   **Bit-Endianness Control:** Choose between little-endian and big-endian representations for your bitarrays.
*   **Sequence-Like Operations:**  Supports slicing, concatenation (`+`), repetition (`*`), in-place operations (`+=`, `*=`), the `in` operator, and `len()`.
*   **Bitwise Operations:** Efficient bitwise operations including `~`, `&`, `|`, `^`, `<<`, `>>`, and their in-place counterparts.
*   **Variable-Length Prefix Code Encoding/Decoding:** Fast methods for encoding and decoding variable-length bit prefix codes, ideal for compression and data representation.
*   **Buffer Protocol Support:**  Supports the buffer protocol for efficient data transfer and integration with other Python libraries.
*   **Integration with Other Formats:** Pack and unpack bitarrays to and from other binary data formats, such as NumPy arrays.
*   **Immutability:** Offers `frozenbitarray` objects for hashable, immutable bitarray storage.
*   **Extensive Utilities:** The `bitarray.util` module includes functions for:
    *   Hexadecimal conversion
    *   Random bitarray generation
    *   Pretty printing
    *   Integer conversion
    *   Huffman code creation
    *   Sparse bitarray compression/decompression
    *   Serialization/deserialization
    *   Counting functions
    *   And other helpful utility functions.
*   **Comprehensive Testing:** Includes an extensive test suite with over 500 unit tests.
*   **Type Hinting:** Supports type hinting for improved code readability and maintainability.

## Installation

Install bitarray using pip:

```bash
pip install bitarray
```

## Usage

bitarray objects behave similarly to Python lists, but are designed for efficient bit-level storage.

```python
from bitarray import bitarray

# Create a bitarray
a = bitarray()
a.append(1)
a.extend([1, 0])
print(a)  # Output: bitarray('110')

# Initialize from a string
b = bitarray('1001011')
print(b)

# Slice assignment
b[2:4] = bitarray('10')
print(b) # Output: bitarray('1010011')
```

## Bitwise Operators

bitarray supports all standard bitwise operators:

```python
a = bitarray('101110001')
b = bitarray('111001011')

print(~a)  # Invert
print(a & b) # Bitwise AND
print(a | b)  # Bitwise OR
print(a ^ b)  # Bitwise XOR
print(a << 2)  # Left shift by 2
print(b >> 1)  # Right shift by 1
```

## Bit-Endianness Explained

Bitarrays can be created with either `big` (default) or `little` endianness.  This determines how bits are arranged within the underlying byte representation.

```python
a = bitarray(b'A') # Big Endian - default
print(a.endian)
print(a)  # Output: bitarray('01000001')

b = bitarray(b'A', endian='little')
print(b.endian)
print(b)  # Output: bitarray('10000010')
```
## Additional Resources

*   **Documentation:** Detailed information is available in the doc folder of the GitHub repository.
*   **Examples:** Explore example code in the `examples` directory.
*   **Change Log:** See the `changelog.rst` file for version history and updates.