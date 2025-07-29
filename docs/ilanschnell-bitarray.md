# bitarray: Efficient Arrays of Booleans in Python

**Need a memory-efficient way to store and manipulate boolean data?** The `bitarray` library provides a fast and flexible Python solution for working with arrays of bits, perfect for handling large datasets with optimized performance.

[Visit the GitHub Repository](https://github.com/ilanschnell/bitarray)

## Key Features

*   **Bit-Endianness Control:** Choose between big-endian and little-endian representations for your bitarrays.
*   **Sequence-like Operations:** Benefit from familiar list-like operations:
    *   Slicing (including assignment and deletion)
    *   Concatenation (+), repetition (\*), augmented assignment (+=, \*=)
    *   Membership testing (in)
    *   Length (len())
*   **Bitwise Operations:** Perform efficient bitwise manipulations:
    *   Inversion (~), AND (&), OR (|), XOR (^), left shift (<<), right shift (>>)
    *   In-place bitwise operations (&=, |=, ^=, <<=, >>=)
*   **Variable-Length Prefix Codec:**  Fast methods for encoding and decoding variable bit length prefix codes, including Huffman coding.
*   **Buffer Protocol Support:** Seamless integration with other Python objects through buffer import and export.
*   **Data Serialization:** Pack and unpack to various binary data formats, including NumPy ndarrays.
*   **Immutability:** Use hashable `frozenbitarray` objects.
*   **Utilities:** The `bitarray.util` module offers tools for:
    *   Hexadecimal conversion
    *   Random bitarray generation
    *   Pretty printing
    *   Integer conversion
    *   Huffman code creation
    *   Sparse bitarray compression
    *   Serialization and deserialization
    *   Counting and other utility functions
*   **Thorough Testing:** Extensive test suite with over 500 unit tests to ensure reliability.
*   **Type Hinting:** For improved code readability and maintainability.

## Installation

Install bitarray easily using pip or conda:

```bash
pip install bitarray
```

```bash
conda install bitarray
```

## Usage

```python
from bitarray import bitarray

# Create a bitarray
a = bitarray('10110')

# Accessing elements (returns integers)
print(a[0])  # Output: 1

# Slicing (returns a bitarray)
print(a[1:3])  # Output: bitarray('01')

# Basic operations
a.append(0)
print(a)  # Output: bitarray('101100')
print(a.count(1))  # Output: 3

# Bitwise operations
b = bitarray('110010')
print(a & b)  # Output: bitarray('100000')

# Setting a slice
a[2:4] = bitarray('01')
print(a) # Output: bitarray('100100')
```

## More Information

See the full reference in the original README for detailed information on:

*   **Bit-Endianness:** How endianness affects your data representation.
*   **Buffer Protocol:** How to integrate `bitarray` with other libraries.
*   **Variable bit length prefix codes:** How to use the `encode` and `decode` methods for data compression.
*   **frozenbitarrays:** How to use hashable bitarrays as keys in dictionaries.
*   **Reference:** All methods and functions in the library.