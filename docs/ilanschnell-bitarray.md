# bitarray: Efficient Arrays of Booleans in Python

**bitarray is a Python library that provides a highly efficient way to represent and manipulate arrays of booleans, offering performance comparable to C implementations.**

[View the source code on GitHub](https://github.com/ilanschnell/bitarray)

Key Features:

*   **Bit-Endianness Control:**  Specify little-endian or big-endian representation for each bitarray object.
*   **Sequence Type Functionality:** Supports slicing (including assignment and deletion), concatenation (`+`), repetition (`*`), and membership testing (`in`), as well as `len()`.
*   **Bitwise Operations:** Includes all standard bitwise operators: `~`, `&`, `|`, `^`, `<<`, `>>`, and their in-place counterparts (`&=`, `|=`, `^=`, `<<=`, `>>=`).
*   **Efficient Prefix Code Encoding/Decoding:** Fast methods for handling variable bit length prefix codes, essential for compression and data representation.
*   **Buffer Protocol Support:**  Seamlessly integrates with the buffer protocol, allowing interaction with other Python objects that expose a buffer.
*   **Data Format Compatibility:** Provides packing and unpacking to other binary data formats, such as NumPy arrays.
*   **Pickling and Unpickling:**  Supports serialization and deserialization for persistence.
*   **Frozenbitarray Objects:** Immutable, hashable `frozenbitarray` objects for use as dictionary keys.
*   **Search and Indexing:** Includes sequential search functionality.
*   **Type Hinting:** Fully typed for enhanced code clarity and maintainability.
*   **Extensive Testing:**  Comprehensive test suite with over 500 unit tests.
*   **Utility Module (bitarray.util):**
    *   Hexadecimal string conversions.
    *   Random bitarray generation.
    *   Pretty printing.
    *   Integer conversions.
    *   Huffman code generation.
    *   Compression of sparse bitarrays.
    *   Serialization and deserialization.
    *   Various count functions.
    *   Other helpful utility functions.

## Installation

Install `bitarray` using pip:

```bash
pip install bitarray
```

or with conda:

```bash
conda install bitarray
```

Verify installation:

```python
import bitarray
bitarray.test()
```

## Usage

```python
from bitarray import bitarray

# Create an empty bitarray and append values
a = bitarray()
a.append(1)
a.extend([1, 0])
print(a)  # Output: bitarray('110')

# Create a bitarray from a string or a list
x = bitarray('1001011')
lst = [1, 0, False, True, True]
b = bitarray(lst)
print(b)  # Output: bitarray('10011')

# Indexing and Slicing
print(b[2])    # Output: 0
print(b[2:4])  # Output: bitarray('01')

# Bitwise operations
c = bitarray('101110001')
print(~c)       # Output: bitarray('010001110')
```

## Bit-endianness

bitarray objects support two ways of representing bits in memory. By default, bitarrays use big-endian representation. 

```python
a = bitarray(b'A')
print(a.endian)  # Output: big
print(a)         # Output: bitarray('01000001')

# Explicitly define the endianness
a = bitarray(b'A', endian='little')
print(a)  # Output: bitarray('10000010')
```

## Other relevant parts from original README

*   **Buffer protocol** Learn more about the buffer protocol [here](https://github.com/ilanschnell/bitarray/blob/master/doc/buffer.rst).
*   **Variable bit length prefix codes**: Check out how to use `.encode()` and `.decode()` to handle prefix codes.
*   **Reference:** This includes a full reference for bitarray objects, methods, data descriptors and functions.