# bitarray: Efficient Arrays of Booleans in Python

**Representing arrays of booleans efficiently, the `bitarray` library provides a Pythonic sequence type with a wide range of features for bitwise operations and data manipulation.**

[View the original repository](https://github.com/ilanschnell/bitarray)

**Key Features:**

*   **Bit-Endianness Control:** Choose between little-endian and big-endian representations for flexibility.
*   **Sequence-Like Functionality:** Supports slicing, concatenation (`+`), repetition (`*`), the `in` operator, and `len()`.
*   **Bitwise Operations:** Implements bitwise operators like `~`, `&`, `|`, `^`, `<<`, `>>`, and their in-place counterparts (`&=`, `|=`, `^=`, `<<=`, `>>=`).
*   **Variable Bit Length Prefix Codes:** Fast methods for encoding and decoding.
*   **Buffer Protocol Support:** Allows importing and exporting buffers for integration with other data structures.
*   **Data Conversion:** Facilitates packing and unpacking to/from binary data formats, including `numpy.ndarray`.
*   **Serialization:** Supports pickling and unpickling of bitarray objects.
*   **Immutable Objects:** Provides `frozenbitarray` objects for hashable, immutable storage.
*   **Additional utilities:** Included features for sequential search, Type hinting, and an extensive test suite
*   **Utilities Module:** Includes a `bitarray.util` module with functions for:
    *   Hexadecimal conversions
    *   Generating random bitarrays
    *   Pretty printing
    *   Integer conversions
    *   Huffman code generation
    *   Sparse bitarray compression
    *   Serialization/deserialization
    *   Counting functions
    *   And many other helpful functions

**Installation:**

Install `bitarray` easily using pip:

```bash
pip install bitarray
```

Or install from conda

```bash
conda install bitarray
```

**Quick Start:**

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
b[2:4] = bitarray('01')
print(b)
```

**Further Information**
*   Usage Examples are provided in the original README.
*   The reference section details all object methods and functions.