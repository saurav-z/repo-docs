# bitarray: Efficient Arrays of Booleans in Python

**Represent and manipulate arrays of booleans with exceptional efficiency using the `bitarray` library, providing fast bitwise operations and flexible data handling.**  [View the original repository](https://github.com/ilanschnell/bitarray)

## Key Features

*   **Bit-Endianness Control:** Choose between little-endian and big-endian representations for your bitarrays.
*   **Sequence-like Behavior:** Utilize familiar sequence methods like slicing, concatenation (`+`), repetition (`*`), `in` operator, and `len()`.
*   **Bitwise Operations:** Perform bitwise operations such as `~`, `&`, `|`, `^`, `<<`, `>>` and their in-place versions.
*   **Variable Bit Length Codes:** Encode and decode data using fast methods for variable bit length prefix codes.
*   **Buffer Protocol Support:** Integrate with other Python objects through buffer import and export.
*   **Data Conversion:** Pack and unpack data to/from binary formats, including `numpy.ndarray`.
*   **Pickling/Unpickling:** Easily serialize and deserialize bitarray objects.
*   **Frozen Bitarrays:** Use immutable, hashable `frozenbitarray` objects as dictionary keys.
*   **Additional Features:** Includes sequential search, type hinting, and a comprehensive test suite.
*   **Utility Module:** `bitarray.util` offers:
    *   Hexadecimal string conversions.
    *   Random bitarray generation.
    *   Pretty printing.
    *   Integer conversions.
    *   Huffman code generation.
    *   Compression of sparse bitarrays.
    *   Serialization/Deserialization.
    *   Various count functions.
    *   And more helpful functions.

## Installation

Install `bitarray` via `pip` or `conda`:

```bash
pip install bitarray
# or
conda install bitarray
```

## Usage

```python
from bitarray import bitarray

# Create an empty bitarray and append bits
a = bitarray()
a.append(1)
a.extend([1, 0])
print(a)  # Output: bitarray('110')

# Initialize from a string
b = bitarray('10101')
print(b)  # Output: bitarray('10101')

# Slicing and indexing
print(b[1])     # Output: 0
print(b[1:3])   # Output: bitarray('01')

# Bitwise operations
c = bitarray('101')
d = bitarray('011')
print(c & d)   # Output: bitarray('001')

# Accessing Machine Representation
e = bitarray(b'A', endian='little')
print(e)       # Output: bitarray('10000010')

# Using as dictionary keys
from bitarray import frozenbitarray
key = frozenbitarray('1100011')
my_dict = {key: 'some value'}
print(my_dict) # Output: {frozenbitarray('1100011'): 'some value'}
```

## Further Information

See the [original repository](https://github.com/ilanschnell/bitarray) for more details on:

*   **Bit-endianness**
*   **Buffer protocol**
*   **Variable bit length prefix codes**
*   **Reference** including methods, data descriptors, other objects and functions