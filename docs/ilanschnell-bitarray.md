# bitarray: Efficient Arrays of Booleans

**Represent and manipulate arrays of booleans with exceptional efficiency using the bitarray library!** ([Original Repository](https://github.com/ilanschnell/bitarray))

## Key Features

*   **Bit-Endianness Control:** Specify little- or big-endianness for each bitarray object.
*   **Sequence Methods:**  Supports standard sequence operations: slicing, concatenation (`+`), repetition (`*`), in-place modifications (`+=`, `*=`), the `in` operator, and `len()`.
*   **Bitwise Operations:** Perform bitwise operations with `~`, `&`, `|`, `^`, `<<`, `>>` and their in-place counterparts (`&=`, `|=`, `^=`, `<<=`, `>>=`).
*   **Variable Bit Length Prefix Codes:** Efficient encoding and decoding methods for variable bit length prefix codes, ideal for compression and data representation.
*   **Buffer Protocol Support:** Integrates with the buffer protocol, enabling seamless data exchange with other Python objects, including NumPy arrays and memory-mapped files.
*   **Data Conversion:** Supports packing and unpacking to various binary data formats, including NumPy `ndarray` and custom formats.
*   **Persistence:** Provides pickling and unpickling capabilities for bitarray objects, enabling storage and retrieval.
*   **Immutability:** Offers `frozenbitarray` objects for hashable and immutable data.
*   **Additional Functionality:** Includes sequential search, type hinting, and a comprehensive test suite.
*   **Utility Module (`bitarray.util`):** Offers utilities for:
    *   Hexadecimal string conversions
    *   Random bitarray generation
    *   Pretty printing
    *   Integer conversions
    *   Huffman code creation
    *   Compression of sparse bitarrays
    *   Serialization and deserialization
    *   Various counting functions
    *   Other helpful functions

## Installation

Install bitarray using pip:

```bash
pip install bitarray
```

To test the installation:

```bash
python -c 'import bitarray; bitarray.test()'
```

## Usage

Bitarray objects behave like Python lists, offering efficient storage and manipulation of boolean data. Key features include:

*   **Initialization:** Create bitarrays from integers, bytes, strings, or iterables.
*   **Indexing and Slicing:** Access and modify individual bits and slices using standard Python indexing.
*   **Bitwise Operations:** Apply bitwise logic directly to bitarrays for fast computation.
*   **Endianness:**  Control the bit-endianness (big or little) during creation for precise control over data representation when interacting with binary formats.

```python
from bitarray import bitarray

# Create and manipulate bitarrays
a = bitarray('101101')
a.append(0)
print(a)  # Output: bitarray('1011010')

# Bitwise operations
b = bitarray('110011')
result = a & b
print(result)  # Output: bitarray('1000010')

#Endianness
c = bitarray(b'\x01', endian='little')
print(c)  # Output: bitarray('10000000')
```

## Bitwise Operators

Bitarray objects support a full suite of bitwise operators, mirroring their functionality in C for optimized performance. These include:
```python
~a  # Invert
a & b # Bitwise AND
a | b # Bitwise OR
a ^ b # Bitwise XOR
a << 2 # In-place left-shift by 2
b >> 1 # Return b right-shifted by 1
```

## Bit-Endianness

Understand and control the bit-endianness (big-endian or little-endian) when working with the machine representation or binary data exchange.

## Buffer Protocol

Leverage the buffer protocol to seamlessly integrate bitarray objects with other libraries and applications that use buffers.

## Variable Bit Length Prefix Codes

Use the `.encode()` and `.decode()` methods for efficient data compression, with support for Huffman coding and the use of  `decodetree` objects for improved performance when decoding.

## Frozenbitarrays

Utilize immutable `frozenbitarray` objects when you need hashable bitarrays, such as for use as dictionary keys.

## Reference

Comprehensive documentation of all methods and attributes are available in the original repository.