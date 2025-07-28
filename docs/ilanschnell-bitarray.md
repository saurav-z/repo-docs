# Bitarray: Efficient Arrays of Booleans

[bitarray](https://github.com/ilanschnell/bitarray) is a Python library that offers an efficient and flexible way to work with arrays of booleans. It provides a memory-efficient alternative to standard Python lists, storing eight bits in a single byte.

**Key Features:**

*   **Bit-Endianness Control:** Specify either little-endian or big-endian representation for each bitarray object.
*   **Sequence-Like Operations:** Supports slicing, indexing, concatenation (+), repetition (*), in-place operations (+=, *=), `in` operator, and `len()`.
*   **Bitwise Operations:** Includes bitwise operators like `~`, `&`, `|`, `^`, `<<`, `>>` and their in-place counterparts (`&=`, `|=`, `^=`, `<<=`, `>>=`).
*   **Variable Bit Length Prefix Code Encoding/Decoding:** Fast methods for encoding and decoding using prefix codes, useful for data compression.
*   **Buffer Protocol Support:** Compatible with the buffer protocol, allowing import and export of buffers and integration with other data structures (e.g., memory-mapped files).
*   **Data Conversion:** Packing and unpacking to and from other binary data formats, such as `numpy.ndarray`.
*   **Pickling:** Supports pickling and unpickling of bitarray objects.
*   **Immutable `frozenbitarray`:** Includes immutable, hashable `frozenbitarray` objects for use as dictionary keys.
*   **Additional Utilities:** Provides utility functions in the `bitarray.util` module for:
    *   Hexadecimal string conversions
    *   Random bitarray generation
    *   Pretty printing
    *   Integer conversions
    *   Huffman code creation
    *   Compression of sparse bitarrays
    *   Serialization/deserialization
    *   Various counting functions
    *   More helpful functions.
*   **Extensive Testing:** Includes a comprehensive test suite with over 500 unit tests.
*   **Type Hinting:** Includes type hints for enhanced code clarity.

**Installation:**

Install bitarray using pip:

```bash
pip install bitarray
```

or with conda:

```bash
conda install bitarray
```

**Usage:**

Bitarray objects behave much like Python lists, providing a similar interface for common operations.  Key advantages of bitarray include efficient memory usage and fast bitwise operations.  See the original README for detailed usage examples.