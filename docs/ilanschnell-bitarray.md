# bitarray: Efficient Arrays of Booleans

**bitarray** provides a fast and efficient way to work with arrays of booleans in Python, offering a memory-efficient alternative to standard lists.  [View the original repository](https://github.com/ilanschnell/bitarray).

**Key Features:**

*   **Bit-Endianness Control:** Choose between little-endian and big-endian representations.
*   **Sequence Type Functionality:** Supports slicing, operations (+, \*, +=, \*=, `in`), and `len()`.
*   **Bitwise Operations:**  Includes `~`, `&`, `|`, `^`, `<<`, `>>` and their in-place equivalents.
*   **Variable Bit Length Prefix Codes:**  Fast methods for encoding and decoding.
*   **Buffer Protocol Support:**  Allows importing and exporting buffers, enabling integration with memory-mapped files and other objects.
*   **Data Conversion:** Packing and unpacking to/from other binary data formats, e.g., `numpy.ndarray`.
*   **Serialization:** Pickling and unpickling of bitarray objects.
*   **Frozen Bitarrays:** Immutable `frozenbitarray` objects for use as dictionary keys.
*   **Search Algorithms:** Includes sequential search.
*   **Type Hinting:** Supports type hints for improved code readability and maintainability.
*   **Extensive Testing:** Comprehensive test suite with over 500 unit tests.
*   **Utility Module (`bitarray.util`):**
    *   Hexadecimal string conversions
    *   Random bitarray generation
    *   Pretty printing
    *   Integer conversions
    *   Huffman code generation
    *   Sparse bitarray compression/decompression
    *   Serialization/Deserialization
    *   Various count functions
    *   Other helpful functions

## Installation

Install `bitarray` using pip:

```bash
pip install bitarray
```

To verify installation, you can run the tests:

```bash
python -c 'import bitarray; bitarray.test()'
```

## Usage

`bitarray` objects largely behave like Python lists, but store booleans efficiently.  The primary difference lies in the ability to control bit-endianness for machine representation, important when working with machine-level byte representations (e.g.  `.tobytes()`, `.frombytes()`).

```python
from bitarray import bitarray

# Create an empty bitarray
a = bitarray()
a.append(1)
a.extend([1, 0])
print(a)  # Output: bitarray('110')

# Create a bitarray from a string (whitespace is ignored)
b = bitarray('1001 011')
print(b)  # Output: bitarray('1001011')

# Initialize from an iterable
lst = [1, 0, False, True, True]
c = bitarray(lst)
print(c)  # Output: bitarray('10011')

# Indexing and Slicing
print(c[2])    # Output: 0
print(c[2:4])  # Output: bitarray('01')
```

Bitwise operators and slice assignments are also supported:

```python
a = bitarray('101110001')
b = bitarray('111001011')

print(~a)  # Invert
print(a ^ b) # XOR
a &= b     # In-place AND
print(a <<= 2) # In-place left-shift by 2
```

## Additional Resources

*   [Bitarray documentation](https://github.com/ilanschnell/bitarray/blob/master/doc/index.rst)
*   [Change Log](https://github.com/ilanschnell/bitarray/blob/master/doc/changelog.rst)