# bitarray: Efficient Arrays of Booleans - Fast, Flexible, and Feature-Rich

**Optimize your binary data manipulation with bitarray, a Python library that provides highly efficient arrays of booleans, offering list-like behavior and powerful bitwise operations.**  [View the project on GitHub](https://github.com/ilanschnell/bitarray)

## Key Features

*   **Endianness Control:** Specify big- or little-endian bit representation.
*   **Sequence-Like Operations:**  Supports slicing (including assignment and deletion), concatenation (`+`), repetition (`*`), membership (`in`), and length (`len()`).
*   **Bitwise Operations:** Includes bitwise NOT (`~`), AND (`&`), OR (`|`), XOR (`^`), left shift (`<<`), and right shift (`>>`), along with their in-place counterparts.
*   **Prefix Code Encoding/Decoding:**  Fast methods for encoding and decoding variable-length prefix codes.
*   **Buffer Protocol Support:**  Supports the buffer protocol for importing and exporting data, enabling integration with other objects and memory-mapped files.
*   **Data Conversion:** Offers packing and unpacking to other binary formats, like `numpy.ndarray`.
*   **Immutability:**  Includes `frozenbitarray` objects for immutable and hashable bit arrays.
*   **Utilities Module:** Provides a rich `bitarray.util` module for:
    *   Hexadecimal conversions
    *   Random bitarray generation
    *   Pretty printing
    *   Integer conversions
    *   Huffman code generation
    *   Sparse bitarray compression
    *   Serialization/Deserialization
    *   Various counting and other helpful functions.
*   **Extensive Testing:**  Comprehensive test suite with over 500 unit tests.
*   **Type Hinting**
*   **Sequential Search**

## Installation

Install bitarray using pip:

```bash
pip install bitarray
```

You can test your installation with the following command:

```bash
python -c 'import bitarray; bitarray.test()'
```

## Usage

bitarray objects function similarly to Python lists, allowing you to store and manipulate arrays of boolean values efficiently.

```python
from bitarray import bitarray

# Create an empty bitarray
a = bitarray()

# Append and extend elements
a.append(1)
a.extend([1, 0])

# Initialize from a string
b = bitarray('1001011')

# Initialize from an iterable
lst = [1, 0, False, True, True]
c = bitarray(lst)

# Access elements
print(c[2])  # Accessing a single item returns an integer
print(c[2:4])  # Slicing returns a bitarray

# Count occurrences
print(c.count(1))

# Slice assignment and deletion
d = bitarray(50)
d.setall(0)
d[11:37:3] = 9 * bitarray('1')
del d[12::3]
```

## Bitwise Operators

bitarray supports bitwise operators, making it easy to perform efficient binary operations.

```python
a = bitarray('101110001')
b = bitarray('111001011')

print(~a)  # Invert
print(a ^ b)  # XOR
a &= b  # In-place AND
print(a <<= 2)  # In-place left-shift
print(b >> 1)  # Right-shift
```

## Bit-Endianness

Control the bit representation of your bitarrays using big- or little-endian format.

```python
a = bitarray(b'A')  # Default is big-endian
print(a.endian)
print(a)
print(a[6] = 1)
print(a.tobytes())

b = bitarray(b'A', endian='little')
print(b)
print(b.endian)
```

**Note:** bitwise operations use the machine representation; bit-endianness only matters when interacting with the underlying byte representation (e.g., using `.tobytes()`).

## Buffer Protocol

bitarray objects support the buffer protocol, allowing efficient integration with other Python libraries that use memory buffers.

## Additional Resources

*   [Bitarray 3 transition](https://github.com/ilanschnell/bitarray/blob/master/doc/bitarray3.rst)
*   [Bitarray representations](https://github.com/ilanschnell/bitarray/blob/master/doc/represent.rst)
*   [Canonical Huffman Coding](https://github.com/ilanschnell/bitarray/blob/master/doc/canonical.rst)
*   [Compression of sparse bitarrays](https://github.com/ilanschnell/bitarray/blob/master/doc/sparse_compression.rst)
*   [Variable length bitarray format](https://github.com/ilanschnell/bitarray/blob/master/doc/variable_length.rst)
*   [Random Bitarrays](https://github.com/ilanschnell/bitarray/blob/master/doc/random_p.rst)