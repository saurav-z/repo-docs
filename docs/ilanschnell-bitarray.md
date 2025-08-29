# bitarray: Efficient Arrays of Booleans

**bitarray is a Python library that provides a memory-efficient and versatile way to work with arrays of boolean values.** It offers a wide range of features and optimizations for various data processing tasks.  [View the original repository](https://github.com/ilanschnell/bitarray).

## Key Features

*   **Bit-Endianness Control:** Specify the bit-endianness (big or little-endian) for each bitarray object.
*   **Sequence Methods:** Supports slicing (including slice assignment and deletion), `+`, `*`, `+=`, `*=`, `in`, and `len()`.
*   **Bitwise Operations:** Includes `~`, `&`, `|`, `^`, `<<`, `>>` and their in-place counterparts `&=`, `|=`, `^=`, `<<=`, `>>=`.
*   **Prefix Code Encoding/Decoding:** Fast methods for encoding and decoding variable bit-length prefix codes.
*   **Buffer Protocol Support:** Allows importing and exporting buffers, enabling interaction with other Python objects and memory-mapped files.
*   **Data Serialization:** Packing and unpacking to other binary data formats, such as NumPy arrays.
*   **Pickling and Hashable Objects:** Supports pickling/unpickling of bitarray objects and provides immutable `frozenbitarray` objects, which are hashable.
*   **Additional Utilities:** `bitarray.util` module provides functions for:
    *   Hexadecimal conversion
    *   Random bitarray generation
    *   Pretty printing
    *   Integer conversion
    *   Huffman code creation
    *   Sparse bitarray compression
    *   (De-)serialization
    *   Counting and other helpful operations

## Installation

Install `bitarray` easily using pip:

```bash
pip install bitarray
```

You can verify the installation and run tests with:

```bash
python -c 'import bitarray; bitarray.test()'
```

## Usage

Bitarray objects behave much like Python lists, but efficiently store booleans as bits.

```python
from bitarray import bitarray

# Create an empty bitarray
a = bitarray()

# Append and extend
a.append(1)
a.extend([1, 0])
print(a)  # Output: bitarray('110')

# Initialize from an iterable
lst = [1, 0, False, True, True]
a = bitarray(lst)
print(a)  # Output: bitarray('10011')

# Accessing single bits and slices
print(a[2])  # Output: 0
print(a[2:4])  # Output: bitarray('01')

# Count occurrences
print(a.count(1)) # Output: 3

# Slice Assignment
a = bitarray(50)
a.setall(0) # set all elements to 0
a[11:37:3] = 9 * bitarray('1')
print(a) # Output: bitarray('00000000000100100100100100100100100100000000000000')
```

## Bitwise Operators

Bitwise operators are supported, providing a way to perform logical operations directly on the bitarrays.

```python
a = bitarray('101110001')
b = bitarray('111001011')

print(~a)  # Output: bitarray('010001110')
print(a ^ b) # Output: bitarray('010111010')
a &= b
print(a)  # Output: bitarray('101000001')
```

## Bit-Endianness

Bit-endianness determines how bits are arranged in memory.  It affects buffer-related operations like `tobytes()`, `frombytes()`, and some functions in `bitarray.util`.

## Contributing

Contributions are welcome! Please see the [contributing guidelines](https://github.com/ilanschnell/bitarray/blob/master/CONTRIBUTING.rst) for details.

## Documentation

Comprehensive documentation can be found in the repository, including detailed information on:

*   [Bit-endianness](https://github.com/ilanschnell/bitarray/blob/master/doc/endianness.rst)
*   [Buffer protocol](https://github.com/ilanschnell/bitarray/blob/master/doc/buffer.rst)
*   [Variable length bitarray format](https://github.com/ilanschnell/bitarray/blob/master/doc/variable_length.rst)
*   [Bitarray indexing](https://github.com/ilanschnell/bitarray/blob/master/doc/indexing.rst)

## License

This project is licensed under the [MIT License](https://github.com/ilanschnell/bitarray/blob/master/LICENSE).