# bitarray: Efficient Arrays of Booleans in Python

**Quickly and efficiently represent arrays of booleans with the `bitarray` library, offering a fast and memory-efficient alternative to standard Python lists.**  

[View the Original Repository](https://github.com/ilanschnell/bitarray)

## Key Features

*   **Bit-Endianness Control:**  Specify either big-endian or little-endian representation for each bitarray.
*   **Sequence Type Functionality:** Supports slicing, concatenation (+), repetition (\*), `in` operator, and `len()`.
*   **Bitwise Operations:** Includes bitwise NOT (~), AND (&), OR (|), XOR (^), left shift (<<), and right shift (>>), with in-place versions (&=, |=, ^=, <<=, >>=).
*   **Variable-Length Prefix Code Encoding/Decoding:**  Fast methods for encoding and decoding prefix codes.
*   **Buffer Protocol Support:** Compatible with the buffer protocol, allowing import/export of buffers.
*   **Integration with Other Libraries:** Works with other binary data formats, such as NumPy's `ndarray`.
*   **Pickling/Unpickling:** Supports pickling and unpickling bitarray objects.
*   **Immutable Frozen Bitarrays:** Provides immutable and hashable `frozenbitarray` objects for use as dictionary keys.
*   **Efficient Search:** Features sequential search capabilities.
*   **Type Hinting:** Includes type hints for better code readability and maintainability.
*   **Extensive Testing:** Comprehensive test suite with over 500 unittests.
*   **Utility Module (bitarray.util):** Offers a range of helpful functions for:
    *   Hexadecimal string conversions
    *   Generating random bitarrays
    *   Pretty-printing bitarrays
    *   Integer conversions
    *   Huffman code creation
    *   Sparse bitarray compression/decompression
    *   Serialization/Deserialization
    *   Various count functions

## Installation

Install `bitarray` using pip:

```bash
pip install bitarray
```

Verify the installation and run tests:

```bash
python -c 'import bitarray; bitarray.test()'
```

## Usage

`bitarray` objects behave much like Python lists, but are optimized for storing booleans. The most significant difference is the ability to work with machine representation.

```python
from bitarray import bitarray

# Create a bitarray
a = bitarray()
a.append(1)
a.extend([1, 0])
print(a)  # Output: bitarray('110')

# Initialize from a string
b = bitarray('10101')
print(b) # Output: bitarray('10101')

# Initialize from an iterable
c = bitarray([1, 0, 0, 1, 1])
print(c) # Output: bitarray('10011')

# Indexing and slicing
print(c[2])    # Output: 0
print(c[2:4])  # Output: bitarray('01')
```

Bitarray objects support slice assignment and deletion:

```python
a = bitarray(50)
a.setall(0)            # set all elements in a to 0
a[11:37:3] = 9 * bitarray('1')
print(a) # Output: bitarray('00000000000100100100100100100100100100000000000000')

del a[12::3]
print(a) # Output: bitarray('0000000000010101010101010101000000000')

a[-6:] = bitarray('10011')
print(a) # Output: bitarray('000000000001010101010101010100010011')
```

## Bitwise Operators

Bitarray objects support bitwise operators:

```python
a = bitarray('101110001')
b = bitarray('111001011')

print(~a) # bitwise NOT
print(a ^ b)  # bitwise XOR
a &= b  # in-place AND
print(a)

a <<= 2  # in-place left-shift by 2
print(a)
```

## Bit-Endianness

Bitarrays support both big-endian and little-endian representations, which affects how the bits are interpreted when interacting with the underlying machine representation.

*   **Big-Endian (Default):** The most significant bit is stored first.
*   **Little-Endian:** The least significant bit is stored first.

```python
a = bitarray(b'A')  # big-endian by default
print(a)  # Output: bitarray('01000001')

a = bitarray(b'A', endian='little')
print(a) # Output: bitarray('10000010')

```

## Frozenbitarrays

Frozenbitarrays are immutable versions of bitarrays. They can be used as dictionary keys.

```python
from bitarray import frozenbitarray

key = frozenbitarray('1100011')
my_dict = {key: 'some value'}
print(my_dict) # Output: {frozenbitarray('1100011'): 'some value'}
```

## Further Information

Refer to the original repository for:

*   [Full Reference](https://github.com/ilanschnell/bitarray/blob/master/README.md)
*   [Change Log](https://github.com/ilanschnell/bitarray/blob/master/doc/changelog.rst)
*   [Documentation](https://github.com/ilanschnell/bitarray)