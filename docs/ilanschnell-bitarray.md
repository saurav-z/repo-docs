# bitarray: Efficient Arrays of Booleans

**bitarray is a Python library that provides an efficient and versatile way to represent and manipulate arrays of bits.**  [View the original repository](https://github.com/ilanschnell/bitarray).

Key Features:

*   **Bit-Endianness Control:** Choose between big-endian and little-endian representations.
*   **Sequence Type Functionality:** Supports slicing, concatenation (`+`), repetition (`*`), the `in` operator, and `len()`.
*   **Bitwise Operations:** Includes `~`, `&`, `|`, `^`, `<<`, `>>` (and in-place versions).
*   **Prefix Code Encoding/Decoding:** Fast methods for encoding and decoding variable bit-length prefix codes, including Huffman codes.
*   **Buffer Protocol Support:** Enables importing and exporting buffers for integration with other objects, including memory-mapped files and NumPy arrays.
*   **Binary Data Format Conversion:** Packing and unpacking to other binary data formats like NumPy.
*   **Pickling:** Supports pickling and unpickling for serialization.
*   **Frozen Bitarrays:** Immutable `frozenbitarray` objects for use as dictionary keys.
*   **Additional Utilities:**
    *   Conversion to and from hexadecimal strings
    *   Random bitarray generation
    *   Pretty printing
    *   Conversion to and from integers
    *   Huffman code creation
    *   Compression of sparse bitarrays
    *   Serialization/Deserialization
    *   Count and utility functions
*   **Extensive Testing:** Includes a comprehensive test suite with over 500 unit tests.
*   **Type Hinting:** Supports type hinting for improved code readability and maintainability.

## Installation

Install bitarray using `pip`:

```bash
pip install bitarray
```

Conda packages are also available:

```bash
conda install bitarray
```

## Usage

Bitarray objects function similarly to Python lists, with a focus on efficient bitwise operations.

```python
from bitarray import bitarray

# Create and modify bitarrays
a = bitarray()
a.append(1)
a.extend([1, 0])
print(a)  # Output: bitarray('110')

# Initialize from string or iterable
x = bitarray('1001011')
lst = [1, 0, False, True, True]
a = bitarray(lst)

# Indexing and slicing
print(a[2])  # Output: 0
print(a[2:4])  # Output: bitarray('01')

# Count and remove
print(a.count(1))  # Output: 3
a.remove(0)

# Slice assignment
a = bitarray(50)
a.setall(0)
a[11:37:3] = 9 * bitarray('1')
del a[12::3]
a[-6:] = bitarray('10011')
a += bitarray('000111')

# Slice assignment with booleans
a = 20 * bitarray('0')
a[1:15:3] = True
```

## Bitwise Operations

```python
a = bitarray('101110001')
print(~a)  # Output: bitarray('010001110')
b = bitarray('111001011')
print(a ^ b)  # Output: bitarray('010111010')
a &= b
print(a)  # Output: bitarray('101000001')
a <<= 2
print(a)  # Output: bitarray('100000100')
print(b >> 1)  # Output: bitarray('011100101')
```

## Bit-Endianness

When dealing with the machine representation, bit-endianness (big-endian or little-endian) is important. The default is big-endian.

```python
a = bitarray(b'A')
print(a.endian)  # Output: big
print(a)  # Output: bitarray('01000001')

a = bitarray(b'A', endian='little')
print(a)  # Output: bitarray('10000010')
```

Bitwise operations act on the machine representation, so bitarrays with different endianness cannot be operated on directly.  However, you can easily create a new bitarray with a different endianness:

```python
a = bitarray('111000', endian='little')
b = bitarray(a, endian='big')
print(a == b) #Output: True
```

## Buffer Protocol

bitarray objects implement the buffer protocol.  This allows efficient interaction with other Python libraries and data structures that support buffer interfaces.

## Variable Bit Length Prefix Codes

```python
from bitarray import bitarray, decodetree

# Encoding and Decoding
d = {'H':bitarray('111'), 'e':bitarray('0'),
     'l':bitarray('110'), 'o':bitarray('10')}
a = bitarray()
a.encode(d, 'Hello')
print(a)  # Output: bitarray('111011011010')
print(''.join(a.decode(d))) # Output: Hello

# Using decodetree for improved performance
t = decodetree({'a': bitarray('0'), 'b': bitarray('1')})
a = bitarray('0110')
print(list(a.decode(t)))  # Output: ['a', 'b', 'b', 'a']
```

## Frozenbitarrays

Immutable and hashable objects that can be used as dictionary keys.

```python
from bitarray import frozenbitarray
key = frozenbitarray('1100011')
my_dict = {key: 'some value'}