# bitarray: Efficient Arrays of Booleans

**bitarray is a Python library that provides a memory-efficient way to represent and manipulate arrays of booleans, offering fast performance through C-based implementations.** [View the source code](https://github.com/ilanschnell/bitarray)

## Key Features

*   **Bit-Endianness Control:** Choose between big-endian and little-endian representations.
*   **Sequence-like Operations:** Supports slicing, concatenation (`+`), repetition (`*`), and membership (`in`), as well as `len()`.
*   **Bitwise Operations:** Includes bitwise operators (`~`, `&`, `|`, `^`, `<<`, `>>`) and in-place versions.
*   **Prefix Code Encoding/Decoding:** Fast methods for variable bit-length prefix code encoding and decoding, including Huffman codes.
*   **Buffer Protocol Support:**  Works with the Python buffer protocol, allowing interaction with other binary data formats.
*   **Data Serialization & Pickling:** Supports packing/unpacking, pickling, and unpickling.
*   **Immutable `frozenbitarray` Objects:**  Provides hashable, immutable bitarrays.
*   **Efficient Search:**  Includes fast sequential search capabilities.
*   **Type Hinting:**  Offers type hinting for improved code clarity and maintainability.
*   **Extensive Testing:**  Comes with a comprehensive test suite of approximately 600 unit tests.
*   **Utility Module `bitarray.util`:**
    *   Conversion to and from hexadecimal strings
    *   Generation of random bitarrays
    *   Pretty printing
    *   Conversion to and from integers
    *   Creation of Huffman codes
    *   Compression of sparse bitarrays
    *   (De-)serialization
    *   Various count functions
    *   Other helpful functions

## Installation

Install `bitarray` using pip:

```bash
pip install bitarray
```

Test your installation:

```python
import bitarray
bitarray.test()
```

## Usage

```python
from bitarray import bitarray

# Create an empty bitarray and append elements
a = bitarray()
a.append(1)
a.extend([1, 0])
print(a)  # Output: bitarray('110')

# Initialize from a string
b = bitarray('1001011')
print(b)

# Initialize from an iterable
lst = [1, 0, False, True, True]
c = bitarray(lst)
print(c)  # Output: bitarray('10011')

# Access elements
print(c[2])   # Output: 0
print(c[2:4]) # Output: bitarray('01')

# Count occurrences
print(c.count(1))  # Output: 3

# Slice assignment and deletion
d = bitarray(50)
d.setall(0)
d[11:37:3] = 9 * bitarray('1')
print(d)
del d[12::3]
print(d)
d[-6:] = bitarray('10011')
print(d)
d += bitarray('000111')
print(d[9:])

#Assign a boolean value to a slice:
a = 20 * bitarray('0')
a[1:15:3] = True
print(a)
```

## Bitwise Operators

```python
a = bitarray('101110001')
print(~a)   # Output: bitarray('010001110')
b = bitarray('111001011')
print(a ^ b)  # Output: bitarray('010111010')
a &= b
print(a)    # Output: bitarray('101000001')
a <<= 2
print(a)    # Output: bitarray('100000100')
print(b >> 1)  # Output: bitarray('011100101')
```

## Bit-Endianness

Bit-endianness affects how the bits are arranged in memory. It's important when working with the bitarray buffer directly (e.g., `tobytes()`, `frombytes()`) or when using utility functions like `ba2hex()` and `ba2int()`.

## Buffer Protocol

`bitarray` objects support the buffer protocol.  See the documentation for more information.

## Variable Bit Length Prefix Codes

The `encode()` method allows encoding symbols using a prefix code (like Huffman coding) to extend the bitarray:

```python
d = {'H':bitarray('111'), 'e':bitarray('0'),'l':bitarray('110'), 'o':bitarray('10')}
a = bitarray()
a.encode(d, 'Hello')
print(a) # Output: bitarray('111011011010')
```

The `decode()` method returns an iterator over the symbols.  For faster decoding, create a `decodetree` object:

```python
from bitarray import bitarray, decodetree
t = decodetree({'a': bitarray('0'), 'b': bitarray('1')})
a = bitarray('0110')
print(list(a.decode(t)))  # Output: ['a', 'b', 'b', 'a']
```

## Frozenbitarrays

```python
from bitarray import frozenbitarray
key = frozenbitarray('1100011')
print({key: 'some value'}) # Output: {frozenbitarray('1100011'): 'some value'}