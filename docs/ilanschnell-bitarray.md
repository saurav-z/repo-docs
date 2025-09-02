# bitarray: Efficient Arrays of Booleans

**bitarray is a Python library providing an efficient, sequence-like object for representing and manipulating arrays of booleans, implemented in C for speed and performance.** [View the original repository](https://github.com/ilanschnell/bitarray)

## Key Features

*   **Bit-Endianness Control:** Specify the bit-endianness (big or little-endian) for each bitarray object.
*   **Sequence Operations:** Supports slicing, assignment, deletion, and operations like `+`, `*`, `+=`, `*=`, and `in`, as well as `len()`.
*   **Bitwise Operations:** Includes bitwise operators such as `~`, `&`, `|`, `^`, `<<`, `>>` and in-place versions.
*   **Variable Bit Length Prefix Codes:** Fast methods for encoding and decoding prefix codes.
*   **Buffer Protocol Support:** Integrates with the Python buffer protocol for efficient data exchange and memory mapping.
*   **Data Interchange:** Supports packing/unpacking with binary data formats (e.g., `numpy.ndarray`).
*   **Serialization:** Offers pickling and unpickling for easy object persistence.
*   **Immutable `frozenbitarray`:** Provides immutable, hashable objects for use as dictionary keys.
*   **Advanced Functionality:**
    *   Sequential Search
    *   Type Hinting
    *   Extensive Testing
    *   Utility module `bitarray.util`:
        *   Conversion to/from hexadecimal strings
        *   Generation of random bitarrays
        *   Pretty printing
        *   Conversion to/from integers
        *   Huffman code creation
        *   Compression of sparse bitarrays
        *   Serialization/Deserialization
        *   Various count functions
        *   and many more helpful functions.

## Installation

Install the `bitarray` package using pip:

```bash
pip install bitarray
```

After installation, verify with a test:

```bash
python -c 'import bitarray; bitarray.test()'
```

The `test()` function is part of the API, and can be used to verify that all tests ran successfully:

```python
import bitarray
assert bitarray.test().wasSuccessful()
```

## Usage

Bitarrays behave similarly to lists, but store boolean values in a compact, efficient manner.  Key differences from lists include the ability to access the machine representation and setting bit-endianness.

```python
from bitarray import bitarray

# Create a bitarray
a = bitarray()
a.append(1)
a.extend([1, 0])  # a is bitarray('110')

# Create a bitarray of a specific length, initialized to zeros
x = bitarray(2 ** 20)  # bitarray of length 1048576 (initialized to 0)

# Initialize from a string (whitespace is ignored)
b = bitarray('1001 011') # bitarray('1001011')

# Initialize from an iterable
lst = [1, 0, False, True, True]
c = bitarray(lst)  # bitarray('10011')

# Indexing and slicing
print(c[2])    # Output: 0
print(c[2:4])  # Output: bitarray('01')

# Count occurrences
print(c.count(1)) # Output: 3

# Remove the first occurrence of a value
c.remove(0)   # c becomes bitarray('1011')

# Slice assignment and deletion
d = bitarray(50)
d.setall(0)  # Set all elements in d to 0
d[11:37:3] = 9 * bitarray('1')
del d[12::3]
d[-6:] = bitarray('10011')
d += bitarray('000111')

# Slice assignment with booleans
e = 20 * bitarray('0')
e[1:15:3] = True
```

## Bitwise Operators

Bitarray objects support standard bitwise operators:

```python
a = bitarray('101110001')
print(~a)  # Output: bitarray('010001110')
b = bitarray('111001011')
print(a ^ b) # Output: bitarray('010111010')
a &= b # Inplace AND
print(a) # Output: bitarray('101000001')
a <<= 2 # Inplace left-shift by 2
print(a) # Output: bitarray('100000100')
print(b >> 1) # Output: bitarray('011100101')
```

## Bit-Endianness

The bit-endianness can affect the buffer representation of the bitarray.  This is important when using `.tobytes()`, `.frombytes()`, `.tofile()`, `.fromfile()` and when importing/exporting buffers.

## Buffer Protocol

Bitarray objects can both export their own buffer, and import another object's buffer.  For more information, see the [buffer protocol documentation](https://github.com/ilanschnell/bitarray/blob/master/doc/buffer.rst).  See the example for [memory-mapping files](https://github.com/ilanschnell/bitarray/blob/master/examples/mmapped-file.py)

## Variable Bit Length Prefix Codes

The `.encode()` and `.decode()` methods enable the use of prefix codes.

```python
from bitarray import bitarray, decodetree

# Huffman Coding Example
d = {'H': bitarray('111'), 'e': bitarray('0'), 'l': bitarray('110'), 'o': bitarray('10')}
a = bitarray()
a.encode(d, 'Hello')  # a is bitarray('111011011010')
print(''.join(a.decode(d))) # Output: Hello

# Using decodetree for faster decoding
t = decodetree({'a': bitarray('0'), 'b': bitarray('1')})
a = bitarray('0110')
print(list(a.decode(t))) # Output: ['a', 'b', 'b', 'a']
```

## Frozenbitarrays

Frozenbitarrays are immutable, hashable versions of bitarrays that can be used as dictionary keys.

```python
from bitarray import frozenbitarray

key = frozenbitarray('1100011')
my_dict = {key: 'some value'}
print(my_dict) # Output: {frozenbitarray('1100011'): 'some value'}
```

## Reference

For detailed information on methods, data descriptors, other objects and functions, please refer to the original documentation.  See bitarray [version: 3.7.1](https://github.com/ilanschnell/bitarray/blob/master/doc/changelog.rst)