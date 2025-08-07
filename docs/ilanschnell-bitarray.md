# bitarray: Efficient Arrays of Booleans

**Representing and manipulating arrays of booleans with exceptional speed and efficiency, the `bitarray` library provides a powerful alternative to standard Python lists.** [View the project on GitHub](https://github.com/ilanschnell/bitarray).

## Key Features

*   **Bit-Endianness Control:** Specify big- or little-endian representation for your bitarrays.
*   **Sequence Operations:** Utilize familiar list-like methods, including slicing, concatenation (`+`), repetition (`*`), in-place operations (`+=`, `*=`), and the `in` operator, as well as `len()`.
*   **Bitwise Operations:** Perform bitwise operations like `~`, `&`, `|`, `^`, `<<`, `>>` (and their in-place counterparts: `&=`, `|=`, `^=`, `<<=`, `>>=`).
*   **Prefix Code Encoding/Decoding:** Fast encoding and decoding of variable bit length prefix codes for efficient data compression and representation.
*   **Buffer Protocol Support:** Seamless integration with other Python objects via the buffer protocol, allowing for efficient data exchange.
*   **Data Serialization/Deserialization:** Pack and unpack bitarrays to other binary data formats, like `numpy.ndarray`.
*   **Persistence with Pickling:** Serialize and deserialize bitarray objects for easy storage and retrieval.
*   **Immutable `frozenbitarray` Objects:** Create hashable, immutable bitarrays suitable for use as dictionary keys.
*   **Sequential Search:** Implement the efficient `search()` method to identify all matches of a sub-bitarray.
*   **Type Hinting:** Benefit from type hints for improved code readability and maintainability.
*   **Comprehensive Testing:** Benefit from a robust test suite with over 500 unit tests.
*   **Utility Module (`bitarray.util`):**
    *   Hexadecimal string conversions (to/from)
    *   Generation of random bitarrays
    *   Pretty printing
    *   Integer conversions (to/from)
    *   Huffman code generation
    *   Compression of sparse bitarrays
    *   Serialization and deserialization
    *   Various count functions
    *   Other useful functions

## Installation

`bitarray` is readily available on PyPI and conda, making installation straightforward.

```bash
# Using pip
pip install bitarray

# Using conda
conda install bitarray
```

After installation, test the library's functionality:

```bash
python -c 'import bitarray; bitarray.test()'
```

## Usage

`bitarray` objects behave similarly to Python lists, offering an intuitive interface with the addition of bit-level manipulation.

```python
from bitarray import bitarray

# Create an empty bitarray
a = bitarray()

# Append elements
a.append(1)
a.extend([1, 0])

print(a)  # Output: bitarray('110')

# Initialize from a string
x = bitarray('1001 011')
print(x)  # Output: bitarray('1001011')

# Initialize from an iterable
lst = [1, 0, False, True, True]
a = bitarray(lst)
print(a)  # Output: bitarray('10011')

# Indexing and slicing
print(a[2])   # Output: 0
print(a[2:4]) # Output: bitarray('01')

# Count occurrences
print(a.count(1)) # Output: 3

# Remove an element
a.remove(0)
print(a)  # Output: bitarray('1011')
```

Bitarrays support slice assignment and deletion, as well as bitwise operators.

```python
a = bitarray(50)
a.setall(0)
a[11:37:3] = 9 * bitarray('1')
print(a) # Output: bitarray('00000000000100100100100100100100100100000000000000')
del a[12::3]
print(a) # Output: bitarray('0000000000010101010101010101000000000')
a[-6:] = bitarray('10011')
print(a) # Output: bitarray('000000000001010101010101010100010011')
a += bitarray('000111')
print(a[9:]) # Output: bitarray('001010101010101010100010011000111')
```

## Bitwise Operators

Use bitwise operators directly on bitarray objects for efficient manipulation.

```python
a = bitarray('101110001')
print(~a)   # Output: bitarray('010001110')

b = bitarray('111001011')
print(a ^ b)  # Output: bitarray('010111010')

a &= b  # In-place AND
print(a)  # Output: bitarray('101000001')

a <<= 2  # In-place left-shift
print(a)  # Output: bitarray('100000100')

print(b >> 1) # Output: bitarray('011100101')
```

## Bit-Endianness

Understand the effect of bit-endianness, especially when interacting with the underlying machine representation using `.tobytes()`, `.frombytes()`, `.tofile()`, or `.fromfile()`. By default, bitarrays use big-endian representation, and the endianness is a property of the bitarray object.

```python
a = bitarray(b'A')
print(a.endian)  # Output: 'big'
print(a)  # Output: bitarray('01000001')

a = bitarray(b'A', endian='little')
print(a) # Output: bitarray('10000010')
```

## Buffer Protocol

`bitarray` objects implement the Python buffer protocol, enabling efficient data transfer with other Python objects. This enables data exchange with objects like `numpy.ndarray`.

## Variable Bit Length Prefix Codes

Encode and decode data using variable bit length prefix codes for compression and efficient data representation.
```python
from bitarray import bitarray, decodetree

d = {'H':bitarray('111'), 'e':bitarray('0'),
     'l':bitarray('110'), 'o':bitarray('10')}

a = bitarray()
a.encode(d, 'Hello')
print(a) # Output: bitarray('111011011010')

t = decodetree({'a': bitarray('0'), 'b': bitarray('1')})
a = bitarray('0110')
print(list(a.decode(t)))  # Output: ['a', 'b', 'b', 'a']
```

## frozenbitarray

Create immutable and hashable `frozenbitarray` objects for use as dictionary keys.

```python
from bitarray import frozenbitarray
key = frozenbitarray('1100011')
my_dict = {key: 'some value'}
print(my_dict) # Output: {frozenbitarray('1100011'): 'some value'}
```

## Reference

Refer to the documentation for a detailed explanation of all methods, functions, and data descriptors.