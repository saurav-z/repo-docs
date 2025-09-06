# bitarray: Efficient Arrays of Booleans

**Representing arrays of booleans efficiently, the `bitarray` library provides a fast and flexible way to work with bit-level data in Python.**  Find the original repo at [https://github.com/ilanschnell/bitarray](https://github.com/ilanschnell/bitarray).

**Key Features:**

*   **Bit-Endianness Control:** Specify `big` or `little` endianness for each `bitarray` object.
*   **Sequence-like Operations:** Supports slicing, `+`, `*`, `+=`, `*=`, `in`, and `len()`.
*   **Bitwise Operations:** Includes `~`, `&`, `|`, `^`, `<<`, `>>`, and their in-place counterparts.
*   **Prefix Code Handling:** Fast encoding and decoding of variable bit-length prefix codes for efficient data compression.
*   **Buffer Protocol Support:**  Compatible with the buffer protocol for importing/exporting buffers, including memory-mapped files.
*   **Data Conversion:** Supports packing/unpacking with other binary formats like NumPy arrays.
*   **Serialization:** Provides pickling and unpickling for persistence.
*   **Immutable Frozen Bitarrays:** Use `frozenbitarray` for hashable objects, ideal for dictionary keys.
*   **Efficient Search:** Includes sequential search capabilities.
*   **Type Hinting:** Improves code readability and maintainability.
*   **Extensive Testing:**  Comprehensive test suite with around 600 unit tests.
*   **Utility Module (bitarray.util):**
    *   Hexadecimal conversion
    *   Random bitarray generation
    *   Pretty printing
    *   Integer conversion
    *   Huffman code generation
    *   Sparse bitarray compression and decompression
    *   Serialization/deserialization
    *   Various counting functions
    *   And more!

## Installation

Easily install `bitarray` using pip:

```bash
pip install bitarray
```

You can also verify your installation with the built-in test function:

```bash
python -c 'import bitarray; bitarray.test()'
```

## Usage

`bitarray` objects behave like lists but store boolean values efficiently. You can create and manipulate them as follows:

```python
from bitarray import bitarray

a = bitarray()         # create empty bitarray
a.append(1)
a.extend([1, 0])
print(a) # Output: bitarray('110')

x = bitarray(2 ** 20)  # bitarray of length 1048576 (initialized to 0)
print(len(x)) # Output: 1048576

a = bitarray('1001 011')   # initialize from string (whitespace is ignored)
print(a) # Output: bitarray('1001011')

lst = [1, 0, False, True, True]
a = bitarray(lst)      # initialize from iterable
print(a) # Output: bitarray('10011')

print(a[2])    # indexing a single item will always return an integer - Output: 0
print(a[2:4])  # slicing will always return a bitarray - Output: bitarray('01')

print(a.count(1)) # Output: 3
a.remove(0)            # removes first occurrence of 0
print(a) # Output: bitarray('1011')
```

`bitarray` also supports slice assignment, deletion, and bitwise operations.

For more advanced usage, refer to the [Reference](#reference) section below.

## Bitwise Operators

```python
a = bitarray('101110001')
print(~a)  # invert - Output: bitarray('010001110')
b = bitarray('111001011')
print(a ^ b)  # bitwise XOR - Output: bitarray('010111010')
a &= b  # inplace AND
print(a) # Output: bitarray('101000001')
a <<= 2  # in-place left-shift by 2
print(a) # Output: bitarray('100000100')
print(b >> 1)  # return b right-shifted by 1 - Output: bitarray('011100101')
```

## Bit-Endianness

The bit-endianness (big-endian or little-endian) is relevant when working with binary representations (e.g., `tobytes()`, `frombytes()`), and with utility functions in `bitarray.util` like `ba2hex()` or `ba2int()`.  See the [bit-endianness documentation](https://github.com/ilanschnell/bitarray/blob/master/doc/endianness.rst) for details.

## Buffer Protocol

`bitarray` supports the buffer protocol, allowing efficient sharing of its internal data with other libraries. Explore the [buffer protocol documentation](https://github.com/ilanschnell/bitarray/blob/master/doc/buffer.rst) and the `mmapped-file.py` example.

## Variable Bit Length Prefix Codes

`bitarray` supports encoding and decoding data with prefix codes, often used in compression.

```python
d = {'H':bitarray('111'), 'e':bitarray('0'), 'l':bitarray('110'), 'o':bitarray('10')}
a = bitarray()
a.encode(d, 'Hello')
print(a)  # Output: bitarray('111011011010')
print(''.join(a.decode(d))) # Output: Hello
```

## Frozenbitarrays

```python
from bitarray import frozenbitarray
key = frozenbitarray('1100011')
print({key: 'some value'}) # Output: {frozenbitarray('1100011'): 'some value'}
# key[3] = 1  # This would raise a TypeError: frozenbitarray is immutable
```

## Reference

This section provides detailed information about the `bitarray` module's classes, methods, and functions.
```
... (Content from original README, starting at the line "bitarray version: 3.7.1 --" and onward)