# bitarray: Efficient Arrays of Booleans

**Representing arrays of booleans efficiently, bitarray provides a Python object type that behaves like a list but stores data at the bit level, saving memory and enabling fast bitwise operations.**  [View the original repository](https://github.com/ilanschnell/bitarray)

## Key Features:

*   **Bit-Endianness:** Choose between little-endian and big-endian representations for each bitarray.
*   **Sequence Operations:** Supports slicing, assignment, deletion, concatenation (`+`), repetition (`*`), in-place operations (`+=`, `*=`), the `in` operator, and `len()`.
*   **Bitwise Operations:** Includes bitwise NOT (`~`), AND (`&`), OR (`|`), XOR (`^`), left shift (`<<`), and right shift (`>>`), as well as in-place versions.
*   **Variable Bit Length Prefix Codes:** Fast methods for encoding and decoding data using prefix codes.
*   **Buffer Protocol Support:** Integrates with the Python buffer protocol for efficient data transfer.
*   **Integration with Other Libraries:** Compatible with formats like NumPy arrays for packing and unpacking.
*   **Immutability:** Provides `frozenbitarray` objects, which are hashable and suitable for use as dictionary keys.
*   **Additional Features:**
    *   Sequential search
    *   Type hinting
    *   Extensive test suite with over 500 unittests
*   **Utility Module (`bitarray.util`):** Offers helpful functions for:
    *   Hexadecimal string conversions
    *   Generating random bitarrays
    *   Pretty printing
    *   Integer conversions
    *   Huffman code creation
    *   Compression of sparse bitarrays
    *   Serialization/Deserialization
    *   Various count functions

## Installation

bitarray is available on PyPI, and you can install it using pip:

```bash
pip install bitarray
```

## Usage

Bitarray objects behave much like Python lists but store boolean values efficiently.

```python
from bitarray import bitarray

# Create an empty bitarray
a = bitarray()

# Append values
a.append(1)
a.extend([1, 0])
print(a) # Output: bitarray('110')

# Initialize from a string
b = bitarray('1001 011')
print(b) # Output: bitarray('1001011')

# Slicing
print(b[2:4]) # Output: bitarray('01')

# Bitwise operations
c = bitarray('101110001')
d = bitarray('111001011')
print(~c) # Output: bitarray('010001110')
print(c ^ d) # Output: bitarray('010111010')
```

## Bit-Endianness

The bit-endianness (big or little) affects how bits are represented in memory. It is important when working with the machine representation using `.tobytes()`, `.frombytes()`, `.tofile()`, `.fromfile()`, or `memoryview()`.  By default, bitarrays use a big-endian representation.

```python
from bitarray import bitarray

# Big-endian (default)
a = bitarray(b'A')
print(a)  # Output: bitarray('01000001')

# Little-endian
b = bitarray(b'A', endian='little')
print(b) # Output: bitarray('10000010')
```

## Further Information

*   **Reference:** Detailed information about bitarray object and its methods is available below.
*   **Buffer Protocol:** Further details on the buffer protocol integration can be found in the `buffer protocol <https://github.com/ilanschnell/bitarray/blob/master/doc/buffer.rst>`__ documentation.
*   **Examples:** Explore more examples, including memory-mapping files with bitarrays at `mmapped-file.py <https://github.com/ilanschnell/bitarray/blob/master/examples/mmapped-file.py>`__.
*   **Change Log:** See what's new and changed in the `change log <https://github.com/ilanschnell/bitarray/blob/master/doc/changelog.rst>`__.
*   **Bitarray representations:** See `Bitarray representations <https://github.com/ilanschnell/bitarray/blob/master/doc/represent.rst>`__
*   **Canonical Huffman Coding:** `Canonical Huffman Coding <https://github.com/ilanschnell/bitarray/blob/master/doc/canonical.rst>`__
*   **Random Bitarrays:** `Random Bitarrays <https://github.com/ilanschnell/bitarray/blob/master/doc/random_p.rst>`__
*   **Compression of sparse bitarrays:** `Compression of sparse bitarrays <https://github.com/ilanschnell/bitarray/blob/master/doc/sparse_compression.rst>`__
*   **Variable length bitarray format:** `Variable length bitarray format <https://github.com/ilanschnell/bitarray/blob/master/doc/variable_length.rst>`__

## Reference

#### The bitarray object:
*   ``bitarray(initializer=0, /, endian='big', buffer=None)`` -> bitarray

#### bitarray methods:
*   ``all()`` -> bool
*   ``any()`` -> bool
*   ``append(item, /)``
*   ``buffer_info()`` -> tuple
*   ``bytereverse(start=0, stop=<end of buffer>, /)``
*   ``clear()``
*   ``copy()`` -> bitarray
*   ``count(value=1, start=0, stop=<end>, step=1, /)`` -> int
*   ``decode(code, /)`` -> iterator
*   ``encode(code, iterable, /)``
*   ``extend(iterable, /)``
*   ``fill()`` -> int
*   ``find(sub_bitarray, start=0, stop=<end>, /, right=False)`` -> int
*   ``frombytes(bytes, /)``
*   ``fromfile(f, n=-1, /)``
*   ``index(sub_bitarray, start=0, stop=<end>, /, right=False)`` -> int
*   ``insert(index, value, /)``
*   ``invert(index=<all bits>, /)``
*   ``pack(bytes, /)``
*   ``pop(index=-1, /)`` -> item
*   ``remove(value, /)``
*   ``reverse()``
*   ``search(sub_bitarray, start=0, stop=<end>, /, right=False)`` -> iterator
*   ``setall(value, /)``
*   ``sort(reverse=False)``
*   ``to01(group=0, sep=' ')`` -> str
*   ``tobytes()`` -> bytes
*   ``tofile(f, /)``
*   ``tolist()`` -> list
*   ``unpack(zero=b'\x00', one=b'\x01')`` -> bytes

#### bitarray data descriptors:
*   ``endian`` -> str
*   ``nbytes`` -> int
*   ``padbits`` -> int
*   ``readonly`` -> bool

#### Other objects:
*   ``frozenbitarray(initializer=0, /, endian='big', buffer=None)`` -> frozenbitarray
*   ``decodetree(code, /)`` -> decodetree

#### Functions defined in the `bitarray` module:
*   ``bits2bytes(n, /)`` -> int
*   ``get_default_endian()`` -> str
*   ``test(verbosity=1)`` -> TextTestResult

#### Functions defined in `bitarray.util` module:
*   ``any_and(a, b, /)`` -> bool
*   ``ba2base(n, bitarray, /, group=0, sep=' ')`` -> str
*   ``ba2hex(bitarray, /, group=0, sep=' ')`` -> hexstr
*   ``ba2int(bitarray, /, signed=False)`` -> int
*   ``base2ba(n, asciistr, /, endian=None)`` -> bitarray
*   ``byteswap(a, n=<buffer size>, /)``
*   ``canonical_decode(bitarray, count, symbol, /)`` -> iterator
*   ``canonical_huffman(dict, /)`` -> tuple
*   ``correspond_all(a, b, /)`` -> tuple
*   ``count_and(a, b, /)`` -> int
*   ``count_n(a, n, value=1, /)`` -> int
*   ``count_or(a, b, /)`` -> int
*   ``count_xor(a, b, /)`` -> int
*   ``deserialize(bytes, /)`` -> bitarray
*   ``gen_primes(n, /, endian=None)`` -> bitarray
*   ``hex2ba(hexstr, /, endian=None)`` -> bitarray
*   ``huffman_code(dict, /, endian=None)`` -> dict
*   ``int2ba(int, /, length=None, endian=None, signed=False)`` -> bitarray
*   ``intervals(bitarray, /)`` -> iterator
*   ``ones(n, /, endian=None)`` -> bitarray
*   ``parity(a, /)`` -> int
*   ``pprint(bitarray, /, stream=None, group=8, indent=4, width=80)``
*   ``random_k(n, /, k, endian=None)`` -> bitarray
*   ``random_p(n, /, p=0.5, endian=None)`` -> bitarray
*   ``sc_decode(stream, /)`` -> bitarray
*   ``sc_encode(bitarray, /)`` -> bytes
*   ``serialize(bitarray, /)`` -> bytes
*   ``strip(bitarray, /, mode='right')`` -> bitarray
*   ``subset(a, b, /)`` -> bool
*   ``sum_indices(a, /, mode=1)`` -> int
*   ``urandom(n, /, endian=None)`` -> bitarray
*   ``vl_decode(stream, /, endian=None)`` -> bitarray
*   ``vl_encode(bitarray, /)`` -> bytes
*   ``xor_indices(a, /)`` -> int
*   ``zeros(n, /, endian=None)`` -> bitarray