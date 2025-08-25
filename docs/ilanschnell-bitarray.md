# bitarray: Efficient Arrays of Booleans

**Representing arrays of booleans efficiently, this library allows for fast bitwise operations, buffer manipulation, and integration with other Python tools.**  Explore the original repository [here](https://github.com/ilanschnell/bitarray).

## Key Features

*   **Bit-Endianness Control:** Specify little-endian or big-endian representation for each bitarray object.
*   **Sequence Operations:** Supports slicing (including assignment and deletion), `+`, `*`, `+=`, `*=`, `in` operator, and `len()`.
*   **Bitwise Operations:** Includes `~`, `&`, `|`, `^`, `<<`, `>>` and in-place versions.
*   **Variable Bit Length Codes:** Fast encoding and decoding methods for prefix codes.
*   **Buffer Protocol Support:**  Import and export buffers.
*   **Integration with Other Formats:** Packing and unpacking to formats like `numpy.ndarray`.
*   **Pickling and Hashable Objects:** Pickling and unpickling for persistence, and immutable `frozenbitarray` objects for use as dictionary keys.
*   **Utilities:**  Conversion to/from hex strings, random bitarray generation, pretty printing, Huffman coding, compression, and more in the `bitarray.util` module.
*   **Extensive Testing:** Comprehensive test suite with nearly 600 unittests.
*   **Type Hinting:** Includes type hints for improved code readability and maintainability.

## Installation

Install the `bitarray` package easily using pip:

```bash
pip install bitarray
```

After installation, verify with:

```bash
python -c 'import bitarray; bitarray.test()'
```

## Usage

`bitarray` objects behave like lists, but with the efficiency of bit-level storage. The key difference is the ability to access and manipulate the underlying machine representation.

```python
from bitarray import bitarray

# Create and modify bitarrays
a = bitarray()
a.append(1)
a.extend([1, 0])
print(a)  # Output: bitarray('110')

x = bitarray(2 ** 20)  # Initialize with length 1048576 (all zeros)

# Initialize from string or iterable
a = bitarray('1001 011')
lst = [1, 0, False, True, True]
a = bitarray(lst)

# Indexing and slicing
print(a[2])
print(a[2:4])

# Count, Remove, Slice Assignment
print(a.count(1))
a.remove(0)
a[11:37:3] = 9 * bitarray('1') # slice assignment
```

## Bitwise Operators

Bitwise operations are supported:

```python
a = bitarray('101110001')
b = bitarray('111001011')

print(~a)    # Invert
print(a ^ b)  # XOR
a &= b       # In-place AND
a <<= 2      # In-place left-shift
```

## Bit-Endianness

Bit-endianness is relevant when working directly with the bitarray buffer (using methods like `tobytes()`) or when using utility functions like `ba2hex()` and `ba2int()`. Read more about it here:  [`bit-endianness <https://github.com/ilanschnell/bitarray/blob/master/doc/endianness.rst>`].

## Buffer Protocol

`bitarray` supports the buffer protocol, allowing interaction with other buffer-compatible objects. Learn more:  [`buffer protocol <https://github.com/ilanschnell/bitarray/blob/master/doc/buffer.rst>`].

## Variable Bit Length Prefix Codes

Encode and decode data using prefix codes (e.g., Huffman codes):

```python
d = {'H': bitarray('111'), 'e': bitarray('0'), 'l': bitarray('110'), 'o': bitarray('10')}
a = bitarray()
a.encode(d, 'Hello')
print(a)  # Output: bitarray('111011011010')
print(''.join(a.decode(d))) # Output: Hello
```

Use `decodetree` for performance improvements.

## Frozenbitarrays

Immutable, hashable `frozenbitarray` objects can be used as dictionary keys.

```python
from bitarray import frozenbitarray

key = frozenbitarray('1100011')
my_dict = {key: 'some value'}
print(my_dict)
```

## Reference

Complete details of the bitarray methods are available in the original repository.