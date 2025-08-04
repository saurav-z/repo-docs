# bitarray: Efficient Arrays of Booleans in Python

**The bitarray library provides an efficient and versatile way to represent and manipulate arrays of boolean values in Python, offering performance benefits and advanced features.**

[View the original repository on GitHub](https://github.com/ilanschnell/bitarray)

Key Features:

*   **Bit-Endianness Control:** Specify the bit-endianness (big or little) for each bitarray object.
*   **Sequence-Like Behavior:** Supports slicing (including assignment and deletion), `+`, `*`, `+=`, `*=`, `in`, and `len()`.
*   **Bitwise Operations:** Includes `~`, `&`, `|`, `^`, `<<`, `>>`, and their in-place counterparts.
*   **Variable Bit Length Prefix Codes:**  Fast encoding and decoding methods.
*   **Buffer Protocol Support:**  Implements the buffer protocol for efficient data transfer with other objects.
*   **Data Conversion:** Packing, unpacking, pickling, and unpickling for interaction with other binary data formats (e.g., NumPy arrays).
*   **Frozen Bitarrays:** Immutable, hashable `frozenbitarray` objects for use as dictionary keys.
*   **Search and Indexing:** Sequential search functionality.
*   **Type Hinting:**  Includes type hints for enhanced code readability and maintainability.
*   **Extensive Testing:**  Comprehensive test suite with over 500 unit tests.
*   **Utility Module:** `bitarray.util` with features like:
    *   Hexadecimal string conversions
    *   Random bitarray generation
    *   Pretty printing
    *   Integer conversions
    *   Huffman code creation
    *   Sparse bitarray compression
    *   Serialization/deserialization
    *   Various counting functions

## Installation

Install the bitarray package using pip:

```bash
pip install bitarray
```

Alternatively, install with conda:

```bash
conda install bitarray
```

You can test your installation with:

```bash
python -c 'import bitarray; bitarray.test()'
```

The tests' results can be verified by checking the return value's `wasSuccessful()` method.

## Usage

Bitarray objects behave similarly to Python lists but are optimized for boolean data.

```python
from bitarray import bitarray

# Create a bitarray
a = bitarray()
a.append(1)
a.extend([1, 0])
print(a)  # Output: bitarray('110')

# Initialize from string
b = bitarray('1001011')
print(b)  # Output: bitarray('1001011')

# Initialize from an iterable
lst = [1, 0, False, True, True]
c = bitarray(lst)
print(c)  # Output: bitarray('10011')

# Indexing, Slicing, and Operations
print(c[2])  # Output: 0
print(c[2:4]) # Output: bitarray('01')
print(c.count(1)) # Output: 3
```

Bitarrays support slice assignment and deletion.  Also, you can assign to booleans:

```python
a = bitarray(50)
a.setall(0)
a[11:37:3] = 9 * bitarray('1')
print(a)
```

## Bitwise Operations

Bitwise operators can be used on bitarray objects.

```python
a = bitarray('101110001')
b = bitarray('111001011')

print(~a) # Output: bitarray('010001110')
print(a ^ b) # Output: bitarray('010111010')
a &= b
print(a) # Output: bitarray('101000001')
```

## Bit-Endianness

When dealing with the machine representation (e.g., using `.tobytes()`), specify the bit-endianness (`big` or `little`).

```python
a = bitarray(b'A')
print(a.endian)  # Output: 'big'
print(a)  # Output: bitarray('01000001')
b = bitarray(b'A', endian='little')
print(b) # Output: bitarray('10000010')
```

## Buffer Protocol

Bitarray objects support the buffer protocol, enabling efficient data exchange with other Python objects.

## Variable Bit Length Prefix Codes

Use `.encode()` and `.decode()` to work with variable bit length prefix codes.

```python
from bitarray import bitarray

d = {'H':bitarray('111'), 'e':bitarray('0'),
     'l':bitarray('110'), 'o':bitarray('10')}

a = bitarray()
a.encode(d, 'Hello')
print(a)  # Output: bitarray('111011011010')

print(''.join(a.decode(d)))  # Output: Hello
```

The immutable `decodetree` object can be used to speed up repetitive decodings.

## Frozenbitarrays

Immutable and hashable.
```python
from bitarray import frozenbitarray
key = frozenbitarray('1100011')
print({key: 'some value'})
```

## Reference

See the original README for the latest bitarray version, changelog, methods, and other objects.
```