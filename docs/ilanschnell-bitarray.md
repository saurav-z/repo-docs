# bitarray: Efficient Arrays of Booleans in Python

**Representing and manipulating boolean arrays efficiently, the bitarray library provides a fast and memory-conscious alternative to Python lists.**  Explore the power of bitarrays, which offer optimized performance and flexibility for various data processing tasks.  [View the original repository](https://github.com/ilanschnell/bitarray).

## Key Features

*   **Bit-Endianness Control:** Specify either little- or big-endian representation for each bitarray object.
*   **Sequence-like Behavior:** Leverage familiar sequence methods such as slicing, concatenation (+), repetition (*), and the `in` operator, along with `len()`.
*   **Bitwise Operations:** Perform bitwise operations including NOT (~), AND (&), OR (|), XOR (^), left shift (<<), and right shift (>>), along with their in-place versions (&=, |=, ^=, <<=, >>=).
*   **Variable-Length Prefix Codes:** Rapidly encode and decode variable bit length prefix codes.
*   **Buffer Protocol Support:** Utilize the buffer protocol for importing and exporting buffers, integrating seamlessly with other Python objects.
*   **Data Conversion:** Pack and unpack to other binary data formats, such as `numpy.ndarray`.
*   **Serialization:** Includes pickling and unpickling of bitarray objects.
*   **Frozen Bitarrays:** Create immutable (`frozenbitarray`) objects for hashable operations.
*   **Additional Utility Modules:** A wide array of utilities, including:
    *   Conversion to and from hexadecimal strings
    *   Random bitarray generation
    *   Pretty printing
    *   Conversion to and from integers
    *   Huffman code creation
    *   Compression of sparse bitarrays
    *   Serialization and deserialization
    *   Various count functions
    *   Other helpful functions
*   **Extensive Testing:** Comprehensive test suite with over 500 unit tests.

## Installation

Install `bitarray` effortlessly using pip:

```bash
pip install bitarray
```

Test your installation with:

```bash
python -c 'import bitarray; bitarray.test()'
```

## Usage Examples

Bitarrays are designed to mimic lists while offering performance advantages for boolean data.

```python
from bitarray import bitarray

# Initialize a bitarray
a = bitarray()         # create empty bitarray
a.append(1)
a.extend([1, 0])
print(a)  # Output: bitarray('110')

# Create a bitarray of a specified length
x = bitarray(2 ** 20)  # bitarray of length 1048576 (initialized to 0)
print(len(x))  # Output: 1048576

# Initialize from a string
print(bitarray('1001 011'))   # Output: bitarray('1001011')

# Initialize from an iterable
lst = [1, 0, False, True, True]
a = bitarray(lst)
print(a)  # Output: bitarray('10011')

# Access individual bits
print(a[2])  # Output: 0
print(a[2:4])  # Output: bitarray('01')

# Perform bitwise operations
a = bitarray('101110001')
b = bitarray('111001011')
print(~a)
print(a ^ b)
```

## Bit-Endianness Explained

Bitarrays support both little- and big-endian representations.  The endianness can be specified upon creation and affects how bits are stored in memory.

```python
# Big-endian (default)
a = bitarray(b'A')
print(a.endian) # Output: big
print(a) # Output: bitarray('01000001')

# Little-endian
a = bitarray(b'A', endian='little')
print(a) # Output: bitarray('10000010')
print(a.endian) # Output: little
```

## Resources

*   [Documentation](https://github.com/ilanschnell/bitarray)
*   [Change Log](https://github.com/ilanschnell/bitarray/blob/master/doc/changelog.rst)
*   [Buffer Protocol](https://github.com/ilanschnell/bitarray/blob/master/doc/buffer.rst)
*   [Examples](https://github.com/ilanschnell/bitarray/tree/master/examples)