# bitarray: Efficient Arrays of Booleans in Python

**Represent and manipulate arrays of boolean values with exceptional efficiency using `bitarray`, a high-performance Python library optimized for speed and memory usage.** ([Original Repo](https://github.com/ilanschnell/bitarray))

## Key Features

*   **Bit-Endianness Control:** Choose between little-endian and big-endian representations for your bitarrays.
*   **Sequence-Like Operations:** Utilize familiar sequence methods like slicing, concatenation (`+`), repetition (`*`), `in` operator, and `len()`.
*   **Bitwise Operations:** Perform bitwise operations (`~`, `&`, `|`, `^`, `<<`, `>>`) with in-place versions available.
*   **Variable Bit Length Prefix Codes:** Encode and decode efficiently.
*   **Buffer Protocol Support:** Import and export buffers, enabling integration with memory-mapped files and other objects.
*   **Data Conversion:** Pack and unpack data to and from other binary formats like `numpy.ndarray`.
*   **Serialization:** Pickle and unpickle bitarray objects.
*   **Frozen Bitarrays:** Create immutable, hashable `frozenbitarray` objects for use as dictionary keys.
*   **Sequential Search:** Implement fast sequential search operations.
*   **Type Hinting:** Enjoy enhanced code readability and maintainability with type hints.
*   **Extensive Test Suite:** Ensure reliability with over 600 unit tests.
*   **Utility Module (`bitarray.util`):**
    *   Hexadecimal string conversion
    *   Random bitarray generation
    *   Pretty printing
    *   Integer conversion
    *   Huffman code generation
    *   Sparse bitarray compression
    *   Serialization/Deserialization
    *   Various count functions
    *   Other helpful functions

## Installation

Install `bitarray` using pip:

```bash
pip install bitarray
```

Verify the installation with:

```bash
python -c 'import bitarray; bitarray.test()'
```

The `test()` function, as part of the API, will return a `unittest.runner.TextTestResult` object, indicating the test success. You can also verify by:

```python
import bitarray
assert bitarray.test().wasSuccessful()
```

## Usage

`bitarray` objects closely resemble Python lists, with the key difference being homogeneity (storing only bits) and machine-level access.

```python
from bitarray import bitarray

# Create an empty bitarray
a = bitarray()

# Append and extend
a.append(1)
a.extend([1, 0])  # a is now bitarray('110')

# Initialize from a string
b = bitarray('1001011')

# Initialize from an iterable
lst = [1, 0, False, True, True]
c = bitarray(lst) # c is now bitarray('10011')

# Indexing and slicing
print(c[2])       # Output: 0
print(c[2:4])     # Output: bitarray('01')

# Count and remove
print(c.count(1)) # Output: 3
c.remove(0)       # Removes first occurrence of 0
print(c)          # Output: bitarray('1011')
```

`bitarray` objects support slice assignment and deletion, and can be assigned to booleans:

```python
a = bitarray(50)
a.setall(0)
a[11:37:3] = 9 * bitarray('1')
```

## Bitwise Operators

Bitwise operators operate in a familiar fashion:

```python
a = bitarray('101110001')
print(~a)       # Output: bitarray('010001110')
b = bitarray('111001011')
print(a ^ b)    # Output: bitarray('010111010')
a &= b
print(a)        # Output: bitarray('101000001')
a <<= 2
print(a)        # Output: bitarray('100000100')
```

Negative shifts and shifts exceeding the bitarray's length are handled as specified, ensuring predictable behavior.

## Bit-Endianness

Bit-endianness becomes important when interacting with the bitarray buffer directly, such as with `tobytes()`, `frombytes()`, or `tofile()`. Refer to the [bit-endianness documentation](https://github.com/ilanschnell/bitarray/blob/master/doc/endianness.rst) for detailed explanations.

## Buffer Protocol

`bitarray` objects support the buffer protocol. For more information, consult the [buffer protocol documentation](https://github.com/ilanschnell/bitarray/blob/master/doc/buffer.rst). There's also an [example](https://github.com/ilanschnell/bitarray/blob/master/examples/mmapped-file.py) showing how to memory-map a file to a bitarray.

## Variable Bit Length Prefix Codes and Frozenbitarrays

Use `encode()` and `decode()` for variable-length encoding and decoding. `frozenbitarray` provides an immutable, hashable version of bitarray.
```python
from bitarray import frozenbitarray
key = frozenbitarray('1100011')
{key: 'some value'} # Use as a dictionary key
```

## Reference

For a complete reference, please consult the documentation.