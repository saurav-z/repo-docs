# bitarray: Efficient Arrays of Booleans

**Optimize your memory usage and speed up your boolean operations with bitarray, a Python library for efficient arrays of booleans.**  ([View on GitHub](https://github.com/ilanschnell/bitarray))

## Key Features

*   **Endianness Control:** Specify bit-endianness (little or big) for each bitarray object.
*   **Sequence Type Support:** Leverage list-like functionality including slicing, concatenation, and the `in` operator.
*   **Bitwise Operations:** Utilize standard bitwise operators (`~`, `&`, `|`, `^`, `<<`, `>>`, and in-place versions).
*   **Prefix Code Encoding/Decoding:**  Fast methods for encoding and decoding variable bit length prefix codes, like Huffman codes.
*   **Buffer Protocol:** Full support for the buffer protocol, enabling interaction with other binary data formats.
*   **Data Conversion:** Easily pack/unpack bitarrays to/from other formats such as `numpy.ndarray`.
*   **Immutability:** Use `frozenbitarray` objects for hashable, immutable arrays.
*   **Utility Module:** Includes `bitarray.util` for hex conversion, random bitarray generation, Huffman code creation, and more.
*   **Extensive Testing:** Rigorously tested with a comprehensive suite of over 600 unit tests.

## Installation

Install bitarray easily using pip:

```bash
pip install bitarray
```

Verify the installation with the included test suite:

```bash
python -c 'import bitarray; bitarray.test()'
```

## Usage

Bitarray objects behave similarly to Python lists but are optimized for storing boolean values efficiently, with each bitarray element representing a single bit.

```python
from bitarray import bitarray

# Create a bitarray
a = bitarray()
a.append(1)
a.extend([1, 0])
print(a)  # Output: bitarray('110')

# Initialize from an iterable
lst = [1, 0, False, True, True]
a = bitarray(lst)
print(a)  # Output: bitarray('10011')

# Slice assignment and deletion
a = bitarray(50)
a.setall(0)
a[11:37:3] = 9 * bitarray('1')
print(a)
```

## Reference

Comprehensive documentation of `bitarray` features, methods and utility functions can be found in the original [README](https://github.com/ilanschnell/bitarray/blob/master/README.md).