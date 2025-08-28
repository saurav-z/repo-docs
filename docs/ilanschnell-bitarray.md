# bitarray: Efficient Arrays of Booleans for Python

**Representing arrays of booleans with exceptional efficiency, the bitarray library provides a versatile and performant way to store and manipulate boolean data in Python.**  [View the original repo](https://github.com/ilanschnell/bitarray)

## Key Features

*   **Bit-Endianness Control:** Choose between big-endian and little-endian representations for each bitarray object.
*   **Sequence-Like Operations:** Supports slicing (including assignment and deletion), concatenation (`+`), repetition (`*`), `in` operator, and `len()`.
*   **Bitwise Operations:** Includes bitwise operations like `~`, `&`, `|`, `^`, `<<`, and `>>`, along with in-place versions.
*   **Variable Bit Length Prefix Codes:** Offers fast encoding and decoding methods for prefix codes, useful for data compression and representation.
*   **Buffer Protocol Support:**  Supports buffer import and export, enabling interaction with other objects.
*   **Integration with Other Libraries:**  Packing and unpacking to formats like `numpy.ndarray`.
*   **Immutability:**  Includes hashable `frozenbitarray` objects suitable for use as dictionary keys.
*   **Additional Utilities:** The `bitarray.util` module offers:
    *   Hexadecimal string conversions
    *   Random bitarray generation
    *   Pretty printing
    *   Integer conversions
    *   Huffman code creation
    *   Sparse bitarray compression
    *   Serialization and deserialization
    *   Various count functions
*   **Extensive Testing:** Comprehensive test suite with about 600 unit tests.
*   **Type Hinting:** Provides type hints for improved code readability and maintainability.

## Installation

Install `bitarray` using pip:

```bash
pip install bitarray
```

Verify the installation by running tests:

```bash
python -c 'import bitarray; bitarray.test()'
```

## Usage

```python
from bitarray import bitarray

# Create a bitarray
a = bitarray()
a.append(1)
a.extend([1, 0])
print(a)  # Output: bitarray('110')

# Initialize from a string
b = bitarray('1001011')
print(b)  # Output: bitarray('1001011')

# Accessing elements
print(b[2])  # Output: 0
print(b[2:4]) # Output: bitarray('01')

# Bitwise operations
c = bitarray('101110001')
d = bitarray('111001011')
print(~c) # Output: bitarray('010001110')
print(c & d) # Output: bitarray('101000001')
```

## Reference

[See the original README for a full reference of methods and functions.](https://github.com/ilanschnell/bitarray)