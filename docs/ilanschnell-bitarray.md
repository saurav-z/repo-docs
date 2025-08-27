# bitarray: Efficient Arrays of Booleans

**Efficiently represent and manipulate arrays of booleans with the Python `bitarray` library, offering performance and flexibility for various applications.**  [View the original repo](https://github.com/ilanschnell/bitarray).

## Key Features

*   **Bit-Endianness Control:** Choose between little-endian and big-endian representations.
*   **Sequence-Like Behavior:**  Supports slicing, concatenation (+), repetition (*), "in" operator, and `len()`.
*   **Bitwise Operations:**  Includes bitwise NOT (~), AND (&), OR (|), XOR (^), left shift (<<), and right shift (>>), with in-place versions (&=, |=, ^=, <<=, >>=).
*   **Prefix Code Encoding/Decoding:**  Fast methods for encoding and decoding variable bit-length prefix codes.
*   **Buffer Protocol Support:** Integrates with the buffer protocol for direct access to the underlying memory representation.
*   **Data Conversion:**  Packing and unpacking to other binary data formats, such as NumPy arrays.
*   **Pickling and Frozen Bitarrays:** Support for pickling/unpickling and immutable `frozenbitarray` objects that are hashable.
*   **Additional Utilities:**  The `bitarray.util` module offers:
    *   Conversion to/from hexadecimal strings.
    *   Random bitarray generation.
    *   Pretty printing.
    *   Conversion to/from integers.
    *   Huffman code generation.
    *   Sparse bitarray compression.
    *   Serialization/Deserialization.
    *   Various counting functions.

## Installation

Install the `bitarray` package using pip:

```bash
pip install bitarray
```

You can then run the test suite to verify the installation:

```bash
python -c 'import bitarray; bitarray.test()'
```

The `test()` function returns a `unittest.runner.TextTestResult` object, allowing you to check if all tests ran successfully.

## Usage

`bitarray` objects behave similarly to Python lists, with a focus on representing bits efficiently.  Key differences include the ability to access the machine representation and control bit-endianness.

```python
from bitarray import bitarray

# Create an empty bitarray
a = bitarray()

# Append bits
a.append(1)
a.extend([1, 0])  # a is now bitarray('110')

# Initialize from a string
b = bitarray('1001011')

# Initialize from an iterable
lst = [1, 0, False, True, True]
c = bitarray(lst)  # c is now bitarray('10011')

# Accessing and slicing
print(c[2])  # Output: 0
print(c[2:4])  # Output: bitarray('01')

# Count occurrences
print(c.count(1))  # Output: 3

# Slice assignment
d = bitarray(50)
d[:] = 0  # set all elements to 0
d[11:37:3] = 9 * bitarray('1')  # set slice to 1
```

Bitwise operators and other list-like features are also supported.

```python
# Bitwise operations
a = bitarray('101110001')
b = bitarray('111001011')
print(~a)     # Output: bitarray('010001110')
print(a ^ b)  # Output: bitarray('010111010')
a &= b        # in-place AND
print(a << 2)   # Output: bitarray('100000100')

```

## Reference

For detailed information on the available methods, data descriptors, and functions, refer to the original README or the documentation within the `bitarray` module.