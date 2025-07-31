# bitarray: Efficient Arrays of Booleans

**Efficiently represent and manipulate arrays of booleans with the `bitarray` library, a fast and versatile Python tool.** [View the original repository](https://github.com/ilanschnell/bitarray)

**Key Features:**

*   **Bit-Endianness Control:** Specify big- or little-endian representation for each bitarray object, providing flexibility in how bits are interpreted.
*   **Sequence-like Operations:** Utilize familiar sequence methods such as slicing (including assignment and deletion), and operators like `+`, `*`, `+=`, `*=`, and `in`. Also, you can easily use `len()`.
*   **Bitwise Operations:** Perform bitwise operations (`~`, `&`, `|`, `^`, `<<`, `>>`, and their in-place versions) for efficient bit manipulation.
*   **Variable Bit Length Codes:**  Fast methods for encoding and decoding variable bit length prefix codes like Huffman coding.
*   **Buffer Protocol Support:** Leverage the buffer protocol for efficient data transfer and interoperability with other Python objects.
*   **Binary Data Conversion:** Pack and unpack data to and from various binary formats, including NumPy arrays.
*   **Serialization:** Easily pickle and unpickle bitarray objects for storage and retrieval.
*   **Frozenbitarray:** Use immutable, hashable `frozenbitarray` objects as dictionary keys.
*   **Efficient Searching:**  Perform sequential searches within bitarrays.
*   **Type Hinting:** Improve code readability and maintainability with type hints.
*   **Extensive Testing:** Benefit from a comprehensive test suite with over 500 unittests to ensure reliability.
*   **Utility Module:**  Use the `bitarray.util` module for:
    *   Conversion to and from hexadecimal strings
    *   Generating random bitarrays
    *   Pretty printing
    *   Conversion to and from integers
    *   Creating Huffman codes
    *   Compression of sparse bitarrays
    *   (De-)serialization
    *   Various count functions
    *   Other helpful functions

## Installation

Install bitarray easily using pip or conda:

```bash
pip install bitarray
```

```bash
conda install bitarray
```

## Usage

```python
from bitarray import bitarray

# Create a bitarray
a = bitarray()
a.append(1)
a.extend([1, 0])
print(a)  # Output: bitarray('110')

# Initialize from string or iterable
x = bitarray('1001011')
lst = [1, 0, False, True, True]
a = bitarray(lst)
print(a)  # Output: bitarray('10011')

# Indexing, slicing, and more
print(a[2])       # Output: 0
print(a[2:4])     # Output: bitarray('01')
print(a.count(1))  # Output: 3
```

## Bitwise Operators

```python
a = bitarray('101110001')
b = bitarray('111001011')

print(~a)  # Output: bitarray('010001110')
print(a ^ b)  # Output: bitarray('010111010')
```

## Bit-endianness

```python
a = bitarray(b'A')
print(a.endian)  # Output: 'big'
print(a)  # Output: bitarray('01000001')

a = bitarray(b'A', endian='little')
print(a)  # Output: bitarray('10000010')
```

## Additional Details
Refer to the full documentation for more information on the methods and details available.