# bitarray: Efficient Arrays of Booleans for Python

**`bitarray` provides a highly efficient and versatile way to represent and manipulate arrays of booleans in Python, optimized for both speed and memory usage.**  Check out the original repo [here](https://github.com/ilanschnell/bitarray)!

## Key Features

*   **Bit-Endianness Control:** Specify `big` or `little` endianness for each bitarray object.
*   **Sequence Type Behavior:** Supports slicing (including assignment and deletion), concatenation (`+`), repetition (`*`), in-place operations (`+=`, `*=`), the `in` operator, and `len()`.
*   **Bitwise Operations:** Includes bitwise operators (`~`, `&`, `|`, `^`, `<<`, `>>`) and their in-place counterparts (`&=`, `|=`, `^=`, `<<=`, `>>=`).
*   **Variable Bit Length Prefix Codes:** Fast methods for encoding and decoding variable-length prefix codes.
*   **Buffer Protocol Support:**  Implements the buffer protocol for efficient data transfer, including importing and exporting buffers.
*   **Integration with Other Formats:** Packing and unpacking to and from binary data formats like `numpy.ndarray`.
*   **Pickling and Unpickling:** Supports saving and loading bitarray objects.
*   **Immutable Frozen Bitarrays:** `frozenbitarray` objects provide a hashable, immutable version for use as dictionary keys.
*   **Sequential Search:** Provides built-in sequential search capabilities.
*   **Type Hinting:**  Includes type hints for improved code readability and maintainability.
*   **Extensive Testing:**  Comes with a comprehensive test suite of over 500 unit tests.
*   **bitarray.util Module:** Offers various utility functions, including:
    *   Hexadecimal string conversions
    *   Random bitarray generation
    *   Pretty printing
    *   Integer conversions
    *   Huffman code generation
    *   Compression of sparse bitarrays
    *   Serialization/Deserialization
    *   Various count functions
    *   Other helpful functions

## Installation

Install `bitarray` easily using pip:

```bash
pip install bitarray
```

Test the installation:

```bash
python -c 'import bitarray; bitarray.test()'
```

## Usage

`bitarray` objects behave like Python lists, but store boolean values efficiently as bits.

```python
from bitarray import bitarray

# Create an empty bitarray
a = bitarray()

# Append items
a.append(1)
a.extend([1, 0])
print(a)  # Output: bitarray('110')

# Initialize from a list
lst = [1, 0, False, True, True]
a = bitarray(lst)
print(a)  # Output: bitarray('10011')

# Indexing
print(a[2])  # Output: 0
print(a[2:4])  # Output: bitarray('01')

# Slice assignment
a[2:3] = True
print(a)  # Output: bitarray('10111')

# Bitwise operations
a = bitarray('101110001')
b = bitarray('111001011')
result = a & b
print(result) # Output: bitarray('101000001')
```

## Bit-Endianness

`bitarray` objects can use either `big` or `little` endianness. This affects how bits are arranged within the underlying byte representation. Understanding endianness is important when interacting with the binary data of the bitarray.

```python
a = bitarray(b'A')  # Default: big-endian
print(a)  # Output: bitarray('01000001')

b = bitarray(b'A', endian='little')
print(b)  # Output: bitarray('10000010')
```

## License

This project is licensed under the [MIT License](https://github.com/ilanschnell/bitarray/blob/master/LICENSE).