# bitarray: Efficient Arrays of Booleans in Python

**Need a fast and memory-efficient way to represent and manipulate arrays of booleans?**  The `bitarray` library offers a powerful solution, optimized for speed and flexibility, with a full Python API. [Visit the original repository](https://github.com/ilanschnell/bitarray) to learn more.

## Key Features

*   **Bit-Endianness Control:** Choose between big-endian and little-endian representations for your bitarrays.
*   **Sequence Type Functionality:** Leverage familiar sequence methods like slicing, concatenation, repetition, and the `in` operator.
*   **Bitwise Operations:** Perform bitwise operations (AND, OR, XOR, NOT, shifts) directly on bitarrays.
*   **Variable Bit Length Prefix Codes:** Efficiently encode and decode data using prefix codes (e.g., Huffman codes).
*   **Buffer Protocol Support:** Integrate seamlessly with other Python libraries that use the buffer protocol, including NumPy.
*   **Packing/Unpacking:** Convert between bitarrays and other binary data formats.
*   **Immutable frozenbitarray Objects:** Use hashable, immutable `frozenbitarray` objects as dictionary keys.
*   **Extensive Utilities:** Includes a `bitarray.util` module for hexadecimal conversion, random bitarray generation, Huffman coding, and more.
*   **Comprehensive Testing:**  Includes a robust test suite with over 500 unit tests.
*   **Type Hinting:** Improves code readability and maintainability.

## Installation

Install `bitarray` easily using pip:

```bash
pip install bitarray
```

You can also verify the installation and test the library:

```bash
python -c 'import bitarray; bitarray.test()'
```

## Usage Examples

`bitarray` objects function much like lists, but with the efficiency of storing bits in a packed format.  Here are some examples:

```python
from bitarray import bitarray

# Create an empty bitarray and append values
a = bitarray()
a.append(1)
a.extend([1, 0])
print(a)  # Output: bitarray('110')

# Initialize from a string
b = bitarray('1001011')
print(b)  # Output: bitarray('1001011')

# Initialize from a list
lst = [1, 0, False, True, True]
c = bitarray(lst)
print(c)  # Output: bitarray('10011')

# Indexing and slicing
print(c[2])     # Output: 0
print(c[2:4])   # Output: bitarray('01')

# Bitwise operations
d = bitarray('101110001')
e = bitarray('111001011')
print(~d)       # Output: bitarray('010001110')
print(d & e)    # Output: bitarray('101000001')

# Slice assignment
f = bitarray(30)
f[:] = 0
f[10:25] = 1
print(f)
# Output: bitarray('000000000011111111111111100000')
```

## Reference

*   **Documentation:** Comprehensive documentation is available in the original repository README.
*   **Version:**  3.7.0
*   **Change Log:**  See the [changelog](https://github.com/ilanschnell/bitarray/blob/master/doc/changelog.rst).
*   **Further details:** Check the [Bitarray 3 transition](https://github.com/ilanschnell/bitarray/blob/master/doc/bitarray3.rst) for the latest changes.