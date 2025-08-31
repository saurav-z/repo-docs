# bitarray: Efficient Arrays of Booleans in Python

**Optimize memory usage and speed up boolean array operations with the `bitarray` library, a high-performance alternative to Python lists for storing and manipulating bits.**  [View the original repository](https://github.com/ilanschnell/bitarray)

## Key Features

*   **Bit-Endianness Control:** Choose between big-endian and little-endian representations.
*   **Sequence Type Behavior:** Supports slicing (including assignment and deletion), `+`, `*`, `+=`, `*=`, and `in` operator, and `len()`.
*   **Bitwise Operations:** Perform bitwise operations like `~`, `&`, `|`, `^`, `<<`, and `>>` (and their in-place counterparts).
*   **Prefix Code Encoding/Decoding:** Fast methods for variable bit length prefix codes, including Huffman coding.
*   **Buffer Protocol Support:**  Import and export buffers, enabling integration with memory-mapped files and other objects.
*   **Data Conversion:** Pack and unpack to various binary data formats, e.g., `numpy.ndarray`.
*   **Pickling & Unpickling:** Serialize and deserialize bitarray objects.
*   **Frozenbitarray:** Create immutable, hashable objects for use as dictionary keys.
*   **Sequential Search:** Efficient search capabilities within the bitarray.
*   **Type Hinting:** Improves code readability and helps with static analysis.
*   **Extensive Testing:** A robust test suite with approximately 600 unit tests ensures reliability.
*   **Utility Module (`bitarray.util`):** Provides helpful functions for:
    *   Conversion to/from hexadecimal strings
    *   Generating random bitarrays
    *   Pretty printing
    *   Conversion to/from integers
    *   Creating Huffman codes
    *   Compression/decompression of sparse bitarrays
    *   Serialization/deserialization
    *   Various count functions
    *   Other helpful utilities

## Installation

Install `bitarray` easily using pip:

```bash
pip install bitarray
```

To verify the installation and run the test suite:

```bash
python -c 'import bitarray; bitarray.test()'
```

## Usage

`bitarray` objects behave similarly to Python lists but are optimized for storing bits (0s and 1s).

```python
from bitarray import bitarray

# Create a bitarray
a = bitarray()

# Append bits
a.append(1)
a.extend([1, 0])
print(a)  # Output: bitarray('110')

# Initialize from a string
b = bitarray('1001011')

# Access elements
print(b[2])  # Output: 0
print(b[2:4]) # Output: bitarray('01')

# Bitwise Operations
c = bitarray('101110001')
d = bitarray('111001011')

result_xor = c ^ d
print(result_xor)  # Output: bitarray('010111010')

# In-place AND
c &= d
print(c) # Output: bitarray('101000001')

# Slice Assignment
e = bitarray(50)
e.setall(0)
e[11:37:3] = 9 * bitarray('1')
print(e) # Output: bitarray('00000000000100100100100100100100100100000000000000')
```

##  Reference

Comprehensive method and function reference is provided in the original README.