# bitarray: Efficient Arrays of Booleans in Python

**Get blazing-fast performance with the bitarray library, designed for efficient storage and manipulation of boolean data, offering a superior alternative to standard Python lists.**  [View the original repo](https://github.com/ilanschnell/bitarray)

## Key Features

*   **Bit-Endianness Control:** Specify bit-endianness (little or big-endian) for each bitarray object.
*   **Sequence-like Functionality:**  Supports slicing, assignment, deletion, and operators like `+`, `*`, `+=`, `*=`, and `in`, and `len()`.
*   **Bitwise Operations:**  Offers bitwise operators: `~`, `&`, `|`, `^`, `<<`, `>>` (including in-place versions).
*   **Variable-Length Prefix Codes:**  Fast encoding and decoding methods for variable bit-length prefix codes.
*   **Buffer Protocol Support:**  Supports the buffer protocol, enabling import/export of buffers, including memory-mapped files.
*   **Data Format Integration:**  Packing/unpacking to binary data formats like `numpy.ndarray`.
*   **Immutability:**  Includes hashable, immutable `frozenbitarray` objects.
*   **Additional Features:**  Sequential search, type hinting, extensive testing (over 500 unit tests), and more.
*   **Utility Module (bitarray.util):** Comprehensive utility module for:
    *   Hexadecimal conversion
    *   Random bitarray generation
    *   Pretty printing
    *   Integer conversion
    *   Huffman codes
    *   Sparse bitarray compression/decompression
    *   Serialization/deserialization
    *   Counting functions and other helpful utilities

## Installation

Install `bitarray` easily using `pip`:

```bash
pip install bitarray
```

After installation, verify the package with:

```bash
python -c 'import bitarray; bitarray.test()'
```

## Usage

Bitarray objects function similarly to Python lists, but are optimized for storing bits. A key difference is the ability to work directly with the machine representation of the object, which is where the bit-endianness comes into play.

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

# Indexing and slicing
print(b[2])  # Output: 0
print(b[2:4]) # Output: bitarray('01')

# Bitwise operations
c = bitarray('101110001')
d = bitarray('111001011')
print(~c) # Output: bitarray('010001110')
print(c & d) # Output: bitarray('101000001')
```

## Further Exploration

*   **Bit-endianness:** For detailed information, refer to the [Bit-endianness section](https://github.com/ilanschnell/bitarray#bit-endianness) in the original README.
*   **Buffer Protocol:** More details are available in the [buffer protocol documentation](https://github.com/ilanschnell/bitarray/blob/master/doc/buffer.rst).
*   **Examples:**  See the example demonstrating [mmapped-file.py](https://github.com/ilanschnell/bitarray/blob/master/examples/mmapped-file.py).
*   **Documentation:** Explore the [reference documentation](https://github.com/ilanschnell/bitarray#reference) for comprehensive method and function descriptions.
```

Key improvements:

*   **SEO-Optimized Title:**  Includes "Efficient Arrays of Booleans" and "Python" in the title, making it searchable.
*   **Compelling Hook:** Starts with a strong one-sentence description highlighting the library's value proposition.
*   **Bulleted Key Features:** Easier to scan and digest key benefits.
*   **Clearer Organization:** Separates sections more clearly.
*   **Actionable Installation and Usage:** Provides quick-start examples.
*   **Includes Relevant Links:** Adds links to the original repo and specific sections.
*   **Concise Summary:** Keeps the summary to the point while retaining essential information.
*   **Improved Readability:** Uses bolding for emphasis and improves overall formatting.