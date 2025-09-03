# bitarray: Efficient Arrays of Booleans

**Represent boolean data efficiently in Python with `bitarray`, offering a fast, flexible, and feature-rich alternative to standard lists.**  ([View on GitHub](https://github.com/ilanschnell/bitarray))

## Key Features

*   **Bit-Endianness Control:** Specify `big` or `little` endianness for each bitarray object, critical for buffer manipulation and interoperability.
*   **Sequence-like Functionality:** Utilize slicing, `+`, `*`, `+=`, `*=`, `in`, and `len()` just like you would with Python lists.
*   **Bitwise Operations:** Apply bitwise operators `~`, `&`, `|`, `^`, `<<`, `>>` (and their in-place counterparts) directly on bitarrays.
*   **Variable-Length Prefix Codes:** Encode and decode data with speed using prefix codes, perfect for compression and data representation.
*   **Buffer Protocol Support:** Integrate seamlessly with other Python objects through buffer importing and exporting, including memory-mapped files.
*   **Data Serialization:** Easily pack and unpack data to and from other binary formats, such as `numpy.ndarray`.
*   **Immutability:** Use `frozenbitarray` objects that are immutable and hashable, ideal for dictionary keys.
*   **Built-in Utilities:** A robust `bitarray.util` module provides conversion to/from hex, random bitarray generation, pretty printing, integer conversions, Huffman coding, compression, serialization, and more.
*   **Comprehensive Testing:** Rigorously tested with an extensive suite of approximately 600 unit tests.
*   **Type Hinting:** Full type hints for improved code readability and maintainability.

## Installation

Install `bitarray` using pip:

```bash
pip install bitarray
```

Verify the installation with a quick test:

```bash
python -c 'import bitarray; bitarray.test()'
```

## Usage

`bitarray` objects behave much like Python lists, but they store boolean values (0 or 1) very efficiently. They also offer direct access to the underlying machine representation, making them ideal for performance-critical applications.

```python
from bitarray import bitarray

# Create a bitarray
a = bitarray()
a.append(1)
a.extend([1, 0])
print(a)  # Output: bitarray('110')

# Initialize from a list
lst = [1, 0, False, True, True]
a = bitarray(lst)
print(a)  # Output: bitarray('10011')

# Accessing elements
print(a[2])  # Output: 0
print(a[2:4])  # Output: bitarray('01')

# Slicing and assignment
a[:] = 0  # Set all elements to 0
a[1:3] = bitarray('11')
print(a) # Output: bitarray('01100')

# Bitwise operations
a = bitarray('101110001')
b = bitarray('111001011')
print(~a)  # Invert
print(a & b) # Bitwise AND
```

## Reference

Complete API and detailed documentation can be found in the original [README](https://github.com/ilanschnell/bitarray/blob/master/README.rst). Includes information on:

*   The `bitarray` object and its methods
*   `frozenbitarray` objects
*   `decodetree` objects
*   Functions available in `bitarray` module
*   Functions available in the `bitarray.util` module