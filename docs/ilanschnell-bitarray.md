# bitarray: Efficient Arrays of Booleans

**Representing boolean data efficiently is made easy with the bitarray Python library, providing a versatile and high-performance alternative to lists.**  [View the original repository](https://github.com/ilanschnell/bitarray)

## Key Features

*   **Bit-Endianness Control:** Specify the bit-endianness (big or little) for each `bitarray` object, enabling flexibility in data representation and compatibility.
*   **Sequence-like Behavior:**  Utilize familiar sequence methods like slicing (including assignment and deletion), concatenation (`+`), repetition (`*`), the `in` operator, and `len()`.
*   **Bitwise Operations:** Perform bitwise operations (`~`, `&`, `|`, `^`, `<<`, `>>`) and their in-place equivalents (`&=`, `|=`, `^=`, `<<=`, `>>=`) for efficient boolean logic.
*   **Variable-Length Prefix Code Encoding/Decoding:** Fast methods for encoding and decoding variable bit length prefix codes, ideal for data compression and representation.
*   **Buffer Protocol Support:** Leverage the Python buffer protocol for efficient import and export of buffers, enabling integration with other libraries and memory-mapped files.
*   **Binary Data Conversion:** Seamlessly pack and unpack bitarrays to and from other binary data formats like `numpy.ndarray`.
*   **Pickling and Unpickling:** Easily serialize and deserialize `bitarray` objects for persistent storage and retrieval.
*   **Immutable Frozen Bitarrays:** Utilize hashable and immutable `frozenbitarray` objects for use as dictionary keys or in situations where immutability is desired.
*   **Sequential Search:**  Efficiently search bitarrays for specific patterns or values.
*   **Type Hinting:** Leverage type hints for improved code readability and maintainability.
*   **Extensive Testing:** Benefit from a comprehensive test suite with approximately 600 unit tests, ensuring reliability and stability.
*   **Utility Module:** Utilize the `bitarray.util` module, providing functions for:
    *   Conversion to and from hexadecimal strings.
    *   Generating random bitarrays.
    *   Pretty-printing bitarrays.
    *   Conversion to and from integers.
    *   Creating Huffman codes.
    *   Compression of sparse bitarrays.
    *   Serialization and deserialization.
    *   Various count functions.
    *   Other helpful functions.

## Installation

Install `bitarray` using pip:

```bash
pip install bitarray
```

Test the installation:

```bash
python -c 'import bitarray; bitarray.test()'
```

## Usage

`bitarray` objects behave like Python lists, with the added benefit of efficient memory usage for boolean data.  Key differences include the ability to access the machine representation and control bit-endianness.

```python
from bitarray import bitarray

a = bitarray()  # Create an empty bitarray
a.append(1)
a.extend([1, 0])
print(a)  # Output: bitarray('110')

x = bitarray(2 ** 20)  # Bitarray of length 1048576 (initialized to 0)
print(len(x))  # Output: 1048576

b = bitarray('1001 011')  # Initialize from string (whitespace is ignored)
print(b)  # Output: bitarray('1001011')

lst = [1, 0, False, True, True]
c = bitarray(lst)  # Initialize from an iterable
print(c)  # Output: bitarray('10011')
print(c[2])  # Indexing a single item returns an integer (Output: 0)
print(c[2:4])  # Indexing a slice returns a bitarray (Output: bitarray('01'))
print(c.count(1))  # Output: 3
```

## Bitwise Operators

`bitarray` objects support standard bitwise operators:

```python
a = bitarray('101110001')
print(~a)  # Output: bitarray('010001110')
b = bitarray('111001011')
print(a ^ b)  # Output: bitarray('010111010')
a &= b
print(a)  # Output: bitarray('101000001')
a <<= 2
print(a)  # Output: bitarray('100000100')
print(b >> 1)  # Output: bitarray('011100101')
```

## Bit-Endianness

Bit-endianness becomes important when interacting with the bitarray's buffer (using `tobytes()`, `frombytes()`, `tofile()`, or `fromfile()`), or when using functions in `bitarray.util` like `ba2hex()` or `ba2int()`.

## Buffer Protocol

Bitarray objects support the Python buffer protocol, allowing for efficient data transfer with other libraries.

## Variable Bit Length Prefix Codes

The `.encode()` and `.decode()` methods enable efficient encoding and decoding using prefix codes (e.g., Huffman coding):

```python
from bitarray import bitarray, decodetree

d = {'H': bitarray('111'), 'e': bitarray('0'), 'l': bitarray('110'), 'o': bitarray('10')}
a = bitarray()
a.encode(d, 'Hello')
print(a)  # Output: bitarray('111011011010')

print(''.join(a.decode(d))) # Output: Hello
```

## Frozenbitarrays

`frozenbitarray` objects are immutable and hashable, making them suitable for use as dictionary keys:

```python
from bitarray import frozenbitarray
key = frozenbitarray('1100011')
my_dict = {key: 'some value'}
print(my_dict)  # Output: {frozenbitarray('1100011'): 'some value'}
```

## Reference

Refer to the [original repository](https://github.com/ilanschnell/bitarray) for detailed documentation and version-specific information.