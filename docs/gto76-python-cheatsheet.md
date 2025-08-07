# Python Cheat Sheet: A Comprehensive Guide for Developers 🔗

**Python** is a versatile and widely used programming language known for its readability and extensive libraries, making it ideal for various applications. This cheat sheet provides a concise and comprehensive overview of Python's key features and functionalities, perfect for both beginners and experienced developers. Explore essential concepts like data structures, syntax, and libraries to enhance your Python programming skills.

## Key Features:

*   **Comprehensive Coverage:** Provides detailed information on essential Python concepts, including collections, data types, syntax, and libraries.
*   **Practical Examples:** Includes clear code snippets and examples to illustrate each concept.
*   **Organized Structure:**  Presents the information in a well-structured manner with clear headings and subheadings.
*   **SEO Optimized:** Optimized for search engines to help you quickly find the information you need.

---

## Table of Contents

*   1.  [**Collections**](#collections)
    *   List
    *   Dictionary
    *   Set
    *   Tuple
    *   Range
    *   Enumerate
    *   Iterator
    *   Generator
*   2.  [**Types**](#types)
    *   Type
    *   String
    *   Regex
    *   Format
    *   Numbers
    *   Combinatorics
    *   Datetime
*   3.  [**Syntax**](#syntax)
    *   Function
    *   Splat Operator
    *   Inline
        *   Lambda
        *   Comprehensions
        *   Map, Filter, Reduce
        *   Any, All
        *   Conditional Expression
        *   And, Or
        *   Walrus Operator
        *   Named Tuple, Enum, Dataclass
    *   Imports
    *   Closure
        *   Partial
        *   Non-Local
    *   Decorator
        *   Debugger Example
        *   Cache
        *   Parametrized Decorator
    *   Class
        *   Subclass
        *   Type Annotations
        *   Dataclass
        *   Property
        *   Slots
        *   Copy
    *   Duck Types
        *   Comparable
        *   Hashable
        *   Sortable
        *   Iterator
        *   Callable
        *   Context Manager
    *   Iterable Duck Types
        *   Iterable
        *   Collection
        *   Sequence
        *   ABC Sequence
        *   Table of required and automatically available special methods:
    *   Enum
        *   Inline
    *   Exceptions
        *   Complex Example
        *   Catching Exceptions
        *   Raising Exceptions
        *   Exception Object
        *   Built-in Exceptions
        *   User-defined Exceptions
*   4.  [**System**](#system)
    *   Exit
    *   Print
    *   Input
    *   Command Line Arguments
    *   Open
        *   Modes
        *   Exceptions
        *   File Object
        *   Read Text from File
        *   Write Text to File
    *   Paths
        *   DirEntry
        *   Path Object
    *   OS Commands
        *   Shell Commands
*   5.  [**Data**](#data)
    *   JSON
        *   Read Collection from JSON File
        *   Write Collection to JSON File
    *   Pickle
        *   Read Object from Pickle File
        *   Write Object to Pickle File
    *   CSV
        *   Write
        *   Parameters
        *   Dialects
        *   Read Rows from CSV File
        *   Write Rows to CSV File
    *   SQLite
        *   Read
        *   Write
        *   Placeholders
        *   SQLAlchemy
    *   Bytes
        *   Encode
        *   Decode
        *   Read Bytes from File
        *   Write Bytes to File
    *   Struct
        *   Format
    *   Array
    *   Memory View
        *   Examples
    *   Deque
*   6.  [**Operator**](#operator)
    *   Match Statement
        *   Patterns
        *   Example
*   7.  [**Logging**](#logging)
        *   Setup
*   8.  [**Introspection**](#introspection)
*   9.  [**Threading**](#threading)
        *   Thread
        *   Lock
        *   Semaphore, Event, Barrier
        *   Queue
        *   Thread Pool Executor
*   10. [**Coroutines**](#coroutines)
*   11. [**Libraries**](#libraries)
    *   Progress Bar
    *   Plot
    *   Table
    *   Console App
    *   GUI App
    *   Scraping
        *   Selenium
            *   XPath — also available in lxml, Scrapy, and browser's console via `'$x("<xpath>")'`:
    *   Web App
        *   Serving Files
        *   Serving HTML
        *   Serving JSON
*   12. [**Profiling**](#profiling)
        *   Timing a Snippet
        *   Profiling by Line
        *   Call and Flame Graphs
        *   Sampling and Memory Profilers
*   13. [**NumPy**](#numpy)
        *   Indexing
        *   Broadcasting
        *   Example
*   14. [**Image**](#image)
        *   Modes
        *   Examples
        *   Image Draw
    *   Animation
*   15. [**Audio**](#audio)
        *   Sample Values
        *   Read Float Samples from WAV File
        *   Write Float Samples to WAV File
        *   Examples
        *   Text to Speech
    *   Synthesizer
*   16. [**Pygame**](#pygame)
    *   Rect
    *   Surface
    *   Sound
    *   Basic Mario Brothers Example
*   17. [**Pandas**](#pandas)
        *   Series
            *   Series — Aggregate, Transform, Map:
        *   DataFrame
            *   DataFrame — Merge, Join, Concat:
            *   DataFrame — Aggregate, Transform, Map:
            *   Multi-Index
            *   File Formats
            *   GroupBy
            *   Rolling
*   18. [**Plotly**](#plotly)

---

## Collections 🔗

### List 🔗

```python
<list> = [<el_1>, <el_2>, ...]  # Creates a list object. Also list(<collection>).
<el>   = <list>[index]          # First index is 0. Last -1. Allows assignments.
<list> = <list>[<slice>]        # Also <list>[from_inclusive : to_exclusive : ±step].
<list>.append(<el>)             # Appends element to the end. Also <list> += [<el>].
<list>.extend(<collection>)     # Appends elements to the end. Also <list> += <coll>.
<list>.sort()                   # Sorts the elements in ascending order.
<list>.reverse()                # Reverses the order of list's elements.
<list> = sorted(<collection>)   # Returns a new list with sorted elements.
<iter> = reversed(<list>)       # Returns reversed iterator of elements.
<el>  = max(<collection>)       # Returns largest element. Also min(<el_1>, ...).
<num> = sum(<collection>)       # Returns sum of elements. Also math.prod(<coll>).
elementwise_sum  = [sum(pair) for pair in zip(list_a, list_b)]
sorted_by_second = sorted(<collection>, key=lambda el: el[1])
sorted_by_both   = sorted(<collection>, key=lambda el: (el[1], el[0]))
flatter_list     = list(itertools.chain.from_iterable(<list>))
<int> = len(<list>)             # Returns number of items. Also works on dict, set and string.
<int> = <list>.count(<el>)      # Returns number of occurrences. Also `if <el> in <coll>: ...`.
<int> = <list>.index(<el>)      # Returns index of the first occurrence or raises ValueError.
<el>  = <list>.pop()            # Removes and returns item from the end or at index if passed.
<list>.insert(<int>, <el>)      # Inserts item at passed index and moves the rest to the right.
<list>.remove(<el>)             # Removes first occurrence of the item or raises ValueError.
<list>.clear()                  # Removes all list's items. Also works on dictionary and set.
```

### Dictionary 🔗

```python
<dict> = {key_1: val_1, key_2: val_2, ...}      # Use `<dict>[key]` to get or set the value.
<view> = <dict>.keys()                          # Collection of keys that reflects changes.
<view> = <dict>.values()                        # Collection of values that reflects changes.
<view> = <dict>.items()                         # Coll. of key-value tuples that reflects chgs.
value  = <dict>.get(key, default=None)          # Returns default argument if key is missing.
value  = <dict>.setdefault(key, default=None)   # Returns and writes default if key is missing.
<dict> = collections.defaultdict(<type>)        # Returns a dict with default value `<type>()`.
<dict> = collections.defaultdict(lambda: 1)     # Returns a dict with default value 1.
<dict> = dict(<collection>)                     # Creates a dict from coll. of key-value pairs.
<dict> = dict(zip(keys, values))                # Creates a dict from two collections.
<dict> = dict.fromkeys(keys [, value])          # Creates a dict from collection of keys.
<dict>.update(<dict>)                           # Adds items. Replaces ones with matching keys.
value = <dict>.pop(key)                         # Removes item or raises KeyError if missing.
{k for k, v in <dict>.items() if v == value}    # Returns set of keys that point to the value.
{k: v for k, v in <dict>.items() if k in keys}  # Filters the dictionary by specified keys.
```

#### Counter 🔗

```python
>>> from collections import Counter
>>> counter = Counter(['blue', 'blue', 'blue', 'red', 'red'])
>>> counter['yellow'] += 1
>>> print(counter.most_common())
[('blue', 3), ('red', 2), ('yellow', 1)]
```

### Set 🔗

```python
<set> = {<el_1>, <el_2>, ...}                   # Use `set()` for empty set.
<set>.add(<el>)                                 # Or: <set> |= {<el>}
<set>.update(<collection> [, ...])              # Or: <set> |= <set>
<set>  = <set>.union(<coll.>)                   # Or: <set> | <set>
<set>  = <set>.intersection(<coll.>)            # Or: <set> & <set>
<set>  = <set>.difference(<coll.>)              # Or: <set> - <set>
<set>  = <set>.symmetric_difference(<coll.>)    # Or: <set> ^ <set>
<bool> = <set>.issubset(<coll.>)                # Or: <set> <= <set>
<bool> = <set>.issuperset(<coll.>)                # Or: <set> >= <set>
<el> = <set>.pop()                              # Raises KeyError if empty.
<set>.remove(<el>)                              # Raises KeyError if missing.
<set>.discard(<el>)                             # Doesn't raise an error.
```

### Frozen Set 🔗

*   **Is immutable and hashable.**
*   **That means it can be used as a key in a dictionary or as an element in a set.**

```python
<frozenset> = frozenset(<collection>)
```

### Tuple 🔗

**Tuple is an immutable and hashable list.**

```python
<tuple> = ()                               # Empty tuple.
<tuple> = (<el>,)                          # Or: <el>,
<tuple> = (<el_1>, <el_2> [, ...])         # Or: <el_1>, <el_2> [, ...]
```

### Named Tuple 🔗

**Tuple's subclass with named elements.**

```python
>>> from collections import namedtuple
>>> Point = namedtuple('Point', 'x y')
>>> p = Point(1, y=2)
>>> print(p)
Point(x=1, y=2)
>>> p.x, p[1]
(1, 2)
```

### Range 🔗

**Immutable and hashable sequence of integers.**

```python
<range> = range(stop)                      # I.e. range(to_exclusive).
<range> = range(start, stop)               # I.e. range(from_inclusive, to_exclusive).
<range> = range(start, stop, ±step)        # I.e. range(from_inclusive, to_exclusive, ±step).
>>> [i for i in range(3)]
[0, 1, 2]
```

### Enumerate 🔗

```python
for i, el in enumerate(<coll>, start=0):   # Returns next element and its index on each pass.
    ...
```

### Iterator 🔗

**Potentially endless stream of elements.**

```python
<iter> = iter(<collection>)                # `iter(<iter>)` returns unmodified iterator.
<iter> = iter(<function>, to_exclusive)    # A sequence of return values until 'to_exclusive'.
<el>   = next(<iter> [, default])          # Raises StopIteration or returns 'default' on end.
<list> = list(<iter>)                      # Returns a list of iterator's remaining elements.
```

### Itertools 🔗

```python
import itertools as it
```

```python
<iter> = it.count(start=0, step=1)         # Returns updated value endlessly. Accepts floats.
<iter> = it.repeat(<el> [, times])         # Returns element endlessly or 'times' times.
<iter> = it.cycle(<collection>)            # Repeats the passed sequence of elements endlessly.
<iter> = it.chain(<coll>, <coll> [, ...])  # Empties collections in order (only figuratively).
<iter> = it.chain.from_iterable(<coll>)    # Empties collections inside a collection in order.
<iter> = it.islice(<coll>, to_exclusive)   # Only returns first 'to_exclusive' elements.
<iter> = it.islice(<coll>, from_inc, …)    # `to_exclusive, +step_size`. Indices can be None.
```

### Generator 🔗

*   **Any function that contains a yield statement returns a generator.**
*   **Generators and iterators are interchangeable.**

```python
def count(start, step):
    while True:
        yield start
        start += step
>>> counter = count(10, 2)
>>> next(counter), next(counter), next(counter)
(10, 12, 14)
```

## Types 🔗

### Type 🔗

*   **Everything is an object.**
*   **Every object has a type.**
*   **Type and class are synonymous.**

```python
<type> = type(<el>)                          # Or: <el>.__class__
<bool> = isinstance(<el>, <type>)            # Or: issubclass(type(<el>), <type>)
>>> type('a'), 'a'.__class__, str
(<class 'str'>, <class 'str'>, <class 'str'>)
```

#### Some types do not have built-in names, so they must be imported:

```python
from types import FunctionType, MethodType, LambdaType, GeneratorType, ModuleType
```

### Abstract Base Classes 🔗

**Each abstract base class specifies a set of virtual subclasses. These classes are then recognized by isinstance() and issubclass() as subclasses of the ABC, although they are really not. ABC can also manually decide whether or not a specific class is its virtual subclass, usually based on which methods the class has implemented. For instance, Iterable ABC looks for method iter(), while Collection ABC looks for iter(), contains() and len().**

```python
>>> from collections.abc import Iterable, Collection, Sequence
>>> isinstance([1, 2, 3], Iterable)
True
```

```text
+------------------+------------+------------+------------+
|                  |  Iterable  | Collection |  Sequence  |
+------------------+------------+------------+------------+
| list, range, str |    yes     |    yes     |    yes     |
| dict, set        |    yes     |    yes     |            |
| iter             |    yes     |            |            |
+------------------+------------+------------+------------+
```

```python
>>> from numbers import Number, Complex, Real, Rational, Integral
>>> isinstance(123, Number)
True
```

```text
+--------------------+----------+----------+----------+----------+----------+
|                    |  Number  |  Complex |   Real   | Rational | Integral |
+--------------------+----------+----------+----------+----------+----------+
| int                |   yes    |   yes    |   yes    |   yes    |   yes    |
| fractions.Fraction |   yes    |   yes    |   yes    |   yes    |          |
| float              |   yes    |   yes    |   yes    |          |          |
| complex            |   yes    |   yes    |          |          |          |
| decimal.Decimal    |   yes    |          |          |          |          |
+--------------------+----------+----------+----------+----------+----------+
```

### String 🔗

**Immutable sequence of characters.**

```python
<str>  = <str>.strip()                       # Strips all whitespace characters from both ends.
<str>  = <str>.strip('<chars>')              # Strips passed characters. Also lstrip/rstrip().
<list> = <str>.split()                       # Splits on one or more whitespace characters.
<list> = <str>.split(sep=None, maxsplit=-1)  # Splits on 'sep' string at most 'maxsplit' times.
<list> = <str>.splitlines(keepends=False)    # On [\n\r\f\v\x1c-\x1e\x85\u2028\u2029] and \r\n.
<str>  = <str>.join(<coll_of_strings>)       # Joins elements by using string as a separator.
<bool> = <sub_str> in <str>                  # Checks if string contains the substring.
<bool> = <str>.startswith(<sub_str>)         # Pass tuple of strings for multiple options.
<int>  = <str>.find(<sub_str>)               # Returns start index of the first match or -1.
<str>  = <str>.lower()                       # Lowers the case. Also upper/capitalize/title().
<str>  = <str>.casefold()                    # Same, but converts ẞ/ß to ss, Σ/ς to σ, etc.
<str>  = <str>.replace(old, new [, count])   # Replaces 'old' with 'new' at most 'count' times.
<str>  = <str>.translate(<table>)            # Use `str.maketrans(<dict>)` to generate table.
<str>  = chr(<int>)                          # Converts passed integer to Unicode character.
<int>  = ord(<str>)                          # Converts passed Unicode character to integer.
```

*   **Use `'unicodedata.normalize("NFC", <str>)'` on strings like `'Motörhead'` before comparing them to other strings, because `'ö'` can be stored as one or two characters.**
*   **`'NFC'` converts such characters to a single character, while `'NFD'` converts them to two.**

### Property Methods 🔗

```python
<bool> = <str>.isdecimal()                   # Checks for [0-9]. Also [०-९] and [٠-٩].
<bool> = <str>.isdigit()                     # Checks for [²³¹…] and isdecimal().
<bool> = <str>.isnumeric()                   # Checks for [¼½¾…], [零〇一…] and isdigit().
<bool> = <str>.isalnum()                     # Checks for [a-zA-Z…] and isnumeric().
<bool> = <str>.isprintable()                 # Checks for [ !#$%…] and isalnum().
<bool> = <str>.isspace()                     # Checks for [ \t\n\r\f\v\x1c-\x1f\x85…].
```

### Regex 🔗

**Functions for regular expression matching.**

```python
import re
<str>   = re.sub(r'<regex>', new, text, count=0)  # Substitutes all occurrences with 'new'.
<list>  = re.findall(r'<regex>', text)            # Returns all occurrences of the pattern.
<list>  = re.split(r'<regex>', text, maxsplit=0)  # Add brackets around regex to keep matches.
<Match> = re.search(r'<regex>', text)             # First occurrence of the pattern or None.
<Match> = re.match(r'<regex>', text)              # Searches only at the beginning of the text.
<iter>  = re.finditer(r'<regex>', text)           # Returns all occurrences as Match objects.
```

*   **Raw string literals do not interpret escape sequences, thus enabling us to use regex-specific escape sequences that cause SyntaxWarning in normal string literals (since 3.12).**
*   **Argument 'new' of re.sub() can be a function that accepts Match object and returns a str.**
*   **Argument `'flags=re.IGNORECASE'` can be used with all functions that are listed above.**
*   **Argument `'flags=re.MULTILINE'` makes `'^'` and `'$'` match the start/end of each line.**
*   **Argument `'flags=re.DOTALL'` makes `'.'` also accept the `'\n'` (besides all other chars).**
*   **`'re.compile(<regex>)'` returns a Pattern object with methods sub(), findall(), etc.**

### Match Object 🔗

```python
<str>   = <Match>.group()                         # Returns the whole match. Also group(0).
<str>   = <Match>.group(1)                        # Returns part inside the first brackets.
<tuple> = <Match>.groups()                        # Returns all bracketed parts as strings.
<int>   = <Match>.start()                         # Returns start index of the whole match.
<int>   = <Match>.end()                           # Returns its exclusive end index.
```

### Special Sequences 🔗

```python
'\d' == '[0-9]'                                   # Also [०-९…]. Matches a decimal character.
'\w' == '[a-zA-Z0-9_]'                            # Also [ª²³…]. Matches an alphanumeric or _.
'\s' == '[ \t\n\r\f\v]'                           # Also [\x1c-\x1f…]. Matches a whitespace.
```

*   **By default, decimal characters and alphanumerics from all alphabets are matched unless `'flags=re.ASCII'` is used. It restricts special sequence matches to the first 128 Unicode characters and also prevents `'\s'` from accepting `'\x1c'`, `'\x1d'`, `'\x1e'` and `'\x1f'` (non-printable characters that divide text into files, tables, rows and fields, respectively).**
*   **Use a capital letter for negation (all non-ASCII characters will be matched when used in combination with ASCII flag).**

### Format 🔗

```perl
<str> = f'{<el_1>}, {<el_2>}'            # Curly braces can also contain expressions.
<str> = '{}, {}'.format(<el_1>, <el_2>)  # Same as '{0}, {a}'.format(<el_1>, a=<el_2>).
<str> = '%s, %s' % (<el_1>, <el_2>)      # Redundant and inferior C-style formatting.
```

### Example 🔗

```python
>>> Person = collections.namedtuple('Person', 'name height')
>>> person = Person('Jean-Luc', 187)
>>> f'{person.name} is {person.height / 100} meters tall.'
'Jean-Luc is 1.87 meters tall.'
```

### General Options 🔗

```python
{<el>:<10}                               # '<el>      '
{<el>:^10}                               # '   <el>   '
{<el>:>10}                               # '      <el>'
{<el>:.<10}                              # '<el>......'
{<el>:0}                                 # '<el>'
```

*   **Objects are rendered by calling the `'format(<el>, "<options>")'` function.**
*   **Options inside curly braces can be generated dynamically: `f'{<el>:{<str/int>}[…]}'`.**
*   **Adding `'='` to the expression prepends it to the output: `f'{1+1=}'` returns `'1+1=2'`.**
*   **Adding `'!r'` to the expression converts object to string by calling its [repr()](#class) method.**

### Strings 🔗

```python
{'abcde':10}                             # 'abcde     '
{'abcde':10.3}                           # 'abc       '
{'abcde':.3}                             # 'abc'
{'abcde'!r:10}                           # "'abcde'   "
```

### Numbers 🔗

```python
{123456:10}                              # '    123456'
{123456:10,}                             # '   123,456'
{123456:10_}                             # '   123_456'
{123456:+10}                             # '   +123456'
{123456:=+10}                            # '+   123456'
{123456: }                               # ' 123456'
{-123456: }                              # '-123456'
```

### Floats 🔗

```python
{1.23456:10.3}                           # '      1.23'
{1.23456:10.3f}                          # '     1.235'
{1.23456:10.3e}                          # ' 1.235e+00'
{1.23456:10.3%}                          # '  123.456%'
```

#### Comparison of presentation types:

```text
+--------------+----------------+----------------+----------------+----------------+
|              |    {<float>}   |   {<float>:f}  |   {<float>:e}  |   {<float>:%}  |
+--------------+----------------+----------------+----------------+----------------+
|  0.000056789 |   '5.6789e-05' |    '0.000057'  | '5.678900e-05' |    '0.005679%' |
|  0.00056789  |   '0.00056789' |    '0.000568'  | '5.678900e-04' |    '0.056789%' |
|  0.0056789   |   '0.0056789'  |    '0.005679'  | '5.678900e-03' |    '0.567890%' |
|  0.056789    |   '0.056789'   |    '0.056789'  | '5.678900e-02' |    '5.678900%' |
|  0.56789     |   '0.56789'    |    '0.567890'  | '5.678900e-01' |   '56.789000%' |
|  5.6789      |   '5.6789'     |    '5.678900'  | '5.678900e+00' |  '567.890000%' |
| 56.789       |  '56.789'      |   '56.789000'  | '5.678900e+01' | '5678.900000%' |
+--------------+----------------+----------------+----------------+----------------+
```

```text
+--------------+----------------+----------------+----------------+----------------+
|              |  {<float>:.2}  |  {<float>:.2f} |  {<float>:.2e} |  {<float>:.2%} |
+--------------+----------------+----------------+----------------+----------------+
|  0.000056789 |    '5.7e-05'   |      '0.00'    |   '5.68e-05'   |      '0.01%'   |
|  0.00056789  |    '0.00057'   |      '0.00'    |   '5.68e-04'   |      '0.06%'   |
|  0.0056789   |    '0.0057'    |      '0.01'    |   '5.68e-03'   |      '0.57%'   |
|  0.056789    |    '0.057'     |      '0.06'    |   '5.68e-02'   |      '5.68%'   |
|  0.56789     |    '0.57'      |      '0.57'    |   '5.68e-01'   |     '56.79%'   |
|  5.6789      |    '5.7'       |      '5.68'    |   '5.68e+00'   |    '567.89%'   |
| 56.789       |    '5.7e+01'   |     '56.79'    |   '5.68e+01'   |   '5678.90%'   |
+--------------+----------------+----------------+----------------+----------------+
```

*   **`'{<float>:g}'` is `'{<float>:.6}'` with stripped zeros, exponent starting at `'1e+06'`.**
*   **When both rounding up and rounding down are possible, the one that returns result with even last digit is chosen. That makes `'{6.5:.0f}'` a `'6'` and `'{7.5:.0f}'` an `'8'`.**
*   **This rule only effects numbers that can be represented exactly by a float (`.5`, `.25`, …).**

### Ints 🔗

```python
{90:c}                                   # 'Z'. Unicode character with value 90.
{90:b}                                   # '1011010'. Binary representation of the int.
{90:X}                                   # '5A'. Hexadecimal with upper-case letters.
```

### Numbers 🔗

```python
<int>      = int(<float/str/bool>)             # Whole number of any size. Truncates floats.
<float>    = float(<int/str/bool>)             # 64-bit decimal number. Also <float>e±<int>.
<complex>  = complex(real=0, imag=0)           # A complex number. Also `<float> ± <float>j`.
<Fraction> = fractions.Fraction(<int>, <int>)  # E.g. `Fraction(1, 2) / 3 == Fraction(1, 6)`.
<Decimal>  = decimal.Decimal(<str/int/tuple>)  # E.g. `Decimal((1, (2, 3), 4)) == -230_000`.
```

*   **`'int(<str>)'` and `'float(<str>)'` raise ValueError if passed string is malformed.**
*   **Decimal objects store numbers exactly, unlike most floats where `'1.1 + 2.2 != 3.3'`.**
*   **Floats can be compared with: `'math.isclose(<float>, <float>, rel_tol=1e-09)'`.**
*   **Precision of decimal operations is set with: `'decimal.getcontext().prec = <int>'`.**
*   **Bools can be used anywhere ints can, because bool is a subclass of int: `'True + 1 == 2'`.**

### Built-in Functions 🔗

```python
<num> = pow(<num>, <num>)                      # E.g. `pow(2, 3) == 2 ** 3 == 8`.
<num> = abs(<num>)                             # E.g. `abs(complex(3, 4)) == 5`.
<num> = round(<num> [, ±ndigits])              # E.g. `round(123, -1) == 120`.
<num> = min(<collection>)                      # Also max(<num>, <num> [, ...]).
<num> = sum(<collection>)                      # Also math.prod(<collection>).
```

### Math 🔗

```python
from math import floor, ceil, trunc            # They convert floats into integers.
from math import pi, inf, nan, isnan           # `inf * 0` and `nan + 1` return nan.
from math import sqrt, factorial               # `sqrt(-1)` will raise ValueError.
from math import sin, cos, tan                 # Also: asin, acos, degrees, radians.
from math import log, log10, log2              # Log accepts base as second argument.
```

### Statistics 🔗

```python
from statistics import mean, median, mode      # Mode returns the most common item.
from statistics import variance, stdev         # Also: pvariance, pstdev, quantiles.
```

### Random 🔗

```python
from random import random, randint, uniform    # Also: gauss, choice, shuffle,