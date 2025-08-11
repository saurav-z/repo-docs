# Comprehensive Python Cheatsheet 🔗

**Quickly master Python with this comprehensive cheatsheet, packed with essential code snippets and practical examples to boost your coding efficiency.**  This guide is your one-stop resource for everything Python, whether you're a beginner or an experienced developer. Check out the original repository on [GitHub](https://github.com/gto76/python-cheatsheet) for updates and more.

**Key Features:**

*   **Collections:** Lists, Dictionaries, Sets, Tuples, and more.
*   **Types:** Strings, Numbers, Dates, and Regular Expressions.
*   **Syntax:** Functions, Classes, Imports, and Control Structures.
*   **Systems:** File handling, command-line arguments, and operating system commands.
*   **Data:** Handling JSON, Pickle, CSV, and Databases.
*   **Advanced:** Introspection, Logging, Threading, and Coroutines.
*   **Libraries:** Tools for plotting, GUI, web scraping, and more.
*   **Multimedia:** NumPy, Image manipulation, Audio, and Pygame.

---

## Table of Contents

1.  [Collections](#collections) 🔗
2.  [Types](#types) 🔗
3.  [Syntax](#syntax) 🔗
4.  [System](#system) 🔗
5.  [Data](#data) 🔗
6.  [Advanced](#advanced) 🔗
7.  [Libraries](#libraries) 🔗
8.  [Multimedia](#multimedia) 🔗
9.  [Appendix](#appendix) 🔗
    *   [Cython](#cython) 🔗
    *   [Virtual Environments](#virtual-environments) 🔗
    *   [Basic Script Template](#basic-script-template) 🔗

---

## 1. Collections 🔗

*   **List** 🔗
    *   `list = [el_1, el_2, ...]` Creates a list object. Also `list(<collection>)`.
    *   `<el> = list[index]` Get/set element. First index is 0, last is -1.
    *   `list = list[from_inclusive : to_exclusive : ±step]` Slicing.
    *   `list.append(<el>)` Adds element to end. Also `list += [<el>]`.
    *   `list.extend(<collection>)` Adds elements to the end. Also `list += <coll>`.
    *   `list.sort()` Sorts elements in ascending order.
    *   `list.reverse()` Reverses order.
    *   `list = sorted(<collection>)` Returns a new sorted list.
    *   `iter = reversed(list)` Returns reversed iterator.
    *   `<el> = max(<collection>)` Returns the largest element.  Also `min(<el_1>, ...)`
    *   `<num> = sum(<collection>)` Returns sum of elements. Also `math.prod(<coll>)`.
    *   `int = len(list)` Returns the number of items. Also works on dict, set and string.
    *   `int = list.count(<el>)` Returns number of occurrences. Also `if <el> in <coll>:`...
    *   `int = list.index(<el>)` Returns index of the first occurrence or raises `ValueError`.
    *   `<el> = list.pop()` Removes and returns item from the end or at index if passed.
    *   `list.insert(<int>, <el>)` Inserts item at index, shifting the rest.
    *   `list.remove(<el>)` Removes first occurrence. Raises `ValueError` if missing.
    *   `list.clear()` Removes all items. Also works on dictionary and set.
*   **Dictionary** 🔗
    *   `dict = {key_1: val_1, key_2: val_2, ...}`  Use `dict[key]` to get/set value.
    *   `view = dict.keys()` Collection of keys reflecting changes.
    *   `view = dict.values()` Collection of values reflecting changes.
    *   `view = dict.items()` Collection of key-value tuples reflecting changes.
    *   `value = dict.get(key, default=None)` Returns `default` if key is missing.
    *   `value = dict.setdefault(key, default=None)` Returns and writes `default` if key is missing.
    *   `dict = collections.defaultdict(<type>)`  Returns dict with default value `<type>()`.
    *   `dict = dict(<collection>)` Creates a dict from collection of key-value pairs.
    *   `dict = dict(zip(keys, values))` Creates a dict from two collections.
    *   `dict = dict.fromkeys(keys [, value])` Creates a dict from keys.
    *   `dict.update(<dict>)` Adds items and replaces ones with matching keys.
    *   `value = dict.pop(key)` Removes item, raises `KeyError` if missing.
*   **Set** 🔗
    *   `set = {el_1, el_2, ...}`  Use `set()` for empty set.
    *   `set.add(<el>)`  Or: `set |= {<el>}`
    *   `set.update(<collection> [, ...])`  Or: `set |= <set>`
    *   `set  = set.union(<coll.>)` Or: `set | <set>`
    *   `set  = set.intersection(<coll.>)` Or: `set & <set>`
    *   `set  = set.difference(<coll.>)` Or: `set - <set>`
    *   `set  = set.symmetric_difference(<coll.>)`  Or: `set ^ <set>`
    *   `bool = set.issubset(<coll.>)`  Or: `set <= <set>`
    *   `bool = set.issuperset(<coll.>)`  Or: `set >= <set>`
    *   `<el> = set.pop()` Raises `KeyError` if empty.
    *   `set.remove(<el>)` Raises `KeyError` if missing.
    *   `set.discard(<el>)` Doesn't raise an error if missing.
    *   **Frozen Set** 🔗
        *   Immutable and hashable, so it can be a dictionary key or set element.
        *   `frozenset = frozenset(<collection>)`
*   **Tuple** 🔗
    *   Immutable and hashable list.
    *   `tuple = ()` (Empty)
    *   `tuple = (el,)` Or: `el,`
    *   `tuple = (el_1, el_2 [, ...])` Or: `el_1, el_2 [, ...]`
    *   **Named Tuple** 🔗
        *   Tuple's subclass with named elements.
        *   ```python
            from collections import namedtuple
            Point = namedtuple('Point', 'x y')
            p = Point(1, y=2)
            print(p) # Point(x=1, y=2)
            p.x, p[1] # (1, 2)
            ```
*   **Range** 🔗
    *   Immutable and hashable sequence of integers.
    *   `range = range(stop)` (To exclusive)
    *   `range = range(start, stop)` (From inclusive, to exclusive)
    *   `range = range(start, stop, ±step)`
    *   ```python
        [i for i in range(3)] # [0, 1, 2]
        ```
*   **Enumerate** 🔗
    *   ```python
        for i, el in enumerate(<coll>, start=0):
            ...
        ```
        Returns next element and its index on each pass.
*   **Iterator** 🔗
    *   Potentially endless stream of elements.
    *   `iter = iter(<collection>)` `iter(<iter>)` returns unmodified iterator.
    *   `iter = iter(<function>, to_exclusive)` A sequence of return values.
    *   `<el> = next(<iter> [, default])` Raises `StopIteration` or returns `default` on end.
    *   `list = list(<iter>)`  Returns a list of the iterator's elements.
        *   **Itertools** 🔗
        *   ```python
            import itertools as it
            ```
        *   `iter = it.count(start=0, step=1)` Returns updated value endlessly. Accepts floats.
        *   `iter = it.repeat(<el> [, times])` Returns element endlessly or 'times' times.
        *   `iter = it.cycle(<collection>)` Repeats the passed sequence of elements endlessly.
        *   `iter = it.chain(<coll>, <coll> [, ...])` Empties collections in order.
        *   `iter = it.chain.from_iterable(<coll>)` Empties collections inside a collection.
        *   `iter = it.islice(<coll>, to_exclusive)`  Returns first `to_exclusive` elements.
        *   `iter = it.islice(<coll>, from_inc, …)` `to_exclusive, +step_size`. Indices can be `None`.
*   **Generator** 🔗
    *   Function with `yield` statement returns a generator. Generators and iterators are interchangeable.
    *   ```python
        def count(start, step):
            while True:
                yield start
                start += step
        ```
        ```python
        counter = count(10, 2)
        next(counter), next(counter), next(counter) # (10, 12, 14)
        ```

---

## 2. Types 🔗

*   Everything is an object, every object has a type. Type and class are synonymous.
    *   `<type> = type(<el>)` Or: `<el>.__class__`
    *   `bool = isinstance(<el>, <type>)` Or: `issubclass(type(<el>), <type>)`
    *   ```python
        type('a'), 'a'.__class__, str # (<class 'str'>, <class 'str'>, <class 'str'>)
        ```
    *   Some types do not have built-in names, so they must be imported:
    *   ```python
        from types import FunctionType, MethodType, LambdaType, GeneratorType, ModuleType
        ```
    *   **Abstract Base Classes** 🔗
        *   Specifies virtual subclasses recognized by `isinstance()` and `issubclass()`.
        *   ```python
            from collections.abc import Iterable, Collection, Sequence
            isinstance([1, 2, 3], Iterable) # True
            ```
        *   ```text
            +------------------+------------+------------+------------+
            |                  |  Iterable  | Collection |  Sequence  |
            +------------------+------------+------------+------------+
            | list, range, str |    yes     |    yes     |    yes     |
            | dict, set        |    yes     |    yes     |            |
            | iter             |    yes     |            |            |
            +------------------+------------+------------+------------+
            ```
        *   ```python
            from numbers import Number, Complex, Real, Rational, Integral
            isinstance(123, Number) # True
            ```
        *   ```text
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
*   **String** 🔗
    *   Immutable sequence of characters.
    *   `<str>  = <str>.strip()` Strips all whitespace from both ends.
    *   `<str>  = <str>.strip('<chars>')` Strips passed characters. Also `lstrip/rstrip()`.
    *   `<list> = <str>.split()` Splits on one or more whitespace characters.
    *   `<list> = <str>.split(sep=None, maxsplit=-1)` Splits on `sep` string.
    *   `<list> = <str>.splitlines(keepends=False)` On `[\n\r\f\v\x1c-\x1e\x85\u2028\u2029]` and `\r\n`.
    *   `<str>  = <str>.join(<coll_of_strings>)` Joins elements using string as separator.
    *   `<bool> = <sub_str> in <str>` Checks if string contains substring.
    *   `<bool> = <str>.startswith(<sub_str>)` Pass tuple of strings for multiple options.
    *   `<int>  = <str>.find(<sub_str>)`  Returns start index or -1.
    *   `<str>  = <str>.lower()` Lowers case. Also `upper/capitalize/title()`.
    *   `<str>  = <str>.casefold()` Same, converts ẞ/ß to ss, Σ/ς to σ, etc.
    *   `<str>  = <str>.replace(old, new [, count])` Replaces 'old' with 'new'.
    *   `<str>  = <str>.translate(<table>)`  Use `str.maketrans(<dict>)` to generate table.
    *   `<str>  = chr(<int>)` Converts integer to Unicode character.
    *   `<int>  = ord(<str>)` Converts Unicode character to integer.
        *   Use `'unicodedata.normalize("NFC", <str>)'` before comparing strings like `'Motörhead'`, because `'ö'` can be stored as one or two characters. `'NFC'` converts to a single character, while `'NFD'` converts to two.
    *   **Property Methods** 🔗
        *   `<bool> = <str>.isdecimal()` Checks for `[0-9]`. Also `[०-९]` and `[٠-٩]`.
        *   `<bool> = <str>.isdigit()` Checks for `[²³¹…]` and `isdecimal()`.
        *   `<bool> = <str>.isnumeric()` Checks for `[¼½¾…]`, `[零〇一…]` and `isdigit()`.
        *   `<bool> = <str>.isalnum()` Checks for `[a-zA-Z…]` and `isnumeric()`.
        *   `<bool> = <str>.isprintable()` Checks for `[ !#$%…]` and `isalnum()`.
        *   `<bool> = <str>.isspace()` Checks for `[ \t\n\r\f\v\x1c-\x1f\x85…].`
*   **Regex** 🔗
    *   Functions for regular expression matching.
    *   ```python
        import re
        <str>   = re.sub(r'<regex>', new, text, count=0)  # Substitutes all occurrences.
        <list>  = re.findall(r'<regex>', text)            # Returns all occurrences.
        <list>  = re.split(r'<regex>', text, maxsplit=0)  # Keeps matches with brackets.
        <Match> = re.search(r'<regex>', text)             # First occurrence or `None`.
        <Match> = re.match(r'<regex>', text)              # Searches only at the beginning.
        <iter>  = re.finditer(r'<regex>', text)           # All occurrences as Match objects.
        ```
        *   Raw string literals (`r'<regex>'`) do not interpret escape sequences.
        *   Argument `new` of `re.sub()` can be a function that accepts `Match` and returns `str`.
        *   Argument `'flags=re.IGNORECASE'` can be used with functions above.
        *   Argument `'flags=re.MULTILINE'` makes `'^'` and `'$'` match start/end of each line.
        *   Argument `'flags=re.DOTALL'` makes `'.'` also accept `'\n'` (besides all other chars).
        *   `'re.compile(<regex>)'` returns `Pattern` object with `sub()`, `findall()`, etc.
    *   **Match Object** 🔗
        *   `<str>   = <Match>.group()` Returns whole match. Also `group(0)`.
        *   `<str>   = <Match>.group(1)` Returns part inside first brackets.
        *   `<tuple> = <Match>.groups()` Returns all bracketed parts as strings.
        *   `<int>   = <Match>.start()` Returns start index of the whole match.
        *   `<int>   = <Match>.end()` Returns its exclusive end index.
    *   **Special Sequences** 🔗
        *   `'\d' == '[0-9]'` Also `[०-९…].` Matches a decimal character.
        *   `'\w' == '[a-zA-Z0-9_]'` Also `[ª²³…].` Matches alphanumeric or `_`.
        *   `'\s' == '[ \t\n\r\f\v]'` Also `[\x1c-\x1f…].` Matches whitespace.
        *   By default, decimal characters and alphanumerics from all alphabets are matched unless `'flags=re.ASCII'` is used.  Use capital letter for negation.
*   **Format** 🔗
    *   ```python
        <str> = f'{<el_1>}, {<el_2>}'  # Curly braces can contain expressions.
        <str> = '{}, {}'.format(<el_1>, <el_2>)
        <str> = '%s, %s' % (<el_1>, <el_2>)  # Inferior C-style formatting.
        ```
    *   **Example**
        *   ```python
            import collections
            Person = collections.namedtuple('Person', 'name height')
            person = Person('Jean-Luc', 187)
            f'{person.name} is {person.height / 100} meters tall.'
            # Jean-Luc is 1.87 meters tall.
            ```
    *   **General Options** 🔗
        *   `{<el>:<10}` '<el>      '
        *   `{<el>:^10}` '   <el>   '
        *   `{<el>:>10}` '      <el>'
        *   `{<el>:.<10}` '<el>......'
        *   `{<el>:0}` '<el>'
        *   Objects are rendered by calling the `'format(<el>, "<options>")'` function.
        *   Options inside curly braces can be generated dynamically: `f'{<el>:{<str/int>}[…]}'`.
        *   Adding `'='` to the expression prepends it to the output: `f'{1+1=}'` returns `'1+1=2'`.
        *   Adding `'!r'` to the expression converts object to string by calling its [repr()](#class) method.
    *   **Strings**
        *   `{'abcde':10}` 'abcde     '
        *   `{'abcde':10.3}` 'abc       '
        *   `{'abcde':.3}` 'abc'
        *   `{'abcde'!r:10}` "'abcde'   "
    *   **Numbers**
        *   `{123456:10}` '    123456'
        *   `{123456:10,}` '   123,456'
        *   `{123456:10_}` '   123_456'
        *   `{123456:+10}` '   +123456'
        *   `{123456:=+10}` '+   123456'
        *   `{123456: }` ' 123456'
        *   `{-123456: }` '-123456'
    *   **Floats**
        *   `{1.23456:10.3}` '      1.23'
        *   `{1.23456:10.3f}` '     1.235'
        *   `{1.23456:10.3e}` ' 1.235e+00'
        *   `{1.23456:10.3%}` '  123.456%'
    *   ```text
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
        *   `'{<float>:g}'` is `'{<float>:.6}'` with stripped zeros, exponent starting at `'1e+06'`.
        *   When both rounding up and rounding down are possible, the one that returns result with even last digit is chosen.  This rule only effects numbers that can be represented exactly by a float.
    *   **Ints**
        *   `{90:c}` 'Z' Unicode character with value 90.
        *   `{90:b}` '1011010' Binary representation of the int.
        *   `{90:X}` '5A' Hexadecimal with upper-case letters.
*   **Numbers** 🔗
    *   `<int>      = int(<float/str/bool>)` Whole number of any size.
    *   `<float>    = float(<int/str/bool>)` 64-bit decimal number. Also `<float>e±<int>`.
    *   `<complex>  = complex(real=0, imag=0)` A complex number. Also `<float> ± <float>j`.
    *   `<Fraction> = fractions.Fraction(<int>, <int>)` E.g. `Fraction(1, 2) / 3 == Fraction(1, 6)`.
    *   `<Decimal>  = decimal.Decimal(<str/int/tuple>)` E.g. `Decimal((1, (2, 3), 4)) == -230_000`.
        *   `'int(<str>)'` and `'float(<str>)'` raise `ValueError` if string is malformed.
        *   Decimal objects store numbers exactly, unlike most floats where `'1.1 + 2.2 != 3.3'`.
        *   Floats can be compared with: `'math.isclose(<float>, <float>, rel_tol=1e-09)'`.
        *   Precision of decimal operations is set with: `'decimal.getcontext().prec = <int>'`.
        *   Bools can be used anywhere ints can, because bool is a subclass of int: `'True + 1 == 2'`.
    *   **Built-in Functions**
        *   `<num> = pow(<num>, <num>)` E.g. `pow(2, 3) == 2 ** 3 == 8`.
        *   `<num> = abs(<num>)` E.g. `abs(complex(3, 4)) == 5`.
        *   `<num> = round(<num> [, ±ndigits])` E.g. `round(123, -1) == 120`.
        *   `<num> = min(<collection>)` Also `max(<num>, <num> [, ...])`.
        *   `<num> = sum(<collection>)` Also `math.prod(<collection>)`.
    *   **Math**
        *   ```python
            from math import floor, ceil, trunc # Convert floats into integers.
            from math import pi, inf, nan, isnan # `inf * 0` and `nan + 1` return nan.
            from math import sqrt, factorial  # `sqrt(-1)` will raise ValueError.
            from math import sin, cos, tan  # Also: asin, acos, degrees, radians.
            from math import log, log10, log2 # Log accepts base as second argument.
            ```
    *   **Statistics**
        *   ```python
            from statistics import mean, median, mode # Mode returns the most common item.
            from statistics import variance, stdev # Also: pvariance, pstdev, quantiles.
            ```
    *   **Random**
        *   ```python
            from random import random, randint, uniform # Also: gauss, choice, shuffle, seed.
            ```
        *   `<float> = random()` Returns a float inside [0, 1).
        *   `<num>   = randint/uniform(a, b)` Returns an int/float inside `[a, b]`.
        *   `<float> = gauss(mean, stdev)` Also `triangular(low, high, mode)`.
        *   `<el>    = choice(<sequence>)` Keeps it intact. Also `sample(p, n)`.
        *   `shuffle(<list>)` Works on all mutable sequences.
    *   **Hexadecimal Numbers**
        *   `<int> = 0x<hex>` E.g. `0xFF == 255`. Also `0b<bin>`.
        *   `<int> = int('±<hex>', 16)` Also `int('±0x<hex>/±0b<bin>', 0)`.
        *   `<str> = hex(<int>)` Returns `[-]0x<hex>`. Also `bin()`.
    *   **Bitwise Operators**
        *   `<int> = <int> & <int>` E.g. `0b1100 & 0b1010 == 0b1000`.
        *   `<int> = <int> | <int>` E.g. `0b1100 | 0b1010 == 0b1110`.
        *   `<int> = <int> ^ <int>` E.g. `0b1100 ^ 0b1010 == 0b0110`.
        *   `<int> = <int> << n_bits` E.g. `0b1111 << 4 == 0b11110000`.
        *   `<int> = ~<int>` E.g. `~0b1 == -0b10 == -(0b1+1)`.
*   **Combinatorics** 🔗
    *   ```python
        import itertools as it
        ```
    *   ```python
        >>> list(it.product('abc', repeat=2))
        [('a', 'a'), ('a', 'b'), ('a', 'c'),
         ('b', 'a'), ('b', 'b'), ('b', 'c'),
         ('c', 'a'), ('c', 'b'), ('c', 'c')]
        ```
    *   ```python
        >>> list(it.permutations('abc', 2))
        [('a', 'b'), ('a', 'c'),
         ('b', 'a'), ('b', 'c'),
         ('c', 'a'), ('c', 'b')]
        ```
    *   ```python
        >>> list(it.combinations('abc', 2))
        [('a', 'b'), ('a', 'c'),
         ('b', 'c')]
        ```
*   **Datetime** 🔗
    *   Provides `date`, `time`, `datetime` and `timedelta` classes. All are immutable and hashable.
    *   ```python
        # $ pip3 install python-dateutil
        from datetime import date, time, datetime, timedelta, timezone
        import zoneinfo, dateutil.tz
        ```
    *   `<D>  = date(year, month, day)` Only accepts valid dates from 1 to 9999 AD.
    *   `<T>  = time(hour=0, minute=0, second=0)` Also: `microsecond=0, tzinfo=None, fold=0`.
    *   `<DT> = datetime(year, month, day, hour=0)` Also: `minute=0, second=0, microsecond=0, …`.
    *   `<TD> = timedelta(weeks=0, days=0, hours=0)` Also: `minutes=0, seconds=0, microseconds=0`.
    *   Times and datetimes that have defined timezone are called aware, and ones that don't, naive. If time or datetime object is naive, it is presumed to be in the system's timezone!
    *   `'fold=1'` means the second pass in case of time jumping back (usually for one hour).
    *   Timedelta normalizes arguments to `±days`, `seconds` (`< 86 400`) and `microseconds` (`< 1M`).  Its `str()` method returns `'[±D, ]H:MM:SS[.…]'` and `total_seconds()` a `float` of all seconds.
    *   Use `'<D/DT>.weekday()'` to get day of the week as an integer, with Monday being 0.
    *   **Now**
        *   `<D/DTn> = D/DT.today()` Current local date or naive `DT`.  Also `DT.now()`.
        *   `<DTa>   = DT.now(<tzinfo>)` Aware `DT` from current time in passed timezone.
        *   To extract time use `'<DTn>.time()'`, `'<DTa>.time()'` or `'<DTa>.timetz()'`.
    *   **Timezones**
        *   `<tzinfo> = timezone.utc` Coordinated universal time (UK without DST).
        *   `<tzinfo> = timezone(<timedelta>)` Timezone with fixed offset from universal time.
        *   `<tzinfo> = dateutil.tz.tzlocal()` Local timezone with dynamic offset from UTC.
        *   `<tzinfo> = zoneinfo.ZoneInfo('<iana_key>')` 'Continent/City\_Name' zone with dynamic offset.
        *   `<DTa>    = <DT>.astimezone([<tzinfo>])` Converts `DT` to the passed or local fixed zone.
        *   `<Ta/DTa> = <T/DT>.replace(tzinfo=<tzinfo>)` Changes timezone without conversion.