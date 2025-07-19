python
# Comprehensive Python Cheatsheet: Your One-Stop Guide to Python Mastery

# Tired of constantly searching for Python syntax? This comprehensive cheatsheet provides a quick reference for everything Python, from basic data structures to advanced libraries.  
# [View the original repository on GitHub](https://github.com/gto76/python-cheatsheet) to learn more!

## Table of Contents 🔗

*   **1. Collections:** [`List`](#list) 🔗, [`Dictionary`](#dictionary) 🔗, [`Set`](#set) 🔗, [`Tuple`](#tuple) 🔗, [`Range`](#range) 🔗, [`Enumerate`](#enumerate) 🔗, [`Iterator`](#iterator) 🔗, [`Generator`](#generator) 🔗
*   **2. Types:** [`Type`](#type) 🔗, [`String`](#string) 🔗, [`Regex`](#regex) 🔗, [`Format`](#format) 🔗, [`Numbers`](#numbers-1) 🔗, [`Combinatorics`](#combinatorics) 🔗, [`Datetime`](#datetime) 🔗
*   **3. Syntax:** [`Function`](#function) 🔗, [`Inline`](#inline) 🔗, [`Imports`](#imports) 🔗, [`Decorator`](#decorator) 🔗, [`Class`](#class) 🔗, [`Duck Types`](#duck-types) 🔗, [`Enum`](#enum) 🔗, [`Exceptions`](#exceptions) 🔗
*   **4. System:** [`Exit`](#exit) 🔗, [`Print`](#print) 🔗, [`Input`](#input) 🔗, [`Command Line Arguments`](#command-line-arguments) 🔗, [`Open`](#open) 🔗, [`Paths`](#paths) 🔗, [`OS Commands`](#os-commands) 🔗
*   **5. Data:** [`JSON`](#json) 🔗, [`Pickle`](#pickle) 🔗, [`CSV`](#csv) 🔗, [`SQLite`](#sqlite) 🔗, [`Bytes`](#bytes) 🔗, [`Struct`](#struct) 🔗, [`Array`](#array) 🔗, [`Memory View`](#memory-view) 🔗, [`Deque`](#deque) 🔗
*   **6. Advanced:** [`Operator`](#operator) 🔗, [`Match Statement`](#match-statement) 🔗, [`Logging`](#logging) 🔗, [`Introspection`](#introspection) 🔗, [`Threading`](#threading) 🔗, [`Coroutines`](#coroutines) 🔗
*   **7. Libraries:** [`Progress Bar`](#progress-bar) 🔗, [`Plot`](#plot) 🔗, [`Table`](#table) 🔗, [`Console App`](#console-app) 🔗, [`GUI App`](#gui-app) 🔗, [`Scraping`](#scraping) 🔗, [`Web App`](#web-app) 🔗, [`Profiling`](#profiling) 🔗
*   **8. Multimedia:** [`NumPy`](#numpy) 🔗, [`Image`](#image) 🔗, [`Animation`](#animation) 🔗, [`Audio`](#audio) 🔗, [`Synthesizer`](#synthesizer) 🔗, [`Pygame`](#pygame) 🔗, [`Pandas`](#pandas) 🔗, [`Plotly`](#plotly) 🔗
*   **Appendix:** [`Cython`](#cython) 🔗, [`Virtual Environments`](#virtual-environments) 🔗, [`Basic Script Template`](#basic-script-template) 🔗, [`Index`](#index) 🔗

---

## 1. Collections 🔗

*   **List:** Ordered, mutable sequence.
    ```python
    <list> = [<el_1>, <el_2>, ...]  # Creates a list object. Also list(<collection>).
    ```
    *   Access: `<el>   = <list>[index]`  | `<list> = <list>[<slice>]`
    *   Append/Extend: `<list>.append(<el>)` | `<list>.extend(<collection>)`
    *   Sorting/Reversing: `<list>.sort()` | `<list>.reverse()` | `<list> = sorted(<collection>)` | `<iter> = reversed(<list>)`
    *   Aggregation: `<el>  = max(<collection>)` | `<num> = sum(<collection>)`
    *   List Comprehension Examples
        ```python
        elementwise_sum  = [sum(pair) for pair in zip(list_a, list_b)]
        sorted_by_second = sorted(<collection>, key=lambda el: el[1])
        sorted_by_both   = sorted(<collection>, key=lambda el: (el[1], el[0]))
        flatter_list     = list(itertools.chain.from_iterable(<list>))
        ```
    *   Length/Count/Index/Manipulation: `<int> = len(<list>)` | `<int> = <list>.count(<el>)` | `<int> = <list>.index(<el>)` | `<el>  = <list>.pop()` | `<list>.insert(<int>, <el>)` | `<list>.remove(<el>)` | `<list>.clear()`

*   **Dictionary:** Unordered, mutable key-value pairs.
    ```python
    <dict> = {key_1: val_1, key_2: val_2, ...}      # Use `<dict>[key]` to get or set the value.
    ```
    *   Views: `<view> = <dict>.keys()` | `<view> = <dict>.values()` | `<view> = <dict>.items()`
    *   Get/Set Default: `value  = <dict>.get(key, default=None)` | `value  = <dict>.setdefault(key, default=None)`
    *   Defaultdict: `<dict> = collections.defaultdict(<type>)` | `<dict> = collections.defaultdict(lambda: 1)`
    *   Creation: `<dict> = dict(<collection>)` | `<dict> = dict(zip(keys, values))` | `<dict> = dict.fromkeys(keys [, value])`
    *   Update/Manipulation: `<dict>.update(<dict>)` | `value = <dict>.pop(key)` | `{k for k, v in <dict>.items() if v == value}` | `{k: v for k, v in <dict>.items() if k in keys}`
    *   Counter
        ```python
        >>> from collections import Counter
        >>> counter = Counter(['blue', 'blue', 'blue', 'red', 'red'])
        >>> counter['yellow'] += 1
        >>> print(counter.most_common())
        [('blue', 3), ('red', 2), ('yellow', 1)]
        ```

*   **Set:** Unordered, mutable collection of unique elements.
    ```python
    <set> = {<el_1>, <el_2>, ...}                   # Use `set()` for empty set.
    ```
    *   Add/Update: `<set>.add(<el>)` | `<set>.update(<collection> [, ...])`
    *   Set Operations: `<set>  = <set>.union(<coll.>)` | `<set>  = <set>.intersection(<coll.>)` | `<set>  = <set>.difference(<coll.>)` | `<set>  = <set>.symmetric_difference(<coll.>)` | `<bool> = <set>.issubset(<coll.>)` | `<bool> = <set>.issuperset(<coll.>)`
    *   Remove/Discard: `<el> = <set>.pop()` | `<set>.remove(<el>)` | `<set>.discard(<el>)`
    *   Frozen Set
        ```python
        <frozenset> = frozenset(<collection>)
        ```

*   **Tuple:** Ordered, *immutable* sequence of elements.
    ```python
    <tuple> = ()                               # Empty tuple.
    <tuple> = (<el>,)                          # Or: <el>,
    <tuple> = (<el_1>, <el_2> [, ...])         # Or: <el_1>, <el_2> [, ...]
    ```
    *   Named Tuple
        ```python
        >>> from collections import namedtuple
        >>> Point = namedtuple('Point', 'x y')
        >>> p = Point(1, y=2)
        >>> print(p)
        Point(x=1, y=2)
        >>> p[0], p.y
        (1, 2)
        ```

*   **Range:** Immutable sequence of integers.
    ```python
    <range> = range(stop)                      # I.e. range(to_exclusive).
    <range> = range(start, stop)               # I.e. range(from_inclusive, to_exclusive).
    <range> = range(start, stop, ±step)        # I.e. range(from_inclusive, to_exclusive, ±step).
    ```
    ```python
    >>> [i for i in range(3)]
    [0, 1, 2]
    ```

*   **Enumerate:** Returns index and element during iteration.
    ```python
    for i, el in enumerate(<coll>, start=0):   # Returns next element and its index on each pass.
        ...
    ```

*   **Iterator:** Potentially endless stream of elements.
    ```python
    <iter> = iter(<collection>)                # `iter(<iter>)` returns unmodified iterator.
    <iter> = iter(<function>, to_exclusive)    # A sequence of return values until 'to_exclusive'.
    <el>   = next(<iter> [, default])          # Raises StopIteration or returns 'default' on end.
    <list> = list(<iter>)                      # Returns a list of iterator's remaining elements.
    ```
    *   Itertools
        ```python
        import itertools as it
        ```
        *   Infinite Iterators: `<iter> = it.count(start=0, step=1)` | `<iter> = it.repeat(<el> [, times])` | `<iter> = it.cycle(<collection>)`
        *   Combining Iterators: `<iter> = it.chain(<coll>, <coll> [, ...])` | `<iter> = it.chain.from_iterable(<coll>)`
        *   Slicing Iterators: `<iter> = it.islice(<coll>, to_exclusive)` | `<iter> = it.islice(<coll>, from_inc, …)`

*   **Generator:** Function containing `yield` statement, creating an iterator.
    ```python
    def count(start, step):
        while True:
            yield start
            start += step
    ```
    ```python
    >>> counter = count(10, 2)
    >>> next(counter), next(counter), next(counter)
    (10, 12, 14)
    ```

---

## 2. Types 🔗

*   **Type:** Every object has a type; class and type are synonymous.
    ```python
    <type> = type(<el>)                          # Or: <el>.__class__
    <bool> = isinstance(<el>, <type>)            # Or: issubclass(type(<el>), <type>)
    ```
    ```python
    >>> type('a'), 'a'.__class__, str
    (<class 'str'>, <class 'str'>, <class 'str'>)
    ```
    *   Abstract Base Classes
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

*   **String:** Immutable sequence of characters.
    ```python
    <str>  = <str>.strip()                       # Strips all whitespace characters from both ends.
    <str>  = <str>.strip('<chars>')              # Strips passed characters. Also lstrip/rstrip().
    ```
    *   Splitting/Joining: `<list> = <str>.split()` | `<list> = <str>.split(sep=None, maxsplit=-1)` | `<list> = <str>.splitlines(keepends=False)` | `<str>  = <str>.join(<coll_of_strings>)`
    *   Searching: `<bool> = <sub_str> in <str>` | `<bool> = <str>.startswith(<sub_str>)` | `<int>  = <str>.find(<sub_str>)`
    *   Case Conversion/Replace: `<str>  = <str>.lower()` | `<str>  = <str>.casefold()` | `<str>  = <str>.replace(old, new [, count])` | `<str>  = <str>.translate(<table>)`
    *   Character Conversion: `<str>  = chr(<int>)` | `<int>  = ord(<str>)`
    *   Property Methods
        ```python
        <bool> = <str>.isdecimal()                   # Checks for [0-9]. Also [०-९] and [٠-٩].
        <bool> = <str>.isdigit()                     # Checks for [²³¹…] and isdecimal().
        <bool> = <str>.isnumeric()                   # Checks for [¼½¾…], [零〇一…] and isdigit().
        <bool> = <str>.isalnum()                     # Checks for [a-zA-Z…] and isnumeric().
        <bool> = <str>.isprintable()                 # Checks for [ !#$%…] and isalnum().
        <bool> = <str>.isspace()                     # Checks for [ \t\n\r\f\v\x1c-\x1f\x85\xa0…].
        ```

*   **Regex (Regular Expressions):** Pattern matching in strings.
    ```python
    import re
    <str>   = re.sub(r'<regex>', new, text, count=0)  # Substitutes all occurrences with 'new'.
    <list>  = re.findall(r'<regex>', text)            # Returns all occurrences of the pattern.
    <list>  = re.split(r'<regex>', text, maxsplit=0)  # Add brackets around regex to keep matches.
    <Match> = re.search(r'<regex>', text)             # First occurrence of the pattern or None.
    <Match> = re.match(r'<regex>', text)              # Searches only at the beginning of the text.
    <iter>  = re.finditer(r'<regex>', text)           # Returns all occurrences as Match objects.
    ```
    *   Match Object
        ```python
        <str>   = <Match>.group()                         # Returns the whole match. Also group(0).
        <str>   = <Match>.group(1)                        # Returns part inside the first brackets.
        <tuple> = <Match>.groups()                        # Returns all bracketed parts as strings.
        <int>   = <Match>.start()                         # Returns start index of the match.
        <int>   = <Match>.end()                           # Returns exclusive end index of the match.
        ```
    *   Special Sequences
        ```python
        '\d' == '[0-9]'                                   # Also [०-९…]. Matches a decimal character.
        '\w' == '[a-zA-Z0-9_]'                            # Also [ª²³…]. Matches an alphanumeric or _.
        '\s' == '[ \t\n\r\f\v]'                           # Also [\x1c-\x1f…]. Matches a whitespace.
        ```

*   **Format:** String formatting.
    ```python
    <str> = f'{<el_1>}, {<el_2>}'            # Curly brackets can also contain expressions.
    <str> = '{}, {}'.format(<el_1>, <el_2>)  # Same as '{0}, {a}'.format(<el_1>, a=<el_2>).
    <str> = '%s, %s' % (<el_1>, <el_2>)      # Redundant and inferior C-style formatting.
    ```
    *   General Options:
        ```python
        {<el>:<10}                               # '<el>      '
        {<el>:^10}                               # '   <el>   '
        {<el>:>10}                               # '      <el>'
        {<el>:.<10}                              # '<el>......'
        {<el>:0}                                 # '<el>'
        ```
        *   Strings/Numbers/Floats/Ints - see CheatSheet

*   **Numbers:** Numeric types.
    ```python
    <int>      = int(<float/str/bool>)             # Whole number of any size. Truncates floats.
    <float>    = float(<int/str/bool>)             # 64-bit decimal number. Also <float>e±<int>.
    <complex>  = complex(real=0, imag=0)           # Complex number. Also `<float> ± <float>j`.
    <Fraction> = fractions.Fraction(<int>, <int>)  # E.g. `Fraction(1, 2) / 3 == Fraction(1, 6)`.
    <Decimal>  = decimal.Decimal(<str/int/tuple>)  # E.g. `Decimal((1, (2, 3), 4)) == -230_000`.
    ```
    *   Built-in Functions
        ```python
        <num> = pow(<num>, <num>)                      # E.g. `pow(2, 3) == 2 ** 3 == 8`.
        <num> = abs(<num>)                             # E.g. `abs(complex(3, 4)) == 5`.
        <num> = round(<num> [, ±ndigits])              # E.g. `round(123, -1) == 120`.
        <num> = min(<collection>)                      # Also max(<num>, <num> [, ...]).
        <num> = sum(<collection>)                      # Also math.prod(<collection>).
        ```
    *   Math
        ```python
        from math import floor, ceil, trunc            # They convert floats into integers.
        from math import pi, inf, nan, isnan           # `inf * 0` and `nan + 1` return nan.
        from math import sqrt, factorial               # `sqrt(-1)` will raise ValueError.
        from math import sin, cos, tan                 # Also: asin, acos, degrees, radians.
        from math import log, log10, log2              # Log accepts base as second argument.
        ```
    *   Statistics
        ```python
        from statistics import mean, median, mode      # Mode returns the most common item.
        from statistics import variance, stdev         # Also: pvariance, pstdev, quantiles.
        ```
    *   Random
        ```python
        from random import random, randint, uniform    # Also: gauss, choice, shuffle, seed.
        ```
        ```python
        <float> = random()                             # Returns a float inside [0, 1).
        <num>   = randint/uniform(a, b)                # Returns an int/float inside [a, b].
        <float> = gauss(mean, stdev)                   # Also triangular(low, high, mode).
        <el>    = choice(<sequence>)                   # Keeps it intact. Also sample(p, n).
        shuffle(<list>)                                # Works on any mutable sequence.
        ```
    *   Hexadecimal Numbers: `<int> = 0x<hex>` | `<int> = int('±<hex>', 16)` | `<str> = hex(<int>)`
    *   Bitwise Operators: `<int> = <int> & <int>` | `<int> = <int> | <int>` | `<int> = <int> ^ <int>` | `<int> = <int> << n_bits` | `<int> = ~<int>`

*   **Combinatorics:** Tools for creating combinations and permutations.
    ```python
    import itertools as it
    ```
    ```python
    >>> list(it.product('abc', repeat=2))        #   a  b  c
    [('a', 'a'), ('a', 'b'), ('a', 'c'),         # a x  x  x
     ('b', 'a'), ('b', 'b'), ('b', 'c'),         # b x  x  x
     ('c', 'a'), ('c', 'b'), ('c', 'c')]         # c x  x  x
    ```
    ```python
    >>> list(it.permutations('abc', 2))          #   a  b  c
    [('a', 'b'), ('a', 'c'),                     # a .  x  x
     ('b', 'a'), ('b', 'c'),                     # b x  .  x
     ('c', 'a'), ('c', 'b')]                     # c x  x  .
    ```
    ```python
    >>> list(it.combinations('abc', 2))          #   a  b  c
    [('a', 'b'), ('a', 'c'),                     # a .  x  x
     ('b', 'c')                                  # b .  .  x
    ]                                            # c .  .  .
    ```

*   **Datetime:** Working with dates and times.
    ```python
    # $ pip3 install python-dateutil
    from datetime import date, time, datetime, timedelta, timezone
    import zoneinfo, dateutil.tz
    ```
    ```python
    <D>  = date(year, month, day)               # Only accepts valid dates from 1 to 9999 AD.
    <T>  = time(hour=0, minute=0, second=0)     # Also: `microsecond=0, tzinfo=None, fold=0`.
    <DT> = datetime(year, month, day, hour=0)   # Also: `minute=0, second=0, microsecond=0, …`.
    <TD> = timedelta(weeks=0, days=0, hours=0)  # Also: `minutes=0, seconds=0, microseconds=0`.
    ```
    *   Now: `<D/DTn> = D/DT.today()` | `<DTa>   = DT.now(<tzinfo>)`
    *   Timezone: `<tzinfo> = timezone.utc` | `<tzinfo> = timezone(<timedelta>)` | `<tzinfo> = dateutil.tz.tzlocal()` | `<tzinfo> = zoneinfo.ZoneInfo('<iana_key>')` | `<DTa>    = <DT>.astimezone([<tzinfo>])` | `<Ta/DTa> = <T/DT>.replace(tzinfo=<tzinfo>)`
    *   Encode: `<D/T/DT> = D/T/DT.fromisoformat(<str>)` | `<DT>     = DT.strptime(<str>, '<format>')` | `<D/DTn>  = D/DT.fromordinal(<int>)` | `<DTn>    = DT.fromtimestamp(<float>)` | `<DTa>    = DT.fromtimestamp(<float>, <tz>)`
    *   Decode: `<str>    = <D/T/DT>.isoformat(sep='T')` | `<str>    = <D/T/DT>.strftime('<format>')` | `<int>    = <D/DT>.toordinal()` | `<float>  = <DTn>.timestamp()` | `<float>  = <DTa>.timestamp()`
    *   Format
        ```python
        >>> dt = datetime.strptime('2025-08-14 23:39:00.00 +0200', '%Y-%m-%d %H:%M:%S.%f %z')
        >>> dt.strftime("%dth of %B '%y (%a), %I:%M %p %Z")
        "14th of August '25 (Thu), 11:39 PM UTC+02:00"
        ```
    *   Arithmetics: `<bool>   = <D/T/DTn> > <D/T/DTn>` | `<bool>   = <DTa>     > <DTa>` | `<TD>     = <D/DTn>   - <D/DTn>` | `<TD>     = <DTa>     - <DTa>` | `<D/DT>   = <D/DT>    ± <TD>` | `<TD>     = <TD>      * <float>` | `<float>  = <TD>      / <TD>`

---

## 3. Syntax 🔗

*   **Function:** Reusable blocks of code.
    ```python
    def <func_name>(<nondefault_args>): ...                  # E.g. `def func(x, y): ...`.
    def <func_name>(<default_args>): ...                     # E.g. `def func(x=0, y=0): ...`.
    def <func_name>(<nondefault_args>, <default_args>): ...  # E.g. `def func(x, y=0): ...`.
    ```
    *   Function Call: `<obj> = <function>(<positional_args>)` | `<obj> = <function>(<keyword_args>)` | `<obj> = <function>(<positional_args>, <keyword_args>)`
    *   Splat Operator
        ```python
        args, kwargs = (1, 2), {'z': 3}
        func(*args, **kwargs)
        ```
        #### Is the same as:
        ```python
        func(1, 2, z=3)
        ```
        *   Splat Inside Function Definition
        ```python
        def add(*a):
            return sum(a)
        ```
        ```python
        >>> add(1, 2, 3)
        6
        ```
        #### Allowed compositions of arguments and the ways they can be called:
        ```text
        +---------------------------+--------------+--------------+----------------+
        |                           |  func(1, 2)  | func(1, y=2) | func(x=1, y=2) |
        +---------------------------+--------------+--------------+----------------+
        | func(x, *args, **kwargs): |     yes      |     yes      |      yes       |
        | func(*args, y, **kwargs): |              |     yes      |      yes       |
        | func(*, x, **kwargs):     |              |              |      yes       |
        +---------------------------+--------------+--------------+----------------+
        ```
        *   Other Uses
        ```python
        <list>  = [*<collection> [, ...]]  # Or: list(<coll>) [+ ...]
        <tuple> = (*<collection>, [...])   # Or: tuple(<coll>) [+ ...]
        <set>   = {*<collection> [, ...]}  # Or: set(<coll>) [| ...]
        <dict>  = {**<dict> [, ...]}       # Or: <dict> | ...
        ```
        ```python
        head, *body, tail = <collection>   # Head or tail can be omitted.
        ```

*   **Inline:** Concise ways to write code.
    *   Lambda
        ```python
        <func> = lambda: <return_value>                     # A single statement function.
        <func> = lambda <arg_1>, <arg_2>: <return_value>    # Also allows default arguments.
        ```
    *   Comprehensions
        ```python
        <list> = [i+1 for i in range(10)]                   # Returns [1, 2, ..., 10].
        <iter> = (i for i in range(10) if i > 5)            # Returns iter([6, 7, 8, 9]).
        <set>  = {i+5 for i in range(10)}                   # Returns {5, 6, ..., 14}.
        <dict> = {i: i*2 for i in range(10)}                # Returns {0: 0, 1: 2, ..., 9: 18}.
        ```
        ```python
        >>> [l+r for l in 'abc' for r in 'abc']             # Inner loop is on the right side.
        ['aa', 'ab', 'ac', ..., 'cc']
        ```
    *   Map, Filter, Reduce
        ```python
        from functools import reduce
        ```
        ```python
        <iter> = map(lambda x: x + 1, range(10))            # Returns iter([1, 2, ..., 10]).
        <iter> = filter(lambda x: x > 5, range(10))         # Returns iter([6, 7, 8, 9]).
        <obj>  = reduce(lambda out, x: out + x, range(10))  # Returns 45.
        ```
    *   Any, All: `<bool> = any(<collection>)` | `<bool> = all(<collection>)`
    *   Conditional Expression: `<obj> = <exp> if <condition> else <exp>`
        ```python
        >>> [i if i else 'zero' for i in (0, 1, 2, 3)]      # `any([0, '', [], None]) == False`
        ['zero', 1, 2, 3]
        ```
    *   And, Or: `<obj> = <exp> and <exp> [and ...]` | `<obj> = <exp> or <exp> [or ...]`
    *   Walrus Operator:
        ```python
        >>> [i for a in '0123' if (i := int(a)) > 0]        # Assigns to variable mid-sentence.
        [1, 2, 3]
        ```
    *   Named Tuple, Enum, Dataclass
        ```python
        from collections import namedtuple
        Point = namedtuple('Point', 'x y')                  # Creates tuple's subclass.
        point = Point(0, 0)                                 # Returns its instance.
        from enum import Enum
        Direction = Enum('Direction', 'N E S W')            # Creates Enum's subclass.
        direction = Direction.N                             # Returns its member.
        from dataclasses import make_dataclass
        Player = make_dataclass('Player', ['loc', 'dir'])   # Creates a class.
        player = Player(point, direction)                   # Returns its instance.
        ```

*   **Imports:** Making code from other files accessible.
    ```python
    import <module>            # Imports a built-in or '<module>.py'.
    import <package>           # Imports a built-in or '<package>/__init__.py'.
    import <package>.<module>  # Imports a built-in or '<package>/<module>.py'.
    ```
    *   Relative Imports: `'from .[…][<pkg/module>[.…]] import <obj>'`

*   **Closure:** Inner function referencing variables from outer function.
    ```python
    def get_multiplier(a):
        def out(b):
            return a * b
        return out
    ```
    ```python
    >>> multiply_by_3 = get_multiplier(3)
    >>> multiply_by_3(10)
    30
    ```
    *   Partial: `from functools import partial` | `<function> = partial(<function> [, <arg_1> [, ...]])`
    *   Non-Local
        ```python
        def get_counter():
            i = 0
            def out():
                nonlocal i
                i += 1
                return i
            return out
        ```
        ```python
        >>> counter = get_counter()
        >>> counter(), counter(), counter()
        (1, 2, 3)
        ```

*   **Decorator:** Modifies function behavior.
    ```python
    @decorator_name
    def function_that_gets_passed_to_decorator():
        ...
    ```
    *   Debugger Example
        ```python
        from functools import wraps
        def debug(func):
            @wraps(func)
            def out(*args, **kwargs):
                print(func.__name__)
                return func(*args, **kwargs)
            return out
        @debug
        def add(x, y):
            return x + y
        ```
    *   Cache
        ```python
        from functools import cache
        @cache
        def fib(n):
            return n if n < 2 else fib(n-2) + fib(n-1)
        ```
    *   Parametrized Decorator
        ```python
        from functools import wraps
        def debug(print_result=False):
            def decorator(func):
                @wraps(func)
                def out(*args, **kwargs):
                    result = func(*args, **kwargs)
                    print(func.__name__, result if print_result else '')
                    return result
                return out
            return decorator
        @debug(print_result=True)
        def add(x, y):
            return x + y
        ```

*   **Class:** Blueprint for creating objects.
    ```python
    class MyClass:
        def __init__(self, a):
            self.a = a
        def __str__(self):
            return str(self.a)
        def __repr__(self):
            class_name = self.__class__.__name__
            return f'{class_name}({self.a!r})'
        @classmethod
        def get_class_name(cls):
            return cls.__name__
    ```
    ```python
    >>> obj = MyClass(1)
    >>> obj.a, str(obj), repr(obj)
    (1, '1', 'MyClass(1)')