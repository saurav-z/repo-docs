# Python Cheat Sheet: A Comprehensive Guide 🔗

This Python cheat sheet is your go-to resource for a wide range of Python topics, from fundamental data structures to advanced concepts, all packed into an easy-to-read format.

*   **Key Features:**
    *   Comprehensive coverage of Python's core features.
    *   Clear and concise code examples.
    *   Organized into logical sections for quick reference.
    *   Suitable for both beginners and experienced Python developers.

## 1. Collections 🔗

**1.  1 List 🔗**

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

**1.  2 Dictionary 🔗**

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

**1.  3 Counter 🔗**

```python
>>> from collections import Counter
>>> counter = Counter(['blue', 'blue', 'blue', 'red', 'red'])
>>> counter['yellow'] += 1
>>> print(counter.most_common())
[('blue', 3), ('red', 2), ('yellow', 1)]
```

**1.  4 Set 🔗**

```python
<set> = {<el_1>, <el_2>, ...}                   # Use `set()` for empty set.

<set>.add(<el>)                                 # Or: <set> |= {<el>}
<set>.update(<collection> [, ...])              # Or: <set> |= <set>

<set>  = <set>.union(<coll.>)                   # Or: <set> | <set>
<set>  = <set>.intersection(<coll.>)            # Or: <set> & <set>
<set>  = <set>.difference(<coll.>)              # Or: <set> - <set>
<set>  = <set>.symmetric_difference(<coll.>)    # Or: <set> ^ <set>
<bool> = <set>.issubset(<coll.>)                # Or: <set> <= <set>
<bool> = <set>.issuperset(<coll.>)              # Or: <set> >= <set>

<el> = <set>.pop()                              # Raises KeyError if empty.
<set>.remove(<el>)                              # Raises KeyError if missing.
<set>.discard(<el>)                             # Doesn't raise an error.
```

**1.  5 Frozen Set 🔗**

```python
<frozenset> = frozenset(<collection>)
```

**1.  6 Tuple 🔗**

```python
<tuple> = ()                               # Empty tuple.
<tuple> = (<el>,)                          # Or: <el>,
<tuple> = (<el_1>, <el_2> [, ...])         # Or: <el_1>, <el_2> [, ...]
```

**1.  7 Named Tuple 🔗**

```python
>>> from collections import namedtuple
>>> Point = namedtuple('Point', 'x y')
>>> p = Point(1, y=2)
>>> print(p)
Point(x=1, y=2)
>>> p.x, p[1]
(1, 2)
```

**1.  8 Range 🔗**

```python
<range> = range(stop)                      # I.e. range(to_exclusive).
<range> = range(start, stop)               # I.e. range(from_inclusive, to_exclusive).
<range> = range(start, stop, ±step)        # I.e. range(from_inclusive, to_exclusive, ±step).

>>> [i for i in range(3)]
[0, 1, 2]
```

**1.  9 Enumerate 🔗**

```python
for i, el in enumerate(<coll>, start=0):   # Returns next element and its index on each pass.
    ...
```

**1.  10 Iterator 🔗**

```python
<iter> = iter(<collection>)                # `iter(<iter>)` returns unmodified iterator.
<iter> = iter(<function>, to_exclusive)    # A sequence of return values until 'to_exclusive'.
<el>   = next(<iter> [, default])          # Raises StopIteration or returns 'default' on end.
<list> = list(<iter>)                      # Returns a list of iterator's remaining elements.
```

**1.  11 Itertools 🔗**

```python
import itertools as it

<iter> = it.count(start=0, step=1)         # Returns updated value endlessly. Accepts floats.
<iter> = it.repeat(<el> [, times])         # Returns element endlessly or 'times' times.
<iter> = it.cycle(<collection>)            # Repeats the passed sequence of elements endlessly.

<iter> = it.chain(<coll>, <coll> [, ...])  # Empties collections in order (only figuratively).
<iter> = it.chain.from_iterable(<coll>)    # Empties collections inside a collection in order.

<iter> = it.islice(<coll>, to_exclusive)   # Only returns first 'to_exclusive' elements.
<iter> = it.islice(<coll>, from_inc, …)    # `to_exclusive, +step_size`. Indices can be None.
```

**1.  12 Generator 🔗**

```python
def count(start, step):
    while True:
        yield start
        start += step

>>> counter = count(10, 2)
>>> next(counter), next(counter), next(counter)
(10, 12, 14)
```

## 2. Types 🔗

**2.  1 Type 🔗**

```python
<type> = type(<el>)                          # Or: <el>.__class__
<bool> = isinstance(<el>, <type>)            # Or: issubclass(type(<el>), <type>)

>>> type('a'), 'a'.__class__, str
(<class 'str'>, <class 'str'>, <class 'str'>)
```

**2.  2 String 🔗**

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

**2.  3 Regex 🔗**

```python
import re
<str>   = re.sub(r'<regex>', new, text, count=0)  # Substitutes all occurrences with 'new'.
<list>  = re.findall(r'<regex>', text)            # Returns all occurrences of the pattern.
<list>  = re.split(r'<regex>', text, maxsplit=0)  # Add brackets around regex to keep matches.
<Match> = re.search(r'<regex>', text)             # First occurrence of the pattern or None.
<Match> = re.match(r'<regex>', text)              # Searches only at the beginning of the text.
<iter>  = re.finditer(r'<regex>', text)           # Returns all occurrences as Match objects.
```

**2.  4 Format 🔗**

```python
<str> = f'{<el_1>}, {<el_2>}'            # Curly braces can also contain expressions.
<str> = '{}, {}'.format(<el_1>, <el_2>)  # Same as '{0}, {a}'.format(<el_1>, a=<el_2>).
<str> = '%s, %s' % (<el_1>, <el_2>)      # Redundant and inferior C-style formatting.

>>> Person = collections.namedtuple('Person', 'name height')
>>> person = Person('Jean-Luc', 187)
>>> f'{person.name} is {person.height / 100} meters tall.'
'Jean-Luc is 1.87 meters tall.'
```

**2.  5 Numbers 🔗**

```python
<int>      = int(<float/str/bool>)             # Whole number of any size. Truncates floats.
<float>    = float(<int/str/bool>)             # 64-bit decimal number. Also <float>e±<int>.
<complex>  = complex(real=0, imag=0)           # A complex number. Also `<float> ± <float>j`.
<Fraction> = fractions.Fraction(<int>, <int>)  # E.g. `Fraction(1, 2) / 3 == Fraction(1, 6)`.
<Decimal>  = decimal.Decimal(<str/int/tuple>)  # E.g. `Decimal((1, (2, 3), 4)) == -230_000`.
```

**2.  6 Combinatorics 🔗**

```python
import itertools as it

>>> list(it.product('abc', repeat=2))        #   a  b  c
[('a', 'a'), ('a', 'b'), ('a', 'c'),         # a x  x  x
 ('b', 'a'), ('b', 'b'), ('b', 'c'),         # b x  x  x
 ('c', 'a'), ('c', 'b'), ('c', 'c')]         # c x  x  x

>>> list(it.permutations('abc', 2))          #   a  b  c
[('a', 'b'), ('a', 'c'),                     # a .  x  x
 ('b', 'a'), ('b', 'c'),                     # b x  .  x
 ('c', 'a'), ('c', 'b')]                     # c x  x  .

>>> list(it.combinations('abc', 2))          #   a  b  c
[('a', 'b'), ('a', 'c'),                     # a .  x  x
 ('b', 'c')                                  # b .  .  x
]                                            # c .  .  .
```

**2.  7 Datetime 🔗**

```python
# $ pip3 install python-dateutil
from datetime import date, time, datetime, timedelta, timezone
import zoneinfo, dateutil.tz

<D>  = date(year, month, day)               # Only accepts valid dates from 1 to 9999 AD.
<T>  = time(hour=0, minute=0, second=0)     # Also: `microsecond=0, tzinfo=None, fold=0`.
<DT> = datetime(year, month, day, hour=0)   # Also: `minute=0, second=0, microsecond=0, …`.
<TD> = timedelta(weeks=0, days=0, hours=0)  # Also: `minutes=0, seconds=0, microseconds=0`.

<D/DTn> = D/DT.today()                      # Current local date or naive DT. Also DT.now().
<DTa>   = DT.now(<tzinfo>)                  # Aware DT from current time in passed timezone.

<tzinfo> = timezone.utc                     # Coordinated universal time (UK without DST).
<tzinfo> = timezone(<timedelta>)            # Timezone with fixed offset from universal time.
<tzinfo> = dateutil.tz.tzlocal()            # Local timezone with dynamic offset from UTC.
<tzinfo> = zoneinfo.ZoneInfo('<iana_key>')  # 'Continent/City_Name' zone with dynamic offset.
<DTa>    = <DT>.astimezone([<tzinfo>])      # Converts DT to the passed or local fixed zone.
<Ta/DTa> = <T/DT>.replace(tzinfo=<tzinfo>)  # Changes object's timezone without conversion.

<D/T/DT> = D/T/DT.fromisoformat(<str>)      # Object from ISO string. Raises ValueError.
<DT>     = DT.strptime(<str>, '<format>')   # Datetime from custom string. See Format.
<D/DTn>  = D/DT.fromordinal(<int>)          # D/DT from days since the Gregorian NYE 1.
<DTn>    = DT.fromtimestamp(<float>)        # Local naive DT from seconds since the Epoch.
<DTa>    = DT.fromtimestamp(<float>, <tz>)  # Aware datetime from seconds since the Epoch.

<str>    = <D/T/DT>.isoformat(sep='T')      # Also `timespec='auto/hours/minutes/seconds…'`.
<str>    = <D/T/DT>.strftime('<format>')    # Custom string representation of the object.
<int>    = <D/DT>.toordinal()               # Days since NYE 1. Ignores DT's time and zone.
<float>  = <DTn>.timestamp()                # Seconds since the Epoch, from local naive DT.
<float>  = <DTa>.timestamp()                # Seconds since the Epoch, from aware datetime.

>>> dt = datetime.strptime('2025-08-14 23:39:00.00 +0200', '%Y-%m-%d %H:%M:%S.%f %z')
>>> dt.strftime("%dth of %B '%y (%a), %I:%M %p %Z")
"14th of August '25 (Thu), 11:39 PM UTC+02:00"
```

## 3. Syntax 🔗

**3.  1 Function 🔗**

```python
def <func_name>(<nondefault_args>): ...                  # E.g. `def func(x, y): ...`.
def <func_name>(<default_args>): ...                     # E.g. `def func(x=0, y=0): ...`.
def <func_name>(<nondefault_args>, <default_args>): ...  # E.g. `def func(x, y=0): ...`.

<obj> = <function>(<positional_args>)                    # E.g. `func(0, 0)`.
<obj> = <function>(<keyword_args>)                       # E.g. `func(x=0, y=0)`.
<obj> = <function>(<positional_args>, <keyword_args>)    # E.g. `func(0, y=0)`.
```

**3.  2 Splat Operator 🔗**

```python
args, kwargs = (1, 2), {'z': 3}
func(*args, **kwargs)
```

#### Is the same as:

```python
func(1, 2, z=3)
```

**3.  3 Inline 🔗**

```python
<func> = lambda: <return_value>                     # A single statement function.
<func> = lambda <arg_1>, <arg_2>: <return_value>    # Also allows default arguments.
```

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

```python
from functools import reduce

<iter> = map(lambda x: x + 1, range(10))            # Returns iter([1, 2, ..., 10]).
<iter> = filter(lambda x: x > 5, range(10))         # Returns iter([6, 7, 8, 9]).
<obj>  = reduce(lambda out, x: out + x, range(10))  # Returns 45.
```

```python
<bool> = any(<collection>)                          # Is `bool(<el>)` True for any el?
<bool> = all(<collection>)                          # True for all? Also True if empty.
```

```python
<obj> = <exp> if <condition> else <exp>             # Only one expression is evaluated.

>>> [i if i else 'zero' for i in (0, 1, 2, 3)]      # `any([0, '', [], None]) == False`
['zero', 1, 2, 3]
```

```python
<obj> = <exp> and <exp> [and ...]                   # Returns first false or last object.
<obj> = <exp> or <exp> [or ...]                     # Returns first true or last object.
```

```python
>>> [i for ch in '0123' if (i := int(ch)) > 0]      # Assigns to variable mid-sentence.
[1, 2, 3]
```

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

**3.  4 Import 🔗**

```python
import <module>            # Imports a built-in or '<module>.py'.
import <package>           # Imports a built-in or '<package>/__init__.py'.
import <package>.<module>  # Imports a built-in or '<package>/<module>.py'.
```

**3.  5 Closure 🔗**

```python
def get_multiplier(a):
    def out(b):
        return a * b
    return out

>>> multiply_by_3 = get_multiplier(3)
>>> multiply_by_3(10)
30
```

**3.  6 Partial 🔗**

```python
from functools import partial
<function> = partial(<function> [, <arg_1> [, ...]])

>>> def multiply(a, b):
...     return a * b
>>> multiply_by_3 = partial(multiply, 3)
>>> multiply_by_3(10)
30
```

**3.  7 Non-Local 🔗**

```python
def get_counter():
    i = 0
    def out():
        nonlocal i
        i += 1
        return i
    return out

>>> counter = get_counter()
>>> counter(), counter(), counter()
(1, 2, 3)
```

**3.  8 Decorator 🔗**

```python
@decorator_name
def function_that_gets_passed_to_decorator():
    ...
```

**3.  9 Class 🔗**

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

>>> obj = MyClass(1)
>>> obj.a, str(obj), repr(obj)
(1, '1', 'MyClass(1)')
```

**3.  10 Duck Types 🔗**

```python
class MyComparable:
    def __init__(self, a):
        self.a = a
    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.a == other.a
        return NotImplemented

class MyHashable:
    def __init__(self, a):
        self._a = a
    @property
    def a(self):
        return self._a
    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.a == other.a
        return NotImplemented
    def __hash__(self):
        return hash(self.a)

from functools import total_ordering
@total_ordering
class MySortable:
    def __init__(self, a):
        self.a = a
    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.a == other.a
        return NotImplemented
    def __lt__(self, other):
        if isinstance(other, type(self)):
            return self.a < other.a
        return NotImplemented

class Counter:
    def __init__(self):
        self.i = 0
    def __next__(self):
        self.i += 1
        return self.i
    def __iter__(self):
        return self

>>> counter = Counter()
>>> next(counter), next(counter), next(counter)
(1, 2, 3)

class Counter:
    def __init__(self):
        self.i = 0
    def __call__(self):
        self.i += 1
        return self.i

>>> counter = Counter()
>>> counter(), counter(), counter()
(1, 2, 3)

class MyOpen:
    def __init__(self, filename):
        self.filename = filename
    def __enter__(self):
        self.file = open(self.filename)
        return self.file
    def __exit__(self, exc_type, exception, traceback):
        self.file.close()

>>> with open('test.txt', 'w') as file:
...     file.write('Hello World!')
>>> with MyOpen('test.txt') as file:
...     print(file.read())
Hello World!
```

**3.  11 Iterable Duck Types 🔗**

```python
class MyIterable:
    def __init__(self, a):
        self.a = a
    def __iter__(self):
        return iter(self.a)
    def __contains__(self, el):
        return el in self.a

>>> obj = MyIterable([1, 2, 3])
>>> [el for el in obj]
[1, 2, 3]
>>> 1 in obj
True

class MyCollection:
    def __init__(self, a):
        self.a = a
    def __iter__(self):
        return iter(self.a)
    def __contains__(self, el):
        return el in self.a
    def __len__(self):
        return len(self.a)

class MySequence:
    def __init__(self, a):
        self.a = a
    def __iter__(self):
        return iter(self.a)
    def __contains__(self, el):
        return el in self.a
    def __len__(self):
        return len(self.a)
    def __getitem__(self, i):
        return self.a[i]
    def __reversed__(self):
        return reversed(self.a)
```

**3.  12 Enum 🔗**

```python
from enum import Enum, auto

class <enum_name>(Enum):
    <member_name> = auto()              # Increment of the last numeric value or 1.
    <member_name> = <value>             # Values don't have to be hashable or unique.
    <member_name> = <el_1>, <el_2>      # Values can be collections. This is a tuple.

<member> = <enum>.<member_name>         # Returns a member. Raises AttributeError.
<member> = <enum>['<member_name>']      # Returns a member. Raises KeyError.
<member> = <enum>(<value>)              # Returns a member. Raises ValueError.
<str>    = <member>.name                # Returns the member's name.
<obj>    = <member>.value               # Returns the member's value.

<list>   = list(<enum>)                 # Returns a list of enum's members.
<list>   = <enum>._member_names_        # Returns a list of member names.
<list>   = [m.value for m in <enum>]    # Returns a list of member values.

<enum>   = type(<member>)               # Returns an enum. Also <memb>.__class__.
<iter>   = itertools.cycle(<enum>)      # Returns an endless iterator of members.
<member> = random.choice(list(<enum>))  # Randomly selects one of the members.

Cutlery = Enum('Cutlery', 'FORK KNIFE SPOON')
Cutlery = Enum('Cutlery', ['FORK', 'KNIFE', 'SPOON'])
Cutlery = Enum('Cutlery', {'FORK': 1, 'KNIFE': 2, 'SPOON': 3})
```

**3.  13 Exceptions 🔗**

```python
try:
    <code>
except <exception>:
    <code>
```

```python
try:
    <code_1>
except <exception_a>:
    <code_2_a>
except <exception_b>:
    <code_2_b>
else:
    <code_2_c>
finally:
    <code_3>
```

```python
except <exception>: ...
except <exception> as <name>: ...
except (<exception>, [...]): ...
except (<exception>, [...]) as <name>: ...

raise <exception>
raise <exception>()
raise <exception>(<obj> [, ...])

class MyError(Exception): pass
class MyInputError(MyError): pass
```

## 4. System 🔗

**4.  1 Exit 🔗**

```python
import sys
sys.exit()                        # Exits with exit code 0 (success).
sys.exit(<int>)                   # Exits with the passed exit code.
sys.exit(<obj>)                   # Prints to stderr and exits with 1.
```

**4.  2 Print 🔗**

```python
print(<el_1>, ..., sep=' ', end='\n', file=sys.stdout, flush=False)

from pprint import pprint
pprint(<collection>, width=80, depth=None, compact=False, sort_dicts=True)
```

**4.  3 Input 🔗**

```python
<str> = input()
```

**4.  4 Command Line Arguments 🔗**

```python
import sys
scripts_path = sys.argv[0]
arguments    = sys.argv[1:]

from argparse import ArgumentParser, FileType
p = ArgumentParser(description=<str>)                             # Returns a parser object.
p.add_argument('-<short_name>', '--<name>', action='store_true')  # Flag (defaults to False).
p.add_argument('-<short_name>', '--<name>', type=<type>)          # Option (defaults to None).
p.add_argument('<name>', type=<type>, nargs=1)                    # Mandatory first argument.
p.add_argument('<name>', type=<type>, nargs='+')                  # Mandatory remaining args.
p.add_argument('<name>', type=<type>, nargs='?/*')                # Optional argument/s.
args  = p.parse_args()                                            # Exits on parsing error.
<obj> = args.<name>                                               # Returns `<type>(<arg>)`.
```

**4.  5 Open 🔗**

```python
<file> = open(<path>, mode='r', encoding=None, newline=None)

<file>.seek(0)                      # Moves current position to the start of file.
<file>.seek(offset)                 # Moves 'offset' chars/bytes from the start.
<file>.seek(0, 2)                   # Moves current position to the end of file.
<bin_file>.seek(±offset, origin)    # Origin: 0 start, 1 current position, 2 end.

<str/bytes> = <file>.read(size=-1)  # Reads 'size' chars/bytes or until the EOF.
<str/bytes> = <file>.readline()     # Returns a line or empty string/bytes on EOF.
<list>      = <file>.readlines()    # Returns remaining lines. Also list(<file>).
<str/bytes> = next(<file>)          # Returns a line using a read-ahead buffer.

<file>.write(<str/bytes>)           # Writes a str or bytes object to write buffer.
<file>.writelines(<collection>)     # Writes a coll. of strings or bytes objects.
<file>.flush()                      # Flushes write buffer. Runs every 4096/8192 B.
<file>.close()                      # Closes the file after flushing write buffer.
```

**4.  6 Path 🔗**

```python
import os, glob
from pathlib import Path

<str>  = os.getcwd()                # Returns working dir. Starts as shell's $PWD.
<str>  = os.path.join(<path>, ...)  # Uses os.sep to join strings or Path objects.
<str>  = os.path.realpath(<path>)   # Resolves symlinks and calls path.abspath().

<str>  = os.path.basename(<path>)   # Returns final component of the path.
<str>  = os.path.dirname(<path>)    # Returns path without the final component.
<tup.> = os.path.splitext(<path>)   # Splits on last period of the final component.

<list> = os.listdir(path='.')       # Returns