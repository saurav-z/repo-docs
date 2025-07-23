# Python Cheatsheet: A Comprehensive Guide for Developers 🔗

**Quickly master Python with this comprehensive and SEO-optimized cheatsheet!** This guide provides a complete reference for Python syntax, data structures, and essential libraries, ensuring you have everything you need at your fingertips.

[View the original repository on GitHub](https://github.com/gto76/python-cheatsheet)

## Key Features

*   **Comprehensive Coverage:** Explore essential topics, including collections, types, syntax, system interactions, data handling, advanced concepts, and popular libraries.
*   **Clear and Concise:** Benefit from easy-to-understand code examples and explanations.
*   **Organized Structure:** Navigate the cheatsheet with ease using headings, subheadings, and cross-references.
*   **Practical Examples:** Learn through real-world code snippets and usage scenarios.

## Table of Contents

1.  [Collections](#collections) 🔗
2.  [Types](#types) 🔗
3.  [Syntax](#syntax) 🔗
4.  [System](#system) 🔗
5.  [Data](#data) 🔗
6.  [Advanced](#advanced) 🔗
7.  [Libraries](#libraries) 🔗
8.  [Multimedia](#multimedia) 🔗

### 1. Collections 🔗

*   [List](#list)
*   [Dictionary](#dictionary)
*   [Set](#set)
*   [Tuple](#tuple)
*   [Range](#range)
*   [Enumerate](#enumerate)
*   [Iterator](#iterator)
*   [Generator](#generator)

### 2. Types 🔗

*   [Type](#type)
*   [String](#string)
*   [Regex](#regex)
*   [Format](#format)
*   [Numbers](#numbers-1)
*   [Combinatorics](#combinatorics)
*   [Datetime](#datetime)

### 3. Syntax 🔗

*   [Function](#function)
*   [Inline](#inline)
*   [Imports](#imports)
*   [Closure](#closure)
*   [Decorator](#decorator)
*   [Class](#class)
*   [Duck Types](#duck-types)
*   [Enum](#enum)
*   [Exceptions](#exceptions)

### 4. System 🔗

*   [Exit](#exit)
*   [Print](#print)
*   [Input](#input)
*   [Command Line Arguments](#command-line-arguments)
*   [Open](#open)
*   [Paths](#paths)
*   [OS Commands](#os-commands)

### 5. Data 🔗

*   [JSON](#json)
*   [Pickle](#pickle)
*   [CSV](#csv)
*   [SQLite](#sqlite)
*   [Bytes](#bytes)
*   [Struct](#struct)
*   [Array](#array)
*   [Memory View](#memory-view)
*   [Deque](#deque)

### 6. Advanced 🔗

*   [Operator](#operator)
*   [Match Statement](#match-statement)
*   [Logging](#logging)
*   [Introspection](#introspection)
*   [Threading](#threading)
*   [Coroutines](#coroutines)

### 7. Libraries 🔗

*   [Progress Bar](#progress-bar)
*   [Plot](#plot)
*   [Table](#table)
*   [Console App](#console-app)
*   [GUI App](#gui-app)
*   [Scraping](#scraping)
*   [Web App](#web-app)
*   [Profiling](#profiling)

### 8. Multimedia 🔗

*   [NumPy](#numpy)
*   [Image](#image)
*   [Animation](#animation)
*   [Audio](#audio)
*   [Synthesizer](#synthesizer)
*   [Pygame](#pygame)
*   [Pandas](#pandas)
*   [Plotly](#plotly)

---

*Remaining sections are edited for brevity and clarity, while maintaining the original content and using SEO-friendly headings.*

## List 🔗

*   **Creation:** `list = [<el_1>, <el_2>, ...]` or `list(<collection>)`
*   **Access:** `<el> = list[index]`, `<list> = list[from:to:step]`
*   **Modification:**
    *   `append(<el>)` (add to end)
    *   `extend(<collection>)` (add multiple elements)
    *   `insert(index, <el>)` (insert at index)
    *   `remove(<el>)` (remove first occurrence)
    *   `pop()` (remove from end)
    *   `pop(index)` (remove at index)
    *   `clear()` (remove all)
*   **Sorting/Reversing:**
    *   `sort()` (ascending order)
    *   `reverse()` (reverse order)
    *   `sorted(<collection>)` (returns new sorted list)
    *   `reversed(<list>)` (returns reversed iterator)
*   **Other:**
    *   `len(<list>)` (number of items)
    *   `count(<el>)` (occurrences)
    *   `index(<el>)` (index of first occurrence)
    *   `max(<collection>)`, `min(<collection>)`, `sum(<collection>)`

## Dictionary 🔗

*   **Creation:** `dict = {key_1: val_1, key_2: val_2, ...}`
*   **Access/Modification:** `dict[key] = value`, `value = dict[key]`
*   **Views:**
    *   `keys()`
    *   `values()`
    *   `items()`
*   **Methods:**
    *   `get(key, default=None)`
    *   `setdefault(key, default=None)`
    *   `update(dict)` (add/update items)
    *   `pop(key)` (remove item)
*   **Counter:** (from `collections`)  `Counter([el1, el2,...])`
    *   `most_common()`

## Set 🔗

*   **Creation:** `set = {<el_1>, <el_2>, ...}` (use `set()` for empty set)
*   **Modification:**
    *   `add(<el>)`
    *   `update(<collection>)`
    *   `remove(<el>)` (raises `KeyError` if missing)
    *   `discard(<el>)` (doesn't raise error)
    *   `pop()` (removes and returns an arbitrary element, raises `KeyError` if empty)
*   **Set Operations:**
    *   `union(collection)` or `|`
    *   `intersection(collection)` or `&`
    *   `difference(collection)` or `-`
    *   `symmetric_difference(collection)` or `^`
    *   `issubset(collection)` or `<=`
    *   `issuperset(collection)` or `>=`
*   **Frozen Set:**
    *   Immutable and hashable: `frozenset(<collection>)`

## Tuple 🔗

*   **Creation:**
    *   `tuple = ()` (empty)
    *   `tuple = (<el>,)` or `<el>,` (single element)
    *   `tuple = (<el_1>, <el_2> [, ...])` or `<el_1>, <el_2> [, ...]`
*   **Named Tuple:** (from `collections`)
    *   Subclass with named elements
    *   `Point = namedtuple('Point', 'x y')`

## Range 🔗

*   **Creation:**
    *   `range(stop)` (from 0 to stop - 1)
    *   `range(start, stop)` (from start to stop - 1)
    *   `range(start, stop, step)`
*   **Characteristics:**
    *   Immutable
    *   Hashable
    *   Sequence of integers

## Enumerate 🔗

*   Iterates with index: `for i, el in enumerate(<collection>, start=0):`

## Iterator 🔗

*   **Creation:**
    *   `iter(<collection>)`
    *   `iter(<function>, to_exclusive)`
*   **Methods:**
    *   `next(<iter> [, default])` (returns next item or default)
    *   `list(<iter>)` (remaining items as a list)

*   **Itertools:** (from `itertools`)
    *   `count(start=0, step=1)` (endless)
    *   `repeat(<el> [, times])`
    *   `cycle(<collection>)` (endless)
    *   `chain(<coll>, <coll> [, ...])`
    *   `chain.from_iterable(<coll>)`
    *   `islice(<coll>, to_exclusive)`
    *   `islice(<coll>, from_inc, to_exc, +step)`

## Generator 🔗

*   Function with `yield`
*   **Usage:**
    *   Iterators and generators are interchangeable
    *   `def count(start, step):`
    *   `while True:`
    *   `yield start`
    *   `start += step`

## Type 🔗

*   **`type(<el>)` or `<el>.__class__`**
*   **`isinstance(<el>, <type>)` or `issubclass(type(<el>), <type>)`**
*   **Abstract Base Classes**
    *   (from `collections.abc`)
        *   `Iterable`, `Collection`, `Sequence`
    *   (from `numbers`)
        *   `Number`, `Complex`, `Real`, `Rational`, `Integral`

## String 🔗

*   **Immutability**
*   **Methods:**
    *   `strip()`, `lstrip()`, `rstrip()` (whitespace/chars)
    *   `split()`, `split(sep, maxsplit)`
    *   `splitlines(keepends=False)`
    *   `join(<coll_of_strings>)`
    *   `in`, `startswith()`, `find()`
    *   `lower()`, `upper()`, `capitalize()`, `title()`
    *   `casefold()`
    *   `replace(old, new [, count])`
    *   `translate(<table>)` (using `str.maketrans(<dict>)`)
    *   `chr(<int>)`, `ord(<str>)`
*   **Property Methods:**
    *   `isdecimal()`, `isdigit()`, `isnumeric()`
    *   `isalnum()`, `isprintable()`, `isspace()`

## Regex 🔗

*   **Import:** `import re`
*   **Functions:**
    *   `sub(r'<regex>', new, text, count=0)` (substitutes)
    *   `findall(r'<regex>', text)` (all occurrences)
    *   `split(r'<regex>', text, maxsplit=0)`
    *   `search(r'<regex>', text)` (first occurrence or None)
    *   `match(r'<regex>', text)` (beginning only)
    *   `finditer(r'<regex>', text)` (iterates match objects)
*   **Match Object:**
    *   `group()`, `group(1)`
    *   `groups()`
    *   `start()`, `end()`
*   **Special Sequences:**
    *   `'\d'`, `'\w'`, `'\s'`
*   **Raw strings:** `r'<regex>'`
*   **Flags:** `re.IGNORECASE`, `re.MULTILINE`, `re.DOTALL`, `re.ASCII`

## Format 🔗

*   **Methods:**
    *   `f'{<el_1>}, {<el_2>}'`
    *   `'{}, {}'.format(<el_1>, <el_2>)`
    *   `'%s, %s' % (<el_1>, <el_2>)` (C-style, discouraged)
*   **General Options:**
    *   `{<el>:10}` (padding)
    *   `{<el>:^10}`, `{<el}:>10}`, `{<el}:.<10}`
    *   `{<el>:0}`
    *   `{<el>!r}` (repr() method)
*   **Numbers, Floats:** (examples of output)
    *   `{123456:10}` (padding)
    *   `{123456:10,}` (with comma)
    *   `{1.23456:10.3f}` (fixed-point notation)
    *   `{1.23456:10.3e}` (exponential notation)
    *   `{1.23456:10.3%}` (percentage)

## Numbers 🔗

*   **Types:**
    *   `int(<float/str/bool>)`
    *   `float(<int/str/bool>)`
    *   `complex(real=0, imag=0)`
    *   `Fraction(<int>, <int>)` (from `fractions`)
    *   `Decimal(<str/int/tuple>)` (from `decimal`)
*   **Built-in Functions:**
    *   `pow(<num>, <num>)`, `abs(<num>)`
    *   `round(<num> [, ±ndigits])`
    *   `min(<collection>)`, `max(<collection>)`
    *   `sum(<collection>)`, `math.prod(<collection>)` (from `math`)
*   **Math:** (from `math`)
    *   `floor`, `ceil`, `trunc`
    *   `pi`, `inf`, `nan`, `isnan`
    *   `sqrt`, `factorial`
    *   `sin`, `cos`, `tan` (and inverse functions)
    *   `log`, `log10`, `log2`
*   **Statistics:** (from `statistics`)
    *   `mean`, `median`, `mode`
    *   `variance`, `stdev`
*   **Random:** (from `random`)
    *   `random`, `randint`, `uniform`
    *   `gauss`, `choice`, `shuffle`, `seed`
*   **Hexadecimal Numbers:**
    *   `0x<hex>`, `0b<bin>`
    *   `int('±<hex>', 16)`
    *   `hex(<int>)`, `bin()`
*   **Bitwise Operators:**
    *   `&`, `|`, `^`, `<<`, `>>`, `~`

## Combinatorics 🔗

*   **Import:** `import itertools as it`
*   **Functions:**
    *   `product('abc', repeat=2)`
    *   `permutations('abc', 2)`
    *   `combinations('abc', 2)`
    *   (See the original documentation for the table of outputs)

## Datetime 🔗

*   **Import:**
    *   (from `datetime`) `date`, `time`, `datetime`, `timedelta`, `timezone`
    *   `import zoneinfo, dateutil.tz` (requires installation)
*   **Classes:**
    *   `date(year, month, day)`
    *   `time(hour=0, minute=0, second=0, microsecond=0, tzinfo=None, fold=0)`
    *   `datetime(year, month, day, hour=0)`
    *   `timedelta(weeks=0, days=0, hours=0)`
*   **Now:**
    *   `today()`, `now()`, `now(<tzinfo>)`
*   **Timezones:**
    *   `timezone.utc`
    *   `timezone(<timedelta>)`
    *   `dateutil.tz.tzlocal()`
    *   `zoneinfo.ZoneInfo('<iana_key>')`
    *   `astimezone([<tzinfo>])`, `replace(tzinfo=<tzinfo>)`
*   **Encode:**
    *   `fromisoformat(<str>)`
    *   `strptime(<str>, '<format>')`
    *   `fromordinal(<int>)`
    *   `fromtimestamp(<float>)`
*   **Decode:**
    *   `isoformat(sep='T')`
    *   `strftime('<format>')`
    *   `toordinal()`, `timestamp()`
*   **Format:**
    *   `strptime('2025-08-14 23:39:00.00 +0200', '%Y-%m-%d %H:%M:%S.%f %z')`
    *   `strftime("%dth of %B '%y (%a), %I:%M %p %Z")`
    *   (see original for examples.)
*   **Arithmetics:**
    *   `D/T/DTn > D/T/DTn`
    *   `DTa > DTa`
    *   `D/DTn - D/DTn`
    *   `DTa - DTa`
    *   `D/DT ± TD`, `TD * float`, `TD / TD`

## Function 🔗

*   **Definition:**
    *   `def <func_name>(<nondefault_args>): ...`
    *   `def <func_name>(<default_args>): ...`
    *   `def <func_name>(<nondefault_args>, <default_args>): ...`
*   **Call:**
    *   `<obj> = <function>(<positional_args>)`
    *   `<obj> = <function>(<keyword_args>)`
    *   `<obj> = <function>(<positional_args>, <keyword_args>)`
*   **Splat Operator:**  `*args`, `**kwargs`
    *   `(1, 2), {'z': 3} -> func(*args, **kwargs) == func(1, 2, z=3)`
    *   Allowed arguments combinations (see original)

## Inline 🔗

*   **Lambda:**
    *   `lambda: <return_value>`
    *   `lambda <arg_1>, <arg_2>: <return_value>`
*   **Comprehensions:**
    *   `[i+1 for i in range(10)]`
    *   `(i for i in range(10) if i > 5)`
    *   `{i+5 for i in range(10)}`
    *   `{i: i*2 for i in range(10)}`
    *   (and more examples in the original)
*   **Map, Filter, Reduce:** (from `functools` `import reduce`)
    *   `map(lambda x: x + 1, range(10))`
    *   `filter(lambda x: x > 5, range(10))`
    *   `reduce(lambda out, x: out + x, range(10))`
*   **Any, All:**
    *   `any(<collection>)`, `all(<collection>)`
*   **Conditional Expression:**
    *   `<obj> = <exp> if <condition> else <exp>`
*   **And, Or:**
    *   `<obj> = <exp> and <exp> [and ...]`
    *   `<obj> = <exp> or <exp> [or ...]`
*   **Walrus Operator:**
    *   `[i for a in '0123' if (i := int(a)) > 0]`

## Imports 🔗

*   `import <module>`
*   `import <package>`
*   `import <package>.<module>`
*   `from .[…][<pkg/module>[.…]] import <obj>`

## Closure 🔗

*   Nested function referencing an enclosing function's value, returned by the enclosing function.
*   **Partial:** (from `functools`)
    *   `<function> = partial(<function> [, <arg_1> [, ...]])`
*   **Non-Local:**
    *   `nonlocal i`

## Decorator 🔗

*   **Decorators add functionality to a function and return it.**
*   **Usage:**
    *   `@decorator_name`
    *   `def function_that_gets_passed_to_decorator():`
    *   `...`
*   **Debugger Example:** (see original)
    *   Uses `@wraps(func)` from `functools`.
*   **Cache Example:**
    *   Uses `@cache` from `functools`.  `cache_clear()`, `@lru_cache(maxsize=<int>)`
*   **Parametrized Decorator:** (see original)

## Class 🔗

*   **Definition:**
    *   `class MyClass:`
    *   `def __init__(self, a):`
    *   `self.a = a`
    *   `def __str__(self):`
    *   `return str(self.a)`
    *   `def __repr__(self):`
    *   `class_name = self.__class__.__name__`
    *   `return f'{class_name}({self.a!r})'`
    *   `@classmethod`
    *   `def get_class_name(cls):`
    *   `return cls.__name__`
*   **Subclass:**
    *   `class Employee(Person):`
    *   `super().__init__(name)`
    *   (See original examples)
*   **Type Annotations:**
    *   `<name>: <type> [| ...] [= <obj>]`
    *   (See original)
*   **Dataclass:** (from `dataclasses`)
    *   `@dataclass(order=False, frozen=False)`
    *   `class <class_name>:`
    *   `<attr_name>: <type> = <default_value>`
    *   `field(default_factory=list/dict/set)`
    *   `make_dataclass()`
*   **Property:**
    *   `@property` (getters and setters)
*   **Slots:**
    *   `__slots__ = ['a']`
*   **Copy:** (from `copy`)
    *   `copy(<object>)`, `deepcopy(<object>)`

## Duck Types 🔗

*   **Comparable:** (`__eq__(self, other)`)
    *   `__ne__()`, `__lt__()`, `__gt__()`, `__le__()`, `__ge__()`
*   **Hashable:** (`__hash__()`, `__eq__()`)
*   **Sortable:** (from `functools` total_ordering)
    *   Uses `__eq__()` and one of `__lt__()`, `__gt__()`, `__le__()`, or `__ge__()`.
*   **Iterator:** (`__next__()`, `__iter__()`)
*   **Callable:** (`__call__()`)
*   **Context Manager:** (`__enter__()`, `__exit__()`)

## Iterable Duck Types 🔗

*   **Iterable:** (`__iter__()`)
    *   `__contains__()`
*   **Collection:** (`__iter__()`, `__len__()`)
*   **Sequence:** (`__getitem__()`, `__len__()`)
    *   `__reversed__()`
*   **ABC Sequence:** (from `collections.abc` )
    *   `MyAbcSequence(abc.Sequence)`
    *   `__len__()`, `__getitem__()`

## Enum 🔗

*   **Import:** (from `enum`) `Enum`, `auto`
*   **Creation:**
    *   `class <enum_name>(Enum):`
    *   `<member_name> = auto()` (automatic increment)
    *   `<member_name> = <value>`
    *   `<member_name> = <el_1>, <el_2>` (tuple)
*   **Access:**
    *   `<member> = <enum>.<member_name>`
    *   `<member> = <enum>['<member_name>']`
    *   `<member> = <enum>(<value>)`
    *   `<str> = <member>.name`, `<obj> = <member>.value`
*   **Other:**
    *   `list(<enum>)`
    *   `[a.name for a in <enum>]`, `[a.value for a in <enum>]`
    *   `type(<member>)`, `itertools.cycle(<enum>)`, `random.choice(list(<enum>))`
*   **Inline:**
    *   `Cutlery = Enum('Cutlery', 'FORK KNIFE SPOON')`
    *   `Enum('Cutlery', ['FORK', 'KNIFE', 'SPOON'])`
    *   `Enum('Cutlery', {'FORK': 1, 'KNIFE': 2, 'SPOON': 3})`

## Exceptions 🔗

*   **Handling:**
    *   `try:`
    *   `<code>`
    *   `except <exception>:`
    *   `<code>`
*   **Multiple Exceptions:**
    *   `except <exception_a>:`
    *   `except <exception_b>:`
    *   `else:`
    *   `finally:`
*   **Catching Exceptions:**
    *   `except <exception>:`
    *   `except <exception> as <name>:`
    *   `except (<exception>, [...]):`
    *   `except (<exception>, [...]) as <name>:`
    *   `traceback.print_exc()`
    *   `print(<name>)`
    *   `logging.exception(<str>)`
    *   `sys.exc_info()`
*   **Raising Exceptions:**
    *   `raise <exception>`
    *   `raise <exception>()`
    *   `raise <exception>(<obj> [, ...])`
*   **Re-raising:**
    *   `raise` (inside `except <exception> as <name>:` block)
*   **Exception Object:**
    *   `arguments = <name>.args`
    *   `exc_type = <name>.__class__`
    *   `filename = <name>.__traceback__.tb_frame.f_code.co_filename`
    *   `func_name = <name>.__traceback__.tb_frame.f_code.co_name`
    *   `line = linecache.getline(filename, <name>.__traceback__.tb_lineno)`
    *   `trace_str = ''.join(traceback.format_tb(<name>.__traceback__))`
    *   `error_msg = ''.join(traceback.format_exception(type(<name>), <name>, <name>.__traceback__))`
*   **Built-in Exceptions:** (see original)
*   **User-defined Exceptions:**
    *   `class MyError(Exception): pass`

## Exit 🔗

*   **Import:** `import sys`
*   **Methods:**
    *   `sys.exit()` (exit code 0)
    *   `sys.exit(<int>)`
    *   `sys.exit(<obj>)`

## Print 🔗

*   `print(<el_1>, ..., sep=' ', end='\n', file=sys.stdout, flush=False)`
*   (use `file=sys.stderr` for errors)
*   **Pretty Print:** (from `pprint`)
    *   `pprint(<collection>, width=80, depth=None, compact=False, sort_dicts=True)`

## Input 🔗

*   `<str> = input()`

## Command Line Arguments 🔗

*   **Import:** `import sys`
    *   `scripts_path = sys.argv[0]`
    *   `arguments = sys.argv[1:]`
*   **Argument Parser:** (from `argparse`)
    *   `p = ArgumentParser(description=<str>)`
    *   `add_argument('-<short_name>', '--<name>', action='store_true')`
    *   `add_argument('-<short_name>', '--<name>', type=<type>)`
    *   `add_argument('<name>', type=<type>, nargs=1)`
    *   `add_argument('<name>', type=<type>, nargs='+')`
    *   `add_argument('<name>', type=<type>, nargs='?/*')`
    *   `args = p.parse_args()`
    *   `<obj> = args.<name>`
    *   `help=<str>`, `default=<obj>`, `type=FileType(<mode>)`

## Open 🔗

*   **Methods:**
    *   `<file> = open(<path>, mode='r', encoding=None, newline=None)`
    *   `encoding="utf-8"`
*   **Modes:**
    *   `'r'`, `'w'`, `'x'`, `'a'`, `'w+'`, `'r+'`, `'a+'`, `'b'`
*   **Exceptions:**
    *   `FileNotFoundError`, `FileExistsError`, `IsADirectoryError`, `PermissionError`, `OSError`
*   **File Object:**
    *   `seek(0), seek(offset, origin)`
    *   `read(size=-1)`, `readline()`, `readlines()`, `next(<file>)`
    *   `write(<str/bytes>)`, `writelines(<collection>)`, `flush()`, `close()`

*   **Read Text from File:**
    *   `def read_file(filename):`
    *   `with open(filename, encoding='utf-8') as file:`
    *   `return file.readlines()`
*   **Write Text to File:**
    *   `def write_to_file(filename, text):`
    *   `with open(filename, 'w', encoding='utf-8') as file:`
    *   `file.write(text)`

## Paths 🔗

*   **Import:** `import os, glob`, `from pathlib import Path`
*   **OS Functions:**
    *   `getcwd()`, `path.join(<path>, ...)`
    *   `path.realpath(<path>)`
    *   `path.basename(<path>)`, `path.dirname(<path>)`, `path.splitext(<path>)`
    *   `listdir(path='.')`, `glob.glob('<pattern>')`
    *   `path.exists(<path>)`, `path.isfile(<path>)`, `path.isdir(<path>)`
    *   `stat(<path>)`, `<stat>.st_mtime/st_size/…`
*   **DirEntry:** (from `os.scandir`)
    *   `scandir(path='.')`
    *   `<DirEntry>.path`, `.name`
    *   `open(<DirEntry>)`
*   **Path Object:**
    *   `Path(<path> [, ...])`
    *   `<path> / <path>`, `resolve()`
    *   `Path()`, `Path.cwd()`, `Path.home()`, `Path(__file__).resolve()`
    *   `.parent`, `.name`, `.suffix`, `.stem`, `.parts`
    *   `.iterdir()`, `.glob('<pattern>')`
    *   `str(<Path>)`, `as_uri()`
    *   `open(<Path>)`

## OS Commands 🔗

*   **Import:** `import os, shutil, subprocess`
*   **Functions:**
    *   `chdir(<path>)`
    *   `mkdir(<path>, mode=0o777)`, `makedirs(<path>, mode=0o777)`
    *   `shutil.copy(from, to)`, `copy2(from, to)`, `copytree(from, to)`
    *   `rename(from, to)`, `replace(from, to)`, `shutil.move(from, to)`
    *   `remove(<path>)`, `rmdir(<path>)`, `shutil.rmtree(<path>)`
*   **Shell Commands:**
    *   `<pipe> = os.popen('<commands>')`
    *   `<pipe>.read(size=-1)`, `readline/s()`
    *   `<pipe>.close()`

## JSON 🔗

*   **Import:** `import json`
*   **Functions:**
    *   `dumps(<list/dict>)`, `loads(<str>)`
*   **Read from File:**
    *   `def read_json_file(filename):`
    *   `with open(filename, encoding='utf-8') as file:`
    *   `return json.load(file)`
*   **Write to File:**
    *   `def write_to_json_file(filename, collection):`
    *   `with open(filename, 'w', encoding='utf-8') as file:`
    *   `json.dump(collection, file, ensure_ascii=False, indent=2)`

## Pickle 🔗

*   **Import:** `import pickle`
*   **Functions:**
    *   `dumps(<object>)`, `loads(<bytes>)`
*   **Read from File:**
    *   `def read_pickle_file(filename):`
    *   `with open(filename, 'rb') as file:`
    *   `return pickle.load(file)`
*   **Write to File:**
    *   `def write_to_pickle_file(filename, an_object):`
    *   `with open(filename, 'wb') as file:`
    *   `pickle.dump(an_object, file)`

## CSV 🔗

*   **Import:** `import csv`
*   **Read:**
    *   `reader(<file>, dialect=',', delimiter=',')`
    *   `next(<reader>)`, `list(<reader>)`
*   **Write:**
    *   `writer(<file>, dialect=',', delimiter=',')`
    *   `writerow(<collection>)`
    *   `writerows(<coll_of_coll>)`
*   **Parameters:**
    *   `dialect`, `delimiter`, `lineterminator`, `quotechar`, `escapechar`, `doublequote`, `quoting`, `skipinitialspace`
*   **Dialects:**
    *   `excel`, `excel-tab`, `unix`
*   **Read from File:**
    *   `def read_csv_file(filename, **csv_params):`
    *   `with open(filename, encoding='utf-8', newline='') as file:`
    *   `return list(csv.reader(file, **csv_params))`
*   **Write to File:**
    *   `def write_to_csv_file(filename, rows, mode='w', **csv_params):`
    *   `with open(filename, mode, encoding='utf-8', newline='') as file:`
    *   `writer = csv.writer(file, **csv_params)`
    *   `writer.writerows(rows)`

## SQLite 🔗

*   **Import:** `import sqlite3`
*   **Connection:**
    *   `<conn> = sqlite3.connect(<path>)`
    *   `<conn>.close()`
*   **Read:**
    *   `<cursor> = <conn>.