# Arrow:  Effortless Date and Time Handling in Python

**Simplify your Python date and time manipulations with Arrow, a powerful library that makes working with dates, times, and timestamps a breeze.**  [Explore the Arrow Repository](https://github.com/arrow-py/arrow)

## Key Features of Arrow

Arrow simplifies date and time management in Python, offering these key benefits:

*   **Intuitive Date and Time Creation:** Easily create Arrow objects from various input formats.
*   **Timezone Awareness:**  Built-in timezone support and UTC as the default.
*   **Flexible Time Manipulation:**  Shift dates/times with relative offsets (e.g., weeks, hours).
*   **Automatic Formatting and Parsing:** Seamlessly format and parse strings using ISO 8601 and more.
*   **ISO 8601 Standard Support:** Extensive support for the widely used ISO 8601 standard.
*   **Timezone Conversion:** Effortlessly convert between timezones.
*   **Human-Friendly Output:** Humanize dates and times into easy-to-understand phrases.
*   **Extensible and Customizable:**  Create your own Arrow-derived types.
*   **Type Hinting Support:** Full support for PEP 484-style type hints for better code quality.

## Why Use Arrow?

Arrow addresses the usability challenges of Python's standard `datetime` and related modules:

*   **Reduces Complexity:** Fewer modules and types to manage.
*   **Simplifies Timezone Handling:**  Makes timezone conversions straightforward.
*   **Provides Missing Functionality:** Includes features like ISO 8601 parsing and humanization.

## Quick Start

### Installation

Install Arrow using `pip`:

```bash
pip install -U arrow
```

### Example Usage

```python
import arrow

# Create an Arrow object
arrow.get('2013-05-11T21:23:58.970460+07:00')
# Output: <Arrow [2013-05-11T21:23:58.970460+07:00]>

# Get UTC time
utc = arrow.utcnow()
# Output: <Arrow [2013-05-11T21:23:58.970460+00:00]>

# Shift time
utc = utc.shift(hours=-1)
# Output: <Arrow [2013-05-11T20:23:58.970460+00:00]>

# Convert to another timezone
local = utc.to('US/Pacific')
# Output: <Arrow [2013-05-11T13:23:58.970460-07:00]>

# Get the timestamp
local.timestamp()
# Output: 1368303838.970460

# Format the date and time
local.format()
# Output: '2013-05-11 13:23:58 -07:00'

local.format('YYYY-MM-DD HH:mm:ss ZZ')
# Output: '2013-05-11 13:23:58 -07:00'

# Humanize the date and time
local.humanize()
# Output: 'an hour ago'

local.humanize(locale='ko-kr')
# Output: '한시간 전'
```

## Documentation

For comprehensive information, see the official documentation: [arrow.readthedocs.io](https://arrow.readthedocs.io)

## Contributing

Your contributions are welcome!  Help improve Arrow by:

1.  Identifying issues on the [issue tracker](https://github.com/arrow-py/arrow/issues).
2.  Forking the repository.
3.  Implementing changes and writing tests.
4.  Running tests and linting checks using `tox` or `make`.
5.  Submitting a pull request.

## Support Arrow

Support the project through the [Open Collective](https://opencollective.com/arrow) to make one-time or recurring donations directly to the project.