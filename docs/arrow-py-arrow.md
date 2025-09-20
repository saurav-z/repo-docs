# Arrow: Python's Intuitive Date and Time Library

**Simplify your Python date and time manipulations with Arrow, a user-friendly library designed to make working with dates and times a breeze.**  [Check out the Arrow GitHub Repository](https://github.com/arrow-py/arrow) for the source code.

## Key Features:

*   **User-Friendly Interface:** A more intuitive approach to date and time handling.
*   **Drop-in Replacement:** Fully implemented as a replacement for Python's datetime module.
*   **Python 3.8+ Compatibility:** Supports the latest Python versions.
*   **Timezone Awareness:** Handles timezones and UTC by default.
*   **Simple Creation:** Easily create date and time objects from various input formats.
*   **Flexible Manipulation:** Utilize the `shift` method for relative offsets, including weeks.
*   **Automatic Formatting and Parsing:** Automatically parses and formats strings.
*   **ISO 8601 Support:** Wide support for the ISO 8601 standard.
*   **Timezone Conversion:** Effortlessly convert between timezones.
*   **Integration with Existing Libraries:** Works seamlessly with `dateutil`, `pytz`, and `ZoneInfo`.
*   **Time Span Generation:** Generate time spans, ranges, and floors/ceilings.
*   **Humanization:** Human-readable date and time representations with locale support.
*   **Extensibility:** Easily extendable for custom Arrow-derived types.
*   **Type Hints:** Full support for PEP 484-style type hints.

## Why Use Arrow?

Python's built-in datetime modules can be complex. Arrow streamlines date and time operations by:

*   **Consolidating Modules:** Reduces the need to import multiple modules.
*   **Simplifying Types:** Offers a more cohesive set of date and time types.
*   **Improving Timezone Handling:** Simplifies timezone conversions and timestamp operations.
*   **Adding Missing Features:** Provides functionalities like ISO 8601 parsing and humanization.

## Quick Start

### Installation

Install Arrow using `pip`:

```bash
pip install -U arrow
```

### Example Usage

```python
import arrow

# Create an Arrow object from a string
arrow.get('2013-05-11T21:23:58.970460+07:00')
# Output: <Arrow [2013-05-11T21:23:58.970460+07:00]>

# Get the current UTC time
utc = arrow.utcnow()
# Output: <Arrow [2013-05-11T21:23:58.970460+00:00]>

# Shift the time by one hour
utc = utc.shift(hours=-1)
# Output: <Arrow [2013-05-11T20:23:58.970460+00:00]>

# Convert to a different timezone
local = utc.to('US/Pacific')
# Output: <Arrow [2013-05-11T13:23:58.970460-07:00]>

# Get a timestamp
local.timestamp()
# Output: 1368303838.970460

# Format the time
local.format()
# Output: '2013-05-11 13:23:58 -07:00'

local.format('YYYY-MM-DD HH:mm:ss ZZ')
# Output: '2013-05-11 13:23:58 -07:00'

# Humanize the time
local.humanize()
# Output: 'an hour ago'

local.humanize(locale='ko-kr')
# Output: '한시간 전'
```

## Documentation

For detailed information and API reference, please visit the official documentation: [arrow.readthedocs.io](https://arrow.readthedocs.io)

## Contributing

Contributions are welcome!  Review the [issue tracker](https://github.com/arrow-py/arrow/issues) and the ["good first issue" label](https://github.com/arrow-py/arrow/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22) to find ways to get involved.  Refer to the original README for specifics on contributing.

## Support Arrow

Consider supporting the project through [Open Collective](https://opencollective.com/arrow) to help ensure its continued development.