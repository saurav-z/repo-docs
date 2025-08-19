# MAVLink: The Lightweight Communication Protocol for Drones and Robotics

MAVLink is a powerful and efficient message library designed for seamless communication between drones, ground control stations, and other robotic systems.  For more information, visit the [MAVLink GitHub repository](https://github.com/mavlink/mavlink).

## Key Features and Benefits

*   **Lightweight and Efficient:** Optimized for resource-constrained systems, ideal for drones and embedded devices.
*   **Header-Only Library:** Simplifies integration and reduces build times.
*   **Cross-Platform Compatibility:** Supports multiple programming languages including C, Python, and others (see supported languages on the [MAVLink website](https://mavlink.io/en/#supported_languages)).
*   **XML-Based Message Definitions:** Allows for easy customization and extension of message sets ("dialects").
*   **Field-Proven:** Widely adopted and deployed in numerous products, ensuring reliability and interoperability.

## Getting Started: Generating C Headers

Here's how to get started with MAVLink, specifically generating C headers:

**1. Install Dependencies (Ubuntu):**

```bash
sudo apt install python3-pip
```

**2. Clone the MAVLink Repository:**

```bash
git clone https://github.com/mavlink/mavlink.git --recursive
cd mavlink
```

**3. Install Python Requirements:**

```bash
python3 -m pip install -r pymavlink/requirements.txt
```

**4. Generate C Headers:**

```bash
python3 -m pymavlink.tools.mavgen --lang=C --wire-protocol=2.0 --output=generated/include/mavlink/v2.0 message_definitions/v1.0/common.xml
```

## Integrating with CMake

**1. Install Headers Locally:**

```bash
cmake -Bbuild -H. -DCMAKE_INSTALL_PREFIX=install -DMAVLINK_DIALECT=common -DMAVLINK_VERSION=2.0
cmake --build build --target install
```

**2.  Find MAVLink in `CMakeLists.txt`:**

```cmake
find_package(MAVLink REQUIRED)
add_executable(my_program my_program.c)
target_link_libraries(my_program PRIVATE MAVLink::mavlink)
```

**3.  Configure CMake:**

```bash
cd ../my_program
cmake -Bbuild -H. -DCMAKE_PREFIX_PATH=../mavlink/install
```

For a complete example, see the `examples/c` directory within the repository.

*Note: MAVLink is a header-only library, so `target_link_libraries` in CMake doesn't actually link to a library.*

## Resources and Further Information

*   [Documentation/Website](https://mavlink.io/en/)
*   [Discussion/Support](https://mavlink.io/en/#support)
*   [Contributing](https://mavlink.io/en/contributing/contributing.html)
*   [License](https://mavlink.io/en/#license)
*   [Using C MAVLink Libraries (mavgen)](https://mavlink.io/en/mavgen_c/)
*   [Installing the MAVLink Toolchain](https://mavlink.io/en/getting_started/installation.html)
*   [Generating MAVLink Libraries](https://mavlink.io/en/getting_started/generate_libraries.html)
*   [Using MAVLink Libraries](https://mavlink.io/en/getting_started/use_libraries.html)