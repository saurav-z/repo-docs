# MAVLink: The Lightweight Communication Protocol for Drones and Robotics

MAVLink is a highly optimized, open-source message library that enables seamless communication between drones, ground control stations, and other robotic systems. ([See the original repo](https://github.com/mavlink/mavlink))

## Key Features of MAVLink

*   **Lightweight and Efficient:** Designed for resource-constrained environments, making it ideal for drones and embedded systems.
*   **Header-Only Library:** Easy to integrate into projects with minimal dependencies.
*   **XML-Based Message Definitions:** Uses XML files to define message sets ("dialects"), promoting interoperability and flexibility.
*   **Multi-Language Support:** Generate code in various languages ([see supported languages](https://mavlink.io/en/#supported_languages)) using Python tools.
*   **Field-Proven:** Widely used in the drone and robotics industry, ensuring reliability and robustness.
*   **Interoperability:** Facilitates communication between components from different manufacturers.

## Getting Started: Generate C Headers (Example)

Here's a quick guide to generating C headers on Ubuntu LTS 20.04 or 22.04:

```bash
# Dependencies
sudo apt install python3-pip

# Clone mavlink
git clone https://github.com/mavlink/mavlink.git --recursive
cd mavlink

python3 -m pip install -r pymavlink/requirements.txt
```

Generate the C library:

```bash
python3 -m pymavlink.tools.mavgen --lang=C --wire-protocol=2.0 --output=generated/include/mavlink/v2.0 message_definitions/v1.0/common.xml
```

## Using with CMake (Example)

1.  **Install Headers:**

    ```bash
    cmake -Bbuild -H. -DCMAKE_INSTALL_PREFIX=install -DMAVLINK_DIALECT=common -DMAVLINK_VERSION=2.0
    cmake --build build --target install
    ```

2.  **Use `find_package` in `CMakeLists.txt`:**

    ```cmake
    find_package(MAVLink REQUIRED)
    add_executable(my_program my_program.c)
    target_link_libraries(my_program PRIVATE MAVLink::mavlink)
    ```

3.  **Pass Install Directory to CMake:**

    ```bash
    cd ../my_program
    cmake -Bbuild -H. -DCMAKE_PREFIX_PATH=../mavlink/install
    ```

    See [examples/c](examples/c) for a full example.

    *Note: MAVLink is a header-only library, so it doesn't truly "link".*

## Further Resources

*   **Documentation:** [mavlink.io/en/](https://mavlink.io/en/)
*   **Support:** [mavlink.io/en/#support](https://mavlink.io/en/#support)
*   **Contributing:** [mavlink.io/en/contributing/contributing.html](https://mavlink.io/en/contributing/contributing.html)
*   **License:** [mavlink.io/en/#license](https://mavlink.io/en/#license)
*   **Using C MAVLink Libraries (mavgen):** [mavlink.io/en/mavgen_c/](https://mavlink.io/en/mavgen_c/)
*   **Installing the MAVLink Toolchain:** [mavlink.io/en/getting_started/installation.html](https://mavlink.io/en/getting_started/installation.html)
*   **Generating MAVLink Libraries:** [mavlink.io/en/getting_started/generate_libraries.html](https://mavlink.io/en/getting_started/generate_libraries.html)
*   **Using MAVLink Libraries:** [mavlink.io/en/getting_started/use_libraries.html](https://mavlink.io/en/getting_started/use_libraries.html)