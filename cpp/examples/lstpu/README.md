# Simple C++ code example

This example shows how to build a simple C++ program that uses the Edge TPU
runtime library to lists available Edge TPU devices. (It does not perform an
inference.)

## Requirements

You need to install [Bazel](https://bazel.build/) in order to build the binary.
Follow the [Bazel install
instructions](https://docs.bazel.build/versions/master/install.html)

The example is configured to use the cross-compilation toolchain definition from
[crosstool](https://github.com/google-coral/crosstool), but you don't need
to download that repo.

## Compile the example

For the native compilation you need to install at least `build-essential`
package:

```
sudo apt-get install -y build-essential
```

Then run `make` command.

For cross-compilation you need to install `build-essential` packages for
the corresponding architectures:

```
sudo apt-get install -y crossbuild-essential-armhf \
                        crossbuild-essential-arm64
```

Then run `make CPU=armv7a` or `make CPU=aarch64`.

Find the output binary file inside `blaze-out` directory.
