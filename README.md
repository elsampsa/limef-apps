# Example Limef apps

Cpp and Python example apps that use Limef library.

## External libraries

### OpenCV

For the OpenCV app you need to run [ext/build_ext.bash](ext/build_ext.bash)
After that you can opt to run [ext/make_opencv_deb.bash](ext/make_opencv_deb.bash) and install the debian package (it includes the .so libraries and also the header files)

Otherwise you need to add `ext/opencv/install/lib` into your `LD_LIBRARY_PATH`.

## With debian package

- Install limef debian package
- Install the opencv debian package

Just do
```bash
mkdir build
cd build
cmake ..
```
Python examples work oob

## Dev install

Maybe set up a staging environment, see [../staging.bash](../staging.bash).
Read the `run_cmake.bash` script of each example and adapt accordingly.
You need to set `LD_LIBRARY_PATH` and for python apps also `PYTHONPATH`.

Contents:
```
cpp/
    base/           # simple rtsp server
    onnx/           # ftm placeholder
    opencv/         # uses cuda opencv - please run first ext/build_ext.bash
    torch/          # ftm placeholder
python/
    # simple rtsp server and opencv examples in python
```

## Copyright

(c) 2026 Sampsa Riikonen

## License

MIT
