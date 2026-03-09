# Example Limef apps

Cpp and Python example apps that use Limef library.

For cpp apps you need to either install the limef debian package or set up a staging environment, see [../staging.bash](../staging.bash).
Read the `run_cmake.bash` script of each example and adapt accordingly.

If you use the staging environment, you need to set `LD_LIBRARY_PATH` and for staging + python also `PYTHONPATH`.

For the OpenCV app you need to run [ext/build_ext.bash](ext/build_ext.bash) and also add `ext/opencv/install/lib` into your `LD_LIBRARY_PATH`
or otherwise install the cuda opencv library globally.

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
