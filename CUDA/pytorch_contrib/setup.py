from setuptools import setup
from torch.utils import cpp_extension


# https://pytorch.org/tutorials/advanced/cpp_custom_ops.html#cpp-custom-ops-tutorial


setup(name="custom_linear",
      ext_modules=[
          cpp_extension.CUDAExtension("custom_linear", ["custom_linear.cu"])
      ],
      cmdclass={'build_ext': cpp_extension.BuildExtension}
)