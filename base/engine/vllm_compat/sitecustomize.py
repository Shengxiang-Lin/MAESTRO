"""Compatibility fixes for Python subprocesses launched by MAESTRO's vLLM manager.

FlashInfer 0.6.16 annotates a function with ``array.array[int]``. Python 3.11
cannot evaluate that expression at import time. Postpone annotations only in
that module, without editing the installation or writing patched bytecode.
"""
from __future__ import annotations

import __future__
import array
import importlib.abc
import importlib.machinery


class _PostponedAnnotationsLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        # Compile directly so cached code cannot bypass the fix and patched code
        # is never saved inside the environment's site-packages directory.
        return compile(
            self.get_data(self.path),
            self.path,
            "exec",
            flags=__future__.annotations.compiler_flag,
            dont_inherit=True,
        )


class _FlashInferAnnotationsFinder(importlib.abc.MetaPathFinder):
    _maestro_flashinfer_annotations = True

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "flashinfer.comm.fd_exchange":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or not isinstance(
            spec.loader, importlib.machinery.SourceFileLoader
        ):
            return None
        spec.loader = _PostponedAnnotationsLoader(fullname, spec.loader.path)
        return spec


def _install():
    if hasattr(array.array, "__class_getitem__"):
        return
    import sys

    if not any(
        getattr(finder, "_maestro_flashinfer_annotations", False)
        for finder in sys.meta_path
    ):
        sys.meta_path.insert(0, _FlashInferAnnotationsFinder())


_install()
