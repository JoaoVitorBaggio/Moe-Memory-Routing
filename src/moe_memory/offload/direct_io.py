"""``pread`` alinhado. ``O_DIRECT`` quando o sistema aceita; senão ``pread``.

``O_DIRECT`` exige offset, tamanho e buffer múltiplos do bloco do
dispositivo. O store só pede blocos de 4 KiB, então o caminho direto
não realinha no meio do expert. Se o open com ``O_DIRECT`` falha
(tmpfs, sistema de arquivos sem suporte), a leitura cai para ``pread``
e o resto do código não muda.
"""

from __future__ import annotations

import ctypes
import os

_libc = ctypes.CDLL(None, use_errno=True)
_libc.pread.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_longlong]
_libc.pread.restype = ctypes.c_ssize_t
_libc.posix_memalign.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_size_t]
_libc.posix_memalign.restype = ctypes.c_int
_libc.free.argtypes = [ctypes.c_void_p]
_libc.free.restype = None


class BlockFile:
    """Arquivo aberto para leitura de blocos. ``pread`` é thread-safe."""

    def __init__(self, path: str, *, direct: bool) -> None:
        flags = os.O_RDONLY
        self.direct = False
        self.path = path
        if direct and hasattr(os, "O_DIRECT"):
            try:
                self.fd = os.open(path, flags | os.O_DIRECT)
                self.direct = True
            except OSError:
                self.fd = os.open(path, flags)
        else:
            self.fd = os.open(path, flags)

    def read_at(self, offset: int, size: int) -> bytes:
        if offset < 0 or size < 0:
            raise ValueError("offset e tamanho precisam ser >= 0")
        if self.direct:
            if offset % 4096 != 0 or size % 4096 != 0:
                raise ValueError("O_DIRECT exige offset e tamanho múltiplos de 4096")
            return _pread_aligned(self.fd, offset, size)
        got = os.pread(self.fd, size, offset)
        if len(got) != size:
            raise OSError(f"leitura curta em {self.path}: {len(got)} de {size} em {offset}")
        return got

    def read_into(self, offset: int, dest_ptr: int, size: int, *, dest_aligned: bool) -> None:
        """Lê para um buffer já alocado. ``dest_ptr`` é ``data_ptr`` do tensor."""

        if self.direct and dest_aligned and dest_ptr % 4096 == 0 and offset % 4096 == 0 and size % 4096 == 0:
            _pread_into(self.fd, dest_ptr, size, offset)
            return
        blob = self.read_at(offset, size)
        ctypes.memmove(dest_ptr, blob, size)

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def __enter__(self) -> BlockFile:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _pread_aligned(fd: int, offset: int, size: int) -> bytes:
    addr = ctypes.c_void_p()
    rc = _libc.posix_memalign(ctypes.byref(addr), 4096, max(size, 1))
    if rc != 0:
        raise MemoryError(f"posix_memalign falhou ({rc})")
    try:
        _pread_into(fd, addr.value or 0, size, offset)
        return ctypes.string_at(addr, size)
    finally:
        _libc.free(addr)


def _pread_into(fd: int, ptr: int, size: int, offset: int) -> None:
    got = _libc.pread(fd, ctypes.c_void_p(ptr), size, offset)
    if got < 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    if got != size:
        raise OSError(f"leitura curta: {got} de {size}")
