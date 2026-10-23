"""half-life sandbox network logger for Python processes.

The sandbox puts this directory first on ``PYTHONPATH``, so every Python
process started inside it imports this module as ``sitecustomize`` at startup.
It wraps ``socket.getaddrinfo``, ``socket.create_connection``, and the
``connect``, ``connect_ex``, and ``sendto`` methods of ``socket.socket``. Each
outbound attempt appends one JSON line ``{"target": "host:port", "via":
"python-socket"}`` to the file named by ``HALFLIFE_NETLOG`` and then lets the
call proceed, where the sandbox's empty network namespace makes it fail. When
``HALFLIFE_NET_BLOCK=1`` (no namespace available) the wrapper raises
``OSError(ENETUNREACH)`` instead of proceeding.

Loopback, unspecified, and own-hostname targets, Unix sockets, and passive
lookups (``host=None``) are not logged, and calls made inside an already
logged ``create_connection`` are not logged twice. This is logging, not
enforcement: ``python -I`` or ``-S`` skips it, and the network namespace is
what actually blocks traffic.
"""

from __future__ import annotations

import contextlib
import errno
import ipaddress
import json
import os
import socket
import threading
from collections.abc import Callable
from typing import Any

VIA = "python-socket"
_STATE = threading.local()


def is_local_host(host: object) -> bool:
    """Whether a host never leaves the machine (loopback, unspecified, own name)."""
    if host is None:
        return True
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    if not isinstance(host, str):
        return False
    name = host.strip().lower().rstrip(".")
    if name in ("", "localhost", "localhost.localdomain") or name.endswith(".localhost"):
        return True
    try:
        if name == socket.gethostname().lower():
            return True
    except OSError:
        pass
    try:
        address = ipaddress.ip_address(name.split("%", 1)[0])
    except ValueError:
        return False
    return address.is_loopback or address.is_unspecified


def format_target(host: object, port: object) -> str:
    """``host:port`` text for a log line (IPv6 literals in brackets)."""
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    text = str(host)
    if ":" in text and not text.startswith("["):
        text = f"[{text}]"
    return f"{text}:{port}"


def record(target: str, path: str | None = None) -> None:
    """Append one attempt to the netlog (no-op when no netlog is configured)."""
    path = os.environ.get("HALFLIFE_NETLOG") if path is None else path
    if not path:
        return
    line = json.dumps({"target": target, "via": VIA}, sort_keys=True)
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError:
        pass


def _depth() -> int:
    return int(getattr(_STATE, "depth", 0))


def attempt(host: object, port: object) -> None:
    """Log an outbound attempt, and refuse it when blocking is on."""
    if _depth() > 0 or is_local_host(host):
        return
    record(format_target(host, port))
    if os.environ.get("HALFLIFE_NET_BLOCK") == "1":
        raise OSError(errno.ENETUNREACH, "network access is disabled in the half-life sandbox")


def _inet_address(family: object, address: object) -> tuple[object, object] | None:
    inet = (socket.AF_INET, socket.AF_INET6)
    if family in inet and isinstance(address, tuple) and len(address) >= 2:
        return address[0], address[1]
    return None


def install(module: Any = socket) -> bool:
    """Wrap the socket entry points of ``module``; returns False if already done."""
    if getattr(module, "_halflife_netlog", False):
        return False
    cls = module.socket
    orig_getaddrinfo: Callable[..., Any] = module.getaddrinfo
    orig_create: Callable[..., Any] = module.create_connection
    orig_connect: Callable[..., Any] = cls.connect
    orig_connect_ex: Callable[..., Any] = cls.connect_ex
    orig_sendto: Callable[..., Any] = cls.sendto

    def getaddrinfo(host: Any, port: Any, *args: Any, **kwargs: Any) -> Any:
        attempt(host, port)
        return orig_getaddrinfo(host, port, *args, **kwargs)

    def create_connection(address: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(address, tuple) and len(address) >= 2:
            attempt(address[0], address[1])
        _STATE.depth = _depth() + 1
        try:
            return orig_create(address, *args, **kwargs)
        finally:
            _STATE.depth = _depth() - 1

    def connect(self: Any, address: Any) -> Any:
        target = _inet_address(getattr(self, "family", None), address)
        if target is not None:
            attempt(*target)
        return orig_connect(self, address)

    def connect_ex(self: Any, address: Any) -> Any:
        target = _inet_address(getattr(self, "family", None), address)
        if target is not None:
            attempt(*target)
        return orig_connect_ex(self, address)

    def sendto(self: Any, data: Any, *args: Any) -> Any:
        if args:
            target = _inet_address(getattr(self, "family", None), args[-1])
            if target is not None:
                attempt(*target)
        return orig_sendto(self, data, *args)

    module.getaddrinfo = getaddrinfo
    module.create_connection = create_connection
    cls.connect = connect
    cls.connect_ex = connect_ex
    cls.sendto = sendto
    module._halflife_netlog = True
    return True


if __name__ == "sitecustomize":  # pragma: no cover  # runs only inside the sandbox
    with contextlib.suppress(Exception):
        install()
