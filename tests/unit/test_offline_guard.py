"""The test suite runs offline: see the guard in tests/conftest.py."""
import socket

import pytest


def test_outbound_connections_are_blocked():
    with pytest.raises(RuntimeError, match="Network access is disabled in tests"):
        socket.create_connection(("93.184.216.34", 80), timeout=2)


def test_loopback_is_allowed():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    socket.create_connection(server.getsockname(), timeout=2).close()
    server.close()


@pytest.mark.network
def test_network_marker_disables_the_guard():
    assert socket.socket.connect.__name__ != "_guarded_connect"
