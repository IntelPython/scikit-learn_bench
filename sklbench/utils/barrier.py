# ===============================================================================
# Copyright 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================

"""TCP socket barrier for synchronizing throughput mode worker processes."""

import socket
from typing import List, Tuple


def create_server() -> Tuple[socket.socket, int]:
    """Create a TCP server socket on localhost with OS-assigned port."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("localhost", 0))
    server.listen(128)
    port = server.getsockname()[1]
    return server, port


def recv_until(sock: socket.socket, expected: bytes):
    """Block until expected message is received on socket."""
    data = b""
    while expected not in data:
        chunk = sock.recv(1024)
        if not chunk:
            raise ConnectionError(
                f"Socket closed before receiving {expected!r}"
            )
        data += chunk


def send_all(connections: List[socket.socket], message: bytes):
    """Send message to all connections."""
    for conn in connections:
        conn.sendall(message)


def accept_and_wait(
    server: socket.socket, num_connections: int, expected: bytes, timeout: float
) -> List[socket.socket]:
    """Accept num_connections and wait for expected message from each."""
    server.settimeout(timeout)
    connections = []
    for _ in range(num_connections):
        conn, _ = server.accept()
        recv_until(conn, expected)
        connections.append(conn)
    return connections


def wait_all(connections: List[socket.socket], expected: bytes, timeout: float):
    """Wait for expected message from all existing connections."""
    for conn in connections:
        conn.settimeout(timeout)
        recv_until(conn, expected)
