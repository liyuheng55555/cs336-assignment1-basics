def connection_to_str(tp: tuple[bytes, bytes]) -> str:
    return "(" + tp[0].decode("utf-8") + ", " + tp[1].decode("utf-8") + ")"

def max_connection(connections: list[tuple[bytes,bytes]]) -> tuple[bytes,bytes]:
    return max(connections, key=lambda x: (x[0].decode("utf-8"), x[1].decode("utf-8")))