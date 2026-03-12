import os

import IsaREPL


def mk_unicode_file(path: str) -> str:
    if not path.endswith(".thy"):
        raise ValueError(f"Expected .thy file, got: {path}")
    unicode_path = path[:-4] + ".unicode.thy"
    if os.path.exists(unicode_path) and os.path.getmtime(unicode_path) > os.path.getmtime(path):
        return unicode_path
    with open(path, "r") as f:
        content = f.read()
    unicode_content = IsaREPL.Client.pretty_unicode(content)
    with open(unicode_path, "w") as f:
        f.write(unicode_content)
    return unicode_path
