import subprocess
import sys
import os
import shlex
import re
import functools
import urllib.parse


def remove_dot_segments(path):
    # Implements RFC3986 5.2.4 remote_dot_segments
    # Pseudo-code: https://tools.ietf.org/html/rfc3986#section-5.2.4
    # https://github.com/urllib3/urllib3/blob/ba49f5c4e19e6bca6827282feb77a3c9f937e64b/src/urllib3/util/url.py#L263
    output = []
    segments = path.split("/")
    for s in segments:
        if s == ".":
            continue
        elif s == "..":
            if output:
                output.pop()
        else:
            output.append(s)
    if not segments[0] and (not output or output[0]):
        output.insert(0, "")
    if segments[-1] in (".", ".."):
        output.append("")
    return "/".join(output)


def escape_rfc3986(s):
    """Escape non-ASCII characters as suggested by RFC 3986"""
    return urllib.parse.quote(s, b"%/;:@&=+$,!~*'()?#[]")


def normalize_url(url):
    """Normalize URL as suggested by RFC 3986"""
    url_parsed = urllib.parse.urlparse(url)
    return url_parsed._replace(
        netloc=url_parsed.netloc.encode("idna").decode("ascii"),
        path=escape_rfc3986(remove_dot_segments(url_parsed.path)),
        params=escape_rfc3986(url_parsed.params),
        query=escape_rfc3986(url_parsed.query),
        fragment=escape_rfc3986(url_parsed.fragment),
    ).geturl()


# Python 3.8+ does not honor %HOME% on windows, but this breaks compatibility with youtube-dl
# See https://github.com/yt-dlp/yt-dlp/issues/792
# https://docs.python.org/3/library/os.path.html#os.path.expanduser
if os.name in ("nt", "ce"):

    def compat_expanduser(path):
        HOME = os.environ.get("HOME")
        if not HOME:
            return os.path.expanduser(path)
        elif not path.startswith("~"):
            return path
        i = path.replace("\\", "/", 1).find("/")  # ~user
        if i < 0:
            i = len(path)
        userhome = os.path.join(os.path.dirname(HOME), path[1:i]) if i > 1 else HOME
        return userhome + path[i:]

else:
    compat_expanduser = os.path.expanduser


def expand_path(s):
    """Expand shell variables and ~"""
    return os.path.expandvars(compat_expanduser(s))


_WINDOWS_QUOTE_TRANS = str.maketrans({'"': R"\""})
_CMD_QUOTE_TRANS = str.maketrans(
    {
        # Keep quotes balanced by replacing them with `""` instead of `\\"`
        '"': '""',
        # These require an env-variable `=` containing `"^\n\n"` (set in `utils.Popen`)
        # `=` should be unique since variables containing `=` cannot be set using cmd
        "\n": "%=%",
        "\r": "%=%",
        # Use zero length variable replacement so `%` doesn't get expanded
        # `cd` is always set as long as extensions are enabled (`/E:ON` in `utils.Popen`)
        "%": "%%cd:~,%",
    }
)


def shell_quote(args, *, shell=False):
    args = list(variadic(args))

    if os.name != "nt":
        return shlex.join(args)

    trans = _CMD_QUOTE_TRANS if shell else _WINDOWS_QUOTE_TRANS
    return " ".join(
        (
            s
            if re.fullmatch(r"[\w#$*\-+./:?@\\]+", s, re.ASCII)
            else re.sub(r'(\\+)("|$)', r"\1\1\2", s).translate(trans).join('""')
        )
        for s in args
    )


class Popen(subprocess.Popen):
    if sys.platform == "win32":
        _startupinfo = subprocess.STARTUPINFO()
        _startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    else:
        _startupinfo = None

    @staticmethod
    def _fix_pyinstaller_issues(env):
        if not hasattr(sys, "_MEIPASS"):
            return

        # Force spawning independent subprocesses for exes bundled with PyInstaller>=6.10
        # Ref: https://pyinstaller.org/en/v6.10.0/CHANGES.html#incompatible-changes
        #      https://github.com/yt-dlp/yt-dlp/issues/11259
        env["PYINSTALLER_RESET_ENVIRONMENT"] = "1"

        # Restore LD_LIBRARY_PATH when using PyInstaller
        # Ref: https://pyinstaller.org/en/v6.10.0/runtime-information.html#ld-library-path-libpath-considerations
        #      https://github.com/yt-dlp/yt-dlp/issues/4573
        def _fix(key):
            orig = env.get(f"{key}_ORIG")
            if orig is None:
                env.pop(key, None)
            else:
                env[key] = orig

        _fix("LD_LIBRARY_PATH")  # Linux
        _fix("DYLD_LIBRARY_PATH")  # macOS

    def __init__(self, args, *remaining, env=None, text=False, shell=False, **kwargs):
        if env is None:
            env = os.environ.copy()
        self._fix_pyinstaller_issues(env)

        self.__text_mode = (
            kwargs.get("encoding")
            or kwargs.get("errors")
            or text
            or kwargs.get("universal_newlines")
        )
        if text is True:
            kwargs["universal_newlines"] = True  # For 3.6 compatibility
            kwargs.setdefault("encoding", "utf-8")
            kwargs.setdefault("errors", "replace")

        if shell and os.name == "nt" and kwargs.get("executable") is None:
            if not isinstance(args, str):
                args = shell_quote(args, shell=True)
            shell = False
            # Set variable for `cmd.exe` newline escaping (see `utils.shell_quote`)
            env["="] = '"^\n\n"'
            args = f'{self.__comspec()} /Q /S /D /V:OFF /E:ON /C "{args}"'

        super().__init__(
            args,
            *remaining,
            env=env,
            shell=shell,
            **kwargs,
            startupinfo=self._startupinfo,
        )

    def __comspec(self):
        comspec = os.environ.get("ComSpec") or os.path.join(
            os.environ.get("SystemRoot", ""), "System32", "cmd.exe"
        )
        if os.path.isabs(comspec):
            return comspec
        raise FileNotFoundError(
            "shell not found: neither %ComSpec% nor %SystemRoot% is set"
        )

    def communicate_or_kill(self, *args, **kwargs):
        try:
            return self.communicate(*args, **kwargs)
        except BaseException:  # Including KeyboardInterrupt
            self.kill(timeout=None)
            raise

    def kill(self, *, timeout=0):
        super().kill()
        if timeout != 0:
            self.wait(timeout=timeout)

    @classmethod
    def run(cls, *args, timeout=None, **kwargs):
        with cls(*args, **kwargs) as proc:
            default = "" if proc.__text_mode else b""
            stdout, stderr = proc.communicate_or_kill(timeout=timeout)
            return stdout or default, stderr or default, proc.returncode


def error_to_str(err):
    return f"{type(err).__name__}: {err}"


def is_path_like(f):
    return isinstance(f, (str, bytes, os.PathLike))


def sanitize_url(url, *, scheme="http"):
    # Prepend protocol-less URLs with `http:` scheme in order to mitigate
    # the number of unwanted failures due to missing protocol
    if url is None:
        return
    elif url.startswith("//"):
        return f"{scheme}:{url}"
    # Fix some common typos seen so far
    COMMON_TYPOS = (
        # https://github.com/ytdl-org/youtube-dl/issues/15649
        (r"^httpss://", r"https://"),
        # https://bx1.be/lives/direct-tv/
        (r"^rmtp([es]?)://", r"rtmp\1://"),
    )
    for mistake, fixup in COMMON_TYPOS:
        if re.match(mistake, url):
            return re.sub(mistake, fixup, url)
    return url


def str_or_none(v, default=None):
    return default if v is None else str(v)


def try_call(*funcs, expected_type=None, args=[], kwargs={}):
    for f in funcs:
        try:
            val = f(*args, **kwargs)
        except (
            AttributeError,
            KeyError,
            TypeError,
            IndexError,
            ValueError,
            ZeroDivisionError,
        ):
            pass
        else:
            if expected_type is None or isinstance(val, expected_type):
                return val


WINDOWS_VT_MODE = False if os.name == "nt" else None


@functools.cache
def supports_terminal_sequences(stream):
    if os.name == "nt":
        if not WINDOWS_VT_MODE:
            return False
    elif not os.getenv("TERM"):
        return False
    try:
        return stream.isatty()
    except BaseException:
        return False


def write_string(s, out=None, encoding=None):
    assert isinstance(s, str)
    out = out or sys.stderr
    # `sys.stderr` might be `None` (Ref: https://github.com/pyinstaller/pyinstaller/pull/7217)
    if not out:
        return

    if os.name == "nt" and supports_terminal_sequences(out):
        s = re.sub(r"([\r\n]+)", r" \1", s)

    enc, buffer = None, out
    # `mode` might be `None` (Ref: https://github.com/yt-dlp/yt-dlp/issues/8816)
    if "b" in (getattr(out, "mode", None) or ""):
        enc = encoding or preferredencoding()
    elif hasattr(out, "buffer"):
        buffer = out.buffer
        enc = encoding or getattr(out, "encoding", None) or preferredencoding()

    buffer.write(s.encode(enc, "ignore") if enc else s)
    out.flush()
