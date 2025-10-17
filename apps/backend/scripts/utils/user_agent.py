import os
import re
import json
import platform
import subprocess
from pathlib import Path
from functools import lru_cache
from loguru import logger

try:
    import plistlib  # macOS
except Exception:
    plistlib = None

try:
    import winreg  # Windows
except Exception:
    winreg = None

try:
    from selenium import webdriver

    _has_selenium = True
except ImportError:
    _has_selenium = False


@lru_cache()
def get_browser_user_agent(
    browser: str = "chrome", profile: str | None = None, use_selenium: bool = False
) -> str:
    """
    Extract User-Agent from browser. Supports extraction from config files or by launching the browser with selenium.

    Args:
        browser: The name of the browser.
        profile: The profile name or path from which to load the user agent.
                 For example, Chrome supports multiple user logins;
                 here, profile specifies which user's user-agent to use.
        use_selenium: Whether to use selenium to extract the user agent.
                      If True, the browser will be launched and the user agent will be extracted from the browser.
                      If False, the user agent will be extracted from the config files.

    Returns:
        A string of the user agent.
    """
    browser = browser.lower()
    # Env override takes highest precedence
    for key in ("WEBTHINKER_USER_AGENT", "BROWSER_USER_AGENT", "USER_AGENT"):
        env_ua = os.environ.get(key)
        if env_ua:
            logger.info(f"[UA] Using user agent from env {key}")
            return env_ua.strip()

    # Allow default profile from env if not provided explicitly
    if profile is None:
        profile = (
            os.environ.get("CHROME_PROFILE")
            or os.environ.get("CHROME_PROFILE_DIR")
            or os.environ.get("CHROMIUM_PROFILE")
            or os.environ.get("BROWSER_PROFILE")
        )

    logger.info(f"[UA] Getting user agent for browser: {browser} (profile={profile})")

    if use_selenium and _has_selenium:
        ua = _get_ua_via_selenium(browser)
        if ua is not None:
            logger.info(f"[UA] Retrieved from selenium: {ua}")
            return ua
        else:
            logger.error(
                f"[UA] Selenium failed for {browser}, fallback to not use selenium"
            )

    try:
        if browser in ("chrome", "brave", "edge", "vivaldi", "whale", "chromium"):
            ua = _get_chromium_ua(browser, profile)
        elif browser == "firefox":
            ua = _get_firefox_ua()
        elif browser == "safari":
            ua = _get_safari_ua()
        else:
            logger.warning(
                f"[UA] Unsupported browser: {browser}, using default user agent"
            )
            return _default_user_agent()
        logger.info(f"[UA] Retrieved from {browser}: {ua}")
        return ua
    except Exception as e:
        logger.warning(f"[UA] Falling back to default UserAgent due to: {e}")
        return _default_user_agent()


def current_os_family() -> str:
    s = platform.system().lower()
    if "darwin" in s or "mac" in s:
        return "mac"
    if "windows" in s:
        return "windows"
    return "linux"  # covers Linux/*nix


class VersionInfo:
    __slots__ = ("chrome_version", "brand_version", "source")

    def __init__(
        self, chrome_version: str | None, brand_version: str | None, source: str
    ):
        self.chrome_version = chrome_version  # Chrome token version Chrome/{x.y.z.w} (needed for Brave/Edge too)
        self.brand_version = brand_version  # Brand token version (e.g., Edg/{ver}, Vivaldi/{ver}, YaBrowser/{ver}, Whale/{ver})
        self.source = source


def mac_detect_versions(browser: str) -> VersionInfo:
    """
    macOS: Read Info.plist
    """
    if plistlib is None:
        return VersionInfo(None, None, "mac:noplistlib")

    bundles = {
        "chrome": [
            "Google Chrome.app",
            "Google Chrome Beta.app",
            "Google Chrome Canary.app",
        ],
        "chromium": ["Chromium.app"],
        "edge": [
            "Microsoft Edge.app",
            "Microsoft Edge Beta.app",
            "Microsoft Edge Canary.app",
        ],
        "brave": ["Brave Browser.app"],
        "vivaldi": ["Vivaldi.app"],
        "whale": ["Naver Whale.app"],
        "yandex": [
            "Yandex.app",
            "Yandex Browser.app",
        ],  # Cover possible alternative app names
    }

    # System-level and user-level application directories
    roots = [Path("/Applications"), Path.home() / "Applications"]

    for appname in bundles.get(browser, []):
        for root in roots:
            plist_path = root / appname / "Contents" / "Info.plist"
            if plist_path.exists():
                try:
                    with plist_path.open("rb") as f:
                        info = plistlib.load(f)
                    ver = info.get("CFBundleShortVersionString") or info.get(
                        "KSVersion"
                    )  # Some distributions use this key
                    if not ver:
                        continue
                    # Chrome token version is usually the same as brand version; Edge/Vivaldi/Yandex/Whale need brand token
                    chrome_ver = ver
                    brand_ver = (
                        ver
                        if browser in {"edge", "vivaldi", "yandex", "whale"}
                        else None
                    )
                    return VersionInfo(chrome_ver, brand_ver, f"mac:plist:{appname}")
                except Exception as e:
                    logger.debug(f"[UA] plist read failed: {plist_path} err={e}")

    return VersionInfo(None, None, "mac:plist:none")


def win_detect_versions(browser: str) -> VersionInfo:
    """
    Windows: Registry & Executable Files
    """
    if winreg is None:
        return VersionInfo(None, None, "win:no-winreg")

    # BLBeacon\version is a common location for Chrome/Edge; other Chromium-based browsers may not have this key
    reg_keys = []
    if browser in {"chrome", "chromium"}:
        reg_keys.append(
            (winreg.HKEY_CURRENT_USER, r"Software\Google\Chrome\BLBeacon", "version")
        )
    if browser == "edge":
        reg_keys.append(
            (winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Edge\BLBeacon", "version")
        )

    for hive, key, valname in reg_keys:
        try:
            with winreg.OpenKey(hive, key) as k:
                ver, _ = winreg.QueryValueEx(k, valname)
            if ver:
                chrome_ver = ver  # Edge UA also needs Chrome/{ver}
                brand_ver = ver if browser == "edge" else None
                return VersionInfo(chrome_ver, brand_ver, f"win:registry:{key}")
        except OSError:
            pass

    return VersionInfo(None, None, "win:registry:none")


def candidate_binaries(browser: str) -> list[str]:
    """
    List of executable filenames for different distributions (ordered by priority).
    """
    if browser == "chrome":
        return [
            "google-chrome",
            "google-chrome-stable",
            "chrome",
            "chrome.exe",
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        ]
    if browser == "chromium":
        return [
            "chromium",
            "chromium-browser",
            "chromium.exe",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
        ]
    if browser == "edge":
        return [
            "microsoft-edge",
            "microsoft-edge-stable",
            "msedge",
            "msedge.exe",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        ]
    if browser == "brave":
        return [
            "brave",
            "brave-browser",
            "brave.exe",
            "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
        ]
    if browser == "vivaldi":
        return [
            "vivaldi",
            "vivaldi.exe",
            "/Applications/Vivaldi.app/Contents/MacOS/Vivaldi",
        ]
    if browser == "whale":
        return [
            "whale",
            "whale.exe",
            "/Applications/Naver Whale.app/Contents/MacOS/Naver Whale",
        ]
    if browser == "yandex":
        return [
            "yandex-browser",
            "yandex",
            "yandex.exe",
            "/Applications/Yandex.app/Contents/MacOS/Yandex",
        ]
    # Default fallback to Chrome
    return ["google-chrome", "google-chrome-stable", "chrome", "chrome.exe"]


def parse_version_from_output(text: str) -> str | None:
    """
    Parse 'Google Chrome 139.0.0.0' or '139.0.0.0' to extract version number.
    """
    import re

    m = re.search(r"\b(\d+\.\d+\.\d+\.\d+)\b", text)
    return m.group(1) if m else None


def generic_binary_versions(browser: str) -> VersionInfo:
    """
    Linux/portable: call binaries with --product-version/--version.
    Probe a candidate executable list and return the first successfully parsed version.
    """
    candidates = candidate_binaries(browser)
    for exe in candidates:
        for arg in ("--product-version", "--version"):
            try:
                out = subprocess.check_output(
                    [exe, arg], stderr=subprocess.STDOUT, timeout=3
                )
                text = out.decode("utf-8", "ignore").strip()
                ver = parse_version_from_output(text)
                if ver:
                    chrome_ver = ver
                    brand_ver = (
                        ver
                        if browser in {"edge", "vivaldi", "yandex", "whale"}
                        else None
                    )
                    return VersionInfo(chrome_ver, brand_ver, f"bin:{exe} {arg}")
            except Exception:
                continue
    return VersionInfo(None, None, "bin:none")


@lru_cache()
def detect_chromium_versions(browser: str) -> VersionInfo:
    """
    Return the Chrome token version and the optional brand token version.
    Order:
      macOS:  Info.plist -> binary --product-version/--version
      Windows: Registry -> binary --version
      Linux:  binary --product-version/--version
    """
    osfam = current_os_family()
    browser = browser.lower()

    # 1) macOS: read app bundle
    if osfam == "mac":
        vi = mac_detect_versions(browser)
        if vi.chrome_version:
            return vi
        # Fallback to executable --version
        vi = generic_binary_versions(browser)
        if vi.chrome_version:
            return vi
        return VersionInfo(None, None, "mac:none")

    # 2) Windows: try registry, then --version
    if osfam == "windows":
        vi = win_detect_versions(browser)
        if vi.chrome_version:
            return vi
        vi = generic_binary_versions(browser)
        if vi.chrome_version:
            return vi
        return VersionInfo(None, None, "win:none")

    # 3) Linux/BSD: run executable directly
    vi = generic_binary_versions(browser)
    if vi.chrome_version:
        return vi
    return VersionInfo(None, None, "nix:none")


def build_standard_ua(
    os_family: str, browser: str, chrome_ver: str, brand_ver: str | None = None
) -> str:
    """
    Compose a modern Chromium UA string:
      - Windows: 'Windows NT 10.0; Win64; x64'
      - macOS:   'Macintosh; Intel Mac OS X 10_15_7'
      - Linux:   'X11; Linux x86_64'
    Append brand tokens when applicable (Edg/Vivaldi/YaBrowser/Whale).
    Brave/Chromium use the same UA as Chrome (brand via UA-CH only).
    """
    if os_family == "windows":
        platform_token = "Windows NT 10.0; Win64; x64"
    elif os_family == "mac":
        platform_token = "Macintosh; Intel Mac OS X 10_15_7"
    else:
        platform_token = "X11; Linux x86_64"

    base = (
        f"Mozilla/5.0 ({platform_token}) "
        f"AppleWebKit/537.36 (KHTML, like Gecko) "
        f"Chrome/{chrome_ver} Safari/537.36"
    )

    # Append brand token (ordering close to real UA)
    browser = browser.lower()
    if browser == "edge" and brand_ver:
        return f"{base} Edg/{brand_ver}"
    if browser == "vivaldi" and brand_ver:
        return f"{base} Vivaldi/{brand_ver}"
    if browser == "yandex" and brand_ver:
        return f"{base} YaBrowser/{brand_ver}"
    if browser == "whale" and brand_ver:
        return f"{base} Whale/{brand_ver}"
    # brave/chromium behave the same as chrome here (brand via UA-CH)
    return base


def _try_read_profile_preferences_user_agent(
    browser: str, profile: str | None
) -> str | None:
    # Only read explicitly specified profile; avoid scanning all profiles to reduce overhead/noise
    try:
        base = _get_chromium_profile_path(browser, profile) if profile else None
        if not base:
            return None
        pref_path = os.path.join(base, "Preferences")
        if not os.path.exists(pref_path):
            return None
        with open(pref_path, encoding="utf-8") as f:
            prefs = json.load(f)
        return prefs.get("user_agent")  # Most recent versions do not store this
    except Exception as e:
        logger.debug(f"[UA] read Preferences failed: {e}")
        return None


def _looks_like_valid_chromium_ua(ua: str) -> bool:
    # Minimal validation to avoid treating odd overrides as UA
    if not ua or "Chrome/" not in ua or "AppleWebKit/537.36" not in ua:
        return False
    return bool(re.search(r"Chrome/\d+\.\d+\.\d+\.\d+", ua))


def _get_chromium_ua(
    browser: str, profile: str | None = None, allow_profile_ua: bool = False
) -> str:
    """
    Cross-platform UA builder for Chromium-family browsers.

    Strategy:
      1) Try reading installed browser version (platform-specific).
      2) Build a standard UA string from version (no profile scraping).
      3) Fallback to default UA if everything fails.
    Note:
      - `profile` is only used for logging/potential consistency needs; implementation does not depend on profile content.
    """
    browser = (browser or "chrome").lower()
    logger.info(
        f"[UA] chromium UserAgent resolving for browser={browser}, profile={profile}"
    )

    # 1) Get browser version (prefer bundle/registry; fallback to executable --version)
    ver_info = detect_chromium_versions(browser)

    if ver_info.chrome_version:
        ua = build_standard_ua(
            os_family=current_os_family(),
            browser=browser,
            chrome_ver=ver_info.chrome_version,
            brand_ver=ver_info.brand_version,  # Brand token for Edge/Vivaldi/Yandex/Whale
        )
        logger.info(f"[UA] Version-derived UserAgent: {ua}")
        return ua

    # 2) Optional: read user_agent from Preferences (compatibility mode, off by default)
    if allow_profile_ua:
        ua = _try_read_profile_preferences_user_agent(browser, profile)
        if ua and _looks_like_valid_chromium_ua(ua):
            logger.info("[UA] Using profile Preferences user_agent (compat mode)")
            return ua
        else:
            logger.info("[UA] Profile user_agent not present or invalid; ignoring")

    # 3) Fallback (default UA)
    ua = _default_user_agent()
    logger.warning("[UA] Falling back to default UserAgent")
    return ua


def _get_chromium_profile_path(browser, profile):
    """
    Get the profile path of the chromium browser.
    """
    system = platform.system().lower()
    if profile:
        return os.path.expanduser(profile)

    if system == "windows":
        base = os.environ.get("LOCALAPPDATA")
        return f"{base}\\Google\\Chrome\\User Data\\Default"
    elif system == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
        return f"{base}/Google/Chrome/Default"
    else:
        base = os.path.expanduser("~/.config")
        return f"{base}/google-chrome/Default"


def get_env_path(key, *suffix):
    """
    Get the path of the environment variable.
    """
    root = os.environ.get(key)
    if not root:
        logger.error(f"Environment variable '{key}' is not set.")
        raise EnvironmentError(f"Environment variable '{key}' is not set.")
    logger.info(f"Environment variable '{key}' is set to {root}")
    return os.path.join(root, *suffix)


def _find_all_chromium_profiles(browser: str):
    """
    Find and sort all profile folders containing Preferences by modification time.
    For Chrome

    Returns:
        A list of profile paths.
        Sort profiles from newest to oldest by Preferences modification time
    """
    system = platform.system().lower()
    if system == "windows":
        base = get_env_path("LOCALAPPDATA", "Google", "Chrome", "User Data")
    elif system == "darwin":
        base = os.path.expanduser("~/Library/Application Support/Google/Chrome")
    else:
        base = os.path.expanduser("~/.config/google-chrome")

    if not os.path.exists(base):
        return []

    candidates = []
    for name in os.listdir(base):
        if name == "Default" or name.startswith("Profile"):
            full_path = os.path.join(base, name)
            pref_path = os.path.join(full_path, "Preferences")
            if os.path.isfile(pref_path):
                mtime = os.path.getmtime(pref_path)
                candidates.append((mtime, full_path))

    # Sort profiles from newest to oldest by Preferences modification time
    sorted_profiles = [path for _, path in sorted(candidates, key=lambda x: -x[0])]
    logger.info(f"[UA] Sorted profiles: {sorted_profiles}")
    return sorted_profiles


def _get_chromium_base_dir(browser: str) -> str:
    system = platform.system().lower()
    if browser == "edge":
        if system == "windows":
            return get_env_path("LOCALAPPDATA", "Microsoft", "Edge", "User Data")
        elif system == "darwin":
            return os.path.expanduser("~/Library/Application Support/Microsoft Edge")
        else:
            return os.path.expanduser("~/.config/microsoft-edge")
    # Default to Google Chrome layout for other Chromium variants
    if system == "windows":
        return get_env_path("LOCALAPPDATA", "Google", "Chrome", "User Data")
    elif system == "darwin":
        return os.path.expanduser("~/Library/Application Support/Google/Chrome")
    else:
        return os.path.expanduser("~/.config/google-chrome")


def _get_chromium_version(browser: str) -> str | None:
    """
    Read browser version from Chromium's Local State file, trying common keys.
    """
    try:
        base = _get_chromium_base_dir(browser)
        local_state = os.path.join(base, "Local State")
        if not os.path.isfile(local_state):
            return None
        with open(local_state, encoding="utf-8") as f:
            data = json.load(f)
        browser_info = data.get("browser", {}) if isinstance(data, dict) else {}
        # Try several plausible keys across platforms/versions
        for key in (
            "last_version",
            "last_known_chrome_version",
            "last_known_google_chrome_version",
            "last_chrome_version",
        ):
            ver = browser_info.get(key)
            if isinstance(ver, str) and ver.strip():
                return ver.strip()
    except Exception as e:
        logger.warning(f"[UA] Failed reading Local State: {e}")
    return None


def _build_chrome_like_ua(version: str) -> str:
    """
    Build a realistic Chrome UA string for the current OS with a provided version.
    """
    system = platform.system().lower()
    if system == "windows":
        platform_token = "Windows NT 10.0; Win64; x64"
    elif system == "darwin":
        mac_ver = platform.mac_ver()[0]
        mac_token = mac_ver.replace(".", "_") if mac_ver else "13_5"
        platform_token = f"Macintosh; Intel Mac OS X {mac_token}"
    elif system == "linux":
        platform_token = "X11; Linux x86_64"
    else:
        platform_token = "X11"
    return (
        f"Mozilla/5.0 ({platform_token}) AppleWebKit/537.36 "
        f"(KHTML, like Gecko) Chrome/{version} Safari/537.36"
    )


# def _get_firefox_ua():
#     return "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0"


def _get_firefox_ua():
    """
    Attempt to infer UA from Firefox's prefs.js.
    Note: Firefox does not save UA, but can determine active version based on file existence within profile.
    """
    base_paths = [
        os.path.expanduser("~/.mozilla/firefox"),  # Linux
        os.path.expanduser("~/Library/Application Support/Firefox/Profiles"),  # macOS
        os.path.join(
            os.environ.get("APPDATA", ""), "Mozilla", "Firefox", "Profiles"
        ),  # Windows
    ]
    profiles = []

    for base in base_paths:
        if os.path.isdir(base):
            for name in os.listdir(base):
                if (
                    ".default" in name
                    or ".release" in name
                    or name.startswith("profile")
                ):
                    profiles.append(os.path.join(base, name))

    for path in profiles:
        prefs_path = os.path.join(path, "prefs.js")
        if os.path.isfile(prefs_path):
            logger.info(f"[UA] Found Firefox profile: {prefs_path}")
            return "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0"

    logger.error("[UA] No Firefox profile found, fallback to default UserAgent")
    # Here raise an error, the caller will use default user agent
    raise RuntimeError("No Firefox profile found, fallback to default UserAgent")


def _get_safari_ua():
    """
    Safari does not save UA, return common macOS UA directly.
    """
    return "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15"


def _get_ua_via_selenium(browser: str = "chrome") -> str | None:
    """
    Use selenium to extract the user agent for various browsers.
    If the browser is not supported, return None, then fallback to extract from config files.

    Args:
        browser: The name of the browser.

    Returns:
        A string of the user agent, or None if the browser is not supported.
    """
    logger.info(f"[UA] Getting user agent for browser: {browser} via selenium")
    try:
        browser = browser.lower()

        if browser == "chrome":
            options = webdriver.ChromeOptions()
            options.add_argument("--headless")
            driver = webdriver.Chrome(options=options)

        elif browser == "firefox":
            options = webdriver.FirefoxOptions()
            options.add_argument("--headless")
            driver = webdriver.Firefox(options=options)

        elif browser == "edge":
            options = webdriver.EdgeOptions()
            options.add_argument("--headless")
            driver = webdriver.Edge(options=options)

        elif browser == "safari":
            # Safari does not support headless mode as of Safari 16.x
            driver = webdriver.Safari()

        else:
            logger.error(f"Unsupported browser for selenium: {browser}")
            return None

        ua = driver.execute_script("return navigator.userAgent;")
        driver.quit()
        return ua

    except Exception as e:
        logger.error(
            f"[UA] Selenium failed for {browser}: {e}, fallback to default UserAgent"
        )
        return None


def _default_user_agent() -> str:
    """
    Generate a more realistic default User-Agent based on the current operating system.
    """
    system = platform.system().lower()
    # arch = platform.machine().lower()

    if system == "windows":
        platform_token = "Windows NT 10.0; Win64; x64"
    elif system == "darwin":
        mac_ver = platform.mac_ver()[0]
        mac_token = mac_ver.replace(".", "_") if mac_ver else "13_5"
        platform_token = f"Macintosh; Intel Mac OS X {mac_token}"
    elif system == "linux":
        platform_token = "X11; Linux x86_64"
    else:
        platform_token = "X11"

    default_ua = (
        f"Mozilla/5.0 ({platform_token}) AppleWebKit/537.36 "
        f"(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
    )
    logger.info(f"[UA] Default UserAgent: {default_ua}")
    return default_ua
