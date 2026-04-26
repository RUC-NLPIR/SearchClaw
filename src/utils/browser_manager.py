"""
Browser lifecycle manager — shared singleton for Playwright browser instances.

Manages browser launch, connection, page pooling, and cleanup.
Two browser contexts are supported:

1. **Search context**: Headless Chromium for Google/DDG search — no user
   profile, stealth scripts injected to avoid bot detection.
2. **Fetch context**: CDP connection to user's real Chrome OR a persistent
   Playwright context with user_data_dir — inherits cookies, extensions,
   and login state for auth-walled pages.

Inspired by MediaCrawler's CDP-first strategy and browser-use's
`Browser.from_system_chrome()` pattern.

This module is only imported when `browser.enabled=true` in settings.yaml
AND playwright is installed. All imports from `playwright` are done inside
methods so the module can be safely imported even if playwright is absent
(it will fail at runtime when a method is actually called).
"""

from __future__ import annotations

import asyncio
import logging
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from playwright.async_api import (
        Browser,
        BrowserContext,
        Page,
        Playwright,
    )

logger = logging.getLogger(__name__)

# Inline stealth JavaScript — hides common Playwright/automation markers.
# Based on puppeteer-extra-plugin-stealth techniques.  A lightweight
# subset is sufficient for search engines; full stealth is only needed
# for aggressive anti-bot platforms.
_STEALTH_JS = """
// Hide webdriver flag
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });

// Chrome runtime stub
if (!window.chrome) { window.chrome = { runtime: {} }; }

// Override permissions query
const originalQuery = window.navigator.permissions.query;
window.navigator.permissions.query = (parameters) =>
    parameters.name === 'notifications'
        ? Promise.resolve({ state: Notification.permission })
        : originalQuery(parameters);

// Hide automation-related properties
Object.defineProperty(navigator, 'plugins', {
    get: () => [1, 2, 3, 4, 5],
});
Object.defineProperty(navigator, 'languages', {
    get: () => ['en-US', 'en'],
});
"""

# Extended stealth JS for fetch context — includes additional anti-detection
# for aggressive sites like WeChat/mp.weixin.qq.com that perform deeper
# fingerprint checks.
_FETCH_STEALTH_JS = """
// ── Core: hide webdriver flag ──
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });

// ── Chrome runtime + app stubs ──
if (!window.chrome) { window.chrome = {}; }
if (!window.chrome.runtime) { window.chrome.runtime = {}; }
if (!window.chrome.app) {
    window.chrome.app = {
        isInstalled: false,
        InstallState: { DISABLED: 'disabled', INSTALLED: 'installed', NOT_INSTALLED: 'not_installed' },
        RunningState: { CANNOT_RUN: 'cannot_run', READY_TO_RUN: 'ready_to_run', RUNNING: 'running' },
    };
}
if (!window.chrome.csi) {
    window.chrome.csi = function() {
        return {
            onloadT: Date.now(),
            startE: Date.now(),
            pageT: Math.random() * 500 + 300,
            tran: 15,
        };
    };
}
if (!window.chrome.loadTimes) {
    window.chrome.loadTimes = function() {
        return {
            commitLoadTime: Date.now() / 1000,
            connectionInfo: 'h2',
            finishDocumentLoadTime: Date.now() / 1000,
            finishLoadTime: Date.now() / 1000,
            firstPaintAfterLoadTime: 0,
            firstPaintTime: Date.now() / 1000,
            navigationType: 'Other',
            npnNegotiatedProtocol: 'h2',
            requestTime: Date.now() / 1000 - 0.16,
            startLoadTime: Date.now() / 1000,
            wasAlternateProtocolAvailable: false,
            wasFetchedViaSpdy: true,
            wasNpnNegotiated: true,
        };
    };
}

// ── Override permissions query ──
const originalQuery = window.navigator.permissions.query;
window.navigator.permissions.query = (parameters) =>
    parameters.name === 'notifications'
        ? Promise.resolve({ state: Notification.permission })
        : originalQuery(parameters);

// ── Navigator properties ──
Object.defineProperty(navigator, 'plugins', {
    get: () => {
        const plugins = [
            { name: 'Chrome PDF Plugin', filename: 'internal-pdf-viewer', description: 'Portable Document Format' },
            { name: 'Chrome PDF Viewer', filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai', description: '' },
            { name: 'Native Client', filename: 'internal-nacl-plugin', description: '' },
        ];
        plugins.length = 3;
        return plugins;
    },
});
Object.defineProperty(navigator, 'languages', {
    get: () => ['zh-CN', 'zh', 'en-US', 'en'],
});
Object.defineProperty(navigator, 'language', {
    get: () => 'zh-CN',
});
Object.defineProperty(navigator, 'maxTouchPoints', {
    get: () => 0,
});
Object.defineProperty(navigator, 'hardwareConcurrency', {
    get: () => 8,
});
Object.defineProperty(navigator, 'deviceMemory', {
    get: () => 8,
});
Object.defineProperty(navigator, 'platform', {
    get: () => 'MacIntel',
});

// ── WebGL vendor/renderer spoofing ──
const getParameterProxyHandler = {
    apply: function(target, thisArg, args) {
        const param = args[0];
        const gl = thisArg;
        // UNMASKED_VENDOR_WEBGL
        if (param === 0x9245) return 'Google Inc. (Apple)';
        // UNMASKED_RENDERER_WEBGL
        if (param === 0x9246) return 'ANGLE (Apple, Apple M1 Pro, OpenGL 4.1)';
        return Reflect.apply(target, thisArg, args);
    }
};
try {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl') || canvas.getContext('webgl2');
    if (gl) {
        const ext = gl.getExtension('WEBGL_debug_renderer_info');
        if (ext) {
            const origGetParam = gl.__proto__.getParameter;
            gl.__proto__.getParameter = new Proxy(origGetParam, getParameterProxyHandler);
        }
    }
} catch(e) {}

// ── Canvas fingerprint noise ──
const originalToDataURL = HTMLCanvasElement.prototype.toDataURL;
HTMLCanvasElement.prototype.toDataURL = function(type) {
    if (this.width === 0 || this.height === 0) {
        return originalToDataURL.apply(this, arguments);
    }
    const ctx = this.getContext('2d');
    if (ctx) {
        const imageData = ctx.getImageData(0, 0, this.width, this.height);
        for (let i = 0; i < imageData.data.length; i += 4) {
            // Inject tiny noise into alpha channel (invisible to human eye)
            imageData.data[i + 3] = imageData.data[i + 3] > 0
                ? Math.max(1, imageData.data[i + 3] + (Math.random() > 0.5 ? 1 : -1))
                : 0;
        }
        ctx.putImageData(imageData, 0, 0);
    }
    return originalToDataURL.apply(this, arguments);
};

// ── Connection type ──
if (navigator.connection === undefined) {
    Object.defineProperty(navigator, 'connection', {
        get: () => ({
            effectiveType: '4g',
            rtt: 50,
            downlink: 10,
            saveData: false,
        }),
    });
}

// ── Window dimensions consistency ──
if (window.outerWidth === 0) {
    Object.defineProperty(window, 'outerWidth', { get: () => window.innerWidth });
    Object.defineProperty(window, 'outerHeight', { get: () => window.innerHeight + 85 });
}

// ── Notification constructor ──
if (typeof Notification !== 'undefined') {
    Object.defineProperty(Notification, 'permission', { get: () => 'default' });
}
"""


@dataclass
class BrowserConfig:
    """Configuration for the browser manager."""
    # Connection mode: "cdp" connects to user's real Chrome,
    # "playwright" launches managed Chromium.
    mode: str = "playwright"
    # CDP debug port (only used when mode is "cdp").
    cdp_port: int = 9222
    # Path to Chrome/Edge binary (auto-detected if empty).
    chrome_path: str = ""
    # Run browser without visible window.
    headless: bool = True
    # User data directory for persistent login state.
    # Set to your Chrome profile path to access auth-walled sites.
    # Leave empty for a fresh profile each session.
    user_data_dir: str = ""
    # Search engine for browser search.
    search_engine: str = "google"
    # Whether browser is enabled for search and/or fetch.
    use_for_search: bool = True
    use_for_fetch: bool = True


class BrowserManager:
    """
    Shared browser instance manager — singleton, lazy-init.

    The browser is NOT launched at construction time.  The first call
    to ``get_search_page()`` or ``get_fetch_page()`` triggers lazy
    initialization.  This keeps startup fast when browser is configured
    but not yet needed.

    Thread-safety: all public methods are coroutines and must be called
    from the same event loop.  The singleton is per-process.
    """

    _instance: BrowserManager | None = None

    def __init__(self, config: BrowserConfig):
        self._config = config

        # Playwright instances (lazy-initialized)
        self._playwright: Playwright | None = None
        self._search_browser: Browser | None = None
        self._search_context: BrowserContext | None = None
        self._fetch_context: BrowserContext | None = None
        self._fetch_browser: Browser | None = None  # Only for CDP mode

        # Page pools — reuse pages instead of creating/closing per request
        self._search_pages: list[Page] = []
        self._fetch_pages: list[Page] = []

        # Initialization locks (prevent double-init under concurrency)
        self._search_init_lock = asyncio.Lock()
        self._fetch_init_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls, config: BrowserConfig | None = None) -> BrowserManager:
        """
        Get or create the singleton BrowserManager.

        Call with config on first use (at startup); subsequent calls
        can omit config and will return the existing instance.
        """
        if cls._instance is None:
            if config is None:
                raise RuntimeError(
                    "BrowserManager not initialized — call get_instance(config) first"
                )
            cls._instance = cls(config)
            logger.info(
                f"BrowserManager created: mode={config.mode}, "
                f"headless={config.headless}, "
                f"search={config.use_for_search}, fetch={config.use_for_fetch}"
            )
        return cls._instance

    @classmethod
    async def shutdown_instance(cls) -> None:
        """Shutdown the singleton instance (if any). Called on app exit."""
        if cls._instance is not None:
            await cls._instance.shutdown()
            cls._instance = None

    @property
    def is_cdp_mode(self) -> bool:
        """Whether the browser is configured to connect via CDP."""
        return self._config.mode == "cdp"

    # ------------------------------------------------------------------
    # Search context (headless, no user profile)
    # ------------------------------------------------------------------

    async def _ensure_search_context(self) -> None:
        """Lazy-init the search browser context."""
        if self._search_context is not None:
            return

        async with self._search_init_lock:
            # Double-check after acquiring lock
            if self._search_context is not None:
                return

            pw = await self._ensure_playwright()
            logger.info("Launching search browser (headless Chromium)...")

            launch_args = [
                "--disable-blink-features=AutomationControlled",
                "--disable-infobars",
                "--no-first-run",
                "--no-default-browser-check",
            ]

            self._search_browser = await pw.chromium.launch(
                headless=self._config.headless,
                args=launch_args,
            )
            self._search_context = await self._search_browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/131.0.0.0 Safari/537.36"
                ),
                locale="en-US",
            )
            # Inject stealth script to all pages in this context
            await self._search_context.add_init_script(script=_STEALTH_JS)
            logger.info("Search browser context ready")

    async def get_search_page(self) -> Page:
        """
        Get a page for search — headless, no user profile.

        Returns a reusable page from the pool, or creates a new one.
        Caller MUST call ``release_page()`` when done.
        """
        await self._ensure_search_context()
        assert self._search_context is not None

        if self._search_pages:
            page = self._search_pages.pop()
            # Navigate to blank to clear previous state
            try:
                await page.goto("about:blank", timeout=5000)
            except Exception:
                # Page might be closed/crashed — create a new one
                page = await self._search_context.new_page()
            return page

        return await self._search_context.new_page()

    # ------------------------------------------------------------------
    # Fetch context (CDP to real Chrome or persistent context)
    # ------------------------------------------------------------------

    async def _ensure_fetch_context(self) -> None:
        """Lazy-init the fetch browser context."""
        if self._fetch_context is not None:
            return

        async with self._fetch_init_lock:
            if self._fetch_context is not None:
                return

            pw = await self._ensure_playwright()

            if self._config.mode == "cdp":
                await self._connect_cdp(pw)
            else:
                await self._launch_persistent_context(pw)

    async def _connect_cdp(self, pw: Playwright) -> None:
        """
        Connect to user's real Chrome via CDP.

        The user must launch Chrome with:
            google-chrome --remote-debugging-port=9222 --user-data-dir=/path/to/profile

        NOTE: Chrome 146+ requires --user-data-dir to be explicitly set to a
        non-default path, otherwise --remote-debugging-port is silently ignored.

        Inherits all cookies, extensions, and fingerprint from the
        real browser — virtually undetectable by anti-bot systems.
        """
        cdp_url = f"http://127.0.0.1:{self._config.cdp_port}"
        logger.info(f"Connecting to Chrome via CDP at {cdp_url}...")

        try:
            self._fetch_browser = await pw.chromium.connect_over_cdp(cdp_url)
            contexts = self._fetch_browser.contexts
            if contexts:
                self._fetch_context = contexts[0]
                logger.info(
                    f"Connected to Chrome CDP — using existing context "
                    f"({len(self._fetch_context.pages)} open pages)"
                )
            else:
                self._fetch_context = await self._fetch_browser.new_context()
                logger.info("Connected to Chrome CDP — created new context")
        except Exception as e:
            logger.warning(
                f"CDP connection failed ({e}). "
                f"Falling back to Playwright persistent context. "
                f"To use CDP, launch Chrome with: "
                f"google-chrome --remote-debugging-port={self._config.cdp_port} "
                f"--user-data-dir=/path/to/profile "
                f"(Chrome 146+ requires --user-data-dir to be set)"
            )
            await self._launch_persistent_context(pw)

    async def _launch_persistent_context(self, pw: Playwright) -> None:
        """
        Launch Playwright with a persistent user data directory.

        Preserves login state across sessions via the user_data_dir.
        Falls back to a temporary profile if no user_data_dir is configured.
        """
        user_data_dir = self._config.user_data_dir
        if not user_data_dir:
            # Use a default directory within the project
            user_data_dir = str(Path("./browser_data").resolve())
            logger.info(f"No user_data_dir configured, using {user_data_dir}")

        Path(user_data_dir).mkdir(parents=True, exist_ok=True)

        # Detect Chrome path if not configured
        chrome_path = self._config.chrome_path or detect_chrome_path()
        if chrome_path:
            logger.info(f"Using Chrome at: {chrome_path}")
        else:
            logger.info("No Chrome detected, using Playwright's bundled Chromium")

        launch_kwargs: dict = {
            "user_data_dir": user_data_dir,
            "headless": self._config.headless,
            "viewport": {"width": 1920, "height": 1080},
            "locale": "zh-CN",
            "user_agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            "args": [
                "--disable-blink-features=AutomationControlled",
                "--disable-infobars",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-dev-shm-usage",
            ],
        }
        if chrome_path:
            launch_kwargs["executable_path"] = chrome_path

        logger.info(
            f"Launching persistent browser context "
            f"(user_data_dir={user_data_dir}, headless={self._config.headless})..."
        )
        self._fetch_context = await pw.chromium.launch_persistent_context(
            **launch_kwargs
        )
        # Inject extended stealth for fetch contexts — more comprehensive
        # than _STEALTH_JS to handle aggressive anti-bot sites (WeChat, etc.)
        await self._fetch_context.add_init_script(script=_FETCH_STEALTH_JS)
        logger.info("Persistent browser context ready")

    async def get_fetch_page(self) -> Page:
        """
        Get a page for fetch — with user profile if configured.

        Returns a reusable page from the pool, or creates a new one.
        Caller MUST call ``release_page()`` when done.
        """
        await self._ensure_fetch_context()
        assert self._fetch_context is not None

        if self._fetch_pages:
            page = self._fetch_pages.pop()
            try:
                await page.goto("about:blank", timeout=5000)
            except Exception:
                page = await self._fetch_context.new_page()
            return page

        return await self._fetch_context.new_page()

    # ------------------------------------------------------------------
    # Page pool management
    # ------------------------------------------------------------------

    async def release_page(self, page: Page, context_type: str = "search") -> None:
        """
        Return a page to the pool for reuse.

        Args:
            page: The page to release.
            context_type: "search" or "fetch" — determines which pool.
        """
        try:
            # Check if page is still usable
            if page.is_closed():
                return

            # Navigate to blank to clear state
            await page.goto("about:blank", timeout=5000)

            pool = (
                self._search_pages
                if context_type == "search"
                else self._fetch_pages
            )

            # Cap pool size to avoid memory bloat
            if len(pool) < 3:
                pool.append(page)
            else:
                await page.close()
        except Exception:
            # Page is in a bad state — discard it
            try:
                await page.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _ensure_playwright(self) -> Playwright:
        """Lazy-init Playwright."""
        if self._playwright is None:
            from playwright.async_api import async_playwright

            self._playwright = await async_playwright().start()
        return self._playwright

    async def shutdown(self) -> None:
        """Close all browsers and Playwright. Called on app shutdown."""
        logger.info("Shutting down BrowserManager...")

        # Close pooled pages
        for page in self._search_pages + self._fetch_pages:
            try:
                await page.close()
            except Exception:
                pass
        self._search_pages.clear()
        self._fetch_pages.clear()

        # Close search browser
        if self._search_context:
            try:
                await self._search_context.close()
            except Exception:
                pass
            self._search_context = None

        if self._search_browser:
            try:
                await self._search_browser.close()
            except Exception:
                pass
            self._search_browser = None

        # Close fetch context/browser
        if self._fetch_context:
            try:
                await self._fetch_context.close()
            except Exception:
                pass
            self._fetch_context = None

        if self._fetch_browser:
            try:
                await self._fetch_browser.close()
            except Exception:
                pass
            self._fetch_browser = None

        # Stop Playwright
        if self._playwright:
            try:
                await self._playwright.stop()
            except Exception:
                pass
            self._playwright = None

        logger.info("BrowserManager shut down")


# ------------------------------------------------------------------
# Chrome detection utilities
# ------------------------------------------------------------------

def detect_chrome_path() -> str | None:
    """
    Auto-detect Chrome/Edge/Chromium binary on the current OS.

    Checks common installation paths on macOS, Linux, and Windows.
    Returns the full path to the binary, or None if not found.
    """
    system = platform.system()

    if system == "Darwin":
        candidates = [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        ]
    elif system == "Linux":
        candidates = [
            "google-chrome",
            "google-chrome-stable",
            "chromium",
            "chromium-browser",
            "microsoft-edge",
        ]
    elif system == "Windows":
        import os
        local_app = os.environ.get("LOCALAPPDATA", "")
        program_files = os.environ.get("PROGRAMFILES", "C:\\Program Files")
        program_files_x86 = os.environ.get(
            "PROGRAMFILES(X86)", "C:\\Program Files (x86)"
        )
        candidates = [
            f"{local_app}\\Google\\Chrome\\Application\\chrome.exe",
            f"{program_files}\\Google\\Chrome\\Application\\chrome.exe",
            f"{program_files_x86}\\Google\\Chrome\\Application\\chrome.exe",
            f"{local_app}\\Microsoft\\Edge\\Application\\msedge.exe",
            f"{program_files}\\Microsoft\\Edge\\Application\\msedge.exe",
        ]
    else:
        return None

    for candidate in candidates:
        # For Linux, check both absolute paths and PATH lookup
        if system == "Linux" and not candidate.startswith("/"):
            found = shutil.which(candidate)
            if found:
                logger.info(f"Detected Chrome at: {found}")
                return found
        else:
            if Path(candidate).exists():
                logger.info(f"Detected Chrome at: {candidate}")
                return candidate

    return None
