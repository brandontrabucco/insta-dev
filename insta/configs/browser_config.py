from dataclasses import dataclass
from typing import Tuple


@dataclass
class BrowserConfig:

    playwright_url: str = "http://localhost:{port}"
    playwright_port: int = 3000
    
    screenshot: bool = True

    restrict_viewport: Tuple[float, float, float, float] = None
    require_visible: bool = True
    require_frontmost: bool = True

    headless: bool = True
    stealthy: bool = True
    proxy: dict = None

    screen_width: int = 1920
    screen_height: int = 1080

    catch_errors: bool = True
    log_errors: bool = True
    max_errors: int = 5

    delays: dict = None


DEFAULT_BROWSER_CONFIG = BrowserConfig(
    restrict_viewport = (0, 0, 1920, 1080),
    require_visible = True,
    require_frontmost = True,
    headless = True,
    stealthy = True,
    proxy = None,
    screen_width = 1920,
    screen_height = 1080,
    catch_errors = True,
    log_errors = True,
    max_errors = 5,
    delays = {
        "observation": 0.5,
    }
)


def get_browser_config(
    **env_config_kwargs
) -> BrowserConfig:
    
    return BrowserConfig(
        playwright_url = env_config_kwargs.get(
            "playwright_url",
            DEFAULT_BROWSER_CONFIG.playwright_url
        ),
        playwright_port = env_config_kwargs.get(
            "playwright_port",
            DEFAULT_BROWSER_CONFIG.playwright_port
        ),
        screenshot = env_config_kwargs.get(
            "screenshot",
            DEFAULT_BROWSER_CONFIG.screenshot
        ),
        restrict_viewport = env_config_kwargs.get(
            "restrict_viewport",
            DEFAULT_BROWSER_CONFIG.restrict_viewport
        ),
        require_visible = env_config_kwargs.get(
            "require_visible",
            DEFAULT_BROWSER_CONFIG.require_visible
        ),
        require_frontmost = env_config_kwargs.get(
            "require_frontmost",
            DEFAULT_BROWSER_CONFIG.require_frontmost
        ),
        headless = env_config_kwargs.get(
            "headless",
            DEFAULT_BROWSER_CONFIG.headless
        ),
        stealthy = env_config_kwargs.get(
            "stealthy",
            DEFAULT_BROWSER_CONFIG.stealthy
        ),
        proxy = env_config_kwargs.get(
            "proxy",
            DEFAULT_BROWSER_CONFIG.proxy
        ),
        screen_width = env_config_kwargs.get(
            "screen_width",
            DEFAULT_BROWSER_CONFIG.screen_width
        ),
        screen_height = env_config_kwargs.get(
            "screen_height",
            DEFAULT_BROWSER_CONFIG.screen_height
        ),
        catch_errors = env_config_kwargs.get(
            "catch_errors",
            DEFAULT_BROWSER_CONFIG.catch_errors
        ),
        log_errors = env_config_kwargs.get(
            "log_errors",
            DEFAULT_BROWSER_CONFIG.log_errors
        ),
        max_errors = env_config_kwargs.get(
            "max_errors",
            DEFAULT_BROWSER_CONFIG.max_errors
        ),
        delays = env_config_kwargs.get(
            "delays",
            DEFAULT_BROWSER_CONFIG.delays
        )
    )
