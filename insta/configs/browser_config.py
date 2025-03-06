from dataclasses import dataclass, asdict
from typing import Tuple


@dataclass
class BrowserConfig:

    playwright_url: str = "http://localhost:{port}"
    playwright_port: int = 3000
    
    screenshot: bool = True

    restrict_viewport: Tuple[float, float, float, float] = None
    require_visible: bool = True
    require_frontmost: bool = True
    
    remove_pii: bool = False
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
    remove_pii = False,
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
    **browser_kwargs
) -> BrowserConfig:
    
    default_browser_kwargs = asdict(DEFAULT_BROWSER_CONFIG)
    default_browser_kwargs.update(browser_kwargs)
    
    return BrowserConfig(
        **default_browser_kwargs
    )
