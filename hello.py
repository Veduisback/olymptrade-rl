import asyncio
from playwright.async_api import async_playwright


CDP_URL = "http://127.0.0.1:9222"
PLATFORM_URL = "https://olymptrade.com/platform"


async def main():
    async with async_playwright() as p:
        print("[BROWSER] Connecting to Chrome...")

        browser = await p.chromium.connect_over_cdp(CDP_URL)

        if not browser.contexts:
            raise RuntimeError("No Chrome context found.")

        context = browser.contexts[0]

        # Find existing OlympTrade platform tab
        page = None

        for existing_page in context.pages:
            print(f"[TAB] {existing_page.url}")

            if "olymptrade.com/platform" in existing_page.url:
                page = existing_page
                break

        if page is None:
            raise RuntimeError(
                "No existing OlympTrade platform tab found.\n"
                "Open https://olymptrade.com/platform first."
            )

        await page.bring_to_front()

        print()
        print("=" * 70)
        print("BUTTON CLICK INSPECTOR")
        print("=" * 70)
        print(f"Using: {page.url}")
        print()
        print("Now manually click ANY button on the website.")
        print("The script will print its details.")
        print()
        print("Press Ctrl+C to stop.")
        print("=" * 70)

        # Install a click listener in the page.
        await page.expose_function(
            "report_python_click",
            lambda data: print_click(data)
        )

        await page.evaluate(
            """
            () => {

                // Avoid installing the listener twice.
                if (window.__python_click_inspector_installed) {
                    return;
                }

                window.__python_click_inspector_installed = true;

                document.addEventListener(
                    "click",
                    (event) => {

                        let element = event.target;

                        // Find the nearest button/input/svg/control.
                        const control = element.closest(
                            "button, input, [role='button'], svg"
                        );

                        if (!control) {
                            return;
                        }

                        const rect = control.getBoundingClientRect();

                        const data = {
                            tag: control.tagName,

                            id: control.id || "",

                            test: control.getAttribute(
                                "data-test"
                            ) || "",

                            anchor: control.getAttribute(
                                "data-anchor"
                            ) || "",

                            role: control.getAttribute(
                                "role"
                            ) || "",

                            ariaLabel: control.getAttribute(
                                "aria-label"
                            ) || "",

                            title: control.getAttribute(
                                "title"
                            ) || "",

                            text: (
                                control.innerText ||
                                control.value ||
                                ""
                            ).trim(),

                            classes: control.className
                                ? String(control.className)
                                : "",

                            x: rect.x,
                            y: rect.y,
                            width: rect.width,
                            height: rect.height,

                            outerHTML: control.outerHTML
                        };

                        window.report_python_click(data);
                    },

                    true
                );
            }
            """
        )

        print("[READY] Click inspector is active.")

        # Keep the script alive.
        while True:
            await asyncio.sleep(1)


def print_click(data):
    print()
    print()
    print("=" * 70)
    print("CLICK DETECTED")
    print("=" * 70)

    print(f"TAG        : {data['tag']}")
    print(f"ID         : {data['id']}")
    print(f"DATA-TEST  : {data['test']}")
    print(f"DATA-ANCHOR: {data['anchor']}")
    print(f"ROLE       : {data['role']}")
    print(f"ARIA LABEL : {data['ariaLabel']}")
    print(f"TITLE      : {data['title']}")
    print(f"TEXT       : {data['text']}")
    print(f"CLASSES    : {data['classes']}")

    print()
    print(
        f"POSITION   : "
        f"x={data['x']:.1f}, "
        f"y={data['y']:.1f}, "
        f"w={data['width']:.1f}, "
        f"h={data['height']:.1f}"
    )

    print()
    print("OUTER HTML:")
    print("-" * 70)
    print(data["outerHTML"])
    print("-" * 70)
    print()


if __name__ == "__main__":
    asyncio.run(main())