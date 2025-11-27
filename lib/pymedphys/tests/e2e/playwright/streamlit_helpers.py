# Copyright (C) 2024 Cancer Care Associates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper functions for Playwright E2E tests of Streamlit apps.

These helpers mirror the custom Cypress commands to provide a consistent
testing API.
"""

from playwright.sync_api import Page, expect


class StreamlitPage:
    """A wrapper around a Playwright Page with Streamlit-specific helpers.

    This class provides convenience methods for interacting with Streamlit
    apps, similar to the custom Cypress commands.
    """

    def __init__(self, page: Page, base_url: str):
        self.page = page
        self.base_url = base_url
        # Increase default timeout for Streamlit operations
        self.page.set_default_timeout(30000)

    def start(self, app: str):
        """Navigate to a specific app in the Streamlit GUI.

        Parameters
        ----------
        app : str
            The app name to navigate to (e.g., "metersetmap").
        """
        # First visit the base URL
        self.page.goto(self.base_url)

        # Then navigate to the specific app
        self.page.goto(f"{self.base_url}/?app={app}")
        self.wait_for_streamlit()

        # Hide the decoration ribbon for consistent screenshots
        self.page.evaluate(
            """
            const decoration = document.querySelector('[data-testid="stDecoration"]');
            if (decoration) {
                decoration.style.display = 'none';
            }
        """
        )

        self.wait_for_streamlit()

    def wait_for_streamlit(self, timeout: int = 120000):
        """Wait for Streamlit to finish computing.

        This waits for the status widget to disappear, indicating that
        Streamlit has finished processing.

        Parameters
        ----------
        timeout : int
            Maximum time to wait in milliseconds.
        """
        # Wait a brief moment for any status widget to appear
        self.page.wait_for_timeout(500)

        # Wait for any running status indicator to disappear
        try:
            status_widget = self.page.locator(".StatusWidget-enter-done")
            status_widget.wait_for(state="hidden", timeout=timeout)
        except Exception:
            # Status widget may not exist if no computation is happening
            pass

    def click_radio(self, title: str, item: str):
        """Select a radio button option.

        Parameters
        ----------
        title : str
            The title/label of the radio group.
        item : str
            The option to select within the radio group.
        """
        # Find the radio group by title, then find the specific option
        radio_group = self.page.locator(f":text('{title}')").locator("..").first
        option = radio_group.locator(f":text('{item}')").locator("input").first
        option.click(force=True)
        self.wait_for_streamlit()

    def click_button(self, text: str):
        """Click a button with the specified text.

        Parameters
        ----------
        text : str
            The button text to click.
        """
        self.page.locator(f".stButton button:has-text('{text}')").click()
        self.wait_for_streamlit()

    def fill_text_input(self, text: str, index: int = 0):
        """Fill a text input field.

        Parameters
        ----------
        text : str
            The text to enter.
        index : int
            The index of the text input (0-based) if there are multiple.
        """
        inputs = self.page.locator(".stTextInput input")
        inputs.nth(index).fill(text)
        inputs.nth(index).press("Enter")
        self.wait_for_streamlit()

    def select_multiselect(self, values: list[str], index: int = 0):
        """Select values in a multiselect widget.

        Parameters
        ----------
        values : list[str]
            The values to select.
        index : int
            The index of the multiselect (0-based) if there are multiple.
        """
        multiselect = self.page.locator(".stMultiSelect").nth(index)
        for value in values:
            multiselect.type(f"{value}")
            multiselect.press("Enter")
        self.wait_for_streamlit()

    def assert_text_match(self, label: str, count: int, value: str | None = None):
        """Assert that text elements match expected values.

        Parameters
        ----------
        label : str
            The label text to search for.
        count : int
            The expected number of matching elements.
        value : str, optional
            The expected value in the code element, if applicable.
        """
        elements = self.page.locator(
            f".element-container .stMarkdown p:has-text('{label}')"
        )
        expect(elements).to_have_count(count)

        if value is not None:
            code_elements = elements.locator("code")
            for i in range(code_elements.count()):
                expect(code_elements.nth(i)).to_have_text(value)

    def scroll_to_bottom(self):
        """Scroll to the bottom of the main content area."""
        self.page.locator(".main").evaluate("el => el.scrollTo(0, el.scrollHeight)")

    def take_screenshot(self, name: str):
        """Take a screenshot of the current page.

        Parameters
        ----------
        name : str
            The name for the screenshot file.
        """
        self.scroll_to_bottom()
        self.wait_for_streamlit()
        self.page.screenshot(path=f"{name}.png")

    def get_index_page(self):
        """Navigate to the index page."""
        self.page.goto(self.base_url)
        self.wait_for_streamlit()

    def click_app_button(self, app_title: str):
        """Click an app button on the index page.

        Parameters
        ----------
        app_title : str
            The title of the app to navigate to.
        """
        self.page.locator(f"button:has-text('{app_title}')").click()
        self.wait_for_streamlit()

    def return_to_index(self):
        """Click the 'Return to Index' button in the sidebar."""
        self.page.locator("text=Return to Index").click()
        self.wait_for_streamlit()

    def filter_apps(self, filter_text: str):
        """Filter apps on the index page.

        Parameters
        ----------
        filter_text : str
            The text to filter apps by.
        """
        self.page.locator("input[type='text']").first.fill(filter_text)
        self.wait_for_streamlit()
