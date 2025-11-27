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

"""Playwright E2E tests for PyMedPhys Streamlit apps."""

import pytest

from playwright.sync_api import Page, expect

from .streamlit_helpers import StreamlitPage


@pytest.mark.playwright
class TestStreamlitIndex:
    """Tests for the Streamlit index page."""

    def test_index_page_loads(
        self, page: Page, streamlit_server, streamlit_base_url: str
    ):
        """Test that the index page loads successfully."""
        st_page = StreamlitPage(page, streamlit_base_url)
        st_page.get_index_page()

        # The page should contain the PyMedPhys title/logo
        expect(page).to_have_title("PyMedPhys")

    def test_index_has_app_categories(
        self, page: Page, streamlit_server, streamlit_base_url: str
    ):
        """Test that the index page displays app categories."""
        st_page = StreamlitPage(page, streamlit_base_url)
        st_page.get_index_page()

        # Should have Production and/or Beta categories
        production = page.locator("text=Production")
        beta = page.locator("text=Beta")

        # At least one category should be visible
        expect(production.or_(beta).first).to_be_visible()

    def test_filter_apps(self, page: Page, streamlit_server, streamlit_base_url: str):
        """Test that the app filter works."""
        st_page = StreamlitPage(page, streamlit_base_url)
        st_page.get_index_page()

        # Find the filter input and type a filter term
        filter_input = page.locator("input[type='text']").first
        expect(filter_input).to_be_visible()

        # Filter for a known app
        filter_input.fill("meter")
        st_page.wait_for_streamlit()

        # The MetersetMap app should be visible
        metersetmap_button = page.locator("button:has-text('MetersetMap')")
        expect(metersetmap_button).to_be_visible()


@pytest.mark.playwright
class TestMetersetMapApp:
    """Tests for the MetersetMap Streamlit app."""

    def test_metersetmap_loads(
        self, page: Page, streamlit_server, streamlit_base_url: str
    ):
        """Test that the MetersetMap app loads successfully."""
        st_page = StreamlitPage(page, streamlit_base_url)
        st_page.start("metersetmap")

        # The page title should contain MetersetMap
        expect(page.locator("h1")).to_contain_text("MetersetMap")

    def test_metersetmap_has_config_mode(
        self, page: Page, streamlit_server, streamlit_base_url: str
    ):
        """Test that the MetersetMap app has a Config Mode selector."""
        st_page = StreamlitPage(page, streamlit_base_url)
        st_page.start("metersetmap")

        # Should have Config Mode radio options
        config_mode = page.locator("text=Config Mode")
        expect(config_mode).to_be_visible()

    def test_metersetmap_return_to_index(
        self, page: Page, streamlit_server, streamlit_base_url: str
    ):
        """Test navigating from MetersetMap back to index."""
        st_page = StreamlitPage(page, streamlit_base_url)
        st_page.start("metersetmap")

        # Click Return to Index button
        return_button = page.locator("text=Return to Index")
        expect(return_button).to_be_visible()
        return_button.click()
        st_page.wait_for_streamlit()

        # Should be back on the index page with categories
        production = page.locator("text=Production")
        expect(production).to_be_visible()


@pytest.mark.playwright
class TestPseudonymiseApp:
    """Tests for the Pseudonymise Streamlit app."""

    def test_pseudonymise_loads(
        self, page: Page, streamlit_server, streamlit_base_url: str
    ):
        """Test that the Pseudonymise app loads successfully."""
        st_page = StreamlitPage(page, streamlit_base_url)
        st_page.start("pseudonymise")

        # The page title should contain Pseudonymise
        expect(page.locator("h1")).to_contain_text("Pseudonymise")

    def test_pseudonymise_has_upload(
        self, page: Page, streamlit_server, streamlit_base_url: str
    ):
        """Test that the Pseudonymise app has a file upload widget."""
        st_page = StreamlitPage(page, streamlit_base_url)
        st_page.start("pseudonymise")

        # Should have a file uploader
        # Streamlit file uploaders have a specific data-testid
        uploader = page.locator("[data-testid='stFileUploader']")
        expect(uploader.first).to_be_visible()


@pytest.mark.playwright
class TestAppNavigation:
    """Tests for navigation between apps."""

    def test_navigate_via_url_parameter(
        self, page: Page, streamlit_server, streamlit_base_url: str
    ):
        """Test navigating to an app via URL query parameter."""
        st_page = StreamlitPage(page, streamlit_base_url)

        # Navigate directly to metersetmap via URL
        page.goto(f"{streamlit_base_url}/?app=metersetmap")
        st_page.wait_for_streamlit()

        expect(page.locator("h1")).to_contain_text("MetersetMap")

    def test_navigate_via_index_button(
        self, page: Page, streamlit_server, streamlit_base_url: str
    ):
        """Test navigating to an app by clicking its button on the index."""
        st_page = StreamlitPage(page, streamlit_base_url)
        st_page.get_index_page()

        # Filter to find the app more easily
        st_page.filter_apps("pseudon")

        # Click the Pseudonymise button
        page.locator("button:has-text('Pseudonymise')").click()
        st_page.wait_for_streamlit()

        expect(page.locator("h1")).to_contain_text("Pseudonymise")

    def test_invalid_app_redirects_to_index(
        self, page: Page, streamlit_server, streamlit_base_url: str
    ):
        """Test that an invalid app parameter redirects to the index."""
        st_page = StreamlitPage(page, streamlit_base_url)

        # Try to navigate to a non-existent app
        page.goto(f"{streamlit_base_url}/?app=nonexistent-app-xyz")
        st_page.wait_for_streamlit()

        # Should be on the index page
        production = page.locator("text=Production")
        expect(production).to_be_visible()
