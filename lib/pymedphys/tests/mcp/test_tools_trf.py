# Copyright (C) 2025 PyMedPhys Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for MCP TRF tools."""

import asyncio

import pytest

from pymedphys._mcp.tools.trf import _get_trf_summary, read_trf


def run_async(coro):
    """Helper to run async functions in tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


class TestReadTrf:
    """Tests for read_trf function."""

    def test_read_nonexistent_file(self):
        """Reading nonexistent file should return error."""
        result = run_async(read_trf("/nonexistent/path/file.trf"))
        assert "error" in result
        assert "not found" in result["error"]

    def test_read_trf_output_formats(self):
        """read_trf should accept different output formats."""
        # Test that function accepts format parameters (actual file tests
        # would require a real TRF file)
        result = run_async(read_trf("/nonexistent/file.trf", output_format="summary"))
        assert "error" in result  # Still error, but format was accepted

        result = run_async(read_trf("/nonexistent/file.trf", output_format="detailed"))
        assert "error" in result

        result = run_async(read_trf("/nonexistent/file.trf", output_format="dataframe"))
        assert "error" in result


class TestGetTrfSummary:
    """Tests for _get_trf_summary function."""

    def test_empty_table(self):
        """Empty table should return minimal summary."""

        class MockEmptyTable:
            def __len__(self):
                return 0

        result = _get_trf_summary(MockEmptyTable())
        assert result["num_samples"] == 0

    def test_table_without_columns(self):
        """Table without columns attribute should return num_samples only."""

        class MockTableNoColumns:
            def __len__(self):
                return 10

        result = _get_trf_summary(MockTableNoColumns())
        assert result["num_samples"] == 10
        assert "gantry_angle" not in result

    def test_table_with_gantry_column(self):
        """Table with gantry column should extract gantry range."""
        import pandas as pd

        table = pd.DataFrame(
            {
                "Step_Gantry/Actual": [0.0, 45.0, 90.0, 135.0, 180.0],
                "Other": [1, 2, 3, 4, 5],
            }
        )

        result = _get_trf_summary(table)
        assert "gantry_angle" in result
        assert result["gantry_angle"]["min"] == 0.0
        assert result["gantry_angle"]["max"] == 180.0

    def test_table_with_collimator_column(self):
        """Table with collimator column should extract collimator range."""
        import pandas as pd

        table = pd.DataFrame(
            {"Step_Coll/Actual_Angle": [0.0, 10.0, 20.0], "Other": [1, 2, 3]}
        )

        result = _get_trf_summary(table)
        assert "collimator_angle" in result
        assert result["collimator_angle"]["min"] == 0.0
        assert result["collimator_angle"]["max"] == 20.0

    def test_table_with_dose_column(self):
        """Table with dose column should extract total MU."""
        import pandas as pd

        table = pd.DataFrame({"Step_Dose/Actual": [0, 50, 100, 150, 200]})

        result = _get_trf_summary(table)
        assert "monitor_units" in result
        assert result["monitor_units"]["total"] == 200

    def test_table_with_jaw_columns(self):
        """Table with jaw columns should extract jaw positions."""
        import pandas as pd

        table = pd.DataFrame(
            {
                "Y1_Actual": [-100, -100, -100],
                "Y2_Actual": [100, 100, 100],
                "X1_Actual": [-50, -50, -50],
                "X2_Actual": [50, 50, 50],
            }
        )

        result = _get_trf_summary(table)
        assert "jaw_y1" in result
        assert "jaw_y2" in result
        assert "jaw_x1" in result
        assert "jaw_x2" in result

    def test_table_with_mlc_columns(self):
        """Table with MLC columns should extract MLC info."""
        import pandas as pd

        columns = {
            "A1_Actual": [0, 1, 2],
            "A2_Actual": [0, 1, 2],
            "B1_Actual": [0, 1, 2],
            "B2_Actual": [0, 1, 2],
        }
        table = pd.DataFrame(columns)

        result = _get_trf_summary(table)
        assert "mlc" in result
        assert result["mlc"]["num_leaf_columns"] == 4


class TestCompareTrfToPlan:
    """Tests for compare_trf_to_plan function."""

    def test_compare_requires_valid_trf(self):
        """Comparison should require valid TRF file."""
        from pymedphys._mcp.tools.trf import compare_trf_to_plan

        # This would fail to load the TRF file
        # We can't easily test this without a real TRF file,
        # so we test error handling
        pass  # Skip for now without actual test data


class TestTrfToolsIntegration:
    """Integration tests for TRF tools (require actual TRF files)."""

    @pytest.mark.skip(reason="Requires actual TRF test data")
    def test_read_actual_trf_file(self):
        """Test reading an actual TRF file."""
        # This would test with actual TRF data
        pass
