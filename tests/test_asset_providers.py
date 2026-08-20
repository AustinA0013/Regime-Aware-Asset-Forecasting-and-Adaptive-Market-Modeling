"""Asset provider adjustment, schema, and cache tests."""

from __future__ import annotations

from datetime import date, datetime

import numpy as np

from market_regimes.assets.providers import (
    ASSET_COLUMNS,
    AssetRequest,
    YahooChartAssetProvider,
)


class StubYahooProvider(YahooChartAssetProvider):
    def _download_json(self, url: str) -> dict[str, object]:
        timestamps = [
            int(datetime(2024, 1, 2, 21).timestamp()),
            int(datetime(2024, 1, 3, 21).timestamp()),
        ]
        return {
            "chart": {
                "error": None,
                "result": [
                    {
                        "timestamp": timestamps,
                        "indicators": {
                            "quote": [
                                {
                                    "open": [99.0, 109.0],
                                    "high": [102.0, 112.0],
                                    "low": [98.0, 108.0],
                                    "close": [100.0, 110.0],
                                    "volume": [1_000_000, 1_200_000],
                                }
                            ],
                            "adjclose": [{"adjclose": [50.0, None]}],
                        },
                    }
                ],
            }
        }


def test_yahoo_provider_uses_adjusted_close_and_marks_fallback(tmp_path) -> None:
    provider = StubYahooProvider(tmp_path)
    result = provider.fetch(
        AssetRequest("AAA", date(2024, 1, 2), date(2024, 1, 3))
    )
    assert list(result.columns) == ASSET_COLUMNS
    assert result["price_for_returns"].tolist() == [50.0, 110.0]
    assert result["adjustment_policy"].tolist() == [
        "yahoo_adjusted_close_total_return_compatible",
        "raw_close_fallback_unadjusted",
    ]
    assert np.isfinite(result["volume"]).all()
    cached = provider.fetch(AssetRequest("AAA", date(2024, 1, 2), date(2024, 1, 3)))
    assert cached.attrs["provider_mode"] == "yahoo-chart-cache"
