from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from typesafe_sdk import SystemOneResponse

from tests.helpers import make_raw_response, make_response


@pytest.fixture
def raw_response() -> Callable[..., dict[str, Any]]:
    return make_raw_response


@pytest.fixture
def response_factory() -> Callable[..., SystemOneResponse]:
    return make_response
