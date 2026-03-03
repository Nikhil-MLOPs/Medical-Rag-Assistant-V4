import pytest
from src.rag.guardrails import Guardrails


def test_guardrails_blocks():

    g = Guardrails()

    with pytest.raises(ValueError):
        g.validate("I want suicide help")