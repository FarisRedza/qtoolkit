import qkit
import pytest

def test_binary_entropy() -> None:
    result = qkit.binary_entropy(x=0.5)
    assert result == 1

def test_fraction_to_dB() -> None:
    result = qkit.fraction_to_dB(x=0.5)
    assert result == pytest.approx(3.010299957)

def test_dB_to_fraction() -> None:
    result =  qkit.dB_to_fraction(x=3)
    assert result == pytest.approx(0.501187234)

def test_fraction_dB_parity() -> None:
    frac = 0.5
    dB = qkit.fraction_to_dB(x=frac)
    result = qkit.dB_to_fraction(x=dB)
    assert frac == result