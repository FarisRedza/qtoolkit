import qkit
import pytest

@pytest.mark.parametrize(
    ('correct', 'incorrect', 'expected'),
    [
        (950, 50, 0.05),
        (1000, 0, 0.0),
        (0, 1000, 1.0),
        (50, 50, 0.5),
    ],
)
def test_qber(correct, incorrect, expected):
    result = qkit.qber(correct, incorrect)

    assert result == pytest.approx(expected)

def test_qz_correlated():
    result = qkit.qz(
        c_hh=450,
        c_hv=25,
        c_vh=25,
        c_vv=500,
    )

    assert result == pytest.approx(0.05)

def test_qz_anticorrelated():
    result = qkit.qz(
        c_hh=25,
        c_hv=450,
        c_vh=500,
        c_vv=25,
        correlated=False,
    )

    assert result == pytest.approx(0.05)