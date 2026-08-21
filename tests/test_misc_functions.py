import pytest

import qtoolkit

# binary_entropy

@pytest.mark.parametrize(
    ('p', 'expected'),
    [
        (0.0, 0.0),
        (0.25, 0.8112781244591328),
        (0.5, 1.0),
        (0.75, 0.8112781244591328),
        (1.0, 0.0),
    ],
)
def test_binary_entropy(
        p: float,
        expected: float,
) -> None:
    assert qtoolkit.binary_entropy(p) == pytest.approx(expected)


@pytest.mark.parametrize(
    'p',
    [
        -0.1,
        1.1,
    ],
)
def test_binary_entropy_invalid_probability(
        p: float,
) -> None:
    with pytest.raises(ValueError):
        qtoolkit.binary_entropy(p)

# fraction - dB conversions

@pytest.mark.parametrize(
    ('fraction', 'expected'),
    [
        (1.0, 0.0),
        (0.5, 3.010299956639812),
        (0.1, 10.0),
        (0.01, 20.0),
    ],
)
def test_fraction_to_db(
        fraction: float,
        expected: float,
) -> None:
    assert (
        qtoolkit.fraction_to_dB(fraction)
        == pytest.approx(expected)
    )


@pytest.mark.parametrize(
    ('db', 'expected'),
    [
        (0.0, 1.0),
        (3.0, 0.5011872336272722),
        (10.0, 0.1),
        (20.0, 0.01),
    ],
)
def test_db_to_fraction(
        db: float,
        expected: float,
) -> None:
    assert (
        qtoolkit.dB_to_fraction(db)
        == pytest.approx(expected)
    )


@pytest.mark.parametrize(
    'fraction',
    [
        1.0,
        0.9,
        0.5,
        0.1,
        0.01,
        1e-6,
    ],
)
def test_fraction_db_round_trip(
        fraction: float,
) -> None:
    db = qtoolkit.fraction_to_dB(fraction)
    result = qtoolkit.dB_to_fraction(db)

    assert result == pytest.approx(fraction)


@pytest.mark.parametrize(
    'db',
    [
        0.0,
        1.0,
        3.0,
        10.0,
        20.0,
        60.0,
    ],
)
def test_db_fraction_round_trip(
        db: float,
) -> None:
    fraction = qtoolkit.dB_to_fraction(db)
    result = qtoolkit.fraction_to_dB(fraction)

    assert result == pytest.approx(db)