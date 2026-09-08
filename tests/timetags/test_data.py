import pathlib

import numpy as np
import pytest

import qtoolkit

# TimetagData

@pytest.fixture
def timetag_data() -> qtoolkit.timetags.TimetagData:
    return qtoolkit.timetags.TimetagData(
        timetags=np.array(
            [
                1000,
                1050,
                2000,
                2050,
                3000,
                3050,
            ],
            dtype=np.int64,
        ),
        channels=np.array(
            [
                0,
                4,
                0,
                5,
                1,
                4,
            ],
            dtype=np.int8,
        ),
    )


def test_timetag_data_length(
        timetag_data: qtoolkit.timetags.TimetagData,
) -> None:
    assert len(timetag_data) == 6


def test_timetag_data_span(
        timetag_data: qtoolkit.timetags.TimetagData,
) -> None:
    assert timetag_data.span_ps == 2050


@pytest.mark.parametrize(
    'number_of_timetags',
    [
        0,
        1,
    ],
)
def test_timetag_data_short_span_is_zero(
        number_of_timetags: int,
) -> None:
    data = qtoolkit.timetags.TimetagData(
        timetags=np.arange(
            number_of_timetags,
            dtype=np.int64,
        ),
        channels=np.zeros(
            number_of_timetags,
            dtype=np.int8,
        ),
    )

    assert data.span_ps == 0


def test_timetag_data_get_channel_timetags(
        timetag_data: qtoolkit.timetags.TimetagData,
) -> None:
    result = timetag_data.get_channel_timetags(0)

    np.testing.assert_array_equal(
        result,
        np.array(
            [1000, 2000],
            dtype=np.int64,
        ),
    )


def test_timetag_data_get_missing_channel(
        timetag_data: qtoolkit.timetags.TimetagData,
) -> None:
    result = timetag_data.get_channel_timetags(100)

    assert len(result) == 0


def test_timetag_data_count_all(
        timetag_data: qtoolkit.timetags.TimetagData,
) -> None:
    assert timetag_data.count() == 6


def test_timetag_data_count_channel(
        timetag_data: qtoolkit.timetags.TimetagData,
) -> None:
    assert timetag_data.count(0) == 2


def test_timetag_data_count_missing_channel(
        timetag_data: qtoolkit.timetags.TimetagData,
) -> None:
    assert timetag_data.count(100) == 0


def test_timetag_data_select_channels(
        timetag_data: qtoolkit.timetags.TimetagData,
) -> None:
    result = timetag_data.select_channels(
        0,
        1,
    )

    np.testing.assert_array_equal(
        result.timetags,
        np.array(
            [1000, 2000, 3000],
            dtype=np.int64,
        ),
    )

    np.testing.assert_array_equal(
        result.channels,
        np.array(
            [0, 0, 1],
            dtype=np.int8,
        ),
    )


def test_timetag_data_rejects_non_1d_timetags() -> None:
    with pytest.raises(
        ValueError,
        match='timetags must be one-dimensional',
    ):
        qtoolkit.timetags.TimetagData(
            timetags=np.array([
                [1, 2],
            ]),
            channels=np.array([
                0,
                1,
            ]),
        )


def test_timetag_data_rejects_non_1d_channels() -> None:
    with pytest.raises(
        ValueError,
        match='channels must be one-dimensional',
    ):
        qtoolkit.timetags.TimetagData(
            timetags=np.array([
                1,
                2,
            ]),
            channels=np.array([
                [0, 1],
            ]),
        )


def test_timetag_data_rejects_different_lengths() -> None:
    with pytest.raises(
        ValueError,
        match='must have the same length',
    ):
        qtoolkit.timetags.TimetagData(
            timetags=np.array([
                1,
                2,
            ]),
            channels=np.array([
                0,
            ]),
        )


def test_timetag_data_from_text_file(
        tmp_path: pathlib.Path,
) -> None:
    file_path = tmp_path.joinpath('timetags.txt')

    file_path.write_text(
        '941575226770542 6\n'
        '941575227172420 0\n'
        '941575227234390 6\n'
        '941575227678840 6\n'
    )

    result = qtoolkit.timetags.TimetagData.from_file(
        file_path
    )

    np.testing.assert_array_equal(
        result.timetags,
        np.array(
            [
                941575226770542,
                941575227172420,
                941575227234390,
                941575227678840,
            ],
            dtype=np.int64,
        ),
    )

    np.testing.assert_array_equal(
        result.channels,
        np.array(
            [6, 0, 6, 6],
            dtype=np.int8,
        ),
    )

    assert result.file_path == file_path


def test_timetag_data_from_csv_file(
        tmp_path: pathlib.Path,
) -> None:
    file_path = tmp_path.joinpath('timetags.csv')

    file_path.write_text(
        '941575226770542,6\n'
        '941575227172420,0\n'
        '941575227234390,6\n'
        '941575227678840,6\n'
    )

    result = qtoolkit.timetags.TimetagData.from_file(
        file_path
    )

    np.testing.assert_array_equal(
        result.timetags,
        np.array(
            [
                941575226770542,
                941575227172420,
                941575227234390,
                941575227678840,
            ],
            dtype=np.int64,
        ),
    )

    np.testing.assert_array_equal(
        result.channels,
        np.array(
            [6, 0, 6, 6],
            dtype=np.int8,
        ),
    )

    assert result.file_path == file_path


def test_timetag_data_from_invalid_file(
        tmp_path: pathlib.Path,
) -> None:
    with pytest.raises(
        ValueError,
        match='Unsupported file type: ',
    ):

        file_path = tmp_path.joinpath('timetags')

        result = qtoolkit.timetags.TimetagData.from_file(
            file_path
        )


def test_timetag_data_from_string_path(
        tmp_path: pathlib.Path,
) -> None:
    file_path = tmp_path.joinpath('timetags.txt')
    file_path.write_text('1000 0\n')

    result = qtoolkit.timetags.TimetagData.from_file(
        str(file_path)
    )

    assert result.file_path == file_path


def test_timetag_data_from_single_row(
        tmp_path: pathlib.Path,
) -> None:
    file_path = tmp_path.joinpath('timetags.txt')
    file_path.write_text('1000 3\n')

    result = qtoolkit.timetags.TimetagData.from_file(
        file_path
    )

    np.testing.assert_array_equal(
        result.timetags,
        np.array(
            [1000],
            dtype=np.int64,
        ),
    )

    np.testing.assert_array_equal(
        result.channels,
        np.array(
            [3],
            dtype=np.int8,
        ),
    )


def test_timetag_data_from_file_rejects_extra_columns(
        tmp_path: pathlib.Path,
) -> None:
    file_path = tmp_path / 'invalid.txt'
    file_path.write_text(
        '1000 0 123\n'
    )

    with pytest.raises(
        ValueError,
        match='exactly two columns',
    ):
        qtoolkit.timetags.TimetagData.from_file(
            file_path
        )

# ProcessedTimetagData

def test_processed_timetag_data_from_timetag_data() -> None:
    data = qtoolkit.timetags.TimetagData(
        timetags=np.array(
            [
                1000,
                1050,
                2000,
                2050,
            ],
            dtype=np.int64,
        ),
        channels=np.array(
            [
                0,
                1,
                0,
                1,
            ],
            dtype=np.int8,
        ),
    )

    result = (
        qtoolkit.timetags.ProcessedTimetagData
        .from_timetag_data(
            timetag_data=data,
            pairs=[
                qtoolkit.timetags.ChannelPair(
                    first=0,
                    second=1,
                ),
            ],
            coincidence_window=100,
        )
    )

    assert result.coincidences == {
        (0, 1): 2,
    }

    assert result.coincidence_window == 100


def test_processed_timetag_data_accepts_tuple_pairs() -> None:
    data = qtoolkit.timetags.TimetagData(
        timetags=np.array(
            [
                1000,
                1050,
            ],
            dtype=np.int64,
        ),
        channels=np.array(
            [
                0,
                1,
            ],
            dtype=np.int8,
        ),
    )

    result = (
        qtoolkit.timetags.ProcessedTimetagData
        .from_timetag_data(
            timetag_data=data,
            pairs=[
                (0, 1),
            ],
            coincidence_window=100,
        )
    )

    assert result.coincidences == {
        (0, 1): 1,
    }


def test_processed_timetag_data_from_file(
        tmp_path: pathlib.Path,
) -> None:
    file_path = tmp_path.joinpath('timetags.txt')

    file_path.write_text(
        '1000 0\n'
        '1050 1\n'
        '2000 0\n'
        '2050 1\n'
    )

    result = (
        qtoolkit.timetags.ProcessedTimetagData
        .from_file(
            file_path=file_path,
            pairs=[
                qtoolkit.timetags.ChannelPair(
                    first=0,
                    second=1,
                ),
            ],
            coincidence_window=100,
        )
    )

    assert result.coincidences == {
        (0, 1): 2,
    }

    assert result.coincidence_window == 100
    assert result.file_path == file_path
