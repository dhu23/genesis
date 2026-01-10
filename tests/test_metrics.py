from genesis.metrics import DataPoint, Drawdown, DrawDownTracker
import pytest

class TestDrawDown(object):
    DATA_1 = DataPoint(1, 1.5)
    DATA_5 = DataPoint(5, -3)
    DATA_10 = DataPoint(10, -2)
    DATA_12 = DataPoint(12, -4)
    DATA_20 = DataPoint(20, 20)

    def test_bad_drawdown_construction(self):
        with pytest.raises(Exception):
            Drawdown(self.DATA_5, self.DATA_20)

        with pytest.raises(Exception):
            Drawdown(self.DATA_20, self.DATA_1)

    def test_good_drawdown_construction(self):
        dd = Drawdown(self.DATA_1, self.DATA_5)
        assert -4.5 == pytest.approx(dd.drawdown_value, abs=0.001) 
        assert 4 == dd.length

        assert not dd.extend(self.DATA_10) # no update to drawdown value
        assert -4.5 == pytest.approx(dd.drawdown_value, abs=0.001)
        assert 9 == dd.length

        assert dd.extend(self.DATA_12)
        assert -5.5 == pytest.approx(dd.drawdown_value, abs=0.001)
        assert 11 == dd.length

        with pytest.raises(Exception):
            dd.extend(self.DATA_20)

        with pytest.raises(Exception):
            dd.extend(self.DATA_5)

        # extension exceptions don't change internal state
        assert -5.5 == pytest.approx(dd.drawdown_value, abs=0.001)
        assert 11 == dd.length
        

class TestDrawDownTracker(object):
    # 0, 1, 2, ..., 18, 19
    SEQUENCE_1: list[int] = list(range(20))

    # 20, 19, 18, ..., 2, 1
    SEQUENCE_2: list[int] = list(range(20, 0, -1))

    # 10, 11, 12, 13, 14
    SEQUENCE_3: list[int] = list(range(10, 15))

    # 15, 14, 13, 12, 11
    SEQUENCE_4: list[int] = list(range(15, 10, -1))

    def test_empty_tracker(self):
        tracker = DrawDownTracker()
        assert not tracker.drawdowns()
        assert not tracker.in_drawdown()
        assert not tracker.max_drawdowns()
        assert not tracker.longest_drawdowns()
        assert tracker.data_point_count() == 0

    # make point by point assertion for such sequence of data
    # 10, 11, 10.5, 8.5, 9.5, 6, 7, 8, 9, 10, 11, 11.5, 11, 11
    #  11.5|                       *
    #  11.0|   *                 * * * *
    #  10.5|   * *               * * * *
    #  10.0| * * *             * * * * *
    #   9.5| * * *   *         * * * * *
    #   9.0| * * *   *       * * * * * *
    #   8.5| * * * * *       * * * * * *
    #   8.0| * * * * *     * * * * * * * 
    #   7.5| * * * * *     * * * * * * *
    #   7.0| * * * * *   * * * * * * * *
    #   6.5| * * * * *   * * * * * * * *
    #   6.0| * * * * * * * * * * * * * *
    #      +-----------------------------
    #      0         5        10 
    # There are two drawdown periods:
    # 1. from 11 down to 6
    def test_tracker_state(self):
        tracker = DrawDownTracker()
        tracker.update(10)


    

        

    def test_sequence_1(self):
        pass