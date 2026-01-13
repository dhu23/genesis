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
        assert -4.5 == pytest.approx(dd.drawdown_depth, abs=0.001) 
        assert 4 == dd.length

        assert not dd.extend(self.DATA_10) # no update to drawdown value
        assert -4.5 == pytest.approx(dd.drawdown_depth, abs=0.001)
        assert 9 == dd.length

        assert dd.extend(self.DATA_12)
        assert -5.5 == pytest.approx(dd.drawdown_depth, abs=0.001)
        assert 11 == dd.length

        with pytest.raises(Exception):
            dd.extend(self.DATA_20)

        with pytest.raises(Exception):
            dd.extend(self.DATA_5)

        # extension exceptions don't change internal state
        assert -5.5 == pytest.approx(dd.drawdown_depth, abs=0.001)
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
        assert not tracker.max_drawdown_value()
        assert not tracker.longest_drawdown_length()
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
    # 1. from 11 down to 6 (closed)
    # 2. from 11.5 down to 11 (not closed)
    def test_tracker_state(self):
        tracker = DrawDownTracker()
        
        tracker.update(10)
        assert not tracker.drawdowns()
        assert not tracker.in_drawdown()
        assert not tracker.max_drawdown_value()
        assert not tracker.longest_drawdown_length()
        assert tracker.data_point_count() == 1

        tracker.update(11)
        assert not tracker.drawdowns()
        assert not tracker.in_drawdown()
        assert not tracker.max_drawdown_value()
        assert not tracker.longest_drawdown_length()
        assert tracker.data_point_count() == 2

        HIGH_WATER_MARK_1 = DataPoint(1, 11)

        tracker.update(10.5)
        current_drawdown = Drawdown(HIGH_WATER_MARK_1, DataPoint(2, 10.5))
        assert tracker.drawdowns() == [current_drawdown]
        assert tracker.in_drawdown()
        assert tracker.max_drawdown_value() == -0.5
        assert tracker.longest_drawdown_length() == 1
        assert tracker.data_point_count() == 3
    
        tracker.update(8.5)
        current_drawdown = Drawdown(HIGH_WATER_MARK_1, DataPoint(3, 8.5))
        assert tracker.drawdowns() == [current_drawdown]
        assert tracker.in_drawdown()
        assert tracker.max_drawdown_value() == -2.5
        assert tracker.longest_drawdown_length() == 2
        assert tracker.data_point_count() == 4

        tracker.update(9.5) # doesn't change drawdown depth
        drawdowns = tracker.drawdowns()
        assert len(drawdowns) == 1
        assert drawdowns[0].start == HIGH_WATER_MARK_1
        assert drawdowns[0].end == DataPoint(4, 9.5)
        assert drawdowns[0].drawdown_depth == -2.5
        assert drawdowns[0].length == 3
        assert tracker.in_drawdown()
        assert tracker.max_drawdown_value() == -2.5
        assert tracker.longest_drawdown_length() == 3
        assert tracker.data_point_count() == 5

        tracker.update(6.0)
        drawdowns = tracker.drawdowns()
        assert len(drawdowns) == 1
        assert drawdowns[0].start == HIGH_WATER_MARK_1
        assert drawdowns[0].end == DataPoint(5, 6.0)
        assert drawdowns[0].drawdown_depth == -5.0
        assert drawdowns[0].length == 4
        assert tracker.in_drawdown()
        assert tracker.max_drawdown_value() == -5.0
        assert tracker.longest_drawdown_length() == 4
        assert tracker.data_point_count() == 6

        tracker.update(7.0)
        tracker.update(8.0)
        tracker.update(9.0)
        tracker.update(10.0)
        tracker.update(11.0)
        drawdowns = tracker.drawdowns()
        assert len(drawdowns) == 1
        assert drawdowns[0].start == HIGH_WATER_MARK_1
        assert drawdowns[0].end == DataPoint(10, 11.0)
        assert drawdowns[0].drawdown_depth == -5.0
        assert drawdowns[0].length == 9
        assert tracker.in_drawdown()
        assert tracker.max_drawdown_value() == -5.0
        assert tracker.longest_drawdown_length() == 9
        assert tracker.data_point_count() == 11

        HIGH_WATER_MARK_2 = DataPoint(11, 11.5)

        # recovered from a drawdown
        tracker.update(11.5)
        drawdowns = tracker.drawdowns()
        assert len(drawdowns) == 1
        assert not tracker.in_drawdown()
        assert tracker.max_drawdown_value() == -5.0
        assert tracker.longest_drawdown_length() == 9
        assert tracker.data_point_count() == 12

        tracker.update(11)
        tracker.update(11)
        drawdowns = tracker.drawdowns()
        assert len(drawdowns) == 2
        assert drawdowns[0].start == HIGH_WATER_MARK_1
        assert drawdowns[0].end == DataPoint(10, 11.0)
        assert drawdowns[0].drawdown_depth == -5.0
        assert drawdowns[0].length == 9
        assert drawdowns[1].start == HIGH_WATER_MARK_2
        assert drawdowns[1].end == DataPoint(13, 11)
        assert drawdowns[1].drawdown_depth == -0.5
        assert drawdowns[1].length == 2
        assert tracker.in_drawdown()
        assert tracker.max_drawdown_value() == -5.0
        assert tracker.longest_drawdown_length() == 9
        assert tracker.data_point_count() == 14


    def test_increasing_sequence(self):
        tracker = DrawDownTracker()
        for i, x in enumerate(self.SEQUENCE_1):
            tracker.update(x)
        
            # no drawdown
            assert not tracker.drawdowns()
            assert not tracker.in_drawdown()
            assert not tracker.max_drawdown_value()
            assert not tracker.longest_drawdown_length()
            assert tracker.data_point_count() == i+1

    def test_decreasing_sequence(self):
        tracker = DrawDownTracker()
        for i, x in enumerate(self.SEQUENCE_2):
            tracker.update(x)

            # other than the first one, all other data points are in drawdown
            if i == 0:
                assert not tracker.drawdowns()
                assert not tracker.in_drawdown()
                assert not tracker.max_drawdown_value()
                assert not tracker.longest_drawdown_length()    
            else:
                assert tracker.drawdowns()
                assert tracker.in_drawdown()
                assert tracker.max_drawdown_value() == x - self.SEQUENCE_2[0]
                assert tracker.longest_drawdown_length() == i

            assert tracker.data_point_count() == i+1

