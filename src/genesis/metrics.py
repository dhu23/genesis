'''
This module contains packages that help calculating time series metrics
such as max drawdown, drawdown period or other patterns.
'''

from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class DataPoint:
    index: int
    value: int | float


@dataclass(frozen=False)
class Drawdown:
    '''
    Tracking a drawdown period. 

    :var consideration: Description
    '''
    start: DataPoint
    end: DataPoint
    _drawdown_value: int | float = field(init=False, repr=False)

    def __post_init__(self):
        if self.start.value < self.end.value:
            raise ValueError(
                f'no down trend {self.start.value} -> {self.end.value}'
            )
        if self.start.index >= self.end.index:
            raise ValueError(
                f'start index ({self.start.index} needs to come before end index {self.end.index})'
            )
        self._drawdown_value = self.end.value - self.start.value 

    def extend(self, data_point: DataPoint) -> bool:
        if data_point.value > self.start.value:
            raise ValueError(
                f'cannot extend beyond starting value {self.start.value}, new value {data_point.value}'
            )
        if self.end.index >= data_point.index:
            raise ValueError(
                f'end index ({self.end.index}) needs to come before new index ({data_point.index})'
            )
        self.end = data_point
        # update drawdown statistics if the new observation trends down lower
        _drawdown_candidate = data_point.value - self.start.value
        if _drawdown_candidate < self._drawdown_value:
            self._drawdown_value = _drawdown_candidate
            return True
        else:
            return False
    
    @property
    def length(self):
        return self.end.index - self.start.index
    
    @property
    def drawdown_value(self):
        return self._drawdown_value


class DrawDownTracker(object):
    '''
    This calculates drawdown metrics in an event-based fashion. 
    The invariant of the program is that it maintains the high-watermark of
    the previous values, and use that to determine whether the new update
    goes above to a new high watermark, or stays belew in the current drawdown.

    Everytime the previous high watermark is breached by a higher update, we
    close the current drawdown period if any and update drawdown statistics

    Every drawdown is identified by the entry point index. It can be reasoned
    that the entry point uniquely defines a drawdown identity.

    Special consideration:
    - going back to exactly the high watermark during a drawdown does not close
      the current drawdown
    '''
    def __init__(self):
        self._high_water_mark: DataPoint | None = None
        
        # past closed drawdown periods + the current drawdown (optional)
        self._drawdowns: list[Drawdown] = []
        # whether the most recent drawdown is still open.
        # this controls if an exsiting drawdown period should be modified
        self._in_drawdown: bool = False
        # data point tracking index
        self._index = 0

        # for tracking drawdown statistics
        self._max_drawdowns: list[Drawdown] = []
        self._longest_drawdowns: list[Drawdown] = []

    def _update_statistics(self, drawdown: Drawdown) -> None:
        self._update_max_drawdowns(drawdown)
        self._update_longest_drawdowns(drawdown)

    def _update_max_drawdowns(self, drawdown: Drawdown) -> None:
        if not self._max_drawdowns:
            self._max_drawdowns.append(drawdown)
            return 
        
        current_max = self._max_drawdowns[-1].drawdown_value
        if drawdown.drawdown_value == current_max:
            self._max_drawdowns.append(drawdown)
        elif drawdown.drawdown_value > current_max:
            self._max_drawdowns = [drawdown]
        else:
            # when the new one is smaller than current max, drop it
            pass

    def _update_longest_drawdowns(self, drawdown: Drawdown) -> None:
        if not self._longest_drawdowns:
            self._longest_drawdowns.append(drawdown)
            return
        
        current_longest = self._longest_drawdowns[-1].length
        if drawdown.length == current_longest:
            self._longest_drawdowns.append(drawdown)
        elif drawdown.length > current_longest:
            self._longest_drawdowns = [drawdown]
        else:
            # when the new one is shorter than the current longest, drop it
            pass    
        
    def update(self, value: int | float):
        '''
        update value with an index, e.g. an integer or date object
        
        :param self: Description
        :param idx: Description
        :type idx: I
        :param value: Description
        :type value: V
        '''
        data = DataPoint(self._index, value)
        self._index += 1

        # drive logic by updating high watermark
        if self._high_water_mark is None:
            # the first data point
            self._high_water_mark = data

        elif self._high_water_mark.value < value:
            self._high_water_mark = data
            # Two scenraios:
            # 1. was in a drawdown and now it breaks out
            # 2. in an uptrending situation
            # close the current drawdown if any
            # no need to update drawdown statistics
            if self._in_drawdown:
                self._in_drawdown = False 

        else:
            # high water mark stays where it is
            # Two scenarios:
            # 1. already in drawdown: extend the current drawdown
            # 2. not in a drawdown: create a new drawdown
            if self._in_drawdown:
                current_drawdown = self._drawdowns[-1]
                current_drawdown.extend(data)
                self._update_statistics(current_drawdown)
            else:
                new_drawdown: Drawdown = Drawdown(self._high_water_mark, data)
                self._drawdowns.append(new_drawdown)
                self._in_drawdown = True
                self._update_statistics(new_drawdown)

    def drawdowns(self):
        return self._drawdowns
    
    def in_drawdown(self):
        return self._in_drawdown

    def max_drawdowns(self) -> list[Drawdown]:
        return self._max_drawdowns

    def longest_drawdowns(self) -> list[Drawdown]:
        return self._longest_drawdowns
    
    def data_point_count(self) -> int:
        '''
        number of data points that are processed by the tracker
        
        :param self: Description
        :return: Description
        :rtype: int
        '''
        return self._index


def get_drawdown_data(data: list[float | int] | np.ndarray) -> DrawDownTracker:
    tracker = DrawDownTracker()
    for x in data:
        tracker.update(x)
    return tracker