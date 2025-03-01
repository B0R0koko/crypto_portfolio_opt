from datetime import datetime, date, timedelta
from typing import List


def start_of_date(day: date) -> datetime:
    return datetime(day.year, day.month, day.day)


def end_of_date(day: date) -> datetime:
    return start_of_date(day=day) + timedelta(days=1) - timedelta(microseconds=1)


class Bounds:

    def __init__(self, start_inclusive: datetime, end_exclusive: datetime):
        self.start_inclusive: datetime = start_inclusive
        self.end_exclusive: datetime = end_exclusive

    @classmethod
    def for_days(cls, start_date: date, end_date: date) -> "Bounds":
        return Bounds(start_of_date(day=start_date), end_of_date(day=end_date))

    @property
    def day0(self) -> date:
        return self.start_inclusive.date()

    @property
    def day1(self) -> date:
        return self.end_exclusive.date()

    def generate_overlapping_bounds(self, step: timedelta, interval: timedelta) -> List["Bounds"]:
        """Returns a list of bounds created from parent Bounds interval with a certain interval size and step"""
        intervals: List["Bounds"] = []

        lb = self.start_inclusive

        while True:
            rb: datetime = lb + interval
            intervals.append(Bounds(start_inclusive=lb, end_exclusive=rb))  # create new overlapping sub-Bounds
            lb += step

            if rb >= self.end_exclusive:
                break

        return intervals
