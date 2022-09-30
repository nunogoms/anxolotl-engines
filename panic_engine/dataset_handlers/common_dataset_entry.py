from typing import List

from numpy import ndarray


class CommonDatasetEntry:
    def __init__(self, timeline, rr, user_gender, user_age, questionnaire):
        self.timeline: List = timeline
        self.rr: List = rr
        self.user_gender: int = user_gender
        self.user_age: int = user_age
        self.anxiety_label: ndarray = questionnaire


