from enum import IntEnum

import numpy as np


# Decision tree
class DecisionTreeNode:
    def __init__(self, name='', left=None, right=None):
        self.name = name
        self.left = left
        self.right = right

    def go_left(self):
        return self.left

    def go_right(self):
        return self.right


class SplitNode(DecisionTreeNode):
    def __init__(self, name='', left=None, right=None, threshold=0.5):
        super().__init__(name, left, right)
        self.threshold = threshold

    def update_threshold(self, new_threshold):
        self.threshold = new_threshold


class DecisionNode(DecisionTreeNode):
    def __init__(self, name='', left=None, right=None, action=None):
        super().__init__(name, left, right)
        self.action = action


# Build specific moving decision tree
def build_moving_decision_tree():
    actions = {'random_move': [None], 'move_away': [None], 'conference_move': [None], 'out_move': [None],
               'friends_move': [None], 'back_seat_move': [None]}

    # Depth 0
    moving_decision_tree = DecisionTree(head=SplitNode(name='there_is_conference', threshold=0), action_dict=actions)
    # Actions
    random_move_action = DecisionNode(name='random_move',
                                      action=moving_decision_tree.action_dict['random_move'])
    move_away_action = DecisionNode(name='move_away',
                                    action=moving_decision_tree.action_dict['move_away'])
    conference_move_action = DecisionNode(name='conference_move',
                                          action=moving_decision_tree.action_dict['conference_move'])
    out_move_action = DecisionNode(name='out_move',
                                   action=moving_decision_tree.action_dict['out_move'])
    friends_move_action = DecisionNode(name='friends_move', action=moving_decision_tree.action_dict['friends_move'])
    back_seat_move_action = DecisionNode(name='back_seat_move',
                                         action=moving_decision_tree.action_dict['back_seat_move'])

    # Known subtrees
    move_away_from_crowd = SplitNode(name='move_away_from_crowd')
    random_or_friends = SplitNode(name='random_or_friends')
    back_seat_or_move_away = SplitNode(name='back_seat_or_move_away')
    back_seat_or_move_out = SplitNode(name='back_seat_or_move_out')
    back_seat_or_friends = SplitNode(name='back_seat_or_friends')

    # Depth 1
    moving_decision_tree.add_node(SplitNode(name='go_to_conference'), parent='there_is_conference', direction='L')
    moving_decision_tree.add_node(move_away_from_crowd, parent='there_is_conference', direction='R')

    # Depth 2
    moving_decision_tree.add_node(SplitNode(name='is_in_conference_1', threshold=0),
                                  parent='go_to_conference', direction='L')
    moving_decision_tree.add_node(SplitNode(name='is_in_conference_2', threshold=0),
                                  parent='go_to_conference', direction='R')

    moving_decision_tree.add_node(back_seat_or_move_away, parent='move_away_from_crowd', direction='L')
    moving_decision_tree.add_node(random_or_friends, parent='move_away_from_crowd', direction='R')

    # Depth 3
    moving_decision_tree.add_node(random_move_action, parent='is_in_conference_1', direction='L')

    moving_decision_tree.add_node(conference_move_action, parent='is_in_conference_1', direction='R')

    moving_decision_tree.add_node(back_seat_or_move_out, parent='is_in_conference_2', direction='L')
    moving_decision_tree.add_node(move_away_from_crowd, parent='is_in_conference_2', direction='R')

    moving_decision_tree.add_node(back_seat_or_friends, parent='random_or_friends', direction='L')
    moving_decision_tree.add_node(random_move_action, parent='random_or_friends', direction='R')

    # Depth 4
    moving_decision_tree.add_node(back_seat_move_action, parent='back_seat_or_friends', direction='L')
    moving_decision_tree.add_node(friends_move_action, parent='back_seat_or_friends', direction='R')

    moving_decision_tree.add_node(back_seat_move_action, parent='back_seat_or_move_out', direction='L')
    moving_decision_tree.add_node(out_move_action, parent='back_seat_or_move_out', direction='R')

    moving_decision_tree.add_node(back_seat_move_action, parent='back_seat_or_move_away', direction='L')
    moving_decision_tree.add_node(move_away_action, parent='back_seat_or_move_away', direction='R')

    return moving_decision_tree


def in_calculate_decision(curr, path):
    if type(curr) == DecisionNode:
        print(curr.name)
        path.append(curr.name)
        return curr.action
    print(curr.name)
    path.append(curr.name)
    if throw_coin(curr.threshold):
        return in_calculate_decision(curr.go_left(), path)
    else:
        return in_calculate_decision(curr.go_right(), path)


class DecisionTree:
    def __init__(self, head=None, action_dict=None):
        if head is None:
            head = DecisionTreeNode('head')
        self.head = head
        self.action_dict = action_dict
        self.nodes_dict = {self.head.name: self.head}

    def add_node(self, new_node: DecisionTreeNode, parent: str, direction='L'):
        if parent in self.nodes_dict:
            if direction == 'L':
                self.nodes_dict[parent].left = new_node
            else:
                self.nodes_dict[parent].right = new_node
        self.nodes_dict[new_node.name] = new_node

    def calculate_decision(self, path):
        return in_calculate_decision(self.head, path)


# Enum classes
class HealthStatus(IntEnum):
    IMMUNE = -2
    RECOVERY = -1
    HEALTHY = 0
    CARRIED = 1


class AirStatus(IntEnum):
    AIR_EXCHANGE = 0
    STANDING_AIR = 1
    AIR_RECYCLING = 2


class CountryStatus(IntEnum):
    LOW_MORBIDITY = 0
    MIDDLE_MORBIDITY = 1
    HIGH_MORBIDITY = 2


class Action(IntEnum):
    DO_NOTHING = 0
    EATING = 1
    MOVING = 2
    DANCING = 3
    TALKING = 4
    DISTANT_PHYSICAL_CONTACT = 5
    CLOSE_PHYSICAL_CONTACT = 6


# Help class
class Queue:
    def __init__(self, n: int):
        self.queue = []
        self.max_len = n

    def insert(self, x):
        self.queue.append(x)
        if len(self.queue) > self.max_len:
            self.queue.pop(0)


# Utils functions
def throw_coin(x: float):
    return np.random.random() < x


def logistic_prob(factors=None, variables=None, expected_value=0.5):
    if factors is None or variables is None:
        return 0
    norm = sum([f for f in factors])
    exponent = (sum([f / norm * v for f, v in zip(factors, variables)]) - expected_value) / 0.1
    return 1 / (1 + np.exp(-exponent))


def dist(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def lower_first(a, b):
    return (a, b) if a < b else (b, a)


def rect_area(coordinates):
    x_1 = coordinates[0]
    y_1 = coordinates[1]
    x_2 = coordinates[2]
    y_2 = coordinates[3]
    return (x_2 - x_1) * (y_2 - y_1)