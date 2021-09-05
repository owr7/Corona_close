import time
from enum import IntEnum
from typing import Any
from mesa import Model, Agent
import numpy as np
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid
from mesa.time import RandomActivation

from itertools import combinations

from utils_for_clean_close_model import throw_coin, DecisionNode, HealthStatus, Action, logistic_prob, CountryStatus, \
    AirStatus, build_moving_decision_tree, dist, rect_area, lower_first, Queue
from utils_for_corona_model import generate_age


class DrawingOnGridAgent(Agent):
    def __init__(self, unique_id: int, model: Model):
        super().__init__(unique_id, model)
        self.health = HealthStatus.HEALTHY


def choose_seat_around_table(pos: (int, int), x: int):
    if x == 10:
        return pos[0] - 1, pos[1] - 1
    if x == 9:
        return pos[0] - 1, pos[1] + 1
    if x == 8:
        return pos[0] + 1, pos[1] + 1
    if x == 7:
        return pos[0] + 1, pos[1] - 1
    if x == 6:
        return pos[0] + 2, pos[1]
    if x == 5:
        return pos[0] - 1, pos[1]
    if x == 4:
        return pos[0], pos[1] - 1
    if x == 3:
        return pos[0], pos[1] + 1
    if x == 2:
        return pos[0] + 2, pos[1] - 1
    if x == 1:
        return pos[0] + 2, pos[1] + 1


def mask_protection(agent_1, agent_2, prob):
    if agent_1.mask and agent_2.mask:
        return prob[0]
    elif agent_1.mask or agent_2.mask:
        return prob[1]
    return prob[3]


def during_of_action(action, waiter=False):
    if action == Action.DO_NOTHING:
        return 0.8
    if action == Action.EATING:
        return 0.1
    if action == Action.TALKING:
        if waiter:
            return 0.5
        return 0.2
    if action == Action.DANCING:
        return 0.1
    if action == Action.DISTANT_PHYSICAL_CONTACT:
        return 0.9
    if action == Action.CLOSE_PHYSICAL_CONTACT:
        return 0.9


def contagious_action(action):
    if action == Action.TALKING:
        return 1.2
    if action == Action.DISTANT_PHYSICAL_CONTACT:
        return 1.4
    if action == Action.DANCING:
        return 1.6
    if action == Action.CLOSE_PHYSICAL_CONTACT:
        return 1.8
    return 0


def influence_action_on_wearing_mask(action, conference):
    if action == Action.EATING:
        return 0.1
    if action == Action.TALKING and conference:
        return 0.6
    elif action == Action.TALKING:
        return 0.8
    if action == Action.DANCING and conference:
        return 0.5
    elif action == Action.DANCING:
        return 0.6
    return 1


class PopAgent(Agent):
    def __init__(self, unique_id: int, model: Model, health=HealthStatus.HEALTHY):
        super().__init__(unique_id, model)

        # Fixed fields
        self.age = generate_age()
        self.infects_others_level = np.random.normal(0.5, 0.25)
        self.infects_by_others_level = np.random.normal(0.5, 0.25)

        # Changeable fields
        self.health = health
        self.mask = False
        self.social_influence = 0
        if self.health == HealthStatus.CARRIED:
            self.infection_generation = 1
        else:
            self.infection_generation = 0
        self.active = False

        # Table fields
        self.base_pos = (0, 0)
        self.last_time_in_seat = 0

        # Action and Interactions fields
        self.current_action = Action.DO_NOTHING
        self.continue_action = 0
        self.partner = self
        self.action_done = False

    # Different moving
    def random_move(self, possible_steps):
        new_position = self.random.choice([pos for pos in possible_steps])
        # Random moving
        self.model.grid.move_agent(self, new_position)

    def conference_move(self, possible_steps):
        chosen_destination = (self.random.randrange(self.model.gathering_area['min_x'],
                                                    self.model.gathering_area['max_x']),
                              self.random.randrange(self.model.gathering_area['min_y'],
                                                    self.model.gathering_area['max_y']))

        new_position = move_forward(possible_steps, chosen_destination)
        self.model.grid.move_agent(self, new_position)

    def back_seat_move(self, possible_steps):
        new_position = move_forward(possible_steps, self.base_pos)
        self.model.grid.move_agent(self, new_position)

    def out_move(self, possible_steps):
        chosen_destination = self.pos
        # The probability complexity of this loop is area/(gathering area) ~O(1) for relevant cases
        while self.model.pos_in_gathering_area(chosen_destination):
            chosen_destination = (self.random.randrange(0, self.model.grid.height),
                                  self.random.randrange(0, self.model.grid.width))

        new_position = move_forward(possible_steps, chosen_destination)
        self.model.grid.move_agent(self, new_position)

    def move_away(self, possible_steps):
        want_stay_at_gathering_area = False
        if self.model.pos_in_gathering_area(self.pos):
            want_stay_at_gathering_area = True
        index = np.argmin([len(self.model.grid.get_cell_list_contents(
            self.model.grid.get_neighborhood(pos, moore=True,
                                             include_center=True,
                                             radius=1))) for pos in possible_steps
            if self.model.pos_in_gathering_area(pos) or (not want_stay_at_gathering_area)])
        new_position = possible_steps[index]
        self.model.grid.move_agent(self, new_position)

    def move_friends(self, possible_steps):
        index = np.argmax([np.average([self.model.relationship_level[
                                           lower_first(self.unique_id, k.unique_id)]
                                       for k in self.model.grid.get_cell_list_contents(
                self.model.grid.get_neighborhood(pos, moore=True,
                                                 include_center=True,
                                                 radius=1)) if k != self and
                                       type(k) == PopAgent]) for pos in possible_steps])
        new_position = possible_steps[index]
        self.model.grid.move_agent(self, new_position)

    def move(self, possible_steps, cellmates):
        # Update tree actions
        self.model.move_decision_tree.action_dict['random_move'][0] = self.random_move
        self.model.move_decision_tree.action_dict['move_away'][0] = self.move_away
        self.model.move_decision_tree.action_dict['conference_move'][0] = self.conference_move
        self.model.move_decision_tree.action_dict['out_move'][0] = self.out_move
        self.model.move_decision_tree.action_dict['friends_move'][0] = self.out_move
        self.model.move_decision_tree.action_dict['back_seat_move'][0] = self.back_seat_move

        # Calculate thresholds
        threshold_go_conference = logistic_prob(factors=[self.model.mask_coeff[0], self.model.mask_coeff[3]],
                                                variables=[self.age / 100,
                                                           1 - self.model.conference_crowded /
                                                           max(rect_area([k for k in
                                                                          self.model.gathering_area.values()]), 1)
                                                           ])
        threshold_get_away_from_people = logistic_prob(factors=[self.model.mask_coeff[0], self.model.mask_coeff[3]],
                                                       variables=[self.age / 100,
                                                                  len(cellmates) / len(possible_steps)
                                                                  ])
        threshold_going_to_friends = max([np.average([self.model.relationship_level[
                                                          (min(self.unique_id, k.unique_id),
                                                           max(self.unique_id, k.unique_id))]
                                                      for k in self.model.grid.get_cell_list_contents(
                self.model.grid.get_neighborhood(pos, moore=True,
                                                 include_center=True,
                                                 radius=1)) if k != self and type(k) != DrawingOnGridAgent
                                                      and type(k) != WaiterAgent])
                                          for pos in possible_steps])

        # Update thresholds
        self.model.move_decision_tree.nodes_dict['there_is_conference'].threshold = 1 if self.model.conference else 0
        self.model.move_decision_tree.nodes_dict['go_to_conference'].threshold = threshold_go_conference
        self.model.move_decision_tree.nodes_dict['move_away_from_crowd'].threshold = threshold_get_away_from_people \
            if self.model.get_away else 0
        self.model.move_decision_tree.nodes_dict['is_in_conference_1'].threshold = \
            self.model.move_decision_tree.nodes_dict['is_in_conference_2'].threshold = \
            1 if self.model.pos_in_gathering_area(self.pos) \
                else 0
        self.model.move_decision_tree.nodes_dict['random_or_friends'].threshold = threshold_going_to_friends
        self.model.move_decision_tree.nodes_dict['back_seat_or_move_away'].threshold = \
            self.model.move_decision_tree.nodes_dict['back_seat_or_move_out'].threshold = \
            self.model.move_decision_tree.nodes_dict['back_seat_or_friends'].threshold = \
            min(float(len(self.model.seating_area)),
                (self.model.time - self.last_time_in_seat) / (self.model.time + self.last_time_in_seat))
        path = []
        chosen_move = self.model.move_decision_tree.calculate_decision(path=path)[0]

        chosen_move(possible_steps)
        # Relevant for actions
        return path

    def choose_interaction(self, cellmates):
        # Choose partner to action
        # sort the neighborhood agents according to relationship with current agent
        cellmates = [j for i, j in sorted(zip([lower_first(self.unique_id, c.unique_id)
                                               for c in cellmates], cellmates)) if not j.action_done]
        if len(cellmates) > 0:
            index = min(np.random.geometric(0.5), len(cellmates) - 1)
            self.partner = cellmates[index]
            self.partner.partner = self
        else:
            self.partner = self

    # Choose action
    def choose_action(self, move_path, cellmates):
        if 'back_seat_move' in move_path and self.close_to_seat() and throw_coin(0.5):
            if throw_coin(0.5):
                self.current_action = Action.EATING
            else:
                self.current_action = Action.DO_NOTHING
        else:
            self.choose_interaction(cellmates)
            if self.partner == self:
                self.current_action = Action.DO_NOTHING
                self.continue_action = during_of_action(self.current_action)
                return
            threshold = logistic_prob(factors=[1, 1, 1, 1, 1],
                                      variables=[self.model.relationship_level[
                                                     lower_first(self.unique_id, self.partner.unique_id)],
                                                 1 - self.age / 100, 1 - self.partner.age / 100,
                                                 sum([1 for k in [self.mask, self.partner.mask] if k]),
                                                 1 - dist(self.pos, self.partner.pos) / np.sqrt(8)])

            if 'back_seat_move' in move_path and self.close_to_seat():
                prob = [0.7, 1]
                possible_actions = [Action.TALKING, Action.DISTANT_PHYSICAL_CONTACT]
            elif 'friends_move' in move_path:
                prob = [0.4, 0.7, 0.9, 1]
                possible_actions = [Action.TALKING, Action.DISTANT_PHYSICAL_CONTACT,
                                    Action.DANCING, Action.CLOSE_PHYSICAL_CONTACT]
            elif 'random_move' in move_path and len(move_path) == 4:
                prob = [0.2, 0.4, 0.8, 1]
                possible_actions = [Action.TALKING, Action.DISTANT_PHYSICAL_CONTACT,
                                    Action.DANCING, Action.CLOSE_PHYSICAL_CONTACT]
            else:
                prob = [0]
                possible_actions = [Action.DO_NOTHING]

            for a, p in zip(possible_actions, prob):
                if threshold < p:
                    self.current_action = a
                    self.partner.current_action = a
                    break

        if self.current_action != Action.DO_NOTHING:
            self.action_done = True
            self.partner.action_done = True
        self.continue_action = during_of_action(self.current_action)
        self.partner.continue_action = during_of_action(self.current_action)

    def close_to_seat(self):
        return self.pos in self.model.grid.get_neighborhood(self.base_pos, moore=True, include_center=True,
                                                            radius=1)

    def contagious(self):
        # The current agent isn't contagious
        if self.health < HealthStatus.CARRIED:
            return
        # All the close cellmates
        cellmates = [c for c in
                     self.model.grid.get_cell_list_contents(self.model.grid.get_neighborhood(self.pos, moore=True,
                                                                                             include_center=True,
                                                                                             radius=1)) if
                     type(c) != DrawingOnGridAgent]
        for c in cellmates:
            # The cellmate is already contagious
            if c.health == HealthStatus.CARRIED and c.infection_generation <= self.infection_generation:
                continue
            # Calculate the threshold for contagious, according to the parameters below
            threshold = logistic_prob(factors=self.model.inf_coeff,
                                      variables=[1 / np.sqrt(self.infection_generation),
                                                 self.infects_others_level,
                                                 c.infects_by_others_level,
                                                 self.model.air_conditioning / AirStatus.AIR_RECYCLING],
                                      expected_value=1)

            # Interaction
            if self.partner != self:
                threshold = threshold * contagious_action(self.current_action) / \
                            ((1 - threshold) + threshold * contagious_action(self.current_action))

            # Mask reduction
            threshold *= mask_protection(self, c, self.model.infRate)
            if throw_coin(threshold):
                c.health = HealthStatus.CARRIED
                c.infection_generation = self.infection_generation + 1

    def wear_mask(self, area_size, cellmates):
        crowded = len(cellmates)
        threshold = logistic_prob(factors=self.model.mask_coeff,
                                  variables=[self.age / 100, self.social_influence,
                                             crowded / area_size,
                                             self.model.country_status / CountryStatus.HIGH_MORBIDITY,
                                             self.model.air_conditioning / AirStatus.AIR_RECYCLING, ])
        threshold *= influence_action_on_wearing_mask(self.current_action, self.model.conference)

        if throw_coin(threshold):
            self.mask = True
        else:
            self.mask = False

    def update_social_influence(self, cellmates):
        wearing_mask = sum([self.model.relationship_level[lower_first(self.unique_id, c.unique_id)]
                            if lower_first(self.unique_id, c.unique_id) != 0
                            else 1 for c in cellmates if c.mask and c != self])
        all = sum([self.model.relationship_level[lower_first(self.unique_id, c.unique_id)]
                   if lower_first(self.unique_id, c.unique_id) != 0
                   else 1 for c in cellmates if c != self])
        # The most of cellmates not wearing mask
        if wearing_mask < 0.5 * all:
            self.social_influence = -1 * np.random.random() * (1 - self.age / 100) + 1 * np.random.random() * (
                    self.age / 100)
        else:
            self.social_influence = np.random.normal(0.5, 0.2)

    def choose_chair(self):
        # Find the level of relationship between the agent and the agents around tables
        contact_level_chair = [sum([self.model.relationship_level[lower_first(self.unique_id, k)]
                                    for k in seat]) for seat in self.model.occupied_chairs if len(seat) <= 10]

        x = self.random.choices([(seat, i) for i, seat in
                                 enumerate(self.model.seating_area)
                                 if len(self.model.occupied_chairs[i]) <= 10],
                                weights=[prob / max(sum(contact_level_chair), 1) for prob in contact_level_chair])
        self.base_pos = x[0][0]
        self.model.occupied_chairs[x[0][1]].append(self.unique_id)

    def active_agent(self):
        if throw_coin(self.model.arrival_rate) and not self.active:
            self.active = True
            if len(self.model.seating_area) > 0:
                self.choose_chair()
        return self.active

    def step(self):
        if not self.active_agent():
            return
        possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True,
                                                          include_center=True,
                                                          radius=2)
        cellmates = [c for c in self.model.grid.get_cell_list_contents(possible_steps
                                                                       ) if type(c) != DrawingOnGridAgent]
        if self.close_to_seat():
            self.last_time_in_seat = self.model.time

        # Continue the previous action
        if throw_coin(self.continue_action):
            self.action_done = True
            self.partner.action_done = True
            self.continue_action *= during_of_action(self.current_action)

        # The agent has not performed any action yet
        if not self.action_done:
            decision_path = self.move(possible_steps, cellmates)
            print(decision_path)
            if self.model.actions:
                self.choose_action(decision_path, cellmates)

        self.update_social_influence(cellmates)
        self.wear_mask(len(possible_steps), cellmates)
        self.contagious()


def count_carried(model: "CoronaCloseModel"):
    return len([1 for k in [a for a in model.schedule.agents
                            if type(a) != DrawingOnGridAgent] if
                k.health == HealthStatus.CARRIED])


def count_mask(model: "CoronaCloseModel"):
    return len([1 for k in model.schedule.agents if type(k) != DrawingOnGridAgent and k.mask])


def move_forward(possible_steps, destination):
    return possible_steps[np.argmin([dist(pos, destination) for pos in possible_steps])]


def count_crowd(model):
    return len([1 for k in model.schedule.agents if model.pos_in_gathering_area(k.pos)
                and type(k) != DrawingOnGridAgent])


class WaiterAgent(PopAgent):
    def move(self, possible_steps, cellmates):
        if throw_coin((self.model.time - self.last_time_in_seat) / (self.model.time + self.last_time_in_seat)):
            new_position = move_forward(possible_steps, self.base_pos)
        else:
            new_position = move_forward(possible_steps, (self.model.grid.height, self.model.grid.width))
        self.model.grid.move_agent(self, new_position)

    def choose_action(self, move_path, cellmates):
        if self.close_to_seat():
            if throw_coin(0.5):
                self.current_action = Action.TALKING
                self.choose_interaction(cellmates)
                self.partner.current_action = Action.TALKING
                self.action_done = True
                self.partner.action_done = True
                self.continue_action = during_of_action(self.current_action, True)
                self.partner.continue_action = during_of_action(self.current_action, True)

    def wear_mask(self, area_size, cellmates):
        if (self.close_to_seat() and throw_coin(0.9)) or throw_coin(0.7):
            self.mask = True
        else:
            self.mask = False


class CoronaCloseModel(Model):
    def __init__(self, N: int, height: int, width: int, country_status: CountryStatus, air_condition: AirStatus,
                 inf_coeff=None, infRate=None, mask_coeff=None, entry_num=0, arrival_rate=0,
                 conference_area=None, relationship=False, get_away=False,
                 tables=False, waiters=False, actions=False, *args: Any, **kwargs: Any, ):
        super().__init__(*args, **kwargs)
        # Fixed system fields
        self.num_agents = N
        self.grid = MultiGrid(width, height, False)
        self.schedule = RandomActivation(self)
        self.running = True
        self.air_conditioning = air_condition
        self.country_status = country_status
        self.arrival_rate = arrival_rate
        self.get_away = get_away
        self.actions = actions

        self.gathering_area = {'min_x': conference_area[0], 'min_y': conference_area[1],
                               'max_x': conference_area[2],
                               'max_y': conference_area[3]}
        self.switch_time = max(conference_area[4], 1)
        self.entry_list = [(0, 0), (0, width - 1), (height - 1, 0), (height, width)][0:entry_num]

        # Changeable system fields
        self.time = 0
        self.conference = False
        self.conference_crowded = 0
        self.save_seven_days_before = Queue(5)
        self.R = 0

        # Coefficients for logistic probability
        self.mask_coeff = mask_coeff
        self.inf_coeff = inf_coeff
        self.infRate = infRate

        # Data Collectors
        self.datacollector = DataCollector(
            model_reporters={"Ills": count_carried},
            agent_reporters={"Health": "health"})

        self.datacollector_1 = DataCollector(
            model_reporters={"Crowd": count_crowd},
            agent_reporters={"Health": "health"})

        self.datacollector_2 = DataCollector(
            model_reporters={"Mask": count_mask},
            agent_reporters={"Health": "health"})

        self.datacollector_3 = DataCollector(
            model_reporters={"R Coeff.": self.get_R},
            agent_reporters={"Health": "health"})

        # Init agents
        # Init carried agents
        percent_ills = max(int(0.025 * float(self.country_status) * self.num_agents), 1)

        for i in range(percent_ills):
            a = PopAgent(i, self, health=HealthStatus.CARRIED)
            self.schedule.add(a)
            if len(self.entry_list) > 0:
                self.grid.place_agent(a, self.random.choice(self.entry_list))
            else:
                x = self.random.randrange(0, self.grid.height)
                y = self.random.randrange(0, self.grid.width)
                self.grid.place_agent(a, (x, y))

        # Init healthy agents
        for i in range(percent_ills, self.num_agents):
            a = PopAgent(i, self)
            self.schedule.add(a)
            if len(self.entry_list) > 0:
                self.grid.place_agent(a, self.random.choice(self.entry_list))
            else:
                x = self.random.randrange(0, self.grid.height)
                y = self.random.randrange(0, self.grid.width)
                self.grid.place_agent(a, (x, y))

        # Init gathering area agents for coloring
        pos_in_gathering_area = [(i, j) for i in range(self.gathering_area['min_x'],
                                                       self.gathering_area['max_x'])
                                 for j in range(self.gathering_area['min_y'],
                                                self.gathering_area['max_x'])]

        for i, k in enumerate(pos_in_gathering_area):
            a = DrawingOnGridAgent(i + self.num_agents * 2, self)
            self.schedule.add(a)
            self.grid.place_agent(a, k)

        # Init seating area
        self.seating_area = []
        if tables:
            self.init_seating_area()
            self.occupied_chairs = [[] for i in range(len(self.seating_area))]

        # Init waiters
        if waiters and tables:
            k = 3000
            for i, table in enumerate(self.seating_area):
                a = WaiterAgent(k + i, self)
                self.schedule.add(a)
                a.active = True
                a.base_pos = table
                self.grid.place_agent(a, table)

        # Init relationship level between agents
        # If declare homogeneous relationship, then the relationship of each couple equal to 0
        # else it define by number that distribute ~N(0.5,0.2)
        self.relationship_level = dict()
        people = [p for p in self.schedule.agents if type(p) != DrawingOnGridAgent]
        for couple in combinations(people, 2):
            if relationship and couple[0].unique_id < 3000 and couple[1].unique_id < 3000:
                self.relationship_level[couple[0].unique_id, couple[1].unique_id] = np.random.normal(0.5, 0.2)
            else:
                self.relationship_level[couple[0].unique_id, couple[1].unique_id] = 0

        # Choose move policy
        self.move_decision_tree = build_moving_decision_tree()

    def init_seating_area(self):
        k = 2000
        while len(self.seating_area) < int(self.num_agents / 10):
            x = self.random.randrange(0, self.grid.height)
            y = self.random.randrange(0, self.grid.width)
            if self.pos_in_gathering_area((x, y)) \
                    or len([1 for seat in self.seating_area
                            if seat in self.grid.get_neighborhood((x, y),
                                                                  moore=True, include_center=True, radius=3)]) > 0 \
                    or x == 0 or y == 0 or x + 2 >= self.grid.height or y + 1 >= self.grid.width:
                continue
            self.seating_area.append((x, y))
            a = DrawingOnGridAgent(k, self)
            b = DrawingOnGridAgent(k + 1, self)
            k += 2
            self.schedule.add(a)
            self.grid.place_agent(a, (x, y))
            self.schedule.add(b)
            self.grid.place_agent(b, (x + 1, y))
        self.occupied_chairs = [10] * len(self.seating_area)

    def pos_in_gathering_area(self, pos):
        x, y = pos
        return self.gathering_area['min_x'] <= x <= self.gathering_area['max_x'] and \
               self.gathering_area['min_y'] <= y <= self.gathering_area['max_y']

    def change_conference(self):
        self.conference = not self.conference

    def get_R(self):
        return self.R

    def clean_done_actions(self):
        for agent in self.schedule.agents:
            if type(agent) != DrawingOnGridAgent:
                agent.action_done = False

    def cal_R(self):
        if len(self.save_seven_days_before.queue) > 0:
            if self.save_seven_days_before.queue[0] > 0:
                self.R = count_carried(self) / self.save_seven_days_before.queue[0]

    def step(self) -> None:
        self.time += 1
        if self.time % self.switch_time == 0 and self.gathering_area['max_x'] > 0:
            self.change_conference()
        self.save_seven_days_before.insert(count_carried(self))
        self.cal_R()
        self.conference_crowded = count_crowd(self)
        self.schedule.step()
        self.datacollector.collect(self)
        self.datacollector_1.collect(self)
        self.datacollector_2.collect(self)
        self.datacollector_3.collect(self)
        if self.actions:
            self.clean_done_actions()


#
# gen = np.zeros(30)
# ills = np.zeros(200)
#
# for i in range(50):
#     print(i)
#     model = CoronaCloseModel(500, 30, 30, percent_ills=0.002)
#     for j in range(50):
#         model.step()
#         if j % 5 == 0:
#             ills[int(j / 5)] += count_carried(model) / 50
#
# print([round(k, 1) for k in ills])
# print([round(k, 1) for k in gen])


def pre_order_print(curr):
    print(curr.name)
    if type(curr) == DecisionNode:
        return
    pre_order_print(curr.left)
    pre_order_print(curr.right)

#
# x = build_moving_decision_tree_with_friends()
# pre_order_print(x.head)
