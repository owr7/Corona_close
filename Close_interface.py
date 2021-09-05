import PySimpleGUIWeb as sg
from Clean_viz import run_sim, CountryStatus
from CleanCloseCorona import AirStatus

default_dict = {
    # General
    "-numOfAgents-": 200,
    "-width-": 30,
    "-Height-": 30,
    "countryStatus": 0,
    "airCondition": 0,

    # Infect coeff.
    'infectGenerationW': 1,
    'infectsOthersW': 1,
    'infectedByOthersW': 1,
    'airConditionW': 1,

    # Mask effects
    "noMaskInfection": 1,
    "oneMaskInfectionI": 0.25,
    "bothMaskInfection": 0.1,

    # Wear mask coeff. for decision
    "wearMaskAgeW": 1,
    "wearMaskSocialInfW": 1,
    "wearMaskCrowdingW": 1,
    "wearMaskCountryStatusW": 1,
    "wearMaskAirCondition": 1,

    # Constrains
    "entryNum": 0,  # 0 means random start point to each agent
    "arrivalRate": 1,  # 1 means all comes together
    "conference_area": '0 0 0 0 0',
    "tables": 0,
    "waiters": 0,
    "actions": 0,

    # Constants
    'Low morbidity': CountryStatus.LOW_MORBIDITY,
    'Medium morbidity': CountryStatus.MIDDLE_MORBIDITY,
    'High morbidity': CountryStatus.HIGH_MORBIDITY,

    'Exchange': AirStatus.AIR_EXCHANGE,
    'Standing': AirStatus.STANDING_AIR,
    'Recycle': AirStatus.AIR_RECYCLING,

    # Expected value
    "numberOfSim": 10,
    "duringOfSim": 30,
}

general = [[
    sg.Text("General parameters", text_color='black', size=(500, 20))],
    [
        sg.Text("Number of agents:"),
        sg.In(size=(10, 1), enable_events=True, key="-numOfAgents-", default_text=default_dict['-numOfAgents-'])],
    [
        sg.Text("Width:"),
        sg.In(size=(10, 1), enable_events=True, key="-width-", default_text=default_dict['-width-'])],
    [
        sg.Text("Height:"),
        sg.In(size=(10, 1), enable_events=True, key="-Height-", default_text=default_dict['-Height-'])],
    [
        sg.Text("The status in the country (effects the wear mask decision and initial number of illness)",
                size=(200, 100)),
        sg.Listbox(values=['Low morbidity', 'Medium morbidity', 'High morbidity'], key='countryStatus',
                   size=(150, 78), default_values=['Low morbidity'], enable_events=True)
    ],
    [
        sg.Text("Ventilation status:"),
        sg.Listbox(values=['Exchange', 'Standing', 'Recycle'], key='airCondition', size=(150, 78),
                   default_values=['Exchange'], enable_events=True, )
    ],
]

probability_of_influence_without_mask = [
    [sg.Text("Probability of influence", text_color='black', size=(400, 20))],
    [sg.Text("We use logistic distribution for transport from linear\ncombination of "
             "parameters to probability value [0,1]\n"
             "Write weights(Not necessarily normalized) for following parameters:", size=(400, 100))],
    [
        sg.Text("Distance from the source of illness (The infectious person infected by others within the simulation):",
                size=(300, 100)),
        sg.In(size=(10, 1), enable_events=True, key="infectGenerationW", default_text=default_dict['infectGenerationW'])
    ],
    [
        sg.Text("Infection level (Infects others):"),
        sg.In(size=(10, 1), enable_events=True, key="infectsOthersW", default_text=default_dict['infectsOthersW'])
    ],
    [
        sg.Text("The level of infection (Infected by others):"),
        sg.In(size=(10, 1), enable_events=True, key="infectedByOthersW", default_text=default_dict['infectedByOthersW'])
    ],
    [
        sg.Text("Ventilation effect on infection:"),
        sg.In(size=(10, 1), enable_events=True, key="airConditionW", default_text=default_dict['airConditionW'])
    ],

]

probability_of_influence = [
    [sg.Text("Probability of protection of mask", text_color='black', size=(400, 20))],
    [
        sg.Text("Reduce of chances of infection with one mask:"),
        sg.In(size=(10, 1), enable_events=True, key="oneMaskInfectionI", default_text=default_dict['oneMaskInfectionI'])
    ],
    [
        sg.Text("Reduce of Chances of infection with one mask on both:"),
        sg.In(size=(10, 1), enable_events=True, key="bothMaskInfection", default_text=default_dict['bothMaskInfection'])
    ],
]

probability_of_wearing_mask = [
    [sg.Text("Level of discipline", text_color='black')],
    [sg.Text("We use logistic distribution for transport from linear\ncombination of "
             "parameters to probability value [0,1]\n"
             "Write weights(Not necessarily normalized) for following parameters:", size=(400, 100))],
    [
        sg.Text("Age:"),
        sg.In(size=(10, 1), enable_events=True, key="wearMaskAgeW", default_text=default_dict['wearMaskAgeW'])
    ],
    [
        sg.Text("Social influence:"),
        sg.In(size=(10, 1), enable_events=True, key="wearMaskSocialInfW",
              default_text=default_dict['wearMaskSocialInfW'])
    ],
    [
        sg.Text("Crowding:"),
        sg.In(size=(10, 1), enable_events=True, key="wearMaskCrowdingW", default_text=default_dict['wearMaskCrowdingW'])
    ],
    [
        sg.Text("Country status:"),
        sg.In(size=(10, 1), enable_events=True, key="wearMaskCountryStatusW",
              default_text=default_dict['wearMaskCountryStatusW'])
    ],
    [
        sg.Text("Ventilation status:"),
        sg.In(size=(10, 1), enable_events=True, key="wearMaskAirCondition",
              default_text=default_dict['wearMaskAirCondition'])
    ],
]

constrains = [
    [sg.Text("Add constrains", text_color='black')],
    [sg.Text('Number of Entries (0 means random start point for each agent)', size=(300, 50)),
     sg.In(size=(10, 1), enable_events=True, key="entryNum", default_text=default_dict['entryNum'])],

    [sg.Text('Arrival rate of agent to the area (0 - never, 1 - immediately)'),
     sg.In(size=(10, 1), enable_events=True, key="arrivalRate", default_text=default_dict['arrivalRate'])],

    [sg.Text('Coordinates of conference area. Input format is: x min, y min, x max, y max, time. If homogeneous area '
             'is '
             'required, '
             'then insert 0 0 0 0 0', size=(320, 70)),
     sg.In(size=(10, 1), enable_events=True, key="conference_area", default_text=default_dict['conference_area'])],

    [sg.Text('Save distance: The agent will try to get away from crowd',
             size=(200, 30), text_color='#CCCCCC')],
    [sg.Checkbox('Add move away action', key='getAwayFromCrowd', enable_events=True)],

    [sg.Text('Relationship: Agents has different relationship (Normal distribution)',
             size=(200, 30), text_color='#CCCCCC')],
    [sg.Checkbox('Add relationship between agents', key='relationship', enable_events=True)],

    [sg.Text('Tables: Small gathering areas with 10 agent around', size=(200, 30), text_color='#CCCCCC')],
    [sg.Checkbox('Add tables', key='tables', enable_events=True)],

    [sg.Text('Waiters: Go towards and return tables', size=(200, 30), text_color='#CCCCCC')],
    [sg.Checkbox('Add waiters', key='waiters', enable_events=True)],

    [sg.Text('Actions: The agents acting actions that effects their behavior maintaining health and level of infection',
             size=(200, 30), text_color='#CCCCCC')],
    [sg.Checkbox('Add actions', key='actions', enable_events=True)],
]

which_data_to_show = [
    [sg.Text('Choose the online data you want to watch', text_color='Black')],
    [
        sg.Checkbox("Number Of Ills", key='numOfIllsCB', default=True)],
    [sg.Checkbox("R Coeff.", key='RmeanCB', default=True)],
    [sg.Checkbox("Number of agent at the gathering area", key='gatheringAreaCB', default=True)],
    [sg.Checkbox("Number Of Wearing Mask.", key='wearingMaskCB', default=True)],
]

send_button = [[sg.Button(button_text='Start Simulation', key='-SUBMIT-')]]

run_avg_simulation = [[sg.Text('No visual version', text_color='black')],
                      [sg.Text(
                          'If you want to get result of average of some simulation,\nplease provide the next,'
                          '\nand press the button below', size=(500, 50))],
                      [sg.Text('Number of simulations:'),
                       sg.In(size=(10, 1), enable_events=True, key="numberOfSim",
                             default_text=default_dict['numberOfSim'])],
                      [sg.Text('During of each simulation:'),
                       sg.In(size=(10, 1), enable_events=True, key="duringOfSim",
                             default_text=default_dict['duringOfSim'])],
                      [sg.Button(button_text='Take Average on Simulations', key='-SUBMIT_AVG-')]]


layout = [[sg.Column(general + probability_of_influence_without_mask + probability_of_influence +
                     probability_of_wearing_mask),
           sg.Column(constrains + which_data_to_show + send_button + run_avg_simulation)]]

window = sg.Window("Corona Simulation", layout)


def input_check():
    for value in values:
        if values[value] == '':
            values[value] = default_dict[value]


while True:
    event, values = window.read()
    if event == '-SUBMIT-' or event == '-SUBMIT_AVG-':
        input_check()

        if values['countryStatus'][0] is None:
            values['countryStatus'][0] = 'Low morbidity'
        if values['airCondition'][0] is None:
            values['airCondition'][0] = 'Exchange'

        infection = [float(values['infectGenerationW']), float(values['infectsOthersW']),
                     float(values['infectedByOthersW']), float(values['airConditionW'])]

        infRate = [float(values['bothMaskInfection']), float(values['oneMaskInfectionI']),
                   float(values['oneMaskInfectionI']), default_dict['noMaskInfection']]

        mask_coeff = [float(values['wearMaskAgeW']), float(values['wearMaskSocialInfW']),
                      float(values['wearMaskCrowdingW']), float(values['wearMaskCountryStatusW']),
                      float(values['wearMaskAirCondition'])]

        conference_area = [int(k) for k in values['conference_area'].split()]
        constrains = [float(values['entryNum']), float(values['arrivalRate']),
                      conference_area, values['tables'],
                      values['waiters'], values['actions']]

        show_online_data = [float(values['numOfIllsCB']),
                            float(values['RmeanCB']),
                            float(values['gatheringAreaCB']),
                            float(values['wearingMaskCB'])]

        run_sim(num_agents=int(values['-numOfAgents-']), height=int(values['-Height-']), width=int(values['-width-']),
                country_status=default_dict[values['countryStatus'][0]],
                air_condition=default_dict[values['airCondition'][0]], inf_coeff=infection, infRate=infRate,
                mask_coeff=mask_coeff, show_online_data=show_online_data, entry_num=int(values['entryNum']),
                arrival_rate=float(values['arrivalRate']), conference_area=conference_area,
                relationship=values['relationship'], get_away=values['getAwayFromCrowd'], tables=values['tables'],
                waiters=values['waiters'], actions=values['actions'], avg_sim=(event == '-SUBMIT_AVG-'),
                num_sim=int(values['numberOfSim']), during_sim=int(values['duringOfSim'])
                )
        break
    if event == "Exit" or event == sg.WIN_CLOSED:
        break

window.close()
