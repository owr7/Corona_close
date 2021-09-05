import matplotlib

from CleanCloseCorona import *
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
import PySimpleGUIWeb as sg
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
import io
import time


results_names = {
    1: "Ills Graph (per day)",
    2: "Number Of People In Conference Area Graph (cumulative)",
    3: "Number of People Wearing Mask Graph (cumulative)",
}


def agent_portrayal(agent):
    if type(agent) == DrawingOnGridAgent:
        portrayal = {"Shape": "rect",
                     "Filled": "true",
                     "h": 1, "w": 1}
        if agent.model.pos_in_gathering_area(agent.pos) and agent.model.conference:
            portrayal['Color'] = '#550000'
            portrayal['Layer'] = 0
        elif agent.pos in agent.model.seating_area:
            portrayal['Color'] = '#000055'
            portrayal['Layer'] = 0
        else:
            portrayal['Color'] = '#FFFFFF'
            portrayal['Layer'] = 0
    else:
        if type(agent) == PopAgent:
            portrayal = {"Shape": "circle",
                         "Filled": "true",
                         "r": 0.5}
        else:
            portrayal = {"Shape": "rect",
                         "Filled": "true",
                         "h": 0.5, "w": 0.5}
        if not agent.active:
            portrayal['Color'] = '#FFFFFF'
            portrayal['Layer'] = 1
        elif agent.health == HealthStatus.HEALTHY:
            portrayal['Color'] = '#00FF00'
            portrayal['Layer'] = 1
        elif agent.health == HealthStatus.RECOVERY:
            portrayal['Color'] = '#00FF00'
            portrayal['Layer'] = 2
        elif agent.health == HealthStatus.CARRIED:
            if agent.infection_generation == 1:
                portrayal['Color'] = '#FF0000'
                portrayal['Layer'] = 3
            else:
                portrayal['Color'] = '#FF00FF'
                portrayal['Layer'] = 3

    return portrayal


def run_avg_sim(model: CoronaCloseModel, num_of_simulations: int, during_of_simulation: int, data_avg, data_var,
                start_hour):
    ills = data_avg[0]
    conference = data_avg[1]
    mask = data_avg[2]

    ills_var = data_var[0]
    conference_var = data_var[1]
    mask_var = data_var[2]

    avg_time_for_step = 0
    for i in range(during_of_simulation):
        start = time.time()
        model.step()
        # Average
        ills[i] += count_carried(model) / num_of_simulations
        conference[i] += count_crowd(model) / num_of_simulations
        mask[i] += count_mask(model) / num_of_simulations

        # Variance
        ills_var[i] += count_carried(model)**2 / num_of_simulations
        conference_var[i] += count_crowd(model)**2 / num_of_simulations
        mask_var[i] += count_mask(model)**2 / num_of_simulations

        avg_time_for_step = avg_time_for_step * i / (i + 1) + (time.time() - start) * 1 / (i + 1)
        finish_at = time.localtime(start_hour + num_of_simulations * during_of_simulation * avg_time_for_step)
        sg.Window("Corona Simulation", [[sg.Text('Finish Time:' + str(finish_at.tm_hour)
                                                 + ':' + str(finish_at.tm_min) + ':' +
                                                 str(finish_at.tm_sec))]]).show()


def draw_figure(canvas, figure):
    plt.close('all')  # erases previously drawn plots
    canv = FigureCanvasAgg(figure)
    buf = io.BytesIO()
    canv.print_figure(buf, format='png')
    if buf is None:
        return None
    buf.seek(0)
    canvas.update(data=buf.read())
    return canv


def show_avg_graphs(figs):
    layout = [[sg.Text('Results of average simulation', text_color='black')],
              [sg.Column([[sg.Text(figs[0][0])],
                          [sg.Image(key='CANVAS_1')],
                          [sg.Text(figs[1][0])],
                          [sg.Image(key='CANVAS_2')],
                          [sg.Text(figs[2][0])],
                          [sg.Image(key='CANVAS_3')], ])],
              [sg.Button('OK')]]

    # create the form and show it without the plot
    window = sg.Window('Results of average simulation', layout, finalize=True,
                       element_justification='center', font='Helvetica 18', )

    # add the plot to the window
    for i, fig in enumerate(figs):
        fig_canvas_agg = draw_figure(window['CANVAS_' + str(i + 1)], fig[1])
    event, values = window.read()

    window.close()


def run_sim(num_agents: int, height: int, width: int, country_status: CountryStatus, air_condition: AirStatus,
            inf_coeff=None, infRate=None, mask_coeff=None, show_online_data=None, entry_num=0, arrival_rate=0,
            conference_area=None, relationship=False, get_away=False, tables=False, waiters=False, actions=False,
            avg_sim=False, num_sim=10, during_sim=30):
    if avg_sim:
        data_avg = [np.zeros(during_sim) for i in range(3)]
        data_var = [np.zeros(during_sim) for i in range(3)]
        start_hour = time.time()
        for i in range(num_sim):
            model = CoronaCloseModel(N=num_agents, width=width, height=height, inf_coeff=inf_coeff,
                                     infRate=infRate, mask_coeff=mask_coeff,
                                     entry_num=entry_num, country_status=country_status, air_condition=air_condition,
                                     get_away=get_away, conference_area=conference_area, relationship=relationship,
                                     tables=tables, waiters=waiters, actions=actions, arrival_rate=arrival_rate)
            run_avg_sim(model, num_sim, during_sim, data_avg, data_var, start_hour)

        figs = []
        for j, (avg, var) in enumerate(zip(data_avg, data_var)):
            fig = matplotlib.figure.Figure(figsize=(5, 4), dpi=100)
            # fig.add_subplot(111).plot(list(range(during_sim)), k)
            fig.add_subplot(111).errorbar(list(range(during_sim)), avg,
                                          yerr=[np.sqrt(v - a**2) for a, v in zip(avg, var)])
            figs.append((results_names[j + 1], fig))
        show_avg_graphs(figs)
        return

    charts = [ChartModule([{"Label": "Ills",
                            "Color": "red"}],
                          data_collector_name='datacollector'),

              ChartModule([{"Label": "Crowd",
                            "Color": "Black"}],
                          data_collector_name='datacollector_1'),

              ChartModule([{"Label": "Mask",
                            "Color": "green"}],
                          data_collector_name='datacollector_2'),
              ChartModule([{"Label": "R Coeff.",
                            "Color": "orange"}],
                          data_collector_name='datacollector_3'),]

    grid = CanvasGrid(agent_portrayal, height, width, 400, 400)
    chart_list = [grid] + [chart for chart, cbox in zip(charts, show_online_data) if cbox == 1]

    server = ModularServer(CoronaCloseModel,
                           chart_list,
                           "Corona Model",
                           {"N": num_agents, "width": width, "height": height, 'country_status': country_status,
                            'air_condition': air_condition,
                            'inf_coeff': inf_coeff, 'infRate': infRate, 'mask_coeff': mask_coeff,
                            'entry_num': entry_num, 'arrival_rate': arrival_rate, 'conference_area': conference_area,
                            'relationship': relationship, 'get_away': get_away, 'tables': tables, 'waiters': waiters,
                            'actions': actions})

    server.port = 8521  # The default
    server.launch()

#
# model = CoronaCloseModel(N=200, height=30, width=30, percent_ills=0.05)
#
# grid = CanvasGrid(agent_portrayal, model.grid.height, model.grid.width, 400, 400)
#
# chart = ChartModule([{"Label": "Ills",
#                       "Color": "red"}],
#                     data_collector_name='datacollector')
#
# chart_1 = ChartModule([{"Label": "Crowd",
#                         "Color": "Black"}],
#                       data_collector_name='datacollector_1')
#
# chart_2 = ChartModule([{"Label": "Mask",
#                         "Color": "green"}],
#                       data_collector_name='datacollector_2')
#
# server = ModularServer(CoronaCloseModel,
#                        [grid, chart, chart_1, chart_2],
#                        "Corona Model",
#                        {"N": model.num_agents, "width": model.grid.width, "height": model.grid.height,
#                         })
#
# server.port = 8521  # The default
# server.launch()
