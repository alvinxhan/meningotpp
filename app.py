from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import apputils

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pandas as pd
import numpy as np
import itertools
from scipy import stats

## -- Initialize app -- ##
# pastorex performance
pastorex_sens = 0.8
pastorex_spec = 0.95

# read master result file
master_result_df = pd.read_csv("./data/master_result_df.csv").set_index("paridx")

# get operating parameters
operating_params_df, baseline_trans_multiplier_arr, sus_per_inv_arr, test_sens_arr, test_spec_arr, test_receptiveness_arr, test_turnaround_time_arr, react_vacc_threshold_arr, react_vacc_turnaround_time_arr, ini_react_vacc_arr = apputils.get_operating_params(list(master_result_df.index.unique()))

# limit ini_react_vacc_arr for now
ini_react_vacc_arr = ini_react_vacc_arr[:1]

external_stylesheets = [dbc.themes.BOOTSTRAP, {
    'href': 'https://fonts.googleapis.com/css2?family=Lato:wght@300&display=swap',
    'rel': 'stylesheet'
}]

app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
server = app.server

## -- Sidebar -- ##
SIDEBAR_STYLE = {
    "position":'fixed',
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "400px",
    "padding":"10px",
    "overflow-y": "scroll",
    "zIndex": 1,
    "background-color": "#f8f9fa",
}
translabels = {0.5:"(Sporadic invasive cases only)", 1.0:"(NmC outbreak in Niger, 2015)", 2.0:"(2x transmissibility as per Niger, 2015)"}
sidebar = html.Div([
    ## -- Disease-related parameters -- ##
    html.H6("Disease"),
    #
    dbc.Label("Relative transmissibility:", style={"font-size":12}),
    dcc.RadioItems(
        id='baseline_trans_multiplier',
        options=[{
            'label':'%.1fx %s'%(val, translabels[val]), 'value':val} for val in baseline_trans_multiplier_arr
            ],
        value=1.0,
        inline=True,
        inputStyle={"margin-right": "10px"},
        labelStyle={'display':'inline-block', 'margin-right': '20px', 'font-size':12}
    ),

    dbc.Label("No. of suspected cases per invasive case", style={"font-size":12}),
    dcc.Slider(
        id="sus_per_inv",
        min=min(sus_per_inv_arr),
        max=max(sus_per_inv_arr),
        step=None,
        marks={val:"%i"%(val) for i, val in enumerate(sus_per_inv_arr)},
        tooltip={"placement":"top", "always_visible":True},
        value=sus_per_inv_arr[0]
    ),

    html.Hr(),
    ## -- Diagnostic test related parameters -- ##
    html.H6("Diagnostic test"),
    dbc.Label("Choose data presentation:", style={"font-size":12}),
    dcc.RadioItems(
        id='display_option',
        options=[
            {'label':'Fix either test sensitivity or specificity', 'value': 'fixss'},
            #{'label':'Compare full ranges of test sensitivity and specificity', 'value': 'compss'}
        ],
        value='fixss',
        inline=True,
        inputStyle={"margin-right": "10px"},
        labelStyle={'display':'inline-block', 'margin-right': '20px', 'font-size':12}
    ),

    html.Div(id='input-container1'),
    html.Div(id='input-container2'),

    dbc.Label("Turnaround time (days) for RDT:", style={"font-size":12}),
    dcc.Slider(
        id="test_turnaround_time",
        min=min(test_turnaround_time_arr),
        max=max(test_turnaround_time_arr),
        step=None,
        marks={val:"%i"%(val) for i, val in enumerate(test_turnaround_time_arr)},
        tooltip={"placement":"top", "always_visible":True},
        value=test_turnaround_time_arr[0],
    ),

    dbc.Label("Turnaround time (days) for Pastorex:", style={"font-size":12}),
    dcc.Slider(
        id="pastorex_turnaround_time",
        min=min(test_turnaround_time_arr),
        max=max(test_turnaround_time_arr),
        step=None,
        marks={val:"%i"%(val) for i, val in enumerate(test_turnaround_time_arr)},
        tooltip={"placement":"top", "always_visible":True},
        value=test_turnaround_time_arr[-1],
    ),

    html.Br(),
    dbc.Label("Cost per rapid diagnostic test (RDT) = $", style={"font-size":12}),
    dcc.Input(
        id='cost_per_test',
        min=0,
        max=int(1e6),
        step=0.01,
        value=4,
        type='number'
    ),
    html.Br(),
    dbc.Label("Cost per Pastorex latex agglutination test = $", style={"font-size":12}),
    dcc.Input(
        id='cost_per_pastorex',
        min=0,
        max=int(1e6),
        step=0.01,
        value=15,
        type='number'
    ),

    html.Hr(),
    ## -- Vaccination-related parameters -- ##
    html.H6("Vaccination"),
    dbc.Label("Proportion of individuals vaccinated (Initial (i.e. before epidemic) and during reactive vaccination campaign)", style={"font-size":12}),
    dcc.RadioItems(
        id='ini_react_vacc_combo',
        options=[
            {'label':'Initial: %i%%; Reactive: %i%%'%(inip, reap), 'value': '%i-%i'%(inip, reap)} for (inip, reap) in ini_react_vacc_arr
            #{'label':'Compare full ranges of test sensitivity and specificity', 'value': 'compss'}
        ],
        value='0-100',
        inline=True,
        inputStyle={"margin-right": "10px"},
        labelStyle={'display':'inline-block', 'margin-right': '20px', 'font-size':12}
    ),
    html.Br(),
    dbc.Label("React vacc. threshold (cumm. cases/100,000 people)", style={"font-size":12}),
    dcc.Slider(
        min=min(react_vacc_threshold_arr),
        max=max(react_vacc_threshold_arr),
        step=None,
        marks={val:"%i"%(val) for i, val in enumerate(react_vacc_threshold_arr) if i == 0 or val%2==0},
        tooltip={"placement":"top", "always_visible":True},
        value=10,
        id="react_vacc_threshold"
    ),
    dbc.Label("Vacc. turnaround time (days)", style={"font-size":12}),
    dcc.Slider(
        min=min(react_vacc_turnaround_time_arr),
        max=max(react_vacc_turnaround_time_arr),
        step=None,
        marks={val:"%i"%(val) for i, val in enumerate(react_vacc_turnaround_time_arr)},
        tooltip={"placement":"top", "always_visible":True},
        value=7,
        id="react_vacc_turnaround_time"
    ),
], style=SIDEBAR_STYLE)

# intro test
results_intro_text = '''
## Modelling to support rapid diagnostic development for meningitis

This Dashboard summarizes the expected impact and costs of using a rapid diagnostic test (RDT) to trigger reactive vaccination campaign for a **population of 1,000,000 individuals over a period of 24 weeks** (i.e. a single dry season lasting 6 months in the meningitis belt). Use the menu on the left to explore how different disease, diagnostic and vaccination parameters can change the epidemic and testing outcomes.

Results shown here are based on a few key assumptions:
- Epidemic intensity is calibrated to reflect similar invasive case rates as per the 2015 *Neisseria meningitidis* (*Nm*) serogroup C outbreak ([Sidikou et al. (2017)](https://doi.org/10.1016/S1473-3099(16)30253-5)). In other words, **results shown in this dashboard are based on a non-*Nm* serogroup A outbreak**.
- The demography of the simulated population, which underlies the contact rates between individuals, is based on Niger.
- Disease confirmation is based only on a single test using the RDT.
- **Testing is performed to trigger reactive vaccination only (the ONLY intervention) and is NOT used for clinical management or non-vaccine based infection control**.
- Reactive vaccination is only targeted at individuals aged between 0 and 29 years of age.
'''

methods_text1 = '''
### Model

Results presented in this Dashboard is generated from **MeningoPATAT**, an agent-based microsimulation model that simulates a bacterial meningitis outbreak, and the use of diagnostic tests to determine the initiation of a reactive vaccination campaign. The model uses demographic data (age, average household size, schooling and employment rates) to parameterize contact rates between individuals. Disease progression (Figure 1) and transmission models embedded within MeningoPATAT are adapted from [Yaesoubi et al. (2018)](https://doi.org/10.1371/journal.pmed.1002495).
'''

methods_text2 = '''
An infected individual will become an infectious carrier but only a proportion of carriers will progress to invasive disease lasting 7 days. Death rate of individuals with invasive disease is assumed to be fixed at 10%. Transmission parameters (e.g. age group-specific transmissiblity and susceptibility), rate of progression to invasive disease, as well as the duration of carriage and recovered states were estimated by calibration. This was done by fitting the simulated epidemic intensitiy, including the number of suspected cases per invasive case, against data collected in Niger during the 2015 *Neisseria meningitidis* serogroup C outbreak ([Sidikou et al. (2017)](https://doi.org/10.1016/S1473-3099(16)30253-5)).
'''

methods_text3 = '''
From the fitted epidemic wave (i.e. 1x in Figure 2), we also simulated two other scenarios where either transmissiblity is twice (2x) of what was observed in Niger, 2015 or that it is halved (0.5x) such that there are only sporadic cases but no actual infection wave.

Currently, we simulated 10 independent epidemic runs for every set of input parameters, including the baseline scenario where there was no reactive vaccination campaign.

### Testing and reactive vaccination
A percentage of samples collected from suspected cases will be processed for testing using a diagnostic test either a rapid diagnostic with the desired sensitivity and specificity or Pastorex ([Uadiale et al. (2016)](https://academic.oup.com/trstmh/article/110/7/381/2427454?login=false)). **We assumed this is the only test used for disease confirmation**. A reactive vaccination campaign targeted at individuals aged between 0 and 29 years is triggered when the cummulative number of positive cases (inclusive of both true and false positives) crosses a pre-specified threshold. The vaccine is assumed to confer 85% protection against infection.
'''

# panel text
panel1_text = '''
#### Invasive cases and deaths averted
The expected proportion of invasive cases and deaths averted during the 6-month period is plotted. Hover over the data points to see the expected number of invasive cases or deaths per million individuals during the period. **For line-plots, the shaded area denotes the 95% confidence interval of the expected epidemic outcome; the wider the shaded area, the more uncertain we are about computed outcome**.
'''

inset_text = '''
#### Accuracy of estimated disease prevalence and likelihood of triggering reactive vaccination
The left plot below shows the average accuracy of the prevalence estimate as measured by its root-mean-squared-error (RMSE) to the true prevalence (i.e. all infected, including carriage and invasive cases, at each point in time) over the entire six-month period. **The larger the RMSE, the less accurate is the disease prevalence estimate using the RDT**.

The right plot shows the expected probability of triggering a reactive vaccination campaign (i.e. cummulative positive cases meeting the reactive vaccination threshold). **The larger the probability, the more likely a reactive vaccination campaign will be triggered when using the RDT**.
'''

inset_text2 = '''
The plot below shows the expected week during which the reactive vaccination threshold is met (i.e 5 or 10 cummulative positive test per 100,000 people). 24 epi-weeks were simulated. As such, if the computed expected week is on or after 24 weeks, the reactive vaccination campaign was not carried out.
'''

panel2_text = '''
#### Testing outcomes and costs
The plots below show the total number of tests administered for 1,000,000 individuals during a six-month period, the costs of testing per correct diagnosis, and the total testing costs incurred for different amount of samples tested. **Error whiskers in cost plots denotes the standard deviation around the expected value; the wider it is, the more uncertain the cost estimate**.
'''

result_page = dbc.Container([
    dbc.Row(
        dcc.Markdown(children=results_intro_text),
    ),
    dcc.Markdown(children=panel1_text),
    dbc.Row(
        dcc.Loading(id='panel1',
            children=[dcc.Graph(id="fig_invdea", style={"width": "100%"})],
            type='default'),
        justify="center",
        align="center",
        style={"overflowX": "hidden", "position": "relative", "zIndex": 0},
    ),
    dcc.Markdown(children=inset_text),
    dbc.Row(
       dcc.Loading(id='panel5',
        children=[dcc.Graph(id="fig_prev_prec", style={"width": "100%"})],
        type='default'),
       justify="center",
       align="center",
       style={"overflowX": "hidden", "position": "relative", "zIndex": 0},
    ),
    dcc.Markdown(children=inset_text2),
    dbc.Row(
       dcc.Loading(id='panel6',
        children=[dcc.Graph(id="fig_vacc_day", style={"width": "100%"})],
        type='default'),
       justify="center",
       align="center",
       style={"overflowX": "hidden", "position": "relative", "zIndex": 0},
    ),
    dcc.Markdown(children=panel2_text),
    dbc.Row(
       dcc.Loading(id='panel2',
        children=[dcc.Graph(id="fig_n_tests", style={"width": "100%"})],
        type='default'),
       justify="center",
       align="center",
       style={"overflowX": "hidden", "position": "relative", "zIndex": 0},
    ),
    dbc.Row(
       dcc.Loading(id='panel3',
        children=[dcc.Graph(id="fig_costs_per_corr_diag", style={"width": "100%"})],
        type='default'),
       justify="center",
       align="center",
       style={"overflowX": "hidden", "position": "relative", "zIndex": 0},
    ),
    dbc.Row(
       dcc.Loading(id='panel4',
        children=[dcc.Graph(id="fig_total_test_costs", style={"width": "100%"})],
        type='default'),
       justify="center",
       align="center",
       style={"overflowX": "hidden", "position": "relative", "zIndex": 0},
    ),
],
    fluid=True,
    style={
        #"marginLeft": "400px",
        "padding": "20px",
        "overflowX": "hidden",
        "position": "relative",
        "zIndex": 0,
        "maxWidth": "calc(100vw - 400px)"
},)

methods_page = dbc.Container([
    dbc.Row(
        dcc.Markdown(children=methods_text1),
        ),
    html.Div([
        html.Img(
            src='assets/DiseaseProgress.png',
            style={
                'height': '50%',
                'width': '50%'
            }),
        html.Br(),
        dbc.Label("Figure 1: Disease progression of those infected with bacterial meningitis.", style={"font-size":12}),
        html.Br(),
        ],
        style={'textAlign': 'center'}
    ),
    dbc.Row(
        dcc.Markdown(children=methods_text2),
        ),
    html.Div([
        html.Img(
            src='assets/calibration.png',
            style={
                'height': '50%',
                'width': '50%'
            }),
        html.Br(),
        dbc.Label("Figure 2: Number of invasive cases per million individuals predicted by MeningoPATAT for different transmissibility values relative to that observed for the  2015 Neisseria meningitidis (Nm) serogroup C outbreak in Niger.", style={"font-size":12}),
        html.Br(),
        ],
        style={'textAlign': 'center'}
    ),
    dbc.Row(
        dcc.Markdown(children=methods_text3),
        ),
    ],
    fluid=True,
    style={
        #"marginLeft": "400px",
        "padding": "20px",
        "overflowX": "hidden",
        "position": "relative",
        "zIndex": 0,
        "maxWidth": "calc(100vw - 400px)"
    }
)

# Define the main content page
content = html.Div([
    dcc.Tabs([
        dcc.Tab(
            label = 'Results',
            children = [result_page]
            ),
        dcc.Tab(
            label = 'Methods',
            children = [methods_page]
            )
        ]),
    ],
        style={
            "marginLeft": "400px",
            #"padding": "20px",
            "overflowX": "hidden",
            "position": "relative",
            "zIndex": 0,
            "maxWidth": "calc(100vw - 400px)"
    })


# Put everything together in the layout
app.layout = html.Div([
    sidebar,
    content
    ],
    style={"overflowX": "hidden"})

## alternate display options
@app.callback(
    Output('input-container1', 'children'),
    Input('display_option', 'value')
)
def update_input_container(selected_radio_item):
    if selected_radio_item == 'fixss':
        # fix either sensitivity of specificity
        subcontainer = html.Div([
            dbc.Label("Fix:", style={"font-size":12}),
            dcc.RadioItems(
                id='par1',
                options=[
                    {'label':'Sensitivity', 'value': 'test_sens'},
                    {'label':'Specificity', 'value': 'test_spec'}
                ],
                value='test_sens',
                inline=True,
                inputStyle={"margin-right": "10px"},
                labelStyle={'display':'inline-block', 'margin-right': '20px', 'font-size':12}
            ),
        ])
    elif selected_radio_item == 'compss':
        subcontainer = html.Div([
            dbc.Label("Processed samples (%)", style={"font-size":12}),
            dcc.Slider(
                id="par1",
                min=min(test_receptiveness_arr),
                max=max(test_receptiveness_arr),
                step=None,
                value=test_receptiveness_arr[0],
                marks={val:"%i%%"%(val) for i, val in enumerate(test_receptiveness_arr)},
                tooltip={"placement":"top", "always_visible":True},
            ),
        ])

    return subcontainer

@app.callback(
    Output('input-container2', 'children'),
    Input('par1', 'value'),
    State('display_option', 'value')
)
def update_input_container2(fixed_par, display_option):
    if display_option == 'fixss':
        if fixed_par == 'test_sens':
            marks = {val:"%i%%%s"%(val, "*" if val == 80 else "") for i, val in enumerate(test_spec_arr)}
            value = int(pastorex_sens*100)
        else:
            marks = {val:"%i%%%s"%(val, "*" if val == 95 else "") for i, val in enumerate(test_sens_arr)}
            value = int(pastorex_spec*100)

        subcontainer = html.Div([
            dcc.Slider(
                id="par2",
                min=60,
                max=100,
                step=None,
                marks=marks,
                tooltip={"placement":"top", "always_visible":True},
                value=value,
            ),
            dbc.Label("*%s for Pastorex"%('Sensitivity' if fixed_par == 'test_sens' else 'Specificity'), style={"font-size":10})
        ])
    else:
        subcontainer = html.Div([
            dcc.Store(
                # store dummy variable for par2
                id='par2',
                data='dummy'
            )
        ])
    return subcontainer

##
@app.callback(
    Output('fig_invdea', 'figure'),
    Output('fig_n_tests', 'figure'),
    Output('fig_costs_per_corr_diag', 'figure'),
    Output('fig_total_test_costs', 'figure'),
    Output('fig_prev_prec', 'figure'),
    Output('fig_vacc_day', 'figure'),
    [Input('baseline_trans_multiplier', 'value'),
     Input('sus_per_inv', 'value'),
     Input('test_turnaround_time', 'value'),
     Input('pastorex_turnaround_time', 'value'),
     Input('cost_per_test', 'value'),
     Input('cost_per_pastorex', 'value'),
     Input('react_vacc_threshold', 'value'),
     Input('react_vacc_turnaround_time', 'value'),
     Input('par1', 'value'),
     Input('par2', 'value'),
     Input('ini_react_vacc_combo', 'value')],
    State('display_option', 'value'))
def update_output(baseline_trans_multiplier, sus_per_inv, test_turnaround_time, pastorex_turnaround_time, cost_per_test, cost_per_pastorex, react_vacc_threshold, react_vacc_turnaround_time, par1, par2, ini_react_vacc_combo, display_option):

    if display_option == 'fixss':
        # generate output fixing either sensitivity or specificity
        par2 = par2/100
        fig_invdea, fig_n_tests, fig_costs_per_corr_diag, fig_total_test_costs, fig_prev_prec, fig_vacc_day = generate_fixss_page(baseline_trans_multiplier, sus_per_inv, test_turnaround_time, pastorex_turnaround_time, cost_per_test, cost_per_pastorex, react_vacc_threshold, react_vacc_turnaround_time, par1, par2, ini_react_vacc_combo)

    #elif display_option == 'compss':
    return fig_invdea, fig_n_tests, fig_costs_per_corr_diag, fig_total_test_costs, fig_prev_prec, fig_vacc_day

# generate output fixing either sensitivity or specificity
def generate_fixss_page(baseline_trans_multiplier, sus_per_inv, test_turnaround_time, pastorex_turnaround_time, cost_per_test, cost_per_pastorex, react_vacc_threshold, react_vacc_turnaround_time, fixed_par, fixed_val, ini_react_vacc_combo):

    ## ------ SUMMARIZE RESULTS ------ ##
    ini_vacc_prop, react_vacc_prop = list(map(lambda x: int(x)/100, ini_react_vacc_combo.split('-')))
    # filter disease params
    filtered_par_df = operating_params_df[np.isclose(operating_params_df['baseline_trans_multiplier'], baseline_trans_multiplier)]
    filtered_par_df = filtered_par_df[filtered_par_df['sus_per_inv'] == sus_per_inv]
    # get baseline par idx
    base_par_idx = filtered_par_df[filtered_par_df['testing_bool']<1].index[0]
    # filter vaccination
    filtered_par_df = filtered_par_df[np.isclose(filtered_par_df['ini_vacc_prop'], ini_vacc_prop)]
    filtered_par_df = filtered_par_df[np.isclose(filtered_par_df['react_vacc_prop'], react_vacc_prop)]
    filtered_par_df = filtered_par_df[filtered_par_df['react_vacc_threshold'] == react_vacc_threshold]
    filtered_par_df = filtered_par_df[filtered_par_df['react_vacc_turnaround_time'] == react_vacc_turnaround_time]

    # filter for pastorex (fixed par and turnaround time)
    if fixed_par == 'test_sens':
        pastorex_par_df = filtered_par_df[np.isclose(filtered_par_df[fixed_par], pastorex_sens)]
    else:
        pastorex_par_df = filtered_par_df[np.isclose(filtered_par_df[fixed_par], pastorex_spec)]
    pastorex_par_df = pastorex_par_df[pastorex_par_df['test_turnaround_time'] == pastorex_turnaround_time]

    # filter for RDT (fixed par and turnaround time)
    filtered_par_df = filtered_par_df[np.isclose(filtered_par_df[fixed_par], fixed_val)]
    filtered_par_df = filtered_par_df[filtered_par_df['test_turnaround_time'] == test_turnaround_time]

    # extract result df from master result df
    base_result_df = master_result_df.loc[base_par_idx]
    ana_result_df = master_result_df.loc[filtered_par_df.index]
    ana_result_df['net_test_costs'] = ana_result_df['n_tests'] * cost_per_test

    pastorex_df = master_result_df.loc[pastorex_par_df.index]
    pastorex_df['net_test_costs'] = pastorex_df['n_tests'] * cost_per_pastorex

    # set var_par
    var_par = 'test_spec' if fixed_par == 'test_sens' else 'test_sens'

    ## ------ cmaps ------ ##
    rgb_cmap = ['179,205,227','140,150,198','136,65,157']
    hex_cmap = ['#b3cde3','#8c96c6','#88419d']
    diag_cmap = {'Incorrect':'#e41a1c', 'Correct':'#377eb8', 'Pastorex':['#fbb4b9','#f768a1','#ae017e']}

    ## ------ FIGURE 1: Proportion of invasive cases and deaths averted vs. variable parameter ------ ##
    fig_invdea = make_subplots(rows=1, cols=2, column_widths=[0.5, 0.5])
    annotation_list = []
    for i, Ylabel in enumerate(['n_invasive', 'n_deaths']):
        # get Y0 data
        Y0 = base_result_df[Ylabel].to_numpy()
        # each line = one test_receptiveness value
        for j, test_receptiveness in enumerate(np.sort(ana_result_df['test_receptiveness'].unique())):
            # line/shading color
            rgb_col, hex_col = rgb_cmap[j], hex_cmap[j]
            # filter for test_receptiveness
            curr_ana_result_df = ana_result_df[np.isclose(ana_result_df['test_receptiveness'], test_receptiveness)]
            # get X and Y data
            X = np.around(curr_ana_result_df[var_par].to_numpy()*100, 0)
            Y = curr_ana_result_df[Ylabel].to_numpy().reshape(len(np.unique(X)), -1) # reshape Y based on each row = one unique X value
            # calculate plotting coordinates
            #vacc_trigger_mask = curr_ana_result_df['vacc_trigger_day'].to_numpy().reshape(len(np.unique(X)), -1) # vaccination must be triggered
            plot_x, plot_y, plot_y_lb, plot_y_ub = apputils.get_linregress_plot_coords(X, Y, Y0=Y0, method='ratio')
            ci = ((plot_y_ub - plot_y) + (plot_y - plot_y_lb))/2

            if np.mean(Y0) < 10:
                plot_y = np.zeros(len(plot_y))
                ci = np.zeros(len(ci))
            # add shading trace to CI band
            fig_invdea.add_trace(
                go.Scatter(
                    x=np.concatenate([plot_x, plot_x[::-1]]),
                    y=np.concatenate([plot_y_ub, plot_y_lb[::-1]]),
                    fill='toself',
                    fillcolor='rgba(%s,0.2)'%(rgb_cmap[j]),
                    line_color='rgba(255,255,255,0)',
                    showlegend=False,
                    hoverinfo='none',
                ),
                row=1, col=i+1,
            )
            # get regressed absolute calues
            annot_y, annot_y_lb, annot_y_ub = apputils.get_linregress_plot_coords(X, Y, method='absolute')[1:]
            annot_ci = ((annot_y_ub - annot_y) + (annot_y - annot_y_lb))/2

            # edit hover text
            if Ylabel == 'n_invasive':
                hovertemplate = 'X: %s = '%("Sensitivity" if var_par == 'test_sens' else 'Specificity') + '%{x:.0f}%<br>Y: ' + '%{text}<extra></extra>'
                hover_text = ["%i%%<span>&#177;</span>%i%% invasive cases averted<br>%i<span>&#177;</span>%i invasive cases expected per mil. people"%(plot_y[k], ci[k], annot_y[k], annot_ci[k]) for k in np.arange(len(plot_y))]
            else:
                hovertemplate = 'X: %s = '%("Sensitivity" if var_par == 'test_sens' else 'Specificity') + '%{x:.0f}%<br>Y: ' + '%{text}<extra></extra>'
                hover_text = ["%i%%<span>&#177;</span>%i%% deaths averted<br>%i<span>&#177;</span>%i deaths expected per mil. people"%(plot_y[k], ci[k], annot_y[k], annot_ci[k]) for k in np.arange(len(plot_y))]

            fig_invdea.add_trace(
                go.Scatter(
                    x=plot_x,
                    y=plot_y,
                    mode='lines+markers',
                    line=dict(color=hex_cmap[j]),
                    hovertemplate=hovertemplate,
                    text=hover_text,
                    name='%i%%'%(100*test_receptiveness),
                    showlegend=True if i == 1 else False,
                ),
                row=1, col=i+1
            )

    # reorder traces
    traces_idx = np.arange(len(fig_invdea.data))
    line_traces = traces_idx[traces_idx%2==0]
    shading_traces = traces_idx[~traces_idx%2==0]
    fig_invdea.data = tuple([fig_invdea.data[idx] for idx in line_traces] + [fig_invdea.data[idx] for idx in shading_traces])

    for i, Ylabel in enumerate(['n_invasive', 'n_deaths']):
        # get Y0 data
        Y0 = base_result_df[Ylabel].to_numpy()
        # each line = one test_receptiveness value
        for j, test_receptiveness in enumerate(np.sort(ana_result_df['test_receptiveness'].unique())):
            # add pastorex reference as a line plot
            # filter for test_receptiveness
            curr_pastorex_df = pastorex_df[np.isclose(pastorex_df['test_receptiveness'], test_receptiveness)]
            # get X and Y data
            X = np.around(curr_pastorex_df[var_par].to_numpy()*100, 0)
            pastorex_Y = curr_pastorex_df[Ylabel].to_numpy().reshape(len(np.unique(X)), -1) # reshape Y based on each row = one unique X value
            # calculate plotting coordinates
            pastorex_Y_result = apputils.get_linregress_plot_coords(X, pastorex_Y, Y0=Y0, method='ratio')[1]

            pastorex_var_val = pastorex_sens if var_par == 'test_sens' else pastorex_spec
            anno_y = pastorex_Y_result[np.isclose(plot_x, 100*pastorex_var_val)][0]
            if np.mean(Y0) < 10:
                anno_y = 0.

            fig_invdea.add_trace(
                go.Scatter(
                    x=[60, 80, 100],
                    y=[anno_y, anno_y, anno_y],
                    mode='lines',
                    line=dict(color=diag_cmap['Pastorex'][j], width=2, dash='dash'),
                    hovertemplate='%{text}<extra></extra>',
                    text=["Pastorex: %i%% expected %s averted"%(anno_y, 'invasive cases' if i == 0 else 'deaths')] * 3,
                    name='%i%% (Pastorex)'%(100*test_receptiveness),
                    showlegend=True if i == 1 else False,
                ),
                row=1, col=i+1
            )

            """
            annotation_list.append(dict(
                x=100,
                y=anno_y-2.75,
                xanchor='right',
                text=f"Pastorex: {int(anno_y):,}%",
                showarrow=False,
                xref='x%i'%(i+1),
                yref='y%i'%(i+1),
            ))
            #"""

    fig_invdea.update_layout(
        # Set the titles for each subplot
        annotations=[
          dict(
              text="Total invasive cases averted",
              font=dict(size=14),
              showarrow=False,
              xref="paper",
              yref="paper",
              x=0.225,
              y=1.15,
              xanchor='center'
          ),
          dict(
              text="Total deaths averted",
              font=dict(size=14),
              showarrow=False,
              xref="paper",
              yref="paper",
              x=0.775,
              y=1.15,
              xanchor='center'
          )
        ] + annotation_list,
        # legend
        legend_title="Proportion of<br>samples tested",
        legend_title_font_size=14,
        # plot background
        plot_bgcolor='rgba(0,0,0,0)',  # set plot background color to transparent
        # axes
        xaxis1=dict(title='Sensitivity (%)' if var_par == 'test_sens' else 'Specificity (%)',
                 tickmode='array',
                 tickvals=100*np.sort(ana_result_df[var_par].unique()),
                 showgrid=False,
                 zeroline=False,
                 showline=True,
                 linewidth=1,
                 linecolor='black'),
        xaxis2=dict(title='Sensitivity (%)' if var_par == 'test_sens' else 'Specificity (%)',
                 tickmode='array',
                 tickvals=100*np.sort(ana_result_df[var_par].unique()),
                 showgrid=False,
                 zeroline=False,
                 showline=True,
                 linewidth=1,
                 linecolor='black'),
        yaxis1=dict(title=r"Proportion of invasive cases averted (%)",
                 #range=[0, i_to_ymax[0]],
                 range=[-5, 100],
                 showgrid=False,
                 zeroline=False,
                 showline=True,
                 linewidth=1,
                 linecolor='black'),
        yaxis2=dict(title=r"Proportion of deaths averted (%)",
                 #range=[0, i_to_ymax[1]],
                 range=[-5, 100],
                 showgrid=False,
                 zeroline=False,
                 showline=True,
                 linewidth=1,
                 linecolor='black'),
        )

    ## ------ FIGURE 5: Total number of tests differentiated by # of correct in and incorrect diagnoses ----- ##
    # cmaps
    fig_prev_prec = make_subplots(rows=1, cols=2)

    ## --- prevalence estimate -- ##
    # each line = one test_receptiveness value
    Ylabel = 'rmse_prev'
    ymax = -1
    annotation_list = []
    for j, test_receptiveness in enumerate(np.sort(ana_result_df['test_receptiveness'].unique())):
        # line/shading color
        rgb_col, hex_col = rgb_cmap[j], hex_cmap[j]
        # filter for test_receptiveness
        curr_ana_result_df = ana_result_df[np.isclose(ana_result_df['test_receptiveness'], test_receptiveness)]
        # get X and Y data
        X = np.around(curr_ana_result_df[var_par].to_numpy()*100, 0)
        Y = curr_ana_result_df[Ylabel].to_numpy().reshape(len(np.unique(X)), -1) # reshape Y based on each row = one unique X value
        # calculate plotting coordinates
        plot_x, plot_y, plot_y_lb, plot_y_ub = apputils.get_linregress_plot_coords(X, Y, method='absolute')
        ci = ((plot_y_ub - plot_y) + (plot_y - plot_y_lb))/2
        if max(plot_y_ub) > ymax:
            ymax = max(plot_y_ub)
        # add shading trace to CI band
        fig_prev_prec.add_trace(
            go.Scatter(
                x=np.concatenate([plot_x, plot_x[::-1]]),
                y=np.concatenate([plot_y_ub, plot_y_lb[::-1]]),
                fill='toself',
                fillcolor='rgba(%s,0.2)'%(rgb_cmap[j]),
                line_color='rgba(255,255,255,0)',
                showlegend=False,
                hoverinfo='none',
            ),
            row=1, col=1,
        )

        # edit hover text
        hovertemplate = 'X: %s = '%("Sensitivity" if var_par == 'test_sens' else 'Specificity') + '%{x:.0f}%<br>Y: ' + '%{text}<extra></extra>'
        hover_text = ["RMSE = %i%%<span>&#177;</span>%i%%"%(plot_y[k], ci[k]) for k in np.arange(len(plot_y))]

        fig_prev_prec.add_trace(
            go.Scatter(
                x=plot_x,
                y=plot_y,
                mode='lines+markers',
                line=dict(color=hex_cmap[j]),
                hovertemplate=hovertemplate,
                text=hover_text,
                name='%i%%'%(100*test_receptiveness),
                showlegend=False,
            ),
            row=1, col=1
        )

    """# reorder traces
    traces_idx = np.arange(len(fig_prev_prec.data))
    line_traces = traces_idx[traces_idx%2==0]
    shading_traces = traces_idx[~traces_idx%2==0]
    fig_prev_prec.data = tuple([fig_prev_prec.data[idx] for idx in line_traces] + [fig_prev_prec.data[idx] for idx in shading_traces])"""

    for j, test_receptiveness in enumerate(np.sort(ana_result_df['test_receptiveness'].unique())):
        # add pastorex reference as a line plot
        # filter for test_receptiveness
        curr_pastorex_df = pastorex_df[np.isclose(pastorex_df['test_receptiveness'], test_receptiveness)]
        # get X and Y data
        X = np.around(curr_pastorex_df[var_par].to_numpy()*100, 0)
        pastorex_Y = curr_pastorex_df[Ylabel].to_numpy().reshape(len(np.unique(X)), -1) # reshape Y based on each row = one unique X value
        # calculate plotting coordinates
        pastorex_Y_result = apputils.get_linregress_plot_coords(X, pastorex_Y, method='absolute')[1]

        pastorex_var_val = pastorex_sens if var_par == 'test_sens' else pastorex_spec
        anno_y = pastorex_Y_result[np.isclose(plot_x, 100*pastorex_var_val)][0]
        if np.mean(Y0) < 10:
            anno_y = 0.

        """fig_prev_prec.add_shape(
            type='line',
            x0=60, # x-axis value at which to draw the line
            x1=100, # x-axis value at which to end the line
            y0=anno_y, # y-axis value at which to start the line
            y1=anno_y, # y-axis value at which to end the line
            line=dict(color=diag_cmap['Pastorex'][j], width=2, dash='dash'),
            row = 1, col = 1
        )"""

        fig_prev_prec.add_trace(
            go.Scatter(
                x=[60, 80, 100],
                y=[anno_y, anno_y, anno_y],
                mode='lines',
                line=dict(color=diag_cmap['Pastorex'][j], width=2, dash='dash'),
                name='%i%% (Pastorex)'%(100*test_receptiveness),
                hovertemplate='%{text}<extra></extra>',
                text=["Pastorex: mean RMSE = %i%%"%(int(anno_y))] * 3,
                showlegend=False,
            ),
            row=1, col=1
        )
        """
        annotation_list.append(dict(
            x=100,
            y=anno_y-((2.75/100)*(ymax*1.05)),
            xanchor='right',
            text=f"Pastorex: {anno_y:.1f}%",
            showarrow=False,
            xref='x1',
            yref='y1',
        ))
        #"""

    ## --- prob. of reactive vaccination campaign -- ##
    # each line = one test_receptiveness value
    for j, test_receptiveness in enumerate(np.sort(ana_result_df['test_receptiveness'].unique())):
        # line/shading color
        rgb_col, hex_col = rgb_cmap[j], hex_cmap[j]
        # filter for test_receptiveness
        curr_ana_result_df = ana_result_df[np.isclose(ana_result_df['test_receptiveness'], test_receptiveness)]
        plot_x = np.sort(curr_ana_result_df[var_par].unique())
        plot_y = np.asarray([len(curr_ana_result_df[(curr_ana_result_df['vacc_trigger_day']>0)&(np.isclose(curr_ana_result_df[var_par], x))])/len(curr_ana_result_df[(np.isclose(curr_ana_result_df[var_par], x))]) for x in plot_x])
        plot_x *= 100
        plot_y *= 100
        slope, intercept = apputils.perform_linear_regression(plot_x, plot_y)[:2]
        plot_y = apputils.linear_fn(plot_x, slope, intercept)

        hover_text = ['X: %s = '%("Sensitivity" if var_par == 'test_sens' else 'Specificity') + '%i%%<br>Y: '%(plot_x[k]) + f"{plot_y[k]:.1f}% chance" for k in np.arange(len(plot_y))]
        fig_prev_prec.add_trace(
            go.Bar(
                x=plot_x,
                y=plot_y,
                name='%i%%'%(100*test_receptiveness),
                marker_color=hex_cmap[j],
                hovertext=hover_text,
                hoverinfo='text',
                showlegend=True,
            ),
            row=1, col=2,
          )

        # add pastorex reference as a line plot
    for j, test_receptiveness in enumerate(np.sort(ana_result_df['test_receptiveness'].unique())):
        # filter for test_receptiveness
        curr_pastorex_df = pastorex_df[np.isclose(pastorex_df['test_receptiveness'], test_receptiveness)]
        plot_x = np.sort(curr_pastorex_df[var_par].unique())
        pastorex_Y = np.asarray([len(curr_pastorex_df[(curr_pastorex_df['vacc_trigger_day']>0)&(np.isclose(curr_pastorex_df[var_par], x))])/len(curr_pastorex_df[(np.isclose(curr_pastorex_df[var_par], x))]) for x in plot_x])
        plot_x *= 100
        pastorex_Y *= 100
        slope, intercept = apputils.perform_linear_regression(plot_x, pastorex_Y)[:2]
        pastorex_Y_result = apputils.linear_fn(plot_x, slope, intercept)
        pastorex_var_val = pastorex_sens if var_par == 'test_sens' else pastorex_spec
        anno_y = pastorex_Y_result[np.isclose(plot_x, 100*pastorex_var_val)][0]

        fig_prev_prec.add_trace(
            go.Scatter(
                x=[60, 80, 100],
                y=[anno_y, anno_y, anno_y],
                mode='lines',
                line=dict(color=diag_cmap['Pastorex'][j], width=2, dash='dash'),
                name='%i%% (Pastorex)'%(100*test_receptiveness),
                hovertemplate='%{text}<extra></extra>',
                text=["Pastorex: %i%% probability"%(int(anno_y))] * 3,
                showlegend=True,
            ),
            row=1, col=2
        )

    fig_prev_prec.update_layout(
        # Set the titles for each subplot
        annotations=[
          dict(
              text="Accuracy of prevalence estimate",
              font=dict(size=14),
              showarrow=False,
              xref="paper",
              yref="paper",
              x=0.225,
              y=1.15,
              xanchor='center'
          ),
          dict(
              text="Probability of trigerring reactive vaccination",
              font=dict(size=14),
              showarrow=False,
              xref="paper",
              yref="paper",
              x=0.775,
              y=1.15,
              xanchor='center'
          )
        ] + annotation_list,
        # plot background
        plot_bgcolor='rgba(0,0,0,0)',  # set plot background color to transparent
        barmode='group',
        # legend
        legend_title="Proportion of<br>samples tested",
        legend_title_font_size=14,
        # axes
        xaxis1=dict(title='Sensitivity (%)' if var_par == 'test_sens' else 'Specificity (%)',
                 tickmode='array',
                 tickvals=100*np.sort(ana_result_df[var_par].unique()),
                 showgrid=False,
                 zeroline=False,
                 showline=True,
                 linewidth=1,
                 linecolor='black'),
        yaxis1=dict(title=r"RMSE of prevalence estimate from<br>test and case positivity rates (%)",
                 #range=[0, i_to_ymax[0]],
                 range=[0, ymax*1.05],
                 rangemode = "nonnegative",
                 showgrid=False,
                 zeroline=False,
                 showline=True,
                 linewidth=1,
                 linecolor='black'),
        xaxis2=dict(title='Sensitivity (%)' if var_par == 'test_sens' else 'Specificity (%)',
                 tickmode='array',
                 tickvals=100*np.sort(ana_result_df[var_par].unique()),
                 showgrid=False,
                 zeroline=False,
                 showline=True,
                 linewidth=1,
                 linecolor='black'),
        yaxis2=dict(title="Probability (%)",
                 #range=[0, i_to_ymax[0]],
                 range=[0, 100],
                 showgrid=False,
                 zeroline=False,
                 showline=True,
                 linewidth=1,
                 linecolor='black'),
        )

    ## ------ FIGURE 6: day when reactive vaccination is triggered ------ ##
    fig_vacc_day = make_subplots(rows=1, cols=2)
    """Ylabel = 'mu_invasive_per_week'
    # each line = one test_receptiveness value
    for j, test_receptiveness in enumerate(np.sort(ana_result_df['test_receptiveness'].unique())):
        # filter for test_receptiveness
        curr_ana_result_df = ana_result_df[np.isclose(ana_result_df['test_receptiveness'], test_receptiveness)]
        # get X and Y data
        X = np.around(curr_ana_result_df[var_par].to_numpy()*100, 0)
        Y = curr_ana_result_df[Ylabel].to_numpy().reshape(len(np.unique(X)), -1) # reshape Y based on each row = one unique X value

        # calculate plotting coordinates
        #vacc_trigger_mask = curr_ana_result_df['vacc_trigger_day'].to_numpy().reshape(len(np.unique(X)), -1) # vaccination must be triggered
        plot_x, plot_y, plot_y_lb, plot_y_ub = apputils.get_linregress_plot_coords(X, Y, method='absolute')
        ci = (plot_y_ub-plot_y_lb)/(2 * abs(stats.norm.ppf((1 - .95) / 2)))

        hover_text = ['X: %s = '%("Sensitivity" if var_par == 'test_sens' else 'Specificity') + '%i%%<br>Y: '%(plot_x[k]) + f"{plot_y[k]:.0f}<span>&#177;</span>{ci[k]:.0f} cases per week" for k in np.arange(len(plot_y))]
        fig_vacc_day.add_trace(
            go.Bar(
                x=plot_x,
                y=plot_y,
                error_y=dict(
                    type='data',
                    array=ci,
                    visible=True
                ),
                name='%i%%'%(100*test_receptiveness),
                marker_color=hex_cmap[j],
                hovertext=hover_text,
                hoverinfo='text',
                showlegend=False,
            ),
            row=1, col=1,
          )

    for j, test_receptiveness in enumerate(np.sort(ana_result_df['test_receptiveness'].unique())):
        # add pastorex reference as a line plot
        # filter for test_receptiveness
        curr_pastorex_df = pastorex_df[np.isclose(pastorex_df['test_receptiveness'], test_receptiveness)]
        # get X and Y data
        X = np.around(curr_pastorex_df[var_par].to_numpy()*100, 0)
        pastorex_Y = curr_pastorex_df[Ylabel].to_numpy().reshape(len(np.unique(X)), -1) # reshape Y based on each row = one unique X value
        # calculate plotting coordinates
        #vacc_trigger_mask = curr_ana_result_df['vacc_trigger_day'].to_numpy().reshape(len(np.unique(X)), -1) # vaccination must be triggered
        pastorex_Y_result = apputils.get_linregress_plot_coords(X, pastorex_Y, method='absolute')[1]
        pastorex_var_val = pastorex_sens if var_par == 'test_sens' else pastorex_spec
        anno_y = pastorex_Y_result[np.isclose(plot_x, 100*pastorex_var_val)][0]

        fig_vacc_day.add_trace(
            go.Scatter(
                x=[60, 80, 100],
                y=[anno_y, anno_y, anno_y],
                mode='lines',
                line=dict(color=diag_cmap['Pastorex'][j], width=2, dash='dash'),
                name='%i%% (Pastorex)'%(100*test_receptiveness),
                hovertemplate='%{text}<extra></extra>',
                text=["Pastorex: %i week"%(int(anno_y))] * 3,
                showlegend=False,
            ),
            row=1, col=1
        )"""

    #############

    Ylabel = 'vacc_trigger_day'
    # each line = one test_receptiveness value
    for j, test_receptiveness in enumerate(np.sort(ana_result_df['test_receptiveness'].unique())):
        # filter for test_receptiveness
        curr_ana_result_df = ana_result_df[np.isclose(ana_result_df['test_receptiveness'], test_receptiveness)]
        # get X and Y data
        X = np.around(curr_ana_result_df[var_par].to_numpy()*100, 0)
        Y = curr_ana_result_df[Ylabel].to_numpy().reshape(len(np.unique(X)), -1) # reshape Y based on each row = one unique X value
        Y[Y<0] = 180
        Y = Y/7
        # calculate plotting coordinates
        #vacc_trigger_mask = curr_ana_result_df['vacc_trigger_day'].to_numpy().reshape(len(np.unique(X)), -1) # vaccination must be triggered
        plot_x, plot_y, plot_y_lb, plot_y_ub = apputils.get_linregress_plot_coords(X, Y, method='absolute')
        ci = (plot_y_ub-plot_y_lb)/(2 * abs(stats.norm.ppf((1 - .95) / 2)))

        hover_text = ['X: %s = '%("Sensitivity" if var_par == 'test_sens' else 'Specificity') + '%i%%<br>Y: '%(plot_x[k]) + f"{plot_y[k]:.0f}<span>&#177;</span>{ci[k]:.0f} week" for k in np.arange(len(plot_y))]
        fig_vacc_day.add_trace(
            go.Bar(
                x=plot_x,
                y=plot_y,
                error_y=dict(
                    type='data',
                    array=ci,
                    visible=True
                ),
                name='%i%%'%(100*test_receptiveness),
                marker_color=hex_cmap[j],
                hovertext=hover_text,
                hoverinfo='text',
                showlegend=True,
            ),
            row=1, col=1,
          )

    for j, test_receptiveness in enumerate(np.sort(ana_result_df['test_receptiveness'].unique())):
        # add pastorex reference as a line plot
        # filter for test_receptiveness
        curr_pastorex_df = pastorex_df[np.isclose(pastorex_df['test_receptiveness'], test_receptiveness)]
        # get X and Y data
        X = np.around(curr_pastorex_df[var_par].to_numpy()*100, 0)
        pastorex_Y = curr_pastorex_df[Ylabel].to_numpy().reshape(len(np.unique(X)), -1) # reshape Y based on each row = one unique X value
        pastorex_Y[pastorex_Y<0] = 180
        pastorex_Y = pastorex_Y/7
        # calculate plotting coordinates
        #vacc_trigger_mask = curr_ana_result_df['vacc_trigger_day'].to_numpy().reshape(len(np.unique(X)), -1) # vaccination must be triggered
        pastorex_Y_result = apputils.get_linregress_plot_coords(X, pastorex_Y, method='absolute')[1]
        pastorex_var_val = pastorex_sens if var_par == 'test_sens' else pastorex_spec
        anno_y = pastorex_Y_result[np.isclose(plot_x, 100*pastorex_var_val)][0]

        fig_vacc_day.add_trace(
            go.Scatter(
                x=[60, 80, 100],
                y=[anno_y, anno_y, anno_y],
                mode='lines',
                line=dict(color=diag_cmap['Pastorex'][j], width=2, dash='dash'),
                name='%i%% (Pastorex)'%(100*test_receptiveness),
                hovertemplate='%{text}<extra></extra>',
                text=["Pastorex: %i week"%(int(anno_y))] * 3,
                showlegend=True,
            ),
            row=1, col=1
        )

    fig_vacc_day.add_trace(
        go.Scatter(
            x=[60, 80, 100],
            y=[25, 25, 25],
            mode='lines',
            line=dict(color='#404040', width=1, dash='dash'),
            name='Last simulated epi week',
            hovertemplate='%{text}<extra></extra>',
            text=["Last simulated epi week"] * 3,
            showlegend=True,
        ),
        row=1, col=1,
      )

    """
    dict(
        text="Expected no. of invasive cases per week",
        font=dict(size=14),
        showarrow=False,
        xref="paper",
        yref="paper",
        x=0.225,
        y=1.15,
        xanchor='center'
    ),
    """
    fig_vacc_day.update_layout(
        # Set the titles for each subplot
        annotations=[
          dict(
              text="Average week when reactive<br>vaccination threshold was met",
              font=dict(size=14),
              showarrow=False,
              xref="paper",
              yref="paper",
              x=0.225,
              y=1.25,
              xanchor='center'
          )
        ], #+ annotation_list,
        # plot background
        plot_bgcolor='rgba(0,0,0,0)',  # set plot background color to transparent
        barmode='group',
        # legend
        legend_title="Proportion of<br>samples tested",
        legend_title_font_size=14,
        # axes
        xaxis1=dict(title='Sensitivity (%)' if var_par == 'test_sens' else 'Specificity (%)',
                 tickmode='array',
                 tickvals=100*np.sort(ana_result_df[var_par].unique()),
                 showgrid=False,
                 zeroline=False,
                 showline=True,
                 linewidth=1,
                 linecolor='black'),
        yaxis1=dict(title=r"Time (week)",
                 #range=[0, i_to_ymax[0]],
                 #range=[0, ymax*1.05],
                 rangemode = "nonnegative",
                 showgrid=False,
                 zeroline=False,
                 showline=True,
                 linewidth=1,
                 linecolor='black'),
        )

    ## ------ FIGURE 2: Total number of tests differentiated by # of correct in and incorrect diagnoses ----- ##
    annotation_list = []
    fig_n_tests = make_subplots(rows=1, cols=3, column_widths=[0.33, 0.33, 0.34], shared_yaxes=True)
    #for i, label in enumerate(['n_tests']):
    for j, test_receptiveness in enumerate(np.sort(ana_result_df['test_receptiveness'].unique())):
        # filter for test_receptiveness
        curr_ana_result_df = ana_result_df[np.isclose(ana_result_df['test_receptiveness'], test_receptiveness)]

        for k, (name, Ylabel) in enumerate(zip(['Incorrect', 'Correct'], ['n_incorr_diag', 'n_corr_diag'])):
            diag_col = diag_cmap[name]
            # get X and Y data
            X = np.around(curr_ana_result_df[var_par].to_numpy()*100, 0)
            Y = curr_ana_result_df[Ylabel].to_numpy().reshape(len(np.unique(X)), -1) # reshape Y based on each row = one unique
            # calculate plotting coordinates
            plot_x, plot_y, plot_y_lb, plot_y_ub = apputils.get_linregress_plot_coords(X, Y, method='absolute')
            ci = (plot_y_ub-plot_y_lb)/(2 * abs(stats.norm.ppf((1 - .95) / 2)))

            hover_text = ['X: %s = '%("Sensitivity" if var_par == 'test_sens' else 'Specificity') + '%i%%<br>Y: '%(plot_x[k]) + "%i<span>&#177;</span>%i %s diagnoses"%(plot_y[k], ci[k], name.lower()) for k in np.arange(len(plot_y))]
            fig_n_tests.add_trace(
              go.Bar(
                  x=plot_x,
                  y=plot_y,
                  name=name,
                  marker_color=diag_col,
                  hovertext=hover_text,
                  hoverinfo='text',
                  showlegend=True if j == 2 else False,
              ),
              row=1, col=j+1,
            )

        # add pastorex reference as a line plot
        # filter for test_receptiveness
        curr_pastorex_df = pastorex_df[np.isclose(pastorex_df['test_receptiveness'], test_receptiveness)]
        # get X and Y data
        X = np.around(curr_pastorex_df[var_par].to_numpy()*100, 0)
        pastorex_Y = curr_pastorex_df['n_tests'].to_numpy().reshape(len(np.unique(X)), -1) # reshape Y based on each row = one unique X value
        # calculate plotting coordinates
        pastorex_Y_result = apputils.get_linregress_plot_coords(X, pastorex_Y, method='absolute')[1]
        pastorex_var_val = pastorex_sens if var_par == 'test_sens' else pastorex_spec
        anno_y = pastorex_Y_result[np.isclose(plot_x, 100*pastorex_var_val)][0]
        fig_n_tests.add_shape(
            type='line',
            x0=60, # x-axis value at which to draw the line
            x1=100, # x-axis value at which to end the line
            y0=anno_y, # y-axis value at which to start the line
            y1=anno_y, # y-axis value at which to end the line
            line=dict(color=diag_cmap['Pastorex'][j], width=2, dash='dash'),
            row = 1, col = j+1
        )
        annotation_list.append(dict(
            x=80,
            y=anno_y,
            text=f"Total no. of Pastorex tests:<br>{int(anno_y):,}",
            showarrow=False,
            xref='x%i'%(j+1),
            yref='y%i'%(j+1),
        ))

    fig_n_tests.update_layout(
        title={
          'text': "Total number of tests administered",
          'font': {'size': 16},
          'x': 0.5, # Set the title to be centered
          'xanchor': 'center'
        },
        # Set the titles for each subplot
        annotations=[
          dict(
              text="<b>%i%% of samples tested</b>"%(test_receptiveness*100),
              font=dict(size=14, color=hex_cmap[j]),
              showarrow=False,
              xref="paper",
              yref="paper",
              x=0.03 + j*0.47,
              y=1.1,
          )
          for j, test_receptiveness in enumerate(np.sort(ana_result_df['test_receptiveness'].unique()))
        ] + annotation_list,
        # legend
        legend_title="Diagnoses",
        legend_title_font_size=14,
        barmode='stack',
        plot_bgcolor='rgba(0,0,0,0)',  # set plot background color to transparent
        xaxis1=dict(title='Sensitivity (%)' if var_par == 'test_sens' else 'Specificity (%)',
                 tickmode='array',
                 tickvals=100*np.sort(ana_result_df[var_par].unique()),
                 showgrid=False,
                 zeroline=False,
                 showline=True,
                 linewidth=1,
                 linecolor='black'),
        xaxis2=dict(title='Sensitivity (%)' if var_par == 'test_sens' else 'Specificity (%)',
                 tickmode='array',
                 tickvals=100*np.sort(ana_result_df[var_par].unique()),
                 showgrid=False,
                 zeroline=False,
                 showline=True,
                 linewidth=1,
                 linecolor='black'),
        xaxis3=dict(title='Sensitivity (%)' if var_par == 'test_sens' else 'Specificity (%)',
                 tickmode='array',
                 tickvals=100*np.sort(ana_result_df[var_par].unique()),
                 showgrid=False,
                 zeroline=False,
                 showline=True,
                 linewidth=1,
                 linecolor='black'),
        yaxis1=dict(title=r"Number of tests<br>(per 1,000,000 individuals)",
                 rangemode='nonnegative',
                 #tickvals=np.arange(0, max(result_df['n_tests'])*1.05, 2e3),
                 showgrid=False,
                 zeroline=False,
                 showline=True,
                 linewidth=1,
                 linecolor='black'),
        )

    ## ------ FIGURE 3: Costs per correct diagnosis ----- ##
    fig_costs_per_corr_diag = make_subplots(rows=1, cols=3, column_widths=[0.33, 0.33, 0.34], shared_yaxes=True)
    #for i, label in enumerate(['n_tests']):
    annotation_list = []
    for j, test_receptiveness in enumerate(np.sort(ana_result_df['test_receptiveness'].unique())):
        # filter for test_receptiveness
        curr_ana_result_df = ana_result_df[np.isclose(ana_result_df['test_receptiveness'], test_receptiveness)]
        # ony for correct diagnoses
        name = 'Correct'
        Ylabel = 'n_corr_diag'
        diag_col = diag_cmap[name]
        # get X and Y data
        X = np.around(curr_ana_result_df[var_par].to_numpy()*100, 0)
        Y = curr_ana_result_df[Ylabel].to_numpy().reshape(len(np.unique(X)), -1) # reshape Y based on each row = one unique
        # calculate plotting coordinates
        # get total costs
        TotalCostsY = curr_ana_result_df['net_test_costs'].to_numpy().reshape(len(np.unique(X)), -1)
        Ydat = np.zeros(Y.shape)
        Ydat[Y>0] = TotalCostsY[Y>0]/Y[Y>0]
        plot_x, plot_y, plot_y_lb, plot_y_ub = apputils.get_linregress_plot_coords(X, Ydat, method='absolute')
        ci = (plot_y_ub - plot_y_lb)/2

        hover_text = ['X: %s = '%("Sensitivity" if var_par == 'test_sens' else 'Specificity') + '%i%%<br>Y: '%(plot_x[k]) + "$%.2f<span>&#177;</span>%.2f %s per correct RDT"%(plot_y[k], ci[k], name.lower()) for k in np.arange(len(plot_y))]
        fig_costs_per_corr_diag.add_trace(
          go.Bar(
              x=plot_x,
              y=plot_y,
              error_y=dict(
                  type='data',
                  array=ci,
                  visible=True
              ),
              name='RDT',
              marker_color=diag_col,
              hovertext=hover_text,
              hoverinfo='text',
              showlegend=False if j < 2 else True,
          ),
          row=1, col=j+1,
        )

        # add pastorex reference as a line plot
        # filter for test_receptiveness
        curr_pastorex_df = pastorex_df[np.isclose(pastorex_df['test_receptiveness'], test_receptiveness)]
        # get X and Y data
        X = np.around(curr_pastorex_df[var_par].to_numpy()*100, 0)
        pastorex_Y = curr_pastorex_df[Ylabel].to_numpy().reshape(len(np.unique(X)), -1) # reshape Y based on each row = one unique X value
        TotalCostsY = curr_pastorex_df['net_test_costs'].to_numpy().reshape(len(np.unique(X)), -1)
        Ydat = np.zeros(Y.shape)
        Ydat[pastorex_Y>0] = TotalCostsY[pastorex_Y>0]/pastorex_Y[pastorex_Y>0]
        # calculate plotting coordinates
        pastorex_Y_result = apputils.get_linregress_plot_coords(X, Ydat, method='absolute')[1]

        pastorex_var_val = pastorex_sens if var_par == 'test_sens' else pastorex_spec
        anno_y = pastorex_Y_result[np.isclose(plot_x, 100*pastorex_var_val)][0]
        fig_costs_per_corr_diag.add_shape(
            type='line',
            x0=60, # x-axis value at which to draw the line
            x1=100, # x-axis value at which to end the line
            y0=anno_y, # y-axis value at which to start the line
            y1=anno_y, # y-axis value at which to end the line
            line=dict(color=diag_cmap['Pastorex'][j], width=2, dash='dash'),
            row = 1, col = j+1
        )
        annotation_list.append(dict(
            x=80,
            y=anno_y,
            text=f"Mean cost per corr. Pastorex:<br>${np.mean(anno_y):.2f}",
            showarrow=False,
            xref='x%i'%(j+1),
            yref='y%i'%(j+1),
        ))

    fig_costs_per_corr_diag.update_layout(
        title={
          'text': "Costs per correct diagnosis",
          'font': {'size': 16},
          'x': 0.5, # Set the title to be centered
          'xanchor': 'center'
        },
        # Set the titles for each subplot
        annotations=[
          dict(
              text="<b>%i%% of samples tested</b>"%(test_receptiveness*100),
              font=dict(size=14, color=hex_cmap[j]),
              showarrow=False,
              xref="paper",
              yref="paper",
              x=0.03 + j*0.47,
              y=1.15,
          )
          for j, test_receptiveness in enumerate(np.sort(ana_result_df['test_receptiveness'].unique()))
        ] + annotation_list,
        # legend
        barmode='group',
        plot_bgcolor='rgba(0,0,0,0)',  # set plot background color to transparent
        xaxis1=dict(title='Sensitivity (%)' if var_par == 'test_sens' else 'Specificity (%)',
                 tickmode='array',
                 tickvals=100*np.sort(ana_result_df[var_par].unique()),
                 showgrid=False,
                 zeroline=False,
                 showline=True,
                 linewidth=1,
                 linecolor='black'),
        xaxis2=dict(title='Sensitivity (%)' if var_par == 'test_sens' else 'Specificity (%)',
                 tickmode='array',
                 tickvals=100*np.sort(ana_result_df[var_par].unique()),
                 showgrid=False,
                 zeroline=False,
                 showline=True,
                 linewidth=1,
                 linecolor='black'),
        xaxis3=dict(title='Sensitivity (%)' if var_par == 'test_sens' else 'Specificity (%)',
                 tickmode='array',
                 tickvals=100*np.sort(ana_result_df[var_par].unique()),
                 showgrid=False,
                 zeroline=False,
                 showline=True,
                 linewidth=1,
                 linecolor='black'),
        yaxis1=dict(title=r"Costs per test ($)",
                 rangemode='nonnegative',
                 #tickvals=np.arange(0, max(ana_result_df['n_tests'])*1.05, 2e3),
                 showgrid=False,
                 zeroline=False,
                 showline=True,
                 linewidth=1,
                 linecolor='black'),
    )

    ## -- Figure 4: total costs -- ##
    fig_total_test_costs = make_subplots(rows=1, cols=3, column_widths=[0.33, 0.33, 0.34], shared_yaxes=True)
    #for i, label in enumerate(['n_tests']):
    annotation_list = []
    for j, test_receptiveness in enumerate(np.sort(ana_result_df['test_receptiveness'].unique())):
        # filter for test_receptiveness
        curr_ana_result_df = ana_result_df[np.isclose(ana_result_df['test_receptiveness'], test_receptiveness)]
        # get X and Y data
        X = np.around(curr_ana_result_df[var_par].to_numpy()*100, 0)
        Y = curr_ana_result_df['net_test_costs'].to_numpy().reshape(len(np.unique(X)), -1)# reshape Y based on each row = one unique
        # calculate plotting coordinates
        plot_x, plot_y, plot_y_lb, plot_y_ub = apputils.get_linregress_plot_coords(X, Y, method='absolute')
        ci = (plot_y_ub-plot_y_lb)/(2 * abs(stats.norm.ppf((1 - .95) / 2)))

        hover_text = ['X: %s = '%("Sensitivity" if var_par == 'test_sens' else 'Specificity') + '%i%%<br>Y: '%(plot_x[k]) + f"${int(plot_y[k]):,} <span>&#177;</span> ${int(ci[k]):,}" +  "%s diagnoses"%(name.lower()) for k in np.arange(len(plot_y))]
        fig_total_test_costs.add_trace(
          go.Bar(
              x=plot_x,
              y=plot_y,
              error_y=dict(
                  type='data',
                  array=ci,
                  visible=True
              ),
              name=name,
              marker_color=diag_cmap['Correct'],
              hovertext=hover_text,
              hoverinfo='text',
              showlegend=False,
          ),
          row=1, col=j+1,
        )

        # add pastorex reference as a line plot
        # filter for test_receptiveness
        curr_pastorex_df = pastorex_df[np.isclose(pastorex_df['test_receptiveness'], test_receptiveness)]
        # get X and Y data
        X = np.around(curr_pastorex_df[var_par].to_numpy()*100, 0)
        pastorex_Y = curr_pastorex_df['net_test_costs'].to_numpy().reshape(len(np.unique(X)), -1) # reshape Y based on each row = one unique X value
        # calculate plotting coordinates
        pastorex_Y_result = apputils.get_linregress_plot_coords(X, pastorex_Y, method='absolute')[1]

        pastorex_var_val = pastorex_sens if var_par == 'test_sens' else pastorex_spec
        anno_y = pastorex_Y_result[np.isclose(plot_x, 100*pastorex_var_val)][0]
        fig_total_test_costs.add_shape(
            type='line',
            x0=60, # x-axis value at which to draw the line
            x1=100, # x-axis value at which to end the line
            y0=anno_y, # y-axis value at which to start the line
            y1=anno_y, # y-axis value at which to end the line
            line=dict(color=diag_cmap['Pastorex'][j], width=2, dash='dash'),
            row = 1, col = j+1
        )
        annotation_list.append(dict(
            x=80,
            y=anno_y,
            text=f"Mean total costs using Pastorex:<br>${int(anno_y):,}",
            showarrow=False,
            xref='x%i'%(j+1),
            yref='y%i'%(j+1),
        ))

    fig_total_test_costs.update_layout(
        title={
          'text': "Total testing costs",
          'font': {'size': 16},
          'x': 0.5, # Set the title to be centered
          'xanchor': 'center'
        },
        # Set the titles for each subplot
        annotations=[
          dict(
              text="<b>%i%% of samples tested</b>"%(test_receptiveness*100),
              font=dict(size=14, color=hex_cmap[j]),
              showarrow=False,
              xref="paper",
              yref="paper",
              x=0.03 + j*0.47,
              y=1.1,
          )
          for j, test_receptiveness in enumerate(np.sort(ana_result_df['test_receptiveness'].unique()))
        ] + annotation_list,
        # legend
        barmode='group',
        plot_bgcolor='rgba(0,0,0,0)',  # set plot background color to transparent
        xaxis1=dict(title='Sensitivity (%)' if var_par == 'test_sens' else 'Specificity (%)',
                 tickmode='array',
                 tickvals=100*np.sort(ana_result_df[var_par].unique()),
                 showgrid=False,
                 zeroline=False,
                 showline=True,
                 linewidth=1,
                 linecolor='black'),
        xaxis2=dict(title='Sensitivity (%)' if var_par == 'test_sens' else 'Specificity (%)',
                 tickmode='array',
                 tickvals=100*np.sort(ana_result_df[var_par].unique()),
                 showgrid=False,
                 zeroline=False,
                 showline=True,
                 linewidth=1,
                 linecolor='black'),
        xaxis3=dict(title='Sensitivity (%)' if var_par == 'test_sens' else 'Specificity (%)',
                 tickmode='array',
                 tickvals=100*np.sort(ana_result_df[var_par].unique()),
                 showgrid=False,
                 zeroline=False,
                 showline=True,
                 linewidth=1,
                 linecolor='black'),
        yaxis1=dict(title=r"Total costs<br>($ per 1,000,000 individuals)",
                 rangemode='nonnegative',
                 showgrid=False,
                 zeroline=False,
                 showline=True,
                 linewidth=1,
                 linecolor='black'),
    )

    return fig_invdea, fig_n_tests, fig_costs_per_corr_diag, fig_total_test_costs, fig_prev_prec, fig_vacc_day


if __name__ == '__main__':
    #app.run_server(debug=True)
    app.run_server(host='127.0.0.1',
                   port=8050,
                   debug=True)
