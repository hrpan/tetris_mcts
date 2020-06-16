import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import time
import threading
import sys
from parseLog import Parser, ModelParser, StatusParser
from dash.dependencies import Input, Output, State
from math import ceil
import numpy as np

update_interval = 500

external_stylesheets = ['./static/css/style.css']


style = {'font'}

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {
    'background': '#000000',
    'text': '#cccccc'
}

fig = go.Figure(
    data=[
        go.Scatter(x=[], y=[], name='Lines Cleared', mode='lines'),
        go.Scatter(x=[], y=[], name='Score', mode='lines', yaxis='y2')
    ],
    layout=go.Layout(
        title={'text': 'Lines Cleared / Score',
               'font': {'color': colors['text']}},
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font={'color': colors['text']},
        xaxis={'title': 'Episode', 'rangemode': 'tozero'},
        yaxis={'title': 'Lines Cleared',
               'showgrid': False,
               'rangemode': 'tozero'},
        yaxis2={'title': 'Score',
                'side': 'right',
                'overlaying': 'y',
                'showgrid': False,
                'rangemode': 'tozero'},
        uirevision=True
    )
)

fig_50 = go.Figure(
    data=[
        go.Scatter(x=[], y=[],
                   error_y=dict(type='data', array=[], visible=True),
                   name='Lines Cleared', mode='lines'),
        go.Scatter(x=[], y=[],
                   error_y=dict(type='data', array=[], visible=True),
                   name='Score', mode='lines', yaxis='y2')
    ],
    layout=go.Layout(
        title={'text': 'Lines Cleared / Score per 50 episodes',
               'font': {'color': colors['text']}},
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font={'color': colors['text']},
        xaxis={'title': '50 Episodes', 'rangemode': 'tozero'},
        yaxis={'title': 'Lines Cleared',
               'showgrid': False,
               'rangemode': 'tozero'},
        yaxis2={'title': 'Score', 'side': 'right', 'overlaying': 'y',
                'showgrid': False, 'rangemode': 'tozero'},
        uirevision=True
    )
)

fig_pt = go.Figure(
    data=[
        go.Scatter(x=[], y=[],
                   error_y=dict(type='data', array=[], visible=True),
                   name='Lines Cleared', mode='lines'),
        go.Scatter(x=[], y=[],
                   error_y=dict(type='data', array=[], visible=True),
                   name='Score', mode='lines', yaxis='y2')
    ],
    layout=go.Layout(
        title={'text': 'Lines Cleared / Score per Training Session',
               'font': {'color': colors['text']}},
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font={'color': colors['text']},
        xaxis={'title': 'Training Session', 'rangemode': 'tozero'},
        yaxis={'title': 'Lines Cleared',
               'showgrid': False,
               'rangemode': 'tozero'},
        yaxis2={'title': 'Score', 'side': 'right', 'overlaying': 'y',
                'showgrid': False, 'rangemode': 'tozero'},
        uirevision=True
    )
)

fig_loss = go.Figure(
    data=[
        go.Scatter(x=[], y=[], name='Training Loss', mode='lines'),
        go.Scatter(x=[], y=[],
            error_y=dict(type='data', array=[], visible=True),
            name='Validation Loss', mode='lines'),
        go.Scatter(x=[], y=[], name='Gradient Norm', mode='lines', yaxis='y2')
    ],
    layout=go.Layout(
        title={'text': 'Training / Validation Loss & Gradient Norm',
               'font': {'color': colors['text']}},
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font={'color': colors['text']},
        xaxis={'title': 'Iteration', 'rangemode': 'tozero'},
        yaxis={'rangemode': 'tozero'},
        yaxis2={'title': 'Norm', 'side': 'right', 'overlaying': 'y',
                'showgrid': False, 'rangemode': 'tozero'},
        uirevision=True
    )
)

fig_data = go.Figure(
    data=[
        go.Scatter(x=[], y=[], name='Data Accumulated', mode='lines')
    ],
    layout=go.Layout(
        title={'text': 'Accumulated Training Data',
               'font': {'color': colors['text']}},
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font={'color': colors['text']},
        xaxis={'rangemode': 'tozero'},
        yaxis={'rangemode': 'tozero'},
        uirevision=True
    )
)

colorscale = [[0, 'rgb(100, 100, 100)'],
              [0.5, 'rgb(0, 0, 0)'],
              [1, 'rgb(255, 255, 255)']]
fig_board = go.Figure(
    data=[
        go.Heatmap(z=[[0] * 10] * 20, hoverinfo='none',
                   colorscale=colorscale, showscale=False,
                   xgap=1, ygap=1)
    ],
    layout=go.Layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font={'color': colors['text']},
        height=700,
        width=300,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=5, t=5, b=5, r=5),
        hovermode=False
    ),
)

fig_weight = go.Figure(
    data=[],
    layout=go.Layout(
        title={'text': 'Weight Distribution',
               'font': {'color': colors['text']}},
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font={'color': colors['text']},
        height=400
    )
)

app.layout = html.Div([
            html.Div([
                dcc.Graph(id='live-ls', figure=fig),
                dcc.Graph(id='live-ls-50', figure=fig_50),
                dcc.Graph(id='live-ls-pt', figure=fig_pt),
                dcc.Graph(id='live-loss', figure=fig_loss),
                dcc.Graph(id='live-data', figure=fig_data),
                dcc.Store(id='last-update', data=0),
            ]),
            html.Div([
                dcc.Graph(id='live-weights', figure=fig_weight),
                dcc.Store(id='last-w-update', data=0),
            ]),
            html.Div([
                html.Div([
                    dcc.Graph(id='live-board', figure=fig_board,
                              config={'displayModeBar': False}),
                ], className='board'),
                html.Div([
                    html.H3('Status:'),
                    html.Ul(children=[
                        html.Li(children=html.H4(children=['Combo: ', html.Data(id='live-combo', children=0)])),
                        html.Li(children=html.H4(children=['Lines Cleared: ', html.Data(id='live-lines', children=0)])),
                        html.Li(children=html.H4(children=['Score: ', html.Data(id='live-score', children=0)])),
                        html.Li(children=html.H4(children=['Single/Double/Triple/Tetris: ', html.Data(id='live-line_stats', children='0/0/0/0')])),
                        html.Li(children=html.H4(children=['Node removals since last game: ', html.Data(id='live-rmslg', children=0)])),
                        html.Li(children=html.H4(children=['Memory usage: ', html.Data(id='live-queue', children='0/0')])),
                    ])
                ], className='status'),
            ], className='box'),
            dcc.Interval(id='interval-component', interval=update_interval, n_intervals=0)
        ])


@app.callback(
    [Output('last-update', 'data'), Output('last-w-update', 'data')],
    [Input('interval-component', 'n_intervals')],
    [State('last-update', 'data'), State('last-w-update', 'data')])
def update_graphs(n, data, data_w):
    if log_parser.last_update != data:
        data = log_parser.last_update
    else:
        data = dash.no_update
    if model_parser.last_update != data_w:
        data_w = model_parser.last_update
    else:
        data_w = dash.no_update
    return [data, data_w]


@app.callback(Output('live-board', 'figure'),
              [Input('interval-component', 'n_intervals')])
def update_board(n):
    return fig_board


@app.callback(
    [Output('live-ls', 'figure'),
     Output('live-ls-50', 'figure'),
     Output('live-ls-pt', 'figure'),
     Output('live-loss', 'figure'),
     Output('live-data', 'figure')],
    [Input('last-update', 'data')])
def update_log_graphs(d):
    return fig, fig_50, fig_pt, fig_loss, fig_data


@app.callback(
    [Output('live-combo', 'children'),
     Output('live-lines', 'children'),
     Output('live-score', 'children'),
     Output('live-line_stats', 'children'),
     Output('live-rmslg', 'children'),
     Output('live-queue', 'children')],
    [Input('interval-component', 'n_intervals')])
def update_status(d):
    _tmp = '{}/{}/{}/{}'.format(*line_stats)
    _tmp2 = '{}/{}'.format(filled, size)
    return int(combo[0]), int(lines[0]), int(score[0]), _tmp, int(rmslg), _tmp2


@app.callback(Output('live-weights', 'figure'),
              [Input('last-w-update', 'data')])
def update_weight_graphs(d):
    return fig_weight


log_parser = Parser('../log_endless_dist')
model_parser = ModelParser()
status_parser = StatusParser()
rmslg, filled, size = 0, 0, 0
combo, lines, score = [0], [0], [0]
line_stats = [0, 0, 0, 0]


def log_parser_update():
    global log_parser
    global fig, fig_50, fig_pt, fig_loss, fig_data, fig_weight
    global rmslg, filled, size
    while True:
        update = log_parser.check_update()
        if update:
            line_cleared = log_parser.data['line_cleared']
            score = log_parser.data['score']
            fig.data[0]['x'] = list(range(len(line_cleared)))
            fig.data[0]['y'] = line_cleared
            fig.data[1]['x'] = list(range(len(score)))
            fig.data[1]['y'] = score

            _lc, _lc_err = [], []
            _sc, _sc_err = [], []
            for i in range((len(line_cleared) + 50) // 50):
                _tmp = line_cleared[i * 50: (i+1) * 50]
                _lc.append(np.mean(_tmp))
                _lc_err.append(np.std(_tmp) / np.sqrt(len(_tmp)))
                _tmp = score[i * 50: (i+1) * 50]
                _sc.append(np.mean(_tmp))
                _sc_err.append(np.std(_tmp) / np.sqrt(len(_tmp)))
            fig_50.data[0]['x'] = list(range((len(line_cleared) + 50) // 50))
            fig_50.data[0]['y'] = _lc
            fig_50.data[0]['error_y']['array'] = _lc_err
            fig_50.data[1]['x'] = list(range((len(line_cleared) + 50) // 50))
            fig_50.data[1]['y'] = _sc
            fig_50.data[1]['error_y']['array'] = _sc_err

            line_cleared_pt = log_parser.data['line_cleared_per_train']
            score_pt = log_parser.data['score_per_train']
            fig_pt.data[0]['x'] = list(range(len(line_cleared_pt)))
            fig_pt.data[0]['y'] = [x[0] for x in line_cleared_pt]
            fig_pt.data[0]['error_y']['array'] = [x[1] for x in line_cleared_pt]
            fig_pt.data[1]['x'] = list(range(len(score_pt)))
            fig_pt.data[1]['y'] = [x[0] for x in score_pt]
            fig_pt.data[1]['error_y']['array'] = [x[1] for x in score_pt]

            loss_train = log_parser.data['training_loss']
            loss_valid = log_parser.data['validation_loss']
            loss_valid_err = log_parser.data['validation_loss_err']
            g_norm = log_parser.data['g_norm']
            fig_loss.data[0]['x'] = list(range(len(loss_train)))
            fig_loss.data[0]['y'] = loss_train
            fig_loss.data[1]['x'] = list(range(len(loss_valid)))
            fig_loss.data[1]['y'] = loss_valid
            fig_loss.data[1]['error_y']['array'] = loss_valid_err
            fig_loss.data[2]['x'] = list(range(len(g_norm)))
            fig_loss.data[2]['y'] = g_norm

            data_acc = log_parser.data['data_accumulated']
            fig_data.data[0]['x'] = list(range(len(data_acc)))
            fig_data.data[0]['y'] = data_acc

            rmslg = log_parser.data['rm_since_last_game']
            filled = log_parser.data['filled']
            size = log_parser.data['size']

        if model_parser.check_update():
            _cols = 4
            d = model_parser.data
            _fig_weight = make_subplots(rows=ceil(len(d) / _cols), cols=_cols)
            for idx, (k, v) in enumerate(d.items()):
                r, c = divmod(idx, _cols)
                h = go.Histogram(x=v, name=k)
                _fig_weight.add_trace(h, row=r+1, col=c+1)
            _fig_weight.update_layout(
                    title={'text': 'Weight Distribution',
                           'font': {'color': colors['text']}},
                    plot_bgcolor=colors['background'],
                    paper_bgcolor=colors['background'],
                    font={'color': colors['text']},
                    height=(r + 1) * 400
                    )
            fig_weight = _fig_weight
            print('Model update done.')

        time.sleep(update_interval / 1000)


def status_parser_update():
    global combo, lines, score, line_stats
    while True:
        fig_board.data[0]['z'] = status_parser.board[::-1, :]
        combo = status_parser.combo
        lines = status_parser.lines
        score = status_parser.score
        line_stats = status_parser.line_stats
        time.sleep(update_interval / 1000)


if __name__ == '__main__':
    distributional = True
    log_parser = Parser(sys.argv[1])
    model_parser = ModelParser(distributional)
    thread_log = threading.Thread(target=log_parser_update)
    thread_log.start()
    thread_status = threading.Thread(target=status_parser_update)
    thread_status.start()

    app.run_server(host='0.0.0.0', port=8050, debug=False)
