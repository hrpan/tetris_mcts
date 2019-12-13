import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import time
import threading
from parseLog import Parser, ModelParser, BoardParser
from dash.dependencies import Input, Output, State
from math import ceil

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
                'rangemode': 'tozero'}
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
                'showgrid': False, 'rangemode': 'tozero'}
    )
)

fig_loss = go.Figure(
    data=[
        go.Scatter(x=[], y=[], name='Training Loss', mode='lines'),
        go.Scatter(x=[], y=[], name='Validation Loss', mode='lines')
    ],
    layout=go.Layout(
        title={'text': 'Training / Validation Loss',
               'font': {'color': colors['text']}},
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font={'color': colors['text']},
        xaxis={'title': 'Iteration', 'rangemode': 'tozero'},
        yaxis={'rangemode': 'tozero'}
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
        yaxis={'rangemode': 'tozero'}
    )
)

colorscale = [[0, 'rgb(100, 100, 100)'],
              [0.5, 'rgb(0, 0, 0)'],
              [1, 'rgb(255, 255, 255)']]
fig_board = go.Figure(
    data=[
        go.Heatmap(z=[[0] * 10] * 22, hoverinfo='none',
                   colorscale=colorscale, showscale=False,
                   xgap=1, ygap=1)
    ],
    layout=go.Layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font={'color': colors['text']},
        height=760,
        width=300,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=5, t=5, b=5, r=5),
        hovermode=False
    ),
)

fig_weight = [{'data': [],
               'layout': {'paper_bgcolor': 'black', 'plot_bgcolor': 'black'}}]

rmslg_str = 'Node removals since last game: {}'
queue_str = 'Queue usage: {} / {}'
app.layout = html.Div([
            html.Div([
                dcc.Graph(id='live-ls', figure=fig),
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
                        html.Li(id='live-rmslg',
                                children=rmslg_str.format(0)),
                        html.Li(id='live-queue',
                                children=queue_str.format(0, 0))
                    ])
                ], className='status'),
            ], className='box'),
            dcc.Interval(id='interval-component', interval=1000, n_intervals=0)
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
     Output('live-ls-pt', 'figure'),
     Output('live-loss', 'figure'),
     Output('live-data', 'figure')],
    [Input('last-update', 'data')])
def update_log_graphs(d):
    return fig, fig_pt, fig_loss, fig_data


@app.callback(
    [Output('live-rmslg', 'children'), Output('live-queue', 'children')],
    [Input('last-update', 'data')])
def update_status(d):
    rmslg = log_parser.data['rm_since_last_game']
    filled = log_parser.data['filled']
    size = log_parser.data['size']
    return rmslg_str.format(rmslg), queue_str.format(filled, size)


@app.callback(Output('live-weights', 'figure'),
              [Input('last-w-update', 'data')])
def update_weight_graphs(d):
    return fig_weight


log_parser = Parser('../log_endless')
model_parser = ModelParser()
board_parser = BoardParser()


def parser_update():
    global log_parser
    global fig, fig_pt, fig_loss, fig_data, fig_weight, fig_board
    while True:
        update = log_parser.check_update()
        if update:
            line_cleared = log_parser.data['line_cleared']
            score = log_parser.data['score']
            fig.data[0]['x'] = list(range(len(line_cleared)))
            fig.data[0]['y'] = line_cleared
            fig.data[1]['x'] = list(range(len(score)))
            fig.data[1]['y'] = score

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
            fig_loss.data[0]['x'] = list(range(len(loss_train)))
            fig_loss.data[0]['y'] = loss_train
            fig_loss.data[1]['x'] = list(range(len(loss_valid)))
            fig_loss.data[1]['y'] = loss_valid

            data_acc = log_parser.data['data_accumulated']
            fig_data.data[0]['x'] = list(range(len(data_acc)))
            fig_data.data[0]['y'] = data_acc

        if model_parser.check_update():
            _cols = 4
            d = model_parser.data
            fig_weight = make_subplots(rows=ceil(len(d) / _cols), cols=_cols)
            idx = 0
            for k, v in d.items():
                r, c = divmod(idx, _cols)
                h = go.Histogram(x=v, name=k)
                fig_weight.add_trace(h, row=r+1, col=c+1)
                idx += 1
            fig_weight.update_layout(
                    title={'text': 'Weight Distribution',
                           'font': {'color': colors['text']}},
                    plot_bgcolor=colors['background'],
                    paper_bgcolor=colors['background'],
                    font={'color': colors['text']},
                    height=ceil(idx / _cols) * 400
                    )

        board_parser.update()
        fig_board.data[0]['z'] = board_parser.data[::-1, :]
        time.sleep(1)


if __name__ == '__main__':
    thread_log = threading.Thread(target=parser_update)
    thread_log.start()

    app.run_server(host='0.0.0.0', port=8050, debug=False)
