import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly
import plotly.graph_objs as go
import re, time, threading, json
import numpy as np
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

fig, fig_pt, fig_loss, fig_data, fig_weight, fig_board = [{'data': [], 'layout': {'paper_bgcolor': 'black', 'plot_bgcolor': 'black'}}] * 6

def serve_layout():
    return html.Div([
            html.Div([
                dcc.Graph(id='live-ls', figure=fig),
                dcc.Graph(id='live-ls-pt', figure=fig_pt),
                dcc.Graph(id='live-loss', figure=fig_loss),
                dcc.Graph(id='live-data', figure=fig_data),
                dcc.Store(id='last-update', data={'t': -1}),
            ]),
            html.Div([
                dcc.Graph(id='live-weights', figure=fig_weight),
                dcc.Store(id='last-w-update', data={'t': -1}),
            ]),
            html.Div([
                dcc.Graph(id='live-board', figure=fig_board, config={'displayModeBar': False}),
            ],
            style={
                'borderWidth': '1px',
                'borderStyle': 'solid',
                'borderColor': 'white',
                'width': '305px'}),
            dcc.Interval(id='interval-component', interval=1000, n_intervals=0)
        ])

app.layout = serve_layout

@app.callback(
    [Output('live-board', 'figure')], 
    [Input('interval-component', 'n_intervals')], 
    [State('live-board', 'figure')])
def update_board_graph(n, f):
    global fig_board
    if f == fig_board:
        return dash.no_update,
    else:
        return fig_board,

@app.callback(
    [Output('live-weights', 'figure'), Output('last-w-update', 'data')],
    [Input('interval-component', 'n_intervals')], 
    [State('last-w-update', 'data')])
def update_weight_graphs(n, data):
    if model_parser.last_update != data['t']:
        data['t'] = model_parser.last_update
        global fig_weight
        return fig_weight, data
    else:
        return [dash.no_update] * 2

@app.callback(
    [Output('live-ls', 'figure'), Output('live-ls-pt', 'figure'), Output('live-loss', 'figure'), Output('live-data', 'figure'), Output('last-update', 'data')], 
    [Input('interval-component', 'n_intervals')],
    [State('last-update', 'data')])
def update_log_graphs(n, data):
    if log_parser.last_log_update != data['t']:
        data['t'] = log_parser.last_log_update
        global fig, fig_pt, fig_loss, fig_data
        return fig, fig_pt, fig_loss, fig_data, data
    else:
        return [dash.no_update] * 5


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
            plot_lc = go.Scatter(x=list(range(len(line_cleared))), y=line_cleared, name='Lines Cleared', mode='lines')
            plot_sc = go.Scatter(x=list(range(len(score))), y=score, name='Score', mode='lines', yaxis='y2')
            layout = go.Layout(
                title={'text': 'Lines Cleared / Score', 'font': {'color': colors['text']}},    
                plot_bgcolor=colors['background'],
                paper_bgcolor=colors['background'],
                font={'color': colors['text']},
                xaxis={'title': 'Episode'},
                yaxis={'title': 'Lines Cleared', 'showgrid': False, 'rangemode': 'tozero'},
                yaxis2={'title': 'Score', 'side': 'right', 'overlaying': 'y', 'showgrid': False, 'rangemode': 'tozero'})

            fig = go.Figure(data=[plot_lc, plot_sc], layout=layout)
             
            line_cleared_pt = log_parser.data['line_cleared_per_train']
            score_pt = log_parser.data['score_per_train']
            lc_pt = np.array(log_parser.data['line_cleared_per_train'])
            if lc_pt.size > 0:
                plot_lc_pt = go.Scatter(x=list(range(len(lc_pt))), y=lc_pt[:,0], error_y=dict(type='data', array=lc_pt[:, 1], visible=True), name='Lines Cleared', mode='lines')
            else:
                plot_lc_pt = go.Scatter(x=[], y=[], error_y=dict(type='data', array=[], visible=True), name='Lines Cleared', mode='lines')
            sc_pt = np.array(log_parser.data['score_per_train'])
            if sc_pt.size > 0:
                plot_sc_pt = go.Scatter(x=list(range(len(sc_pt))), y=sc_pt[:,0], error_y=dict(type='data', array=sc_pt[:, 1], visible=True), name='Score', mode='lines', yaxis='y2')
            else:
                plot_sc_pt = go.Scatter(x=[], y=[], error_y=dict(type='data', array=[], visible=True), name='Score', mode='lines', yaxis='y2')
            layout_pt = go.Layout(
                title={'text': 'Lines Cleared / Score per Training Session', 'font': {'color': colors['text']}},    
                plot_bgcolor=colors['background'],
                paper_bgcolor=colors['background'],
                font={'color': colors['text']},
                xaxis={'title': 'Training Session'},
                yaxis={'title': 'Lines Cleared', 'showgrid': False},
                yaxis2={'title': 'Score', 'side': 'right', 'overlaying': 'y', 'showgrid': False})

            fig_pt = go.Figure(data=[plot_lc_pt, plot_sc_pt], layout=layout_pt)

            loss_train = log_parser.data['training_loss']
            loss_valid = log_parser.data['validation_loss']
            plot_train = go.Scatter(x=list(range(len(loss_train))), y=loss_train, name='Training Loss', mode='lines')
            plot_valid = go.Scatter(x=list(range(len(loss_train))), y=loss_valid, name='Validation Loss', mode='lines')
            layout_loss = go.Layout(
                title={'text': 'Training / Validation Loss', 'font': {'color': colors['text']}},
                plot_bgcolor=colors['background'],
                paper_bgcolor=colors['background'],
                font={'color': colors['text']})
            
            fig_loss = go.Figure(data=[plot_train, plot_valid], layout=layout_loss)

            data_acc = log_parser.data['data_accumulated']
            if data_acc:
                plot_data = go.Scatter(x=list(range(len(data_acc))), y=data_acc, name='Data Accumulated', mode='lines')
                layout_data = go.Layout(
                    title={'text': 'Accumulated Training Data', 'font': {'color': colors['text']}},
                    plot_bgcolor=colors['background'],
                    paper_bgcolor=colors['background'],
                    font={'color': colors['text']})
     
                fig_data = go.Figure(data=[plot_data], layout=layout_data)
            
        if model_parser.check_update():
            _cols = 4
            d = model_parser.data
            fig_weight = plotly.subplots.make_subplots(rows=ceil(len(d) / _cols), cols=_cols)
            idx = 0
            for k, v in d.items():
                r, c = divmod(idx, _cols)
                h = go.Histogram(x=v, name=k)
                fig_weight.add_trace(h, row=r+1, col=c+1)
                idx += 1
            fig_weight.update_layout(
                    title={'text': 'Weight Distribution', 'font': {'color': colors['text']}},   
                    plot_bgcolor=colors['background'],
                    paper_bgcolor=colors['background'],
                    font={'color': colors['text']},
                    height = ceil(idx / _cols) * 400
                    )

        board_parser.update()
        d = board_parser.data[::-1,:]
        colorscale=[[0, 'rgb(100, 100, 100)'], [0.5, 'rgb(0, 0, 0)'], [1, 'rgb(255, 255, 255)']]
        hmap = go.Heatmap(z=d, hoverinfo='none', colorscale=colorscale, showscale=False, xgap=1, ygap=1)
        layout_board = go.Layout(
                plot_bgcolor=colors['background'],
                paper_bgcolor=colors['background'],
                font={'color': colors['text']},
                height=760,
                width=300,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                margin=dict(l=5, t=5, b=5, r=5),
                hovermode=False)
        fig_board = go.Figure(data=[hmap], layout=layout_board)
        time.sleep(.5)

if __name__ == '__main__':
    thread_log = threading.Thread(target=parser_update)
    thread_log.start()

    app.run_server(host='0.0.0.0', port=8050, debug=False)
