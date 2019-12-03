import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly
import plotly.graph_objs as go
import re
import numpy as np
from parseLog import Parser
from dash.dependencies import Input, Output


external_stylesheets = ['./static/css/style.css']

colors = {'background': '#000000', 'text': '#D3D3D3'}

style = {'font'}

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {
    'background': '#000000',
    'text': '#646464'
}


app.layout = html.Div(
    children=[
    html.Div([
        dcc.Graph(id='live-ls'),
        dcc.Graph(id='live-ls-pt'),
        dcc.Graph(id='live-loss'),
        dcc.Interval(
            id='interval-component',
            interval=1000,
            n_intervals=0
        )
    ])
    ],
)

log_parser = Parser('../log_endless')
@app.callback([Output('live-ls', 'figure'), Output('live-ls-pt', 'figure'), Output('live-loss', 'figure')], [Input('interval-component', 'n_intervals')])
def update_log_graphs(n):
    update = log_parser.check_update()
    if update:
        line_cleared = log_parser.data['line_cleared']
        plot_lc = go.Scatter(x=list(range(len(line_cleared))), y=line_cleared, name='Lines Cleared', mode='lines')
        score = log_parser.data['score']
        plot_sc = go.Scatter(x=list(range(len(score))), y=score, name='Score', mode='lines', yaxis='y2')
        layout = go.Layout(
            title={'text': 'Lines Cleared / Score', 'font': {'color': colors['text']}},    
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font={'color': colors['text']},
            xaxis={'title': 'Episode'},
            yaxis={'title': 'Lines Cleared', 'showgrid': False},
            yaxis2={'title': 'Score', 'side': 'right', 'overlaying': 'y', 'showgrid': False})

        fig = go.Figure(data=[plot_lc, plot_sc], layout=layout)

        lc_pt = np.array(log_parser.data['line_cleared_per_train'])
        plot_lc_pt = go.Scatter(x=list(range(len(lc_pt))), y=lc_pt[:,0], error_y=dict(type='data', array=lc_pt[:, 1], visible=True), name='Lines Cleared', mode='lines')
        sc_pt = np.array(log_parser.data['score_per_train'])
        plot_sc_pt = go.Scatter(x=list(range(len(sc_pt))), y=sc_pt[:,0], error_y=dict(type='data', array=sc_pt[:, 1], visible=True), name='Score', mode='lines', yaxis='y2')
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
        plot_train = go.Scatter(x=list(range(len(loss_train))), y=loss_train, name='Training Loss', mode='lines')
        loss_valid = log_parser.data['validation_loss']
        plot_valid = go.Scatter(x=list(range(len(loss_train))), y=loss_valid, name='Validation Loss', mode='lines')
        layout_loss = go.Layout(
            title={'text': 'Training / Validation Loss', 'font': {'color': colors['text']}},
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font={'color': colors['text']})
        
        fig_loss = go.Figure(data=[plot_train, plot_valid], layout=layout_loss)
        print(fig, fig_pt, fig_loss)
        return [fig, fig_pt, fig_loss]
    else:
        return [dash.no_update] * 3




if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050, debug=True)
