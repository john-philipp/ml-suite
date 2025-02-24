import colorsys
import hashlib
import os

import numpy as np
from dash import Dash, html, dash_table, dcc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc


def string_to_color(string: str) -> str:
    hash_object = hashlib.sha256(string.encode())
    hex_dig = hash_object.hexdigest()
    seed = int(hex_dig[:8], 16)
    hue = (seed % 360) / 360.0  # Ensure the hue is within [0, 1]
    saturation = 0.8  # Vibrant colors
    lightness = 0.5   # Medium lightness to avoid overly bright/dark colors

    # Convert HSL to RGB
    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)

    # Convert RGB values to HEX format
    r_hex = int(r * 255)
    g_hex = int(g * 255)
    b_hex = int(b * 255)

    return f"#{r_hex:02x}{g_hex:02x}{b_hex:02x}"


def string_to_shaded_color(string: str, shade_string: str) -> str:
    hash_object = hashlib.sha256(string.encode())
    hex_dig = hash_object.hexdigest()
    seed = int(hex_dig[:8], 16)
    hue = (seed % 360) / 360.0  # Map to [0, 1]

    # Hash the shade string to adjust the lightness
    shade_hash_object = hashlib.sha256(shade_string.encode())
    shade_hex_dig = shade_hash_object.hexdigest()
    shade_seed = int(shade_hex_dig[:8], 16)
    lightness = 0.3 + (shade_seed % 70) / 100.0  # Map to [0.3, 1.0] for visible shades

    # Set fixed saturation for vibrant colors
    saturation = 0.8

    # Convert HSL to RGB
    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)

    # Convert RGB values to HEX
    r_hex = int(r * 255)
    g_hex = int(g * 255)
    b_hex = int(b * 255)

    return f"#{r_hex:02x}{g_hex:02x}{b_hex:02x}"


def plot_1(fig, data_frame, type_, unique_on, x, y, row, col, y_max_min=0, shade_on_y=False):
    if isinstance(x, tuple):
        x_display_name = x[1]
        x = x[0]
    else:
        x_display_name = x
    if isinstance(y, tuple):
        y_display_name = y[1]
        y = y[0]
    else:
        y_display_name = y
    data_frame = data_frame[data_frame["type"] == type_]
    unique_values = data_frame[unique_on].unique()
    y_maxes = [y_max_min]
    for unique_value in unique_values:
        data_frame_ = data_frame[data_frame[unique_on] == unique_value].groupby(x).agg(**{
            x: (x, "mean"),
            y: (y, "mean")}
                                                                                       )
        # This allows shading within the same colour.
        # Keeps graphs visually linked while distinguishing
        # particular graphs between each other.
        unique_value_s = f"{unique_value}"
        properties = dict()
        if shade_on_y:
            properties["color"] = string_to_shaded_color(unique_value_s, y)
        else:
            properties["color"] = string_to_color(unique_value_s)

        fig.add_trace(
            go.Scatter(
                x=data_frame_[x],
                y=data_frame_[y],
                mode="markers+lines",
                line=properties,
                marker=properties,
                name=f"{y} ({unique_on}={unique_value})", legendgroup=f"{unique_value}"),
            row=row, col=col)

        y_maxes.append(max(data_frame_[y].values))

    y_max = max(y_maxes)

    fig.update_yaxes(title=y_display_name, range=[0, y_max + 0.1 * y_max], row=row, col=col)
    fig.update_xaxes(title=x_display_name, row=row, col=col)

    return y_max


if __name__ == '__main__':
    results_csv = "../.results/results-all.csv"
    assert os.path.exists(results_csv)

    df = pd.read_csv(results_csv, delimiter=";")

    app = Dash()

    # fig = make_subplots(rows=3, cols=2)
    # kwargs = dict(fig=fig, data_frame=df, unique_on="points", x="epochs_trained", shade_on_y=True)
    # y_max = plot_1(**kwargs, type_="training", y="mean_loss", row=1, col=1)
    # y_max = plot_1(**kwargs, type_="training", y="mean_acc", row=1, col=1, y_max_min=y_max)
    # plot_1(**kwargs, type_="training", y=("mean_iou", "mean_train_loss/acc/iou"), row=1, col=1, y_max_min=y_max)
    #
    # y_max = plot_1(**kwargs, type_="inference", y=("mean_acc", "mean_inf_acc"), row=1, col=2)
    # plot_1(**kwargs, type_="inference", y=("mean_iou", "mean_inf_acc/iou"), row=1, col=2, y_max_min=y_max)
    #
    # kwargs["shade_on_y"] = False
    # plot_1(**kwargs, type_="training", y="time_s", row=2, col=1)
    # plot_1(**kwargs, type_="inference", y=("mean_inf_time", "mean_inf_time_s"), row=2, col=2)
    # plot_1(**kwargs, type_="inference", y=("mem_used_mb", "mem_used_mb_inf"), row=3, col=2)
    # plot_1(**kwargs, type_="training", y=("mem_used_mb", "mem_used_mb_train"), row=3, col=1)

    available_points = [x for x in df["points"].unique() if x and not np.isnan(x)]
    app.layout = dbc.Container([
        dbc.Row([
            html.Div("Unique On"),
            dbc.RadioItems(
                options=[{"label": x, "value": x} for x in available_points],
                value=available_points[0],
                inline=False,
                id='radio-buttons-final')
        ]),

        dbc.Row([
            dbc.Col([
                dash_table.DataTable(data=df.to_dict('records'), page_size=12, style_table={'overflowX': 'auto'})
            ], width=6),

            dbc.Col([
                dcc.Graph(figure={}, id='my-first-graph-final')
            ], width=6),
        ]),
    ])

    app.run(debug=True)
