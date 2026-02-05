import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import numpy as np

# Function to read all CSV files in the current directory
def read_csv_files():
    csv_files = [file for file in os.listdir('.') if file.endswith('.csv')]
    data = {file: pd.read_csv(file) for file in csv_files}
    for file in data:
        df = data[file]
        df['Step'] = df.index  # Add step column for easy plotting
    latest_file = max(csv_files, key=os.path.getmtime)
    return data, data[latest_file]

# Function to plot a metric
def plot_metric(fig, data, metric_name, row, col, ylabel=None, transform=None):
    for file, df in data.items():
        y = df[metric_name][1:] if transform is None else transform(df)
        fig.add_trace(
            go.Scatter(x=df['Step'][1:], y=y, name=metric_name, mode='lines'),
            row=row, col=col
        )
    fig.update_xaxes(title_text='Step', row=row, col=col)
    fig.update_yaxes(title_text=ylabel if ylabel else metric_name, row=row, col=col)
    fig.update_layout(title_text=metric_name, showlegend=True)

# Read CSV files
data, last_df = read_csv_files()

# Define metrics to plot
metrics = [
    ("Count episodes", None),
    ("Old episodes, %", lambda df: df["Count episodes with old"][1:] / df["Count episodes"][1:] * 100),
    ("Old ticks, %", lambda df: df["Count ticks with old"][1:] / (df["Count ticks with current"][1:] + df["Count ticks with old"][1:]) * 100),
    ("Average freeze time", None),
    ("Average ball hits", lambda df: df["Cumulative ball hits"][1:] / (df["Count episodes"][1:] / 2)),
    ("Average ball absolute velocity", None),
    ("Average ball velocity(x)", None),
    ("Average first bot reward", None),
    ("Average second bot reward", None),
    ("Average bots reward difference", lambda df: df["Average second bot reward"][1:] - df["Average first bot reward"][1:]),
    ("Highest reward per tick", None),
    ("Average current-old score difference", lambda df: (df["Current bot cumulative score"][1:] - df["Old bot cumulative score"][1:]) / df["Count episodes with old"][1:]),
    ("Actor loss", None),
    ("Critic loss", None),
    ("Critic Mean Absolute Error", None),
    ("Critic Correlation Coefficient", None),
    ("Training loss", None),
    ("Policy loss", lambda df: df["Critic loss"][1:] * 0.5 - df["Actor loss"][1:]),
    ("Entropy loss", lambda df: df["Entropy"][1:] * df["Entropy coefficient"][1:]),
    (["Entropy", "Minimal Entropy", "Maximal Entropy"], None),
    (["Angle entropy", "Minimal Angle entropy", "Maximal Angle entropy"], None),
    (["Hook entropy", "Minimal Hook entropy", "Maximal Hook entropy"], None),
    (["Hammer entropy", "Minimal Hammer entropy", "Maximal Hammer entropy"], None),
    (["Direction entropy", "Minimal Direction entropy", "Maximal Direction entropy"], None),
    ("Entropy coefficient", None),
    ("Actor grad norm", None),
    ("Critic grad norm", None),
    ("Actor weight norm", None),
    ("Critic weight norm", None),
    ("Actor activation mean", None),
    ("Actor activation std", None),
    (["Mean Ratio", "Std Ratio", "Minimal Ratio", "Maximal Ratio"], None),
    ("Approximate KL Divergence", None),
    ("Learning rate", None),
    ("Time to update", None)
]

# Number of time-related plots
num_add_plots = 5  # Time to process and Decide profiling

# Total number of plots
total_plots = len(metrics) + num_add_plots

# Calculate nrows and ncols dynamically
ncols = 3  # Fixed number of columns
nrows = math.ceil(total_plots / ncols)  # Calculate rows based on total plots and columns

# Create subplots
fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=[metric[0] if not isinstance(metric[0], list) else metric[0][0] for metric in metrics] + 
                    ['Average First/Second Bot Scores', 'Average bots score', 'Ticks per second', 'Time to process', 'Decide profiling'])

# Plot metrics
for i, (metric_name, transform) in enumerate(metrics):
    row = (i // ncols) + 1
    col = (i % ncols) + 1
    try:
        if not isinstance(metric_name, list):
            plot_metric(fig, data, metric_name, row, col, transform=transform)
        else:
            for metric in metric_name:
                plot_metric(fig, data, metric, row, col, transform=transform)
    except Exception as e:
        print("Unable to plot metric:", metric_name, "error:", e)

# Plot bot scores (calculated dynamically)
row = (len(metrics) // ncols) + 1
col = (len(metrics) % ncols) + 1
for file, df in data.items():
    fig.add_trace(
        go.Scatter(x=df['Step'], y=df['First bot cumulative score'] / (df['Count episodes'] / 2), 
                   name=f'First bot {file[:-4]}', mode='lines'),
        row=row, col=col
    )
    fig.add_trace(
        go.Scatter(x=df['Step'], y=df['Second bot cumulative score'] / (df['Count episodes'] / 2), 
                   name=f'Second bot {file[:-4]}', mode='lines'),
        row=row, col=col
    )
fig.update_xaxes(title_text='Step', row=row, col=col)
fig.update_yaxes(title_text='Average Score', row=row, col=col)
fig.update_layout(title_text='Average Bot Scores', showlegend=True)

def ema_tb(y, smoothing=0.9):
    y = np.asarray(y, dtype=float)
    alpha = 1 - smoothing
    s = np.zeros_like(y)
    s[0] = y[0]
    for i in range(1, len(y)):
        s[i] = alpha * y[i] + (1 - alpha) * s[i-1]
    return s

# Plot bot average score
row = ((len(metrics) + 1) // ncols) + 1
col = ((len(metrics) + 1) % ncols) + 1
for file, df in data.items():
    first_bot_avg_score = df['First bot cumulative score'] / (df['Count episodes'] / 2)
    second_bot_avg_score = df['Second bot cumulative score'] / (df['Count episodes'] / 2)
    fig.add_trace(
        go.Scatter(x=df['Step'], y=(first_bot_avg_score + second_bot_avg_score) / 2, 
                   name=f'Average score {file[:-4]}', mode='lines'),
        row=row, col=col
    )
    fig.add_trace(
        go.Scatter(x=df['Step'], y=ema_tb((first_bot_avg_score + second_bot_avg_score) / 2), 
                   name=f'Average score smoothed {file[:-4]}', mode='lines'),
        row=row, col=col
    )
fig.update_xaxes(title_text='Step', row=row, col=col)
fig.update_yaxes(title_text='Average Score', row=row, col=col)
fig.update_layout(title_text='Average Bots Score', showlegend=True)

# Plot Ticks per second
row = ((len(metrics) + 2) // ncols) + 1
col = ((len(metrics) + 2) % ncols) + 1
for file, df in data.items():
    fig.add_trace(
        go.Scatter(x=df['Step'], y=df['TPS'], name=f'Real {file[:-4]}', mode='lines'),
        row=row, col=col
    )
    fig.add_trace(
        go.Scatter(x=df['Step'], y=1000 / (df['Time to decide'] + df['Time to tick'] + df['Time rest']), 
                   name=f'Expected {file[:-4]}', mode='lines'),
        row=row, col=col
    )
fig.update_xaxes(title_text='Step', row=row, col=col)
fig.update_yaxes(title_text='TPS', row=row, col=col)
fig.update_layout(title_text='Ticks per second', showlegend=True)

# Plot time-related metrics
row = ((len(metrics) + 3) // ncols) + 1
col = ((len(metrics) + 3) % ncols) + 1
fig.add_trace(
    go.Scatter(x=last_df['Step'], y=last_df['Time to decide'], name='Time to decide', mode='lines'),
    row=row, col=col
)
fig.add_trace(
    go.Scatter(x=last_df['Step'], y=last_df['Time to tick'], name='Time to tick', mode='lines', line=dict(color='green')),
    row=row, col=col
)
fig.add_trace(
    go.Scatter(x=last_df['Step'], y=last_df['Time rest'], name='Time rest', mode='lines', line=dict(color='yellow')),
    row=row, col=col
)
fig.update_xaxes(title_text='Step', row=row, col=col)
fig.update_yaxes(title_text='Time taken, ms', row=row, col=col)
fig.update_layout(title_text='Time to process', showlegend=True)

# Plot Decide profiling
row = ((len(metrics) + 4) // ncols) + 1
col = ((len(metrics) + 4) % ncols) + 1
fig.add_trace(
    go.Scatter(x=last_df['Step'], y=last_df['Time pre forward'], name='Time pre forward', mode='lines'),
    row=row, col=col
)
fig.add_trace(
    go.Scatter(x=last_df['Step'], y=last_df['Time forward'], name='Time forward', mode='lines', line=dict(color='green')),
    row=row, col=col
)
fig.add_trace(
    go.Scatter(x=last_df['Step'], y=last_df['Time normal'], name='Time normal', mode='lines', line=dict(color='yellow')),
    row=row, col=col
)
fig.add_trace(
    go.Scatter(x=last_df['Step'], y=last_df['Time to cpu'], name='Time to cpu', mode='lines', line=dict(color='black')),
    row=row, col=col
)
fig.add_trace(
    go.Scatter(x=last_df['Step'], y=last_df['Time process last'], name='Time process last', mode='lines', line=dict(color='orange')),
    row=row, col=col
)
fig.update_xaxes(title_text='Step', row=row, col=col)
fig.update_yaxes(title_text='Time taken, ms', row=row, col=col)
fig.update_layout(title_text='Decide profiling', showlegend=True)

# Hide unused subplots
for i in range(total_plots, nrows * ncols):
    fig.update_xaxes(visible=False, row=(i // ncols) + 1, col=(i % ncols) + 1)
    fig.update_yaxes(visible=False, row=(i // ncols) + 1, col=(i % ncols) + 1)

# Adjust layout for full screen
fig.update_layout(
    autosize=True,  # Automatically resize to fit the screen
    margin=dict(l=0, r=0, t=40, b=0),  # Remove margins
    height=300 * nrows,  # Let Plotly handle height
    width=None,  # Let Plotly handle width
    title_text="Training Metrics", 
    showlegend=True
)

# Save and show plot
fig.write_html("data.html", auto_open=True)
