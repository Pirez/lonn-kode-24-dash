import re
import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import dash_bootstrap_components as dbc

from dash import Input, Output, html, dcc 
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression

# Setup dash
app = dash.Dash(external_stylesheets=[dbc.themes.VAPOR])
server = app.server

df = pd.read_csv("kode24_salary_2020.csv", sep=";")
df.salary = df.salary.apply(
    lambda x: float(re.sub(r"[^a-zA-Z0-9\._-]", "", x.replace(",00 kr", "")))
)


df.arbeidserfaring = df.arbeidserfaring.apply(lambda x: int(x))

# Remove all salary above 1.6 mill
df = df[df.salary < 1_600_000]
df = df[df.salary > 400_000]

# Set an upper threashold
limit_arbeidserfaring = 20
df.arbeidserfaring = df.arbeidserfaring.apply(lambda x: min(x, limit_arbeidserfaring))

# Round up to closest 5 000 kr
df.salary = df.salary.apply(lambda x: round(x / 5_000) * 5_000)


whatOtherJobType = list(
    set(df.jobtype.unique()) - set(["fullstack", "backend", "frontend"])
)

df.jobtype = df.jobtype.apply(
    lambda x: x if x in ["fullstack", "backend", "frontend"] else "annet"
)

fylker = df.fylke.unique()
fylker.sort()
jobtitle = df.jobtitle.unique()
jobtitle.sort()
jobtype = df.jobtype.unique()
jobtype.sort()

df["jobtype_encoded"] = preprocessing.LabelEncoder().fit_transform(df.jobtype.values)
df["jobtitle_encoded"] = preprocessing.LabelEncoder().fit_transform(df.jobtitle.values)

df_sub_fylker = pd.get_dummies(df.fylke, prefix="fylke")
df_sub_jobtitle = pd.get_dummies(df.jobtitle, prefix="jobtitle")
df_sub_jobtype = pd.get_dummies(df.jobtype, prefix="jobtype")

fylker_values = df_sub_fylker.columns
jobtile_values = df_sub_jobtitle.columns
jobtype_values = df_sub_jobtype.columns

df_train = pd.concat(
    [df[["arbeidserfaring"]], df_sub_fylker, df_sub_jobtitle, df_sub_jobtype], axis=1
)

X = df_train.values
y = df.salary.values.reshape(-1, 1)

reg = LinearRegression().fit(X, y)
rscore = reg.score(X, y)

input_df = df_train.iloc[0].copy()
input_df[df_train.columns] = 0


form = dbc.FormFloating(
    [
        dbc.Input(id="arbeidserfaring", type="username", style={"width": "150px"}),
        dbc.Label("Antall år erfaring", style={"width": "150px"}),
    ]
)


dropdown_fylker = html.Div(
    [
        dbc.Label("Fylker", html_for="dropdown"),
        dcc.Dropdown(
            id="dropdown_fylker",
            value="fylke_Oslo",
            options=[
                {"label": label, "value": value}
                for label, value in zip(fylker, fylker_values)
            ],
        ),
    ],
    className="mb-3",
    style={"width": "50%"},
)


dropdown_jobtitle = html.Div(
    [
        dbc.Label("Jobbtype", html_for="dropdown"),
        dcc.Dropdown(
            id="dropdown_jobtitle",
            value="jobtitle_in-house",
            options=[
                {"label": label, "value": value}
                for label, value in zip(jobtitle, jobtile_values)
            ],
        ),
    ],
    className="mb-3",
    style={"width": "50%"},
)

dropdown_jobtype = html.Div(
    [
        dbc.Label("Beskrivelse av jobben", html_for="dropdown"),
        dcc.Dropdown(
            id="dropdown_jobtype",
            value="jobtype_fullstack",
            options=[
                {"label": label, "value": value}
                for label, value in zip(jobtype, jobtype_values)
            ],
        ),
    ],
    className="mb-3",
    style={"width": "50%"},
)


dropdown = dbc.DropdownMenu(
    [dbc.DropdownMenuItem(id=fylke.lower(), children=fylke) for fylke in fylker],
    label="Fylke",
    className="m-1",
    toggle_style={
        "textTransform": "uppercase",
        "background": "#FB79B3",
    },
    toggleClassName="fst border border-dark",
)

radioitems = html.Div(
    [
        dbc.Label("Sektor"),
        dbc.RadioItems(
            options=[
                {"label": "Offentlig", "value": 1},
                {"label": "Privat", "value": 2},
            ],
            value=1,
            id="radioitems-input",
        ),
    ]
)

slider = html.Div(
    [
        dbc.Label("Antall år erfaring", html_for="slider"),
        dcc.Slider(
            id="slider",
            min=0,
            max=limit_arbeidserfaring,
            step=1,
            value=2,
            updatemode="drag",
            tooltip={"placement": "bottom", "always_visible": False},
        ),
    ],
    className="mb-3",
    style={"width": "50%"},
)


histogram = dcc.Graph(
    id="histogram_salary",
    config={"displayModeBar": False},
    figure={
        "data": [
            {
                "x": df[df.fylke != "Oslo"]["salary"],
                "name": "Utenfor Oslo",
                "type": "histogram",
            },
            {
                "x": df[df.fylke == "Oslo"]["salary"],
                "name": "Oslo",
                "type": "histogram",
            },
        ],
        "layout": {
            "title": "Lønn basert på arbeidserfaring",
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "yaxis": {"showgrid": False},
        },
    },
)


linear = dcc.Graph(
    id="linear",
    config={"displayModeBar": False},
    figure={
        "data": [
            # px.scatter(df, x="arbeidserfaring", y="salary", color='fylke'),
            go.Scatter(
                x=df["arbeidserfaring"],
                y=df["salary"],
                mode="markers",
                marker_color=df["jobtitle_encoded"],
                text=[
                    f"{f} - {k} - {j} "
                    for f, j, k in zip(df["fylke"], df["jobtype"], df["jobtitle"])
                ],
            )
        ],
        "layout": {
            "title": "Fordeling basert på arbeidserfaring og jobbtype ",
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "yaxis": {"showgrid": False},
        },
    },
)


alert = dbc.Alert(
    [
        html.P(
            "Denne modellen er basert på en linær regresjon med en kontinuerlig variable (arbeidserfaring) "
            "og tre one-hot-encoding variabler (Fylker, Jobbtype, Beskrivelse av jobben). "
            "All lønn under 400k og over 1.6 mill ble fjernet. "
            f"Modellen har en R^2={rscore:0.2f}."
        ),
        html.Hr(),
        html.H4("Kilde: Kode24.no Lønnstall 2021", className="alert-heading"),
    ],
    color="primary",
)


app.layout = dbc.Container(
    [
        html.H2(children=f"Kode24 Lønnskalkulator 2021"),
        html.Br(),
        slider,
        dropdown_fylker,
        dropdown_jobtitle,
        dropdown_jobtype,
        # radioitems,
        html.Br(),
        html.H6(children=f"Forventet lønn"),
        html.H1(
            id="salaryOutput", children=f"{round(df.salary.mean()/5_000)*5_000:0.0f}"
        ),
        histogram,
        linear,
        # dbc.Alert("", color="primary"),
        alert,
        # html.H1(children=f"{round(df.salary.median()/10_000)*10_000:0.0f}"),
    ],
    className="p-5",
)


@app.callback(
    Output(component_id="histogram_salary", component_property="figure"),
    Input(component_id="slider", component_property="value"),
)
def histogram(aef):
    """."""
    return {
        "data": [
            {
                "x": df[(df.arbeidserfaring == aef) & (df.fylke != "Oslo")].salary,
                "type": "histogram",
                "name": "Utenfor Oslo",
            },
            {
                "x": df[(df.arbeidserfaring == aef) & (df.fylke == "Oslo")].salary,
                "type": "histogram",
                "name": "Oslo",
            },
        ],
        "layout": {
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "yaxis": {"showgrid": False},
            "title": f"Lønn ved arbeidserfaring {aef} år",
        },
    }


@app.callback(
    Output(component_id="salaryOutput", component_property="children"),
    [
        Input(component_id="slider", component_property="value"),
        Input(component_id="dropdown_fylker", component_property="value"),
        Input(component_id="dropdown_jobtitle", component_property="value"),
        Input(component_id="dropdown_jobtype", component_property="value"),
    ],
)
def calculate_salary(input_value, fylke_value, jobtitle_value, jobtype_value):
    """ Calulcated the expected salary based on the observation given as input """
    input_df.arbeidserfaring = input_value
    input_df[fylke_value] = 1
    input_df[jobtitle_value] = 1
    input_df[jobtype_value] = 1

    # Calculated the expected data
    y = reg.predict(input_df.values.reshape(1, -1))[0][0]
    expected_salary = f"{round(y/5_000)*5_000:0.0f},- kr"

    # Reset data
    input_df[df_train.columns] = 0

    return expected_salary


if __name__ == "__main__":
    app.run_server(debug=True)
