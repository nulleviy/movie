import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State

def load_and_preprocess(file_path):
    try:
        # Определим названия колонок на основе примера файла
        columns = [
            "title", "rating", "genre", "year", "release_date", "score", "votes",
            "director", "writer", "actor", "country", "budget", "gross", "studio", "runtime"
        ]

        # Чтение данных с обработкой ошибок
        df = pd.read_csv(file_path, on_bad_lines='skip', names=columns, header=None, engine='python')

        # Удаление дубликатов
        df = df.drop_duplicates(subset=['title', 'year'])

        # Приведение типов
        df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)
        df['score'] = pd.to_numeric(df['score'], errors='coerce')
        df['gross'] = pd.to_numeric(df['gross'], errors='coerce')
        df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
        df['votes'] = pd.to_numeric(df['votes'], errors='coerce')

        # Обработка даты релиза
        df['release_date'] = pd.to_datetime(
            df['release_date'].str.extract(r'(\w+ \d{1,2}, \d{4})')[0],
            errors='coerce'
        )

        # Расчет ROI (Return on Investment)
        df['roi'] = (df['gross'] - df['budget']) / df['budget']

        # Категоризация тональности
        df['sentiment'] = df['score'].apply(
            lambda x: 'Positive' if x >= 6.5 else ('Neutral' if 5 <= x < 6.5 else 'Negative')
        )

        # Разделение жанров (если их несколько через запятую)
        df['genres'] = df['genre'].str.split(',')
        df = df.explode('genres')
        df['genres'] = df['genres'].str.strip()

        # Удаление пропусков в ключевых столбцах
        df = df.dropna(subset=['score', 'genres', 'year'])

        return df

    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return pd.DataFrame()


# --- 2. Улучшенные визуализации ---
def plot_rating_distribution(df):
    fig = px.histogram(
        df, x='score', nbins=20,
        title='Распределение рейтингов фильмов',
        labels={'score': 'Рейтинг IMDB'},
        template='plotly_white',
        color='sentiment',
        color_discrete_map={
            'Positive': 'green',
            'Neutral': 'blue',
            'Negative': 'red'
        }
    )
    fig.update_layout(bargap=0.1)
    return fig


def plot_avg_rating_by_genre(df, top_n=15):
    genre_ratings = df.groupby('genres')['score'].agg(['mean', 'count'])
    genre_ratings = genre_ratings[genre_ratings['count'] >= 10]  # Фильтр по количеству фильмов
    genre_ratings = genre_ratings.sort_values('mean', ascending=False).head(top_n).reset_index()

    fig = px.bar(
        genre_ratings, x='genres', y='mean',
        title=f'Топ-{top_n} жанров по среднему рейтингу (минимум 10 фильмов)',
        labels={'mean': 'Средний рейтинг', 'genres': 'Жанр'},
        template='plotly_white',
        text='mean',
        color='mean',
        color_continuous_scale='Viridis'
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    return fig


def plot_sentiment_distribution(df):
    sentiment_counts = df['sentiment'].value_counts().reset_index()
    fig = px.pie(
        sentiment_counts, values='count', names='sentiment',
        title='Распределение тональности отзывов',
        hole=0.3,
        color='sentiment',
        color_discrete_map={
            'Positive': 'green',
            'Neutral': 'blue',
            'Negative': 'red'
        }
    )
    fig.update_traces(textinfo='percent+label')
    return fig


def plot_rating_trend_over_years(df):
    yearly_stats = df.groupby('year').agg({
        'score': 'mean',
        'title': 'count'
    }).reset_index()
    yearly_stats = yearly_stats[yearly_stats['title'] >= 5]  # Фильтр по количеству фильмов

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Средний рейтинг
    fig.add_trace(
        go.Scatter(
            x=yearly_stats['year'], y=yearly_stats['score'],
            name='Средний рейтинг', mode='lines+markers',
            line=dict(color='royalblue', width=2)
        ),
        secondary_y=False,
    )

    # Количество фильмов
    fig.add_trace(
        go.Bar(
            x=yearly_stats['year'], y=yearly_stats['title'],
            name='Количество фильмов', opacity=0.3,
            marker=dict(color='lightgray')
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title_text='Динамика среднего рейтинга и количества фильмов по годам',
        template='plotly_white'
    )
    fig.update_xaxes(title_text="Год")
    fig.update_yaxes(title_text="Средний рейтинг", secondary_y=False)
    fig.update_yaxes(title_text="Количество фильмов", secondary_y=True)

    return fig


def plot_budget_vs_gross(df):
    fig = px.scatter(
        df, x='budget', y='gross', color='score',
        title='Зависимость сборов от бюджета',
        labels={'budget': 'Бюджет ($)', 'gross': 'Сборы ($)', 'score': 'Рейтинг'},
        template='plotly_white',
        hover_data=['title', 'year', 'genres'],
        color_continuous_scale='Viridis'
    )
    fig.update_layout(
        xaxis_type="log",
        yaxis_type="log",
        coloraxis_colorbar=dict(title="Рейтинг")
    )
    return fig


def plot_top_directors(df, top_n=10):
    director_stats = df.groupby('director').agg({
        'score': 'mean',
        'title': 'count',
        'gross': 'sum'
    }).reset_index()
    director_stats = director_stats[director_stats['title'] >= 3]  # Минимум 3 фильма
    director_stats = director_stats.sort_values('score', ascending=False).head(top_n)

    fig = px.bar(
        director_stats, x='director', y='score',
        title=f'Топ-{top_n} режиссеров по среднему рейтингу (минимум 3 фильма)',
        labels={'score': 'Средний рейтинг', 'director': 'Режиссер'},
        template='plotly_white',
        hover_data=['title', 'gross'],
        color='score',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(xaxis={'categoryorder': 'total descending'})
    return fig


# --- 3. Улучшенная интерактивная панель с Dash ---
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# Загрузка данных
df_full = load_and_preprocess("movies.csv")

# Получаем уникальные жанры для фильтров
available_genres = sorted(df_full['genres'].unique())
min_year = int(df_full['year'].min())
max_year = int(df_full['year'].max())

app.layout = html.Div([
    html.Div([
        html.H1("Анализ фильмов IMDB", style={'textAlign': 'center'}),
        html.P("Интерактивная панель для анализа фильмов по различным параметрам",
               style={'textAlign': 'center', 'color': 'gray'})
    ], className='header'),

    html.Div([
        html.Div([
            html.Label("Диапазон годов"),
            dcc.RangeSlider(
                id='year-slider',
                min=min_year,
                max=max_year,
                value=[min_year, max_year],
                marks={y: str(y) for y in range(min_year, max_year + 1, 5)},
                step=1,
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], className='slider-container'),

        html.Div([
            html.Label("Выберите жанры"),
            dcc.Dropdown(
                id='genre-dropdown',
                options=[{'label': genre, 'value': genre} for genre in available_genres],
                multi=True,
                placeholder="Все жанры"
            )
        ], className='dropdown-container'),

        html.Div([
            html.Label("Минимальный рейтинг"),
            dcc.Slider(
                id='rating-slider',
                min=0,
                max=10,
                value=0,
                step=0.5,
                marks={i: str(i) for i in range(0, 11, 2)},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], className='slider-container'),

        html.Div([
            html.Label("Минимальное количество голосов"),
            dcc.Input(
                id='votes-input',
                type='number',
                placeholder='Минимум голосов',
                min=0,
                value=0
            )
        ], className='input-container'),
    ], className='controls'),

    dcc.Tabs(id='tabs', value='tab-ratings', children=[
        dcc.Tab(label='Рейтинги', value='tab-ratings'),
        dcc.Tab(label='Финансы', value='tab-finance'),
        dcc.Tab(label='Режиссеры', value='tab-directors'),
    ]),

    html.Div(id='graphs-container')
])


@app.callback(
    Output('graphs-container', 'children'),
    [Input('tabs', 'value'),
     Input('year-slider', 'value'),
     Input('genre-dropdown', 'value'),
     Input('rating-slider', 'value'),
     Input('votes-input', 'value')]
)
def update_tab_content(tab, years, genres, min_rating, min_votes):
    # Фильтрация данных
    filtered_df = df_full[
        (df_full['year'] >= years[0]) &
        (df_full['year'] <= years[1]) &
        (df_full['score'] >= min_rating) &
        (df_full['votes'] >= min_votes)
        ]

    if genres:
        filtered_df = filtered_df[filtered_df['genres'].isin(genres)]

    if tab == 'tab-ratings':
        return html.Div([
            html.Div([
                dcc.Graph(figure=plot_rating_distribution(filtered_df)),
                dcc.Graph(figure=plot_sentiment_distribution(filtered_df))
            ], className='row'),
            html.Div([
                dcc.Graph(figure=plot_avg_rating_by_genre(filtered_df)),
                dcc.Graph(figure=plot_rating_trend_over_years(filtered_df))
            ], className='row')
        ])

    elif tab == 'tab-finance':
        return html.Div([
            html.Div([
                dcc.Graph(figure=plot_budget_vs_gross(filtered_df))
            ], className='row'),
            html.Div([
                # Можно добавить дополнительные финансовые графики
            ], className='row')
        ])

    elif tab == 'tab-directors':
        return html.Div([
            html.Div([
                dcc.Graph(figure=plot_top_directors(filtered_df))
            ], className='row')
        ])


# --- 4. Запуск приложения ---
if __name__ == '__main__':
    app.run(debug=True, port=8050)