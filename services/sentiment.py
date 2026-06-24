import plotly.graph_objects as go

# Financial sentiment lexicon
SENTIMENT_DICT = {
    # Positive words
    "surge": 0.8,
    "breakout": 0.7,
    "rally": 0.8,
    "growth": 0.5,
    "gain": 0.5,
    "support": 0.4,
    "bullish": 0.7,
    "high": 0.4,
    "jump": 0.6,
    "soar": 0.8,
    "optimism": 0.6,
    "profit": 0.5,
    "accumulate": 0.4,
    "upside": 0.6,
    # Negative words
    "plunge": -0.8,
    "crash": -0.9,
    "drop": -0.5,
    "pullback": -0.4,
    "bearish": -0.7,
    "low": -0.4,
    "fear": -0.6,
    "loss": -0.5,
    "selloff": -0.7,
    "warning": -0.5,
    "downside": -0.6,
    "panic": -0.8,
    "risk": -0.4,
    "pressure": -0.4,
    "deficit": -0.5,
}

def analyze_sentiment(headlines: list[str]) -> float:
    """
    Perform a lexicon-based sentiment analysis on a list of headlines.
    Returns a score between -1.0 (Extreme Fear) and +1.0 (Extreme Greed).
    """
    if not headlines:
        return 0.0 # Neutral

    total_score = 0.0
    word_count = 0

    for headline in headlines:
        headline_lower = headline.lower()
        score = 0.0
        matches = 0
        for word, val in SENTIMENT_DICT.items():
            if word in headline_lower:
                score += val
                matches += 1
        if matches > 0:
            total_score += (score / matches)
            word_count += 1

    if word_count > 0:
        return max(-1.0, min(1.0, total_score / word_count))
    return 0.0 # Neutral fallback

def build_sentiment_gauge(score: float, is_dark: bool = False) -> go.Figure:
    """
    Build a semi-circular speedometer gauge chart for the Fear & Greed index.
    - score: float between -1.0 and +1.0
    - is_dark: whether to style the chart for dark mode
    """
    # Map -1.0 -> 1.0 to 0 -> 100
    gauge_value = (score + 1.0) / 2.0 * 100
    
    # Determine label
    if gauge_value < 25:
        label = "Extreme Fear 😱"
        color = "#ef5350"
    elif gauge_value < 45:
        label = "Fear 😨"
        color = "#ff7043"
    elif gauge_value < 55:
        label = "Neutral 😐"
        color = "#ffca28"
    elif gauge_value < 75:
        label = "Greed 🤑"
        color = "#66bb6a"
    else:
        label = "Extreme Greed 🚀"
        color = "#26a69a"

    text_color = "#eaecef" if is_dark else "#1e2329"
    gauge_bg = "rgba(255, 255, 255, 0.05)" if is_dark else "#f0f2f5"
    gauge_border = "#2b3139" if is_dark else "#eaecef"
    tick_color = "#848e9c" if is_dark else "#707a8a"

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = gauge_value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Macro Sentiment: {label}", 'font': {'size': 16, 'family': 'Inter, sans-serif', 'weight': 'bold', 'color': text_color}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': tick_color},
            'bar': {'color': color, 'thickness': 0.25},
            'bgcolor': gauge_bg,
            'borderwidth': 1,
            'bordercolor': gauge_border,
            'steps': [
                {'range': [0, 25], 'color': 'rgba(239, 83, 80, 0.15)'},
                {'range': [25, 45], 'color': 'rgba(255, 112, 67, 0.12)'},
                {'range': [45, 55], 'color': 'rgba(255, 202, 40, 0.12)'},
                {'range': [55, 75], 'color': 'rgba(102, 187, 106, 0.12)'},
                {'range': [75, 100], 'color': 'rgba(38, 166, 154, 0.15)'}
            ],
            'threshold': {
                'line': {'color': color, 'width': 4},
                'thickness': 0.75,
                'value': gauge_value
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': text_color, 'family': "Inter, sans-serif"},
        height=220,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig
