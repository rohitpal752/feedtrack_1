
import plotly.express as px
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import pandas as pd

def kpi_cards(c1, c2, c3, c4, df):
    c1.metric("Revenue", f"₹{df['revenue'].sum():,.0f}")
    c2.metric("Orders", f"{df['order_id'].nunique() if 'order_id' in df.columns else df.shape[0]}")
    c3.metric("Unique Customers", f"{df['user_id'].nunique() if 'user_id' in df.columns else 'N/A'}")
    c4.metric("Avg Rating", f"{df['rating'].dropna().mean():.2f}" if 'rating' in df.columns else "N/A")

def plot_timeseries_px(df, metric='revenue', freq='D'):
    df2 = df.set_index('order_date').resample(freq)[metric].sum().reset_index()
    fig = px.line(df2, x='order_date', y=metric, title=f"{metric.title()} over time ({freq})")
    return fig

def plot_top_products_px(df, groupby='platform', top_n=10):
    grp = df.groupby([groupby, 'product_name'])[ 'revenue'].sum().reset_index()
    top = grp.sort_values('revenue', ascending=False).groupby(groupby).head(top_n)
    fig = px.bar(top, x='product_name', y='revenue', color=groupby, title="Top Products by Revenue", hover_data=['revenue'])
    fig.update_layout(xaxis={'categoryorder':'total descending'})
    return fig

def plot_sentiment_pie(df_sent):
    counts = df_sent['sentiment'].value_counts().reset_index()
    counts.columns = ['sentiment','count']
    fig = px.pie(counts, values='count', names='sentiment', title='Sentiment distribution')
    return fig

def plot_wordcloud_matplotlib(df, text_col='review_text'):
    text = " ".join(df[text_col].dropna().astype(str).tolist())
    stopwords = set(STOPWORDS)
    wc = WordCloud(width=800, height=300, stopwords=stopwords, collocations=False).generate(text)
    fig, ax = plt.subplots(figsize=(10,3))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)
