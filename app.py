
import streamlit as st
import pandas as pd
from src.data_utils import load_all_and_merge, prepare_filters
from src.viz import kpi_cards, plot_timeseries_px, plot_top_products_px, plot_sentiment_pie, plot_wordcloud_matplotlib
from src.nlp_utils import compute_sentiment
from src.ml_utils import create_rfm, cluster_customers, build_item_similarity_recommender

st.set_page_config(page_title="Advanced Multi-Platform Reviews Dashboard", layout="wide")

@st.cache_data
def load_data():
    paths = {
        "meesho": "data/meesho_reviews.csv",
        "nykaa": "data/nykaa_reviews.csv",
        "cred": "data/cred_reviews.csv",
        "dunzo": "data/dunzo_reviews.csv",
        "razorpay": "data/razorpay_reviews.csv",
        "swiggy": "data/swiggy_reviews.csv",
        "zomato": "data/zomato_reviews.csv"
    }
    df = load_all_and_merge(paths)
    return df

df = load_data()

st.title("Advanced Multi-Platform Reviews & Sales Dashboard")
st.markdown("Datasets: meesho | nykaa | cred | dunzo | razorpay | swiggy | zomato")

# sidebar filters
st.sidebar.header("Filters & Controls")
date_min, date_max, platform_sel, city_sel, text_search = prepare_filters(df)

# apply filters
mask = (df['order_date'] >= pd.to_datetime(date_min)) & (df['order_date'] <= pd.to_datetime(date_max))
if platform_sel:
    mask &= df['platform'].isin(platform_sel)
if city_sel:
    mask &= df['city'].isin(city_sel)
if text_search:
    mask &= df['review_text'].fillna("").str.contains(text_search, case=False, na=False)
df_f = df[mask].copy()

# KPIs
k1, k2, k3, k4 = st.columns(4)
kpi_cards(k1, k2, k3, k4, df_f)

tabs = st.tabs(["Overview", "Platforms", "Products", "Reviews (NLP)", "Customers (RFM & Clusters)", "Recommender", "Export & Notes"])

with tabs[0]:
    st.header("Overview")
    st.write("Daily revenue / reviews time-series")
    if 'order_date' in df_f.columns:
        st.plotly_chart(plot_timeseries_px(df_f, metric='revenue', freq='D'), use_container_width=True)
    st.write("Top products overall")
    st.plotly_chart(plot_top_products_px(df_f, groupby='platform', top_n=15), use_container_width=True)

with tabs[1]:
    st.header("Platform Comparison")
    st.write("Platform-level KPIs and comparison")
    st.dataframe(df_f.groupby('platform').agg({
        'revenue':'sum','order_id':'nunique','user_id':'nunique','rating':'mean'
    }).rename(columns={'order_id':'orders','user_id':'unique_customers'}).sort_values('revenue', ascending=False))

with tabs[2]:
    st.header("Products")
    prod = df_f.groupby(['platform','product_id','product_name']).agg({'revenue':'sum','order_id':'nunique','rating':'mean'}).reset_index()
    st.dataframe(prod.sort_values('revenue', ascending=False).head(200))
    st.download_button("Download filtered products CSV", prod.to_csv(index=False), file_name="filtered_products.csv")

with tabs[3]:
    st.header("Reviews & NLP")
    if 'review_text' in df_f.columns:
        with st.spinner("Computing sentiment..."):
            df_sent = compute_sentiment(df_f, text_col='review_text')
        st.plotly_chart(plot_sentiment_pie(df_sent), use_container_width=True)
        st.write("Sample reviews by sentiment")
        st.dataframe(df_sent[['platform','order_date','rating','sentiment','review_text']].sort_values('order_date', ascending=False).head(300))
        st.markdown("Wordcloud of most common words (filtered)")
        plot_wordcloud_matplotlib(df_sent, text_col='review_text')
    else:
        st.info("No review_text column found in filtered data.")

with tabs[4]:
    st.header("Customer Segmentation (RFM + KMeans)")
    if 'user_id' in df_f.columns:
        rfm = create_rfm(df_f)
        clusters = cluster_customers(rfm, n_clusters=4)
        st.dataframe(clusters.head(200))
        st.download_button("Download RFM+clusters CSV", clusters.to_csv(index=False), file_name="rfm_clusters.csv")
    else:
        st.info("No user_id to build RFM.")

with tabs[5]:
    st.header("Item-Item Similarity Recommender (popularity + cosine)")
    if 'product_id' in df_f.columns:
        rec = build_item_similarity_recommender(df_f)
        sel_prod = st.selectbox("Choose a product (product_name)", rec['product_name'].unique().tolist()[:500])
        st.write("Recommendations (top 10 similar items):")
        st.table(rec[rec['product_name']==sel_prod].iloc[0]['recommendations'])
    else:
        st.info("No product_id present for recommendations.")

with tabs[6]:
    st.header("Export & Notes")
    st.markdown("You can download the cleaned merged dataset for presentation or further analysis.")
    st.download_button("Download merged CSV", df_f.to_csv(index=False), file_name="merged_filtered_data.csv")
    st.markdown("Notes: This dashboard includes sentiment, RFM clustering, item-item similarity recommender, and interactive filters. You can extend forecasting, anomaly detection, and topic modeling in the `src/` modules.")
