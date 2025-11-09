
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def create_rfm(df, user_col='user_id', date_col='order_date', revenue_col='revenue'):
    snapshot = df[date_col].max() + pd.Timedelta(days=1)
    agg = df.groupby(user_col).agg({
        date_col: lambda x: (snapshot - x.max()).days,
        'order_id': 'nunique',
        revenue_col: 'sum'
    }).reset_index().rename(columns={date_col:'recency','order_id':'frequency', revenue_col:'monetary'})
    agg = agg.fillna(0)
    return agg

def cluster_customers(rfm_df, n_clusters=4):
    km = KMeans(n_clusters=n_clusters, random_state=42)
    X = rfm_df[['recency','frequency','monetary']].fillna(0)
    km.fit(X)
    rfm_df['cluster'] = km.labels_
    return rfm_df

def build_item_similarity_recommender(df, top_n=10):
    # build a simple product->product similarity based on product_name text + co-purchase
    prod = df[['product_id','product_name']].drop_duplicates().reset_index(drop=True)
    prod['product_name_clean'] = prod['product_name'].fillna("").astype(str)
    vec = TfidfVectorizer(max_features=2000, stop_words='english')
    X = vec.fit_transform(prod['product_name_clean'])
    sim = cosine_similarity(X)
    recommendations = []
    for i in range(sim.shape[0]):
        idxs = sim[i].argsort()[::-1][1:top_n+1]
        recs = prod.iloc[idxs][['product_id','product_name']].to_dict(orient='records')
        recommendations.append(recs)
    prod['recommendations'] = recommendations
    return prod
