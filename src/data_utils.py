import pandas as pd
import streamlit as st
from pathlib import Path

def _safe_read(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def normalize_cols(df):
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def harmonize_df(df, platform_name):
    df = normalize_cols(df)
    df['platform'] = platform_name

    # Dates
    for col in ['order_date','date','created_at']:
        if col in df.columns:
            df['order_date'] = pd.to_datetime(df[col], errors='coerce')
            break
    if 'order_date' not in df.columns:
        df['order_date'] = pd.NaT

    # Price, quantity, discount
    for col in ['price','amount','cost']:
        if col in df.columns:
            df['price'] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            break
    if 'price' not in df.columns:
        df['price'] = 0.0
    df['quantity'] = pd.to_numeric(df.get('quantity', 1), errors='coerce').fillna(1)
    df['discount'] = pd.to_numeric(df.get('discount', 0), errors='coerce').fillna(0)
    df['revenue'] = df['price'] * df['quantity'] - df['discount']

    # IDs
    df['order_id'] = df.get('order_id', pd.Series(df.index.astype(str)))
    df['user_id'] = df.get('user_id', df.get('customer_id', pd.NA))

    # ✅ Safe product_name extraction (fixed)
    if 'product_name' not in df.columns:
        if 'product' in df.columns:
            df['product_name'] = df['product'].astype(str)
        elif 'title' in df.columns:
            df['product_name'] = df['title'].astype(str)
        elif 'name' in df.columns:
            df['product_name'] = df['name'].astype(str)
        else:
            df['product_name'] = pd.Series(['Unknown'] * len(df))

    # Rating & reviews
    df['rating'] = pd.to_numeric(df.get('rating', df.get('stars', None)), errors='coerce')
    df['review_text'] = df.get('review_text', df.get('review', pd.NA)).astype(str)
    df['city'] = df.get('city', df.get('location', pd.NA))
    return df

def load_all_and_merge(paths_dict):
    frames = []
    for name, path in paths_dict.items():
        p = Path(path)
        if p.exists():
            df = harmonize_df(_safe_read(p), name)
            frames.append(df)
    merged = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    merged['order_date'] = pd.to_datetime(merged['order_date'], errors='coerce')
    merged['revenue'] = pd.to_numeric(merged['revenue'], errors='coerce').fillna(0)
    return merged

def prepare_filters(df):
    min_date = pd.to_datetime(df['order_date'].min(), errors='coerce')
    max_date = pd.to_datetime(df['order_date'].max(), errors='coerce')
    date_range = st.sidebar.date_input("Date range", (min_date, max_date))
    platforms = st.sidebar.multiselect("Platform", df['platform'].dropna().unique().tolist(), default=df['platform'].unique().tolist())
    cities = st.sidebar.multiselect("City", df['city'].dropna().unique().tolist())
    text_search = st.sidebar.text_input("Search in reviews")
    return date_range[0], date_range[1], platforms, cities, text_search
