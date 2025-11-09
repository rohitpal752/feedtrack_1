
import pandas as pd
from pathlib import Path

def _safe_read(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"Failed to read {path}: {e}")
        return pd.DataFrame()

def normalize_cols(df):
    df = df.copy()
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    return df

def harmonize_df(df, platform_name):
    df = normalize_cols(df)
    df['platform'] = platform_name
    # try common columns
    for col in ['order_date','date','created_at']:
        if col in df.columns:
            df['order_date'] = pd.to_datetime(df[col], errors='coerce')
            break
    if 'order_date' not in df.columns:
        df['order_date'] = pd.NaT
    for col in ['price','amount','cost']:
        if col in df.columns:
            df['price'] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            break
    if 'price' not in df.columns:
        df['price'] = 0.0
    if 'quantity' not in df.columns:
        df['quantity'] = 1
    else:
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(1)
    if 'discount' not in df.columns:
        df['discount'] = 0
    else:
        df['discount'] = pd.to_numeric(df['discount'], errors='coerce').fillna(0)
    # revenue calc
    df['revenue'] = df['price'] * df['quantity'] - df['discount']
    # standard ids
    if 'order_id' not in df.columns:
        df['order_id'] = df.index.astype(str)
    if 'product_id' not in df.columns and 'product' in df.columns:
        df['product_id'] = df['product'].astype(str)
    if 'product_name' not in df.columns:
        if 'product' in df.columns:
            df['product_name'] = df['product'].astype(str)
        else:
            df['product_name'] = df.get('title', df.get('name', '')).astype(str)
    if 'user_id' not in df.columns:
        df['user_id'] = df.get('customer_id', df.get('uid', pd.NA))
    # rating & review_text
    if 'rating' not in df.columns:
        if 'stars' in df.columns:
            df['rating'] = pd.to_numeric(df['stars'], errors='coerce')
        else:
            df['rating'] = pd.NA
    if 'review_text' not in df.columns:
        if 'review' in df.columns:
            df['review_text'] = df['review'].astype(str)
        else:
            df['review_text'] = pd.NA
    # city
    if 'city' not in df.columns:
        df['city'] = df.get('location', df.get('city_name', pd.NA))
    return df

def load_all_and_merge(paths_dict):
    frames = []
    for name, path in paths_dict.items():
        p = Path(path)
        if p.exists():
            df = _safe_read(path)
            df = harmonize_df(df, name)
            frames.append(df)
        else:
            print(f"Path not found: {path}")
    if frames:
        merged = pd.concat(frames, ignore_index=True, sort=False)
    else:
        merged = pd.DataFrame()
    # ensure order_date type and fill some defaults
    merged['order_date'] = pd.to_datetime(merged['order_date'], errors='coerce')
    # create month/week/day
    merged['order_month'] = merged['order_date'].dt.to_period('M')
    merged['order_week'] = merged['order_date'].dt.to_period('W')
    merged['order_day'] = merged['order_date'].dt.date
    # fix numeric revenue
    merged['revenue'] = pd.to_numeric(merged['revenue'], errors='coerce').fillna(0)
    return merged

def prepare_filters(df):
    import pandas as pd
    min_date = pd.to_datetime(df['order_date'].min()) if not df['order_date'].isna().all() else pd.to_datetime("2000-01-01")
    max_date = pd.to_datetime(df['order_date'].max()) if not df['order_date'].isna().all() else pd.to_datetime("2099-12-31")
    date_range = st.sidebar.date_input("Order date range", value=(min_date.date(), max_date.date()))
    platforms = st.sidebar.multiselect("Platform", options=sorted(df['platform'].dropna().unique().tolist()), default=sorted(df['platform'].dropna().unique().tolist()))
    cities = st.sidebar.multiselect("City", options=sorted(df['city'].dropna().unique().tolist()), default=[])
    text_search = st.sidebar.text_input("Search in reviews (free text)", value="")
    return date_range[0], date_range[1], platforms, cities, text_search
