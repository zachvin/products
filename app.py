"""
Product Recommender — Streamlit frontend.

Run:
    poetry run streamlit run app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from api.services import recommendations, search

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Product Recommender",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_users(top_n: int = 500) -> pd.DataFrame:
    df = pd.read_parquet("feature_store/user_features")
    return df.sort_values("total_purchases", ascending=False).head(top_n).reset_index(drop=True)


@st.cache_data
def load_metadata() -> pd.DataFrame:
    return pd.read_parquet("data/processed/metadata.parquet").set_index("parent_asin")


@st.cache_data
def load_item_features() -> pd.DataFrame:
    return pd.read_parquet("feature_store/item_features").set_index("item_id")


@st.cache_resource(show_spinner="Loading recommendation models…")
def init_models() -> None:
    recommendations.load()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def enrich(asins: list[str]) -> list[dict]:
    """Attach display metadata and item stats to a list of ASINs."""
    rows = []
    for asin in asins:
        row: dict = {"asin": asin, "title": asin, "price": None, "category": "", "rating": None, "rating_count": 0, "purchases": 0}
        if asin in _meta.index:
            m = _meta.loc[asin]
            row["title"]    = m["title"] or asin
            row["price"]    = m["price"]
            row["category"] = m["main_category"] or ""
        if asin in _items.index:
            it = _items.loc[asin]
            if pd.notna(it["avg_rating"]):
                row["rating"] = round(float(it["avg_rating"]), 1)
            row["rating_count"] = int(it["rating_count"]) if pd.notna(it["rating_count"]) else 0
            row["purchases"]    = int(it["total_purchases"]) if pd.notna(it["total_purchases"]) else 0
        rows.append(row)
    return rows


def product_card(p: dict, user_id: str, key_prefix: str = "") -> None:
    with st.container(border=True):
        if p["category"]:
            st.caption(p["category"].upper())

        title = p["title"]
        st.markdown(f"**{title[:80]}{'…' if len(title) > 80 else ''}**")

        meta_parts = []
        if p["price"]:
            meta_parts.append(f"**{p['price']}**")
        if p["rating"]:
            stars = "★" * round(p["rating"]) + "☆" * (5 - round(p["rating"]))
            meta_parts.append(f"{stars} {p['rating']} ({p['rating_count']:,})")
        if p["purchases"]:
            meta_parts.append(f"🛒 {p['purchases']:,} bought")

        if meta_parts:
            st.markdown("  \n".join(meta_parts))

        up, down, _ = st.columns([1, 1, 4])
        if up.button("👍", key=f"{key_prefix}up_{p['asin']}"):
            try:
                recommendations.record_feedback(user_id, p["asin"], 1.0)
                st.toast("Feedback recorded!")
            except KeyError as e:
                st.toast(str(e), icon="⚠️")
        if down.button("👎", key=f"{key_prefix}down_{p['asin']}"):
            try:
                recommendations.record_feedback(user_id, p["asin"], 0.0)
                st.toast("Feedback recorded!")
            except KeyError as e:
                st.toast(str(e), icon="⚠️")


def render_grid(products: list[dict], user_id: str, cols: int = 4, key_prefix: str = "") -> None:
    if not products:
        st.info("No results found.")
        return
    columns = st.columns(cols)
    for i, p in enumerate(products):
        with columns[i % cols]:
            product_card(p, user_id, key_prefix=key_prefix)


# ---------------------------------------------------------------------------
# Initialise
# ---------------------------------------------------------------------------

_users = load_users()
_meta  = load_metadata()
_items = load_item_features()
init_models()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Product Recommender")
    st.divider()

    st.subheader("User")
    selected_user = st.selectbox(
        "Select user",
        _users["user_id"].tolist(),
        label_visibility="collapsed",
    )

    u = _users[_users["user_id"] == selected_user].iloc[0]
    st.caption(f"🛍️ **{int(u['total_purchases'])}** purchases")
    st.caption(f"⭐ Avg rating given: **{u['avg_rating']:.1f}**")
    st.caption(f"🏷️ Favourite: **{u['favorite_category'] or '—'}**")
    st.caption(f"📅 Days active: **{int(u['days_active'])}**")

    st.divider()

    st.subheader("Recommendation System")
    mode = st.radio(
        "mode",
        ["Two-Tower + LinUCB", "Two-Tower only", "Compare side-by-side"],
        label_visibility="collapsed",
    )

    k = st.slider("Results", min_value=4, max_value=20, value=8, step=4)

# ---------------------------------------------------------------------------
# Main — search bar
# ---------------------------------------------------------------------------

query = st.text_input(
    "Search",
    placeholder="🔍  Search products (e.g. wireless earbuds, phone case…)",
    label_visibility="collapsed",
)

# ---------------------------------------------------------------------------
# Search results
# ---------------------------------------------------------------------------

if query:
    st.subheader(f'Search  ·  "{query}"')
    with st.spinner("Searching…"):
        hits     = search.hybrid_search(query, k=k)
        products = enrich([h["parent_asin"] for h in hits])
    render_grid(products, selected_user)

# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

else:
    if mode == "Compare side-by-side":
        left, right = st.columns(2)

        with left:
            st.subheader("Two-Tower + LinUCB")
            with st.spinner("Loading…"):
                try:
                    hits     = recommendations.get_recommendations(selected_user, k=k, use_linucb=True)
                    products = enrich([h["parent_asin"] for h in hits])
                    render_grid(products, selected_user, cols=2, key_prefix="linucb_")
                except KeyError:
                    st.warning("User not in training set.")

        with right:
            st.subheader("Two-Tower only")
            with st.spinner("Loading…"):
                try:
                    hits     = recommendations.get_recommendations(selected_user, k=k, use_linucb=False)
                    products = enrich([h["parent_asin"] for h in hits])
                    render_grid(products, selected_user, cols=2, key_prefix="tt_")
                except KeyError:
                    st.warning("User not in training set.")

    else:
        use_linucb = mode == "Two-Tower + LinUCB"
        label      = "Recommended for you" if not use_linucb else "Recommended for you  ·  LinUCB re-ranked"
        st.subheader(label)

        with st.spinner("Loading recommendations…"):
            try:
                hits     = recommendations.get_recommendations(selected_user, k=k, use_linucb=use_linucb)
                products = enrich([h["parent_asin"] for h in hits])
            except KeyError:
                st.warning(f"User '{selected_user}' was not seen during training.")
                products = []

        render_grid(products, selected_user)
