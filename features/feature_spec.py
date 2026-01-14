INTERACTIONS_COLS = [
    "customer_id",
    "article_id",
    "t_dat",
    "price",
    "sales_channel_id",
]

USER_FEATURES_COLS = [
    "customer_id",
    "age",
    "club_member_status",
    "fashion_news_frequency",
    "active",
]

ITEM_FEATURES_COLS = [
    "article_id",
    "product_type_name",
    "product_group_name",
    "department_name",
    "colour_group_name",
    "section_name",
    "garment_group_name",
]

USER_HISTORY_AGG_COLS = [
    "customer_id",
    "total_purchases",
    "last_purchase_date",
    "avg_price",
    "top_product_group_name",
]

ITEM_POPULARITY_COLS = [
    "article_id",
    "purchase_count",
    "popularity_rank",
]
