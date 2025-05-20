import streamlit as st
import pandas as pd
import io, csv, re
from transformers import BertTokenizerFast, BertForTokenClassification, pipeline
import altair as alt


# ─── 1.  NER PIPELINE ───────────────────────────────────────────────────
@st.cache_resource
def load_ner(model_dir: str = "bert/results/checkpoint-45"):
    tok = BertTokenizerFast.from_pretrained(model_dir)
    mdl = BertForTokenClassification.from_pretrained(model_dir)
    return pipeline(
        "ner",
        model=mdl,
        tokenizer=tok,
        aggregation_strategy="first",
        device=0 if mdl.device.type == "cuda" else -1,
    )

ner_pipeline = load_ner()

st.title("Charity Fund: Detailed Transactions + Full Monthly Summary")

# ─── 2.  UPLOAD TABLES ──────────────────────────────────────────────────
uploads = {
    "detail":   st.sidebar.file_uploader("1) Transactions (CSV or XLSX)", type=["csv","xlsx","xls"]),
    "summary":  st.sidebar.file_uploader("2) Income & Expenses (CSV or XLSX)", type=["csv","xlsx","xls"]),
    "movement": st.sidebar.file_uploader("3) Cash Movement (CSV or XLSX)", type=["csv","xlsx","xls"]),
}
if not all(uploads.values()):
    st.info("Please upload all three tables.")
    st.stop()

# ─── 3.  GLOBALS ────────────────────────────────────────────────────────
MONTH_MAP = {
    name: idx for idx, name in enumerate(
        ["январь","февраль","март","апрель","май","июнь",
         "июль","август","сентябрь","октябрь","ноябрь","декабрь"],
        start=1
    )
}

INCOME_CATS = [
    "Поступление благотворительных взносов",
    "Доходы в виде вознаграждения по депозитам",
    "Прочие операционные доходы (в т.ч. доходы от переоценки валюты)",
    "Итого доходы",
]
EXPENSE_CATS = [
  "Оказание благотворительной помощи",
    "Административные расходы Фонда, в том числе",
    "Услуги сторонних организаций",
    "Услуги по договорам гражданско-правового характера (ГПХ)",
    "Оплата труда, включая налоги и обязательные взносы",
    "Налоги и обязательные социальные платежи",
    "Командировочные расходы",
    "Прочие расходы (вкл. амортизацию)",
    "Прочие операционные расходы (в т.ч. от переоценки валюты)",
    "Итого расходы",
    "Численность Фонда",
    "Численность работников по договорам гражданско- правового характера (ГПХ)",
    ]

MOVEMENT_CATS = [
    "Сальдо на начало периода",
    "Поступление благотворительных взносов",
    "Выплаты благотворительной помощи",
    "Сальдо на конец периода",
]

# ─── 4.  GENERIC PIVOT ─────────────────────────────────────────────────
def pivot_from_df(df: pd.DataFrame, wanted: list[str]) -> pd.DataFrame:
    # normalize first column into "category"
    raw = df.iloc[:,0].astype(str)
    cat = raw.str.replace(r"\s+"," ",regex=True).str.strip().str.rstrip(":")
    df2 = df.iloc[:,1:].copy()
    df2["category"] = cat

    month_map = {
        col: MONTH_MAP[col.lower()]
        for col in df2.columns
        if isinstance(col, str) and col.lower() in MONTH_MAP
    }
    df2 = df2.rename(columns=month_map)

    df2 = df2[df2["category"].isin(wanted)]
    m = df2.melt(id_vars="category", var_name="month", value_name="value")
    return m.pivot(index="month", columns="category", values="value").reset_index()

# ─── 5.  ROBUST CSV READER ───────────────────────────────────────────────
def read_any_csv(f) -> pd.DataFrame:
    """
    Read a CSV Streamlit-UploadedFile robustly:
    - sniff the delimiter on the first 2KB
    - skip bad lines
    - reset file pointer for caching
    """
    raw_bytes = f.read()
    text = raw_bytes.decode("utf-8", errors="replace")
    # sniff
    try:
        dialect = csv.Sniffer().sniff(text[:2048])
        sep = dialect.delimiter
    except csv.Error:
        sep = ","
    df = pd.read_csv(
        io.StringIO(text),
        sep=sep,
        engine="python",
        on_bad_lines="skip",
        dtype=str,       # read everything as string first
    )
    f.seek(0)
    return df

# ─── 6.  FILE LOADERS ───────────────────────────────────────────────────
@st.cache_data
def load_transactions_file(f) -> pd.DataFrame:
    # 1) load
    if f.name.lower().endswith(".csv"):
        df = read_any_csv(f)
    else:
        df = pd.read_excel(f, sheet_name=0)
    # 2) normalize headers
    df.columns = df.columns.str.strip()
    # 3) require keys
    expected = ["Дата платежа","Назначение платежа","Сумма"]
    missing  = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Transactions is missing columns: {missing}")
    # 4) parse date & month
    df["Дата платежа"] = pd.to_datetime(df["Дата платежа"], dayfirst=True, errors="coerce")
    df["month"]        = df["Дата платежа"].dt.month
    return df[["Дата платежа","month","Назначение платежа","Сумма"]]

@st.cache_data
def load_income_file(f) -> pd.DataFrame:
    if f.name.lower().endswith(".csv"):
        df = read_any_csv(f)
    else:
        xls = pd.ExcelFile(f)
        df  = pd.read_excel(xls, sheet_name=0)
    return pivot_from_df(df, INCOME_CATS)

@st.cache_data
def load_expense_file(f) -> pd.DataFrame:
    if f.name.lower().endswith(".csv"):
        df = read_any_csv(f)
    else:
        xls = pd.ExcelFile(f)
        idx = 1 if len(xls.sheet_names)>1 else 0
        df  = pd.read_excel(xls, sheet_name=idx)
    return pivot_from_df(df, EXPENSE_CATS)

@st.cache_data
def load_movement_file(f) -> pd.DataFrame:
    if f.name.lower().endswith(".csv"):
        df = read_any_csv(f)
    else:
        xls = pd.ExcelFile(f)
        df  = pd.read_excel(xls, sheet_name=0)
    return pivot_from_df(df, MOVEMENT_CATS)

# ─── 7.  ENTITY EXTRACTION ─────────────────────────────────────────────
def extract_entities(text: str) -> tuple[str,str,str]:
    if not isinstance(text, str):
        return None, None, None
    
    ents = ner_pipeline(text)
    by_lbl = {}
    for e in ents:
        by_lbl.setdefault(e["entity_group"], []).append(e["word"])
    return (
        by_lbl.get("ORG", [None])[0],
        by_lbl.get("PAYMENT_TYPE", [None])[0],
        by_lbl.get("PURPOSE", [None])[0],
    )

# ─── 8.  LOAD & MERGE ──────────────────────────────────────────────────
tx_df  = load_transactions_file( uploads["detail"] )
inc_df = load_income_file(      uploads["summary"] )
exp_df = load_expense_file(     uploads["summary"] )
mov_df = load_movement_file(    uploads["movement"] )

# prefix to avoid name collisions
inc_df = inc_df.add_prefix("inc_").rename(columns={"inc_month":"month"})
exp_df = exp_df.add_prefix("exp_").rename(columns={"exp_month":"month"})
mov_df = mov_df.add_prefix("mov_").rename(columns={"mov_month":"month"})

# apply NER & clean amounts
ner_rows = []
for _,r in tx_df.iterrows():
    org,ptype,purp = extract_entities(r["Назначение платежа"])
    amt = int(re.sub(r"[^\d]","", str(r["Сумма"])) or 0)
    ner_rows.append({
        "Дата платежа":       r["Дата платежа"],
        "Назначение платежа": r["Назначение платежа"],
        "Сумма":              amt,
        "organisation":       org,
        "payment_type":       ptype,
        "purpose":            purp,
        "month":              r["month"],
    })
ner_df = pd.DataFrame(ner_rows)

# one‐to‐many merge
final = (
    ner_df
      .merge(inc_df, on="month", how="left")
      .merge(exp_df, on="month", how="left")
      .merge(mov_df, on="month", how="left")
      .sort_values("Дата платежа")
      .reset_index(drop=True)
)

# serial & dedupe
final.insert(0, "№ п/п", final.index+1)
final = final.loc[:, ~final.columns.duplicated()]


# ─── 9. TRANSLATE COLUMNS TO ENGLISH ───────────────────────────────────
rename_map = {
    "№ п/п":                                    "Record No.",
    "Дата платежа":                             "Payment Date",
    "Назначение платежа":                       "Payment Purpose",
    "Сумма":                                    "Amount",

    # income columns (prefixed “inc_”)
    "inc_Поступление благотворительных взносов":                    "Donation Inflows",
    "inc_Доходы в виде вознаграждения по депозитам":               "Deposit Interest Income",
    "inc_Прочие операционные доходы (в т.ч. доходы от переоценки валюты)": 
                                                                       "Other Operating Income (e.g. FX Reval.)",
    "inc_Итого доходы":                                            "Total Income",

    # expense columns (prefixed “exp_”)
    "exp_Оказание благотворительной помощи":                       "Charitable Disbursements",
    "exp_Административные расходы Фонда, в том числе":            "Administrative Expenses (of which)",
    "exp_Услуги сторонних организаций":                           "Third-Party Services",
    "exp_Услуги по договорам гражданско-правового характера (ГПХ)": 
                                                                       "Civil-Contract Services",
    "exp_Оплата труда, включая налоги и обязательные взносы":      "Payroll (inc. Taxes & Contrib.)",
    "exp_Налоги и обязательные социальные платежи":               "Taxes and Social Contributions",
    "exp_Командировочные расходы":                                "Travel Expenses",
    "exp_Прочие расходы (вкл. амортизацию)":                       "Other Expenses (incl. Deprec.)",
    "exp_Прочие операционные расходы (в т.ч. от переоценки валюты)":
                                                                       "Other Operating Expenses (e.g. FX Reval.)",
    "exp_Итого расходы":                                          "Total Expenses",
    "exp_Численность Фонда":                                      "Staff Headcount (Fund)",
    "exp_Численность работников по договорам гражданско- правового характера (ГПХ)":
                                                                       "Headcount (GPH Contractors)",

    # movement columns (prefixed “mov_”)
    "mov_Сальдо на начало периода":                              "Opening Balance",
    "mov_Поступление благотворительных взносов":                  "Donation Inflows (roll-forward)",
    "mov_Выплаты благотворительной помощи":                      "Charitable Payouts (roll-forward)",
    "mov_Сальдо на конец периода":                               "Closing Balance",
}

# apply the rename
final = final.rename(columns=rename_map)

# now display
st.subheader("Final Detailed Table")
st.dataframe(final, use_container_width=True)



# ─── 10. BUTTON-TRIGGERED SUMMARY + VISUALIZATION ────────────────────────

choice = st.selectbox("Show top 10 by:", ["Total Paid", "Transaction Count"], key="metric_choice")

if st.button("Generate Summary by Attribute"):
    # 10.1) Compute group-by
    org_summary = (
        final
          .groupby("organisation")
          .agg(
            total_paid      = pd.NamedAgg(column="Amount", aggfunc="sum"),
            transaction_cnt = pd.NamedAgg(column="Amount", aggfunc="count"),
            first_month     = pd.NamedAgg(column="month", aggfunc="min"),
            last_month      = pd.NamedAgg(column="month", aggfunc="max"),
          )
          .reset_index()
    )

    # 10.2) Choose which metric to sort by and display top 10
    sort_col = "total_paid" if choice=="Total Paid" else "transaction_cnt"

    # get top-10
    top10 = org_summary.sort_values(by=sort_col, ascending=False).head(10)

    st.subheader(f"Top 10 Organisations by {choice}")
    st.dataframe(top10, use_container_width=True)

    # 10.3) Plot the same top-10
    chart = (
        alt.Chart(top10)
           .mark_bar()
           .encode(
             x=alt.X(f"{sort_col}:Q", title=choice),
             y=alt.Y("organisation:N", sort="-x", title="Organisation"),
             tooltip=["organisation", sort_col],
           )
           .properties(title=f"Top 10 Organisations by {choice}")
    )
    st.altair_chart(chart, use_container_width=True)
