import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
from pathlib import Path
import re

st.set_page_config(page_title="Spa Dashboard", layout="wide")

# ====== Styling ======
st.markdown(
    """
    <style>
    .topbar {display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;}
    .title {font-size:22px; font-weight:700;}
    .kpi-card {border:1px solid #1f2937; padding:14px 16px; border-radius:12px; background: var(--background-color, #0b1220);}
    .kpi-label {font-size:12px; color:#9ca3af; margin-bottom:6px;}
    .kpi-value {font-size:22px; font-weight:700;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ====== Helpers ======
def read_csv_robust(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    # Robust sep detection
    for sep in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            if df.shape[1] == 1 and df.iloc[:10, 0].astype(str).str.contains(sep).any():
                continue
            return df
        except Exception:
            pass
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def pick_col(cands, cols):
    # exact match first
    for cand in cands:
        for col in cols:
            if str(col).lower().strip() == str(cand).lower().strip():
                return col
    # then contains match
    for cand in cands:
        for col in cols:
            if str(cand).lower().strip() in str(col).lower():
                return col
    return None

def coerce_money(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace(r"[\$,]", "", regex=True)
    s = s.str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")

def date_range_presets():
    return ["Last 7 days", "Last 30 days", "Last 90 days", "This month", "Last month", "All time", "Custom range"]

def apply_date_filter(df, col, sel_time, custom_start, custom_end, period_anchor, all_dates: pd.Series):
    if col is None or col not in df.columns:
        return df, None
    if period_anchor == "Latest date in data" and not all_dates.dropna().empty:
        anchor_date = max(all_dates.dropna())
    else:
        anchor_date = date.today()
    f = df.copy()
    if sel_time == "Last 7 days":
        start = anchor_date - timedelta(days=7)
        f = f[f[col] >= pd.to_datetime(start)]
    elif sel_time == "Last 30 days":
        start = anchor_date - timedelta(days=30)
        f = f[f[col] >= pd.to_datetime(start)]
    elif sel_time == "Last 90 days":
        start = anchor_date - timedelta(days=90)
        f = f[f[col] >= pd.to_datetime(start)]
    elif sel_time == "This month":
        start = date(anchor_date.year, anchor_date.month, 1)
        f = f[f[col] >= pd.to_datetime(start)]
    elif sel_time == "Last month":
        first_this = date(anchor_date.year, anchor_date.month, 1)
        last_month_end = first_this - timedelta(days=1)
        start = date(last_month_end.year, last_month_end.month, 1)
        f = f[(f[col] >= pd.to_datetime(start)) & (f[col] <= pd.to_datetime(last_month_end))]
    elif sel_time == "Custom range" and custom_start and custom_end:
        f = f[(f[col] >= pd.to_datetime(custom_start)) & (f[col] <= pd.to_datetime(custom_end))]
    return f, anchor_date

# ====== Default repo-friendly paths ======
DATA_DIR = (Path(__file__).parent / "data").resolve()
APPTS_NAME = "appointments_with_google_flag.csv"
LINES_NAME = "Checkout Line Items.csv"

DEFAULT_APPTS = DATA_DIR / APPTS_NAME
DEFAULT_LINES = DATA_DIR / LINES_NAME

# Optional fallbacks for local notebook runs
FALLBACK_APPTS = Path("/mnt/data/appointments_with_google_flag.csv")
FALLBACK_LINES = Path("/mnt/data/Checkout Line Items.csv")

st.caption(f"Looking for data in: `{DATA_DIR}`")
if DATA_DIR.exists():
    try:
        st.caption("Found in /data: " + ", ".join(sorted(p.name for p in DATA_DIR.iterdir())))
    except Exception:
        pass
else:
    st.warning("`data/` folder not found next to spa_dashboard.py. Create it and put your CSVs inside.")

appts = read_csv_robust(DEFAULT_APPTS) if DEFAULT_APPTS.exists() else read_csv_robust(FALLBACK_APPTS)
lines = read_csv_robust(DEFAULT_LINES) if DEFAULT_LINES.exists() else read_csv_robust(FALLBACK_LINES)

st.markdown('<div class="topbar"><div class="title">Spa Dashboard</div></div>', unsafe_allow_html=True)

colA, colB, colC = st.columns([2, 2, 1])
with colA:
    up_appts = st.file_uploader("Appointments CSV", type=["csv"], key="u_appts")
    if up_appts is not None:
        try:
            appts = pd.read_csv(up_appts)
        except Exception as e:
            st.error(f"Could not read appointments CSV: {e}")
with colB:
    up_lines = st.file_uploader("Checkout Line Items CSV", type=["csv"], key="u_lines")
    if up_lines is not None:
        try:
            lines = pd.read_csv(up_lines)
        except Exception as e:
            st.error(f"Could not read line items CSV: {e}")
with colC:
    q = st.text_input("🔎 Search…", key="q", placeholder="Search client, provider, service…")

if appts.empty and lines.empty:
    st.warning("Upload your Appointments and Line Items CSVs to get started.")
    st.stop()

# ====== Normalize / detect columns ======
appts.columns = [c.strip() for c in appts.columns]
lines.columns = [c.strip() for c in lines.columns]

# Appointments
APPT_DATE    = pick_col(["Appointment Date", "Date", "Start Time", "Start", "Booked At", "Appt Date", "Appointment Start", "Date of Appointment"], appts.columns)
APPT_CLIENT  = pick_col(["Client", "Customer", "Client Name", "Customer Name", "Name"], appts.columns)
APPT_SERVICE = pick_col(["Service", "Services", "Service Name", "Menu Item", "Item", "Category"], appts.columns)
APPT_PROVIDER= pick_col(["Provider", "Employee", "Staff", "Service Provider", "Artist", "Technician", "Therapist"], appts.columns)
APPT_ID      = pick_col(["Appointment ID", "Appt ID", "ID", "Ticket ID", "Order ID", "Invoice ID"], appts.columns)

# Status / No-show (fallback to column E)
STATUS_COL = pick_col(["Status", "Appointment Status", "Appt Status", "Booking Status"], appts.columns)
NO_SHOW_COL = pick_col(["No Show", "No-Show", "Noshow", "Is No Show", "NoShow"], appts.columns)
NO_SHOW_FALLBACK_IDX = 4  # column E (0-based index 4)
if NO_SHOW_COL is None and appts.shape[1] > NO_SHOW_FALLBACK_IDX:
    NO_SHOW_COL = appts.columns[NO_SHOW_FALLBACK_IDX]

# Google Ads: use Column L raw for display (and derive boolean)
ADS_COL = pick_col(["Booked via Google Ads", "Google Ads", "GoogleAds", "Is Google Ads", "Google Flag", "Source", "Booking Source", "Lead Source", "Acquisition Channel"], appts.columns)
if ADS_COL is None and appts.shape[1] > 11:  # column L (0-based index 11)
    ADS_COL = appts.columns[11]
appts["__GoogleRaw"] = appts[ADS_COL].astype(str).str.strip() if ADS_COL in appts.columns else ""
appts["__GoogleFlag"] = appts["__GoogleRaw"].str.lower().isin(["yes", "true", "1", "y", "google ads", "google"])

# Dates
if APPT_DATE and APPT_DATE in appts.columns:
    appts[APPT_DATE] = pd.to_datetime(appts[APPT_DATE], errors="coerce", infer_datetime_format=True)

# Line Items — explicitly favor Descriptor (col D) and Price (col F)
SALE_DATE    = pick_col(["Date", "Sale Date", "Checkout Date", "Created At", "Closed At", "Completed At", "Date Succeeded"], lines.columns)
SALE_CLIENT  = pick_col(["Client", "Customer", "Client Name", "Customer Name", "Name"], lines.columns)
SALE_SERVICE = pick_col(["Descriptor", "Service", "Item", "Product", "Line Item", "Service Name", "Menu Item", "Category"], lines.columns)  # column D is "Descriptor"
SALE_PROVIDER= pick_col(["Provider", "Employee", "Staff", "Service Provider", "Artist", "Technician", "Therapist"], lines.columns)
SALE_AMOUNT  = pick_col(["Price", "Total", "Amount", "Line Total", "Subtotal", "Net", "Revenue"], lines.columns)  # column F is "Price"
SALE_TICKET  = pick_col(["Appointment ID", "Appt ID", "Ticket ID", "Order ID", "Invoice ID", "Sale ID", "Checkout ID"], lines.columns)

if SALE_DATE and SALE_DATE in lines.columns:
    lines[SALE_DATE] = pd.to_datetime(lines[SALE_DATE], errors="coerce", infer_datetime_format=True)

# Clean money
if SALE_AMOUNT and SALE_AMOUNT in lines.columns:
    lines[SALE_AMOUNT] = coerce_money(lines[SALE_AMOUNT])
else:
    lines["__AMT__"] = 0.0
    SALE_AMOUNT = "__AMT__"

# ====== Build facts safely (even if lines is empty) ======
if not lines.empty:
    facts = pd.DataFrame({
        "Date":     lines[SALE_DATE]    if SALE_DATE    in lines.columns else pd.NaT,
        "Client":   lines[SALE_CLIENT]  if SALE_CLIENT  in lines.columns else "",
        "Service_Line": lines[SALE_SERVICE] if SALE_SERVICE in lines.columns else "",  # <-- Descriptor
        "Provider": lines[SALE_PROVIDER]if SALE_PROVIDER in lines.columns else "",
        "Amount":   lines[SALE_AMOUNT]  if SALE_AMOUNT  in lines.columns else 0.0,
    })
else:
    facts = pd.DataFrame({
        "Date": pd.Series(dtype="datetime64[ns]"),
        "Client": pd.Series(dtype="object"),
        "Service_Line": pd.Series(dtype="object"),
        "Provider": pd.Series(dtype="object"),
        "Amount": pd.Series(dtype="float"),
    })

# Ensure join-derived columns exist
if "Service_Appt" not in facts.columns:
    facts["Service_Appt"] = ""
if "Booked via Google Ads" not in facts.columns:
    facts["Booked via Google Ads"] = False
if "Booked via Google Ads (raw)" not in facts.columns:
    facts["Booked via Google Ads (raw)"] = ""

# Join appointment fields (service, Google raw+flag) by ID if present
if SALE_TICKET and APPT_ID and (SALE_TICKET in lines.columns) and (APPT_ID in appts.columns):
    try:
        # Service from appointments (for table context only)
        if APPT_SERVICE:
            svc_map = appts[[APPT_ID, APPT_SERVICE]].drop_duplicates().rename(
                columns={APPT_ID: "__JOIN__", APPT_SERVICE: "__ApptService"}
            )
            tmp_svc = lines[[SALE_TICKET]].rename(columns={SALE_TICKET: "__JOIN__"}).join(
                svc_map.set_index("__JOIN__"), on="__JOIN__"
            )
            facts["Service_Appt"] = tmp_svc.get("__ApptService", "")

        # Google raw + boolean
        g_map = appts[[APPT_ID, "__GoogleRaw", "__GoogleFlag"]].rename(columns={APPT_ID: "__JOIN__"})
        tmp_g = lines[[SALE_TICKET]].rename(columns={SALE_TICKET: "__JOIN__"}).join(
            g_map.set_index("__JOIN__"), on="__JOIN__"
        )
        facts["Booked via Google Ads (raw)"] = tmp_g.get("__GoogleRaw", "").fillna("")
        facts["Booked via Google Ads"] = tmp_g.get("__GoogleFlag", False).fillna(False).astype(bool)
    except Exception:
        pass

# Prefer appointment service for display in the table; chart will use Service_Line (Descriptor)
facts["Service_XAxis"] = facts["Service_Appt"].astype(str).where(
    facts["Service_Appt"].astype(str).str.strip() != "",
    facts["Service_Line"].astype(str)
)

# Backfill provider via appointments if missing
if (facts.get("Provider") is None or facts["Provider"].astype(str).str.strip().eq("").all()) and SALE_TICKET and APPT_ID and APPT_PROVIDER:
    try:
        prov_map = appts[[APPT_ID, APPT_PROVIDER]].drop_duplicates().rename(
            columns={APPT_ID: "__JOIN__", APPT_PROVIDER: "__ApptProv"}
        )
        tmp3 = lines[[SALE_TICKET]].rename(columns={SALE_TICKET: "__JOIN__"}).join(
            prov_map.set_index("__JOIN__"), on="__JOIN__"
        )
        if "Provider" not in facts.columns:
            facts["Provider"] = ""
        facts["Provider"] = facts["Provider"].mask(facts["Provider"].astype(str).str.strip().eq(""), tmp3.get("__ApptProv"))
    except Exception:
        pass

# ====== Filters ======
time_options = date_range_presets()
providers = ["All Providers"] + sorted([str(x) for x in facts.get("Provider", pd.Series(dtype=object)).dropna().unique().tolist()])
services_line = ["All Services"] + sorted([str(x) for x in facts.get("Service_Line", pd.Series(dtype=object)).dropna().unique().tolist()])  # Descriptor options

c1, c2, c3, c4 = st.columns([1.2, 1.2, 1, 1])
with c1:
    sel_time = st.selectbox("Time Range", options=time_options, index=5)
with c2:
    period_anchor = st.selectbox(
        "Period Anchor",
        options=["Today (server time)", "Latest date in data"],
        index=0,
        help="Controls how preset ranges are computed.",
    )
with c3:
    sel_provider = st.selectbox("Provider", options=providers, index=0)
with c4:
    sel_service_line = st.selectbox("Service (from Line Items: Descriptor)", options=services_line, index=0)

only_google = st.checkbox("Only Google Ads bookings", value=False)

custom_start = custom_end = None
if sel_time == "Custom range":
    cst, cen = st.columns(2)
    with cst:
        default_start = (appts[APPT_DATE].min().date() if APPT_DATE and appts[APPT_DATE].notna().any() else date.today() - timedelta(days=30))
        custom_start = st.date_input("Start date", value=default_start)
    with cen:
        default_end = (appts[APPT_DATE].max().date() if APPT_DATE and appts[APPT_DATE].notna().any() else date.today())
        custom_end = st.date_input("End date", value=default_end)

# ====== Apply filters ======
facts["Date"] = pd.to_datetime(facts["Date"], errors="coerce")
facts["Amount"] = pd.to_numeric(facts["Amount"], errors="coerce").fillna(0.0)

all_dates = facts["Date"].dt.date if "Date" in facts else pd.Series([], dtype="object")
filtered, anchor_date = apply_date_filter(facts, "Date", sel_time, custom_start, custom_end, period_anchor, all_dates)

if sel_provider != "All Providers" and "Provider" in filtered.columns:
    filtered = filtered[filtered["Provider"].astype(str) == sel_provider]
if sel_service_line != "All Services" and "Service_Line" in filtered.columns:
    filtered = filtered[filtered["Service_Line"].astype(str) == sel_service_line]
if only_google and "Booked via Google Ads" in filtered.columns:
    filtered = filtered[filtered["Booked via Google Ads"] == True]
if q:
    ql = q.lower().strip()
    mask = pd.Series(False, index=filtered.index)
    for c in ["Client", "Provider", "Service_Line", "Service_XAxis"]:
        if c in filtered.columns:
            mask |= filtered[c].astype(str).str.lower().str.contains(ql, na=False)
    filtered = filtered[mask]

# ====== Also filter appointments for appointment-based KPIs ======
appts_filtered = appts.copy()
if APPT_DATE and APPT_DATE in appts_filtered.columns:
    appts_filtered[APPT_DATE] = pd.to_datetime(appts_filtered[APPT_DATE], errors="coerce")
    appts_filtered, _ = apply_date_filter(appts_filtered, APPT_DATE, sel_time, custom_start, custom_end, period_anchor, appts_filtered[APPT_DATE].dt.date)
if sel_provider != "All Providers" and APPT_PROVIDER and APPT_PROVIDER in appts_filtered.columns:
    appts_filtered = appts_filtered[appts_filtered[APPT_PROVIDER].astype(str) == sel_provider]

# ====== KPIs ======
unique_clients = int(filtered.get("Client", pd.Series(dtype=object)).astype(str).str.strip().replace({"": "__NA__"}).nunique())
total_revenue = float(filtered.get("Amount", pd.Series(dtype=float)).sum())

# ----- 90-day lookback return rate & Client Type -----
recurring_clients = 0
return_rate = 0.0
ft_clients = rt_clients = 0
ft_rev = rt_rev = 0.0

if APPT_CLIENT and APPT_DATE and not appts.empty:
    # 1) Full history: previous date per client, then flag if within 90 days
    ap_all = appts.dropna(subset=[APPT_CLIENT, APPT_DATE]).copy()
    ap_all[APPT_DATE] = pd.to_datetime(ap_all[APPT_DATE], errors="coerce")
    ap_all = ap_all.dropna(subset=[APPT_DATE]).sort_values([APPT_CLIENT, APPT_DATE])
    ap_all["prev_date"] = ap_all.groupby(APPT_CLIENT)[APPT_DATE].shift(1)
    ap_all["had_prev_90"] = ap_all["prev_date"].notna() & ((ap_all[APPT_DATE] - ap_all["prev_date"]).dt.days <= 90)

    # 2) Current window (+ provider) appointments
    ap_win = appts_filtered[[APPT_CLIENT, APPT_DATE]].copy().dropna()
    ap_win = ap_win.merge(ap_all[[APPT_CLIENT, APPT_DATE, "had_prev_90"]], on=[APPT_CLIENT, APPT_DATE], how="left")
    ap_win["had_prev_90"] = ap_win["had_prev_90"].fillna(False)

    client_returning_map = ap_win.groupby(APPT_CLIENT)["had_prev_90"].any()
    base_clients = int(ap_win[APPT_CLIENT].nunique())
    recurring_clients = int(client_returning_map.sum())
    return_rate = (recurring_clients / base_clients) if base_clients else 0.0

    # 3) Apply 90-day rule to revenue-side client typing
    if "Client" in filtered.columns:
        filtered["Client Type"] = filtered["Client"].map(client_returning_map.rename_axis("Client").astype(bool)).fillna(False).map(lambda x: "Returning" if x else "First-time")
        ft_clients = filtered.loc[filtered["Client Type"] == "First-time", "Client"].nunique()
        rt_clients = filtered.loc[filtered["Client Type"] == "Returning", "Client"].nunique()
        ft_rev = float(filtered.loc[filtered["Client Type"] == "First-time", "Amount"].sum())
        rt_rev = float(filtered.loc[filtered["Client Type"] == "Returning", "Amount"].sum())
else:
    if "Client" in filtered.columns:
        filtered["Client Type"] = "Returning"

# ----- Canceled / No-show from appointments -----
cancel_appts = noshow_appts = 0
if not appts_filtered.empty:
    status_series = appts_filtered[STATUS_COL].astype(str).str.lower() if STATUS_COL else pd.Series([""] * len(appts_filtered), index=appts_filtered.index)
    raw_noshow = appts_filtered[NO_SHOW_COL] if NO_SHOW_COL and NO_SHOW_COL in appts_filtered.columns else pd.Series([""] * len(appts_filtered), index=appts_filtered.index)

    cancel_flag = status_series.str.contains("cancel", na=False)
    noshow_flag = raw_noshow.astype(str).str.strip().str.lower().isin(["no-show", "no show", "noshow", "true", "1", "yes", "y"])
    noshow_flag = noshow_flag | status_series.str.contains(r"no[\s-]?show", na=False)

    cancel_appts = int(cancel_flag.sum())
    noshow_appts = int(noshow_flag.sum())

# ====== KPI Renders ======
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Unique Clients</div><div class="kpi-value">{unique_clients:,}</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Returning Clients (prior appt ≤ 90 days)</div><div class="kpi-value">{recurring_clients:,}</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">90-Day Return Rate</div><div class="kpi-value">{return_rate:.1%}</div></div>', unsafe_allow_html=True)
with k4:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Total Revenue</div><div class="kpi-value">${total_revenue:,.2f}</div></div>', unsafe_allow_html=True)

k5, k6, k7, k8 = st.columns(4)
with k5:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">First-time Clients</div><div class="kpi-value">{ft_clients:,}</div></div>', unsafe_allow_html=True)
with k6:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Returning Clients</div><div class="kpi-value">{rt_clients:,}</div></div>', unsafe_allow_html=True)
with k7:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">First-time Revenue</div><div class="kpi-value">${ft_rev:,.2f}</div></div>', unsafe_allow_html=True)
with k8:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Returning Revenue</div><div class="kpi-value">${rt_rev:,.2f}</div></div>', unsafe_allow_html=True)

k9, k10 = st.columns(2)
with k9:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Canceled Appointments</div><div class="kpi-value">{cancel_appts:,}</div></div>', unsafe_allow_html=True)
with k10:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">No-Show Appointments</div><div class="kpi-value">{noshow_appts:,}</div></div>', unsafe_allow_html=True)

# ====== Google Ads Revenue caption ======
google_revenue = float(filtered.loc[filtered.get("Booked via Google Ads", pd.Series([], dtype=bool)).fillna(False), "Amount"].sum()) if not filtered.empty else 0.0
st.caption(f"Google Ads Revenue in view: ${google_revenue:,.2f}")

# ====== Charts ======
c1, c2 = st.columns(2)

# Chart MUST use Line Items: Descriptor (D) + Price (F)
with c1:
    if "Service_Line" in filtered.columns and "Amount" in filtered.columns and not filtered.empty:
        by_service_line = filtered.groupby("Service_Line", dropna=False)["Amount"].sum().sort_values(ascending=False).head(30)
        st.markdown("#### Revenue by Service (from Line Items: Descriptor × Price)")
        st.bar_chart(by_service_line)
    else:
        st.info("No line-item service/amount data to chart.")

with c2:
    if "Provider" in filtered.columns and not filtered.empty:
        by_provider = filtered.groupby("Provider", dropna=False)["Amount"].sum().sort_values(ascending=False).head(30)
        st.markdown("#### Revenue by Provider")
        st.bar_chart(by_provider)
    else:
        st.info("No provider data available.")

# ====== Revenue Over Time ======
st.markdown("#### Revenue Over Time")
if "Date" in filtered.columns and not filtered.empty:
    by_day = filtered.dropna(subset=["Date"]).copy()
    by_day["Day"] = pd.to_datetime(by_day["Date"]).dt.date
    by_day = by_day.groupby("Day")["Amount"].sum().sort_index()
    st.line_chart(by_day)
else:
    st.info("No dated revenue rows to chart.")

# ====== Table with Google Ads (from column L) highlighted ======
st.markdown("#### Line Items (Filtered) — Google Ads highlighted")
display = filtered.copy()
if "Date" in display.columns:
    display["Date"] = pd.to_datetime(display["Date"], errors="coerce").dt.strftime("%b %d, %Y")

# Keep both: Appointment service for context, and Line Items descriptor (what powers the chart)
display = display.rename(columns={
    "Service_XAxis": "Service (from Appointments)",
    "Service_Line": "Descriptor (from Line Items)",
})

# Build display column for Ads using raw (column L)
def yes_no(x):
    xl = str(x).strip().lower()
    return "Yes" if xl in ["yes", "true", "1", "y", "google ads", "google"] else "No"

if "Booked via Google Ads (raw)" in display.columns:
    display["Booked via Google Ads"] = display["Booked via Google Ads (raw)"].apply(yes_no)
else:
    display["Booked via Google Ads"] = display.get("Booked via Google Ads", False).apply(lambda b: "Yes" if bool(b) else "No")

wanted_cols = ["Client", "Client Type", "Provider", "Descriptor (from Line Items)", "Service (from Appointments)", "Amount", "Date", "Booked via Google Ads"]
for col in wanted_cols:
    if col not in display.columns:
        display[col] = ""
display = display[wanted_cols]

def highlight_ads(row):
    # Prefer boolean if present on filtered; fallback to Yes/No string
    try:
        flag = bool(filtered.loc[row.name, "Booked via Google Ads"]) if "Booked via Google Ads" in filtered.columns else (row["Booked via Google Ads"] == "Yes")
    except Exception:
        flag = (row["Booked via Google Ads"] == "Yes")
    color = "#2a3a22" if flag else ""
    return [f"background-color: {color}" for _ in row]

styled = display.style.apply(highlight_ads, axis=1)
st.dataframe(styled, use_container_width=True, height=520)

# ====== Client Lifetime Value ======
st.markdown("#### Client Lifetime Value (Total Spend)")
ltv = pd.DataFrame()
if "Client" in filtered.columns and "Amount" in filtered.columns:
    ltv = filtered.groupby("Client", dropna=False)["Amount"].sum().reset_index().sort_values("Amount", ascending=False)
    ltv.rename(columns={"Client": "Client", "Amount": "Total Spend"}, inplace=True)
st.dataframe(ltv, use_container_width=True, height=360)

st.caption("Notes: Revenue-by-service chart uses Line Items (Descriptor × Price). Client return rate uses a 90-day lookback. No-show uses Status/No-Show or column E fallback. Ads display mirrors column L; toggle filter uses boolean.")

