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
    """, unsafe_allow_html=True
)

# ====== Helpers ======
def read_csv_robust(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    # Robust sep detection
    for sep in [",",";","\t","|"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            if df.shape[1] == 1 and df.iloc[:10,0].astype(str).str.contains(sep).any():
                continue
            return df
        except Exception:
            pass
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def pick_col(cands, cols):
    for cand in cands:
        for col in cols:
            if str(col).lower().strip() == str(cand).lower().strip():
                return col
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
    return ["Last 7 days","Last 30 days","Last 90 days","This month","Last month","All time","Custom range"]

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

# Optional fallbacks for local runs
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

colA, colB, colC = st.columns([2,2,1])
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

APPT_DATE = pick_col(["Appointment Date","Date","Start Time","Start","Booked At","Appt Date","Appointment Start","Date of Appointment"], appts.columns)
APPT_CLIENT = pick_col(["Client","Customer","Client Name","Customer Name","Name"], appts.columns)
APPT_SERVICE = pick_col(["Service","Services","Service Name","Menu Item","Item","Category"], appts.columns)
APPT_PROVIDER = pick_col(["Provider","Employee","Staff","Service Provider","Artist","Technician","Therapist"], appts.columns)
APPT_ID = pick_col(["Appointment ID","Appt ID","ID","Ticket ID","Order ID","Invoice ID"], appts.columns)

STATUS_COL = pick_col(["Status","Appointment Status","Appt Status","Booking Status"], appts.columns)
NO_SHOW_COL = pick_col(["No Show","No-Show","Noshow","Is No Show","NoShow"], appts.columns)

GOOGLE_FLAG_COL = pick_col(["Booked via Google Ads","Google Ads","GoogleAds","Is Google Ads","Google Flag"], appts.columns)
BOOKING_SOURCE_COL = pick_col(["Source","Booking Source","Lead Source","Acquisition Channel"], appts.columns)

def derive_google_flag(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series([], dtype=bool)
    if GOOGLE_FLAG_COL and GOOGLE_FLAG_COL in df.columns:
        s = df[GOOGLE_FLAG_COL]
        if s.dtype == bool:
            return s
        return s.astype(str).str.strip().str.lower().isin(["true","1","yes","y","google ads","google"])
    if BOOKING_SOURCE_COL and BOOKING_SOURCE_COL in df.columns:
        return df[BOOKING_SOURCE_COL].astype(str).str.strip().str.lower().eq("google ads")
    return pd.Series([False]*len(df), index=df.index)

APPT_GOOGLE_FLAG = "__GoogleAds"
appts[APPT_GOOGLE_FLAG] = derive_google_flag(appts)

SALE_DATE = pick_col(["Date","Sale Date","Checkout Date","Created At","Closed At","Completed At","Date Succeeded"], lines.columns)
SALE_CLIENT = pick_col(["Client","Customer","Client Name","Customer Name","Name"], lines.columns)
SALE_SERVICE = pick_col(["Service","Item","Product","Line Item","Service Name","Menu Item","Category"], lines.columns)
SALE_PROVIDER = pick_col(["Provider","Employee","Staff","Service Provider","Artist","Technician","Therapist"], lines.columns)
SALE_AMOUNT = pick_col(["Total","Amount","Price","Line Total","Subtotal","Net","Revenue","Price"], lines.columns)
SALE_TICKET = pick_col(["Appointment ID","Appt ID","Ticket ID","Order ID","Invoice ID","Sale ID","Checkout ID"], lines.columns)

# Dates
if APPT_DATE and APPT_DATE in appts.columns:
    appts[APPT_DATE] = pd.to_datetime(appts[APPT_DATE], errors="coerce", infer_datetime_format=True)
if SALE_DATE and SALE_DATE in lines.columns:
    lines[SALE_DATE] = pd.to_datetime(lines[SALE_DATE], errors="coerce", infer_datetime_format=True)

# Clean money
if SALE_AMOUNT and SALE_AMOUNT in lines.columns:
    lines[SALE_AMOUNT] = coerce_money(lines[SALE_AMOUNT])
else:
    lines["__AMT__"] = 0.0
    SALE_AMOUNT = "__AMT__"

# ====== Build facts safely (even if lines is empty) ======
facts = pd.DataFrame()
if not lines.empty:
    facts["Date"] = lines[SALE_DATE] if SALE_DATE in lines.columns else pd.NaT
    facts["Client"] = lines[SALE_CLIENT] if SALE_CLIENT in lines.columns else ""
    facts["Service_Line"] = lines[SALE_SERVICE] if SALE_SERVICE in lines.columns else ""
    facts["Provider"] = lines[SALE_PROVIDER] if SALE_PROVIDER in lines.columns else ""
    facts["Amount"] = lines[SALE_AMOUNT] if SALE_AMOUNT in lines.columns else 0.0
else:
    facts = pd.DataFrame({
        "Date": pd.Series(dtype="datetime64[ns]"),
        "Client": pd.Series(dtype="object"),
        "Service_Line": pd.Series(dtype="object"),
        "Provider": pd.Series(dtype="object"),
        "Amount": pd.Series(dtype="float"),
    })

# Ensure join-derived columns exist before use
if "Service_Appt" not in facts.columns:
    facts["Service_Appt"] = ""
if "Booked via Google Ads" not in facts.columns:
    facts["Booked via Google Ads"] = False

# Join appointment service + google flag by ID, if present
if SALE_TICKET and APPT_ID and (SALE_TICKET in lines.columns) and (APPT_ID in appts.columns):
    try:
        # Service from appointments
        if APPT_SERVICE:
            svc_map = appts[[APPT_ID, APPT_SERVICE]].drop_duplicates().rename(
                columns={APPT_ID: "__JOIN__", APPT_SERVICE: "__ApptService"}
            )
            tmp = lines[[SALE_TICKET]].rename(columns={SALE_TICKET: "__JOIN__"}).join(
                svc_map.set_index("__JOIN__"), on="__JOIN__"
            )
            facts["Service_Appt"] = tmp.get("__ApptService", "")
        # Google flag
        flag_map = appts[[APPT_ID, APPT_GOOGLE_FLAG]].rename(
            columns={APPT_ID: "__JOIN__", APPT_GOOGLE_FLAG: "__GoogleFlag"}
        )
        tmp2 = lines[[SALE_TICKET]].rename(columns={SALE_TICKET: "__JOIN__"}).join(
            flag_map.set_index("__JOIN__"), on="__JOIN__"
        )
        facts["Booked via Google Ads"] = tmp2.get("__GoogleFlag", False).fillna(False).astype(bool)
    except Exception:
        pass

# Prefer appointment service; fallback to line-item service
facts["Service_XAxis"] = facts["Service_Appt"].astype(str).where(
    facts["Service_Appt"].astype(str).str.strip() != "",
    facts["Service_Line"].astype(str)
)

# Backfill provider via appointments if missing
if (facts["Provider"].astype(str).str.strip().eq("").all() if "Provider" in facts.columns else True) and SALE_TICKET and APPT_ID and APPT_PROVIDER:
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
providers = ["All Providers"] + (sorted([str(x) for x in facts.get("Provider", pd.Series(dtype=object)).dropna().unique().tolist()]))
services = ["All Services"] + (sorted([str(x) for x in facts.get("Service_XAxis", pd.Series(dtype=object)).dropna().unique().tolist()]))

c1, c2, c3, c4 = st.columns([1.2,1.2,1,1])
with c1:
    sel_time = st.selectbox("Time Range", options=time_options, index=5)
with c2:
    period_anchor = st.selectbox("Period Anchor", options=["Today (server time)", "Latest date in data"], index=0,
                                 help="Controls how preset ranges are computed.")
with c3:
    sel_provider = st.selectbox("Provider", options=providers, index=0)
with c4:
    sel_service = st.selectbox("Service (from Appointments)", options=services, index=0)

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
if sel_service != "All Services" and "Service_XAxis" in filtered.columns:
    filtered = filtered[filtered["Service_XAxis"].astype(str) == sel_service]
if only_google and "Booked via Google Ads" in filtered.columns:
    filtered = filtered[filtered["Booked via Google Ads"] == True]
if q:
    ql = q.lower().strip()
    mask = pd.Series(False, index=filtered.index)
    for c in ["Client","Provider","Service_XAxis","Service_Line"]:
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
unique_clients = int(filtered.get("Client", pd.Series(dtype=object)).astype(str).str.strip().replace({"":"__NA__"}).nunique())
total_revenue = float(filtered.get("Amount", pd.Series(dtype=float)).sum())

# Recurring / return rate
recurring_clients = 0
return_rate = 0.0
if APPT_CLIENT and APPT_DATE and not appts_filtered.empty:
    visits = appts_filtered.groupby(APPT_CLIENT)[APPT_DATE].nunique()
    recurring_clients = int((visits >= 2).sum())
    base_clients = int(appts_filtered[APPT_CLIENT].astype(str).nunique())
    return_rate = (recurring_clients / base_clients) if base_clients else 0.0

# First-time vs Returning (based on first-ever appointment date)
ft_clients = rt_clients = 0
ft_rev = rt_rev = 0.0
if APPT_CLIENT and APPT_DATE and not appts.empty and "Client" in filtered and "Date" in filtered:
    appts_nonnull = appts.dropna(subset=[APPT_CLIENT, APPT_DATE]).copy()
    appts_nonnull[APPT_DATE] = pd.to_datetime(appts_nonnull[APPT_DATE], errors="coerce")
    first_seen = appts_nonnull.groupby(APPT_CLIENT)[APPT_DATE].min()
    min_f = pd.to_datetime(filtered["Date"]).min()
    max_f = pd.to_datetime(filtered["Date"]).max()
    def classify(c):
        fs = first_seen.get(c, pd.NaT)
        if pd.isna(fs) or pd.isna(min_f) or pd.isna(max_f):
            return "Returning"
        return "First-time" if (fs >= min_f and fs <= max_f) else "Returning"
    filtered["Client Type"] = filtered["Client"].astype(str).apply(classify)
    ft_clients = filtered.loc[filtered["Client Type"]=="First-time","Client"].nunique()
    rt_clients = filtered.loc[filtered["Client Type"]=="Returning","Client"].nunique()
    ft_rev = float(filtered.loc[filtered["Client Type"]=="First-time","Amount"].sum())
    rt_rev = float(filtered.loc[filtered["Client Type"]=="Returning","Amount"].sum())
else:
    filtered["Client Type"] = "Returning"

# Canceled / No-show KPIs from appointments
cancel_appts = noshow_appts = 0
if not appts_filtered.empty:
    status_series = appts_filtered[STATUS_COL].astype(str).str.lower() if STATUS_COL else pd.Series([""]*len(appts_filtered), index=appts_filtered.index)
    no_show_series = appts_filtered[NO_SHOW_COL] if NO_SHOW_COL else pd.Series([None]*len(appts_filtered), index=appts_filtered.index)
    cancel_flag = status_series.str.contains("cancel", na=False)
    noshow_flag = status_series.str.contains("no[-\\s]?show", na=False)
    if no_show_series.notna().any():
        if pd.api.types.is_bool_dtype(no_show_series):
            noshow_flag = noshow_flag | no_show_series.fillna(False)
        else:
            noshow_flag = noshow_flag | no_show_series.astype(str).str.lower().isin(["true","1","yes","y","no show","no-show","noshow"])
    cancel_appts = int(cancel_flag.sum())
    noshow_appts = int(noshow_flag.sum())

# ====== KPI Renders ======
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Unique Clients</div><div class="kpi-value">{unique_clients:,}</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Recurring Clients (≥2 dates)</div><div class="kpi-value">{recurring_clients:,}</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Return Rate</div><div class="kpi-value">{return_rate:.1%}</div></div>', unsafe_allow_html=True)
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
with c1:
    by_service = filtered.groupby("Service_XAxis", dropna=False)["Amount"].sum().sort_values(ascending=False).head(20)
    st.markdown("#### Revenue by Service (Service from Appointments)")
    st.bar_chart(by_service)
with c2:
    if "Provider" in filtered and not filtered.empty:
        by_provider = filtered.groupby("Provider", dropna=False)["Amount"].sum().sort_values(ascending=False).head(20)
        st.markdown("#### Revenue by Provider")
        st.bar_chart(by_provider)
    else:
        st.info("No provider data available.")

# ====== Revenue Over Time ======
st.markdown("#### Revenue Over Time")
if "Date" in filtered and not filtered.empty:
    by_day = filtered.dropna(subset=["Date"]).copy()
    by_day["Day"] = pd.to_datetime(by_day["Date"]).dt.date
    by_day = by_day.groupby("Day")["Amount"].sum().sort_index()
    st.line_chart(by_day)
else:
    st.info("No dated revenue rows to chart.")

# ====== Table with Google Ads highlight ======
st.markdown("#### Line Items (Filtered) — Google Ads highlighted")
display = filtered.copy()
if "Date" in display.columns:
    display["Date"] = pd.to_datetime(display["Date"], errors="coerce").dt.strftime("%b %d, %Y")
display = display.rename(columns={"Service_XAxis":"Service (from Appointments)"})
wanted_cols = ["Client","Client Type","Provider","Service (from Appointments)","Amount","Date","Booked via Google Ads"]
for col in wanted_cols:
    if col not in display.columns:
        display[col] = "" if col != "Booked via Google Ads" else False
display = display[wanted_cols]

def highlight_ads(row):
    try:
        flag = bool(row.get("Booked via Google Ads", False))
    except Exception:
        flag = False
    color = "#2a3a22" if flag else ""
    return [f"background-color: {color}" for _ in row]

styled = display.style.apply(highlight_ads, axis=1)
st.dataframe(styled, use_container_width=True, height=520)

# ====== Client Lifetime Value ======
st.markdown("#### Client Lifetime Value (Total Spend)")
ltv = pd.DataFrame()
if "Client" in filtered and "Amount" in filtered:
    ltv = filtered.groupby("Client", dropna=False)["Amount"].sum().reset_index().sort_values("Amount", ascending=False)
    ltv.rename(columns={"Client":"Client","Amount":"Total Spend"}, inplace=True)
st.dataframe(ltv, use_container_width=True, height=360)

st.caption("Tips: Put CSVs in ./data. First-time = first-ever appt within the current date filter. Cancel/No-show derived from Status/No-Show columns.")
