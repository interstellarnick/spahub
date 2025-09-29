
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
    for sep in [",",";","\\t","|"]:
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
    s = s.astype(str).str.replace(r"[\\$,]", "", regex=True)
    s = s.str.replace(r"[^0-9.\\-]", "", regex=True)
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

# ====== Load data ======
DEFAULT_APPTS = Path("/mnt/data/Salon Appointments-2025-09-25 (1).csv")
DEFAULT_LINES = Path("/mnt/data/Checkout Line Items-2025-09-25.csv")

appts = read_csv_robust(DEFAULT_APPTS)
lines = read_csv_robust(DEFAULT_LINES)

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

# ====== Build facts (sales rows) and attach APPOINTMENT service via join ======
facts = pd.DataFrame()
if not lines.empty:
    facts["Date"] = lines[SALE_DATE] if SALE_DATE in lines.columns else pd.NaT
    facts["Client"] = lines[SALE_CLIENT] if SALE_CLIENT in lines.columns else ""
    # Start with the line-item service (fallback)
    facts["Service_Line"] = lines[SALE_SERVICE] if SALE_SERVICE in lines.columns else ""
    facts["Provider"] = lines[SALE_PROVIDER] if SALE_PROVIDER in lines.columns else ""
    facts["Amount"] = lines[SALE_AMOUNT] if SALE_AMOUNT in lines.columns else 0.0

    # Try to attach the Appointment's service using ticket/id join
    facts["Service_Appt"] = ""
    if SALE_TICKET and APPT_ID and APPT_SERVICE and SALE_TICKET in lines.columns and APPT_ID in appts.columns and APPT_SERVICE in appts.columns:
        try:
            svc_map = appts[[APPT_ID, APPT_SERVICE]].drop_duplicates()
            svc_map = svc_map.rename(columns={APPT_ID: "__JOIN__", APPT_SERVICE: "__ApptService"})
            tmp = lines[[SALE_TICKET]].rename(columns={SALE_TICKET: "__JOIN__"})
            tmp = tmp.join(svc_map.set_index("__JOIN__"), on="__JOIN__")
            facts["Service_Appt"] = tmp["__ApptService"].values
        except Exception:
            pass

# Pick the X-axis service label: prefer appointment service, fallback to line-item service
facts["Service_XAxis"] = facts["Service_Appt"].astype(str).where(facts["Service_Appt"].astype(str).str.strip() != "", facts["Service_Line"].astype(str))

# Derive provider on sales if missing via join on appointment id (after facts exist)
if (("Provider" not in facts.columns) or (facts["Provider"].astype(str).str.strip().eq("").all())) and SALE_TICKET and APPT_ID and APPT_PROVIDER:
    try:
        prov_map = appts[[APPT_ID, APPT_PROVIDER]].drop_duplicates().rename(columns={APPT_ID:"__JOIN__", APPT_PROVIDER:"__ApptProv"})
        tmp2 = lines[[SALE_TICKET]].rename(columns={SALE_TICKET:"__JOIN__"}).join(prov_map.set_index("__JOIN__"), on="__JOIN__")
        facts["Provider"] = facts["Provider"].mask(facts["Provider"].astype[str].str.strip().eq(""), tmp2["__ApptProv"])
    except Exception:
        pass

# ====== Filters ======
time_options = date_range_presets()
providers = ["All Providers"]
if "Provider" in facts.columns:
    providers += sorted([str(x) for x in facts["Provider"].dropna().unique().tolist()])
services = ["All Services"]
if "Service_XAxis" in facts.columns:
    services += sorted([str(x) for x in facts["Service_XAxis"].dropna().unique().tolist()])

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

custom_start = custom_end = None
if sel_time == "Custom range":
    cst, cen = st.columns(2)
    with cst:
        custom_start = st.date_input("Start date", value=(appts[APPT_DATE].min().date() if APPT_DATE and appts[APPT_DATE].notna().any() else date.today() - timedelta(days=30)))
    with cen:
        custom_end = st.date_input("End date", value=(appts[APPT_DATE].max().date() if APPT_DATE and appts[APPT_DATE].notna().any() else date.today()))

# ====== Apply filters ======
facts["Date"] = pd.to_datetime(facts["Date"], errors="coerce")
facts["Amount"] = pd.to_numeric(facts["Amount"], errors="coerce").fillna(0.0)

all_dates = facts["Date"].dt.date if "Date" in facts else pd.Series([], dtype="object")
filtered, anchor_date = apply_date_filter(facts, "Date", sel_time, custom_start, custom_end, period_anchor, all_dates)

if sel_provider != "All Providers" and "Provider" in filtered.columns:
    filtered = filtered[filtered["Provider"].astype(str) == sel_provider]
if sel_service != "All Services" and "Service_XAxis" in filtered.columns:
    filtered = filtered[filtered["Service_XAxis"].astype(str) == sel_service]
if q:
    ql = q.lower().strip()
    mask = pd.Series(False, index=filtered.index)
    for c in ["Client","Provider","Service_XAxis","Service_Line"]:
        if c in filtered.columns:
            mask |= filtered[c].astype(str).str.lower().str.contains(ql, na=False)
    filtered = filtered[mask]

# ====== KPIs ======
unique_clients = int(filtered["Client"].astype(str).str.strip().replace({"":"__NA__"}).nunique()) if "Client" in filtered else 0
# Recurring via appointments if available
recurring_clients = 0
return_rate = 0.0
if not appts.empty and APPT_CLIENT and APPT_DATE:
    appts_copy = appts.copy()
    appts_copy[APPT_DATE] = pd.to_datetime(appts_copy[APPT_DATE], errors="coerce")
    appts_filtered, _ = apply_date_filter(appts_copy, APPT_DATE, sel_time, custom_start, custom_end, period_anchor, appts_copy[APPT_DATE].dt.date)
    if sel_provider != "All Providers" and APPT_PROVIDER in appts_filtered.columns:
        appts_filtered = appts_filtered[appts_filtered[APPT_PROVIDER].astype(str) == sel_provider]
    visits = appts_filtered.groupby(APPT_CLIENT)[APPT_DATE].nunique()
    recurring_clients = int((visits >= 2).sum())
    base_clients = int(appts_filtered[APPT_CLIENT].astype(str).nunique())
    return_rate = (recurring_clients / base_clients) if base_clients else 0.0
else:
    if "Client" in filtered and "Date" in filtered:
        visits = filtered.groupby("Client")["Date"].nunique()
        recurring_clients = int((visits >= 2).sum())
        base_clients = int(filtered["Client"].astype(str).nunique())
        return_rate = (recurring_clients / base_clients) if base_clients else 0.0

total_revenue = float(filtered["Amount"].sum()) if "Amount" in filtered else 0.0

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Unique Clients</div><div class="kpi-value">{unique_clients:,}</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Recurring Clients (≥2 dates)</div><div class="kpi-value">{recurring_clients:,}</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Return Rate</div><div class="kpi-value">{return_rate:.1%}</div></div>', unsafe_allow_html=True)
with k4:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Total Revenue</div><div class="kpi-value">${total_revenue:,.2f}</div></div>', unsafe_allow_html=True)

if anchor_date is not None:
    st.caption(f"Using **{period_anchor}** as {anchor_date:%b %d, %Y}.")

# ====== First-time vs Returning ======
ft_map = pd.Series(dtype="datetime64[ns]")
if APPT_CLIENT and APPT_DATE and not appts.empty:
    appts_nonnull = appts.dropna(subset=[APPT_CLIENT, APPT_DATE]).copy()
    appts_nonnull[APPT_DATE] = pd.to_datetime(appts_nonnull[APPT_DATE], errors="coerce")
    first_seen = appts_nonnull.groupby(APPT_CLIENT)[APPT_DATE].min()
    ft_map = first_seen

filtered_clients = filtered.dropna(subset=["Client"]).copy() if "Client" in filtered else pd.DataFrame(columns=["Client"])
filtered_clients = filtered_clients[["Client"]].drop_duplicates()
if not filtered_clients.empty and not ft_map.empty and "Date" in filtered:
    min_f = pd.to_datetime(filtered["Date"]).min()
    max_f = pd.to_datetime(filtered["Date"]).max()
    def classify(c):
        fs = ft_map.get(c, pd.NaT)
        if pd.isna(fs) or pd.isna(min_f) or pd.isna(max_f):
            return "Returning"
        return "First-time" if (fs >= min_f and fs <= max_f) else "Returning"
    filtered["Client Type"] = filtered["Client"].astype(str).apply(classify)
else:
    filtered["Client Type"] = "Returning"

k5, k6 = st.columns(2)
with k5:
    by_service = filtered.groupby("Service_XAxis", dropna=False)["Amount"].sum().sort_values(ascending=False).head(20)
    st.markdown("#### Revenue by Service (Service from Appointments)")
    st.bar_chart(by_service)
with k6:
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

# ====== Tables ======
st.markdown("#### Line Items (Filtered)")
display = filtered.copy()
if "Date" in display.columns:
    display["Date"] = pd.to_datetime(display["Date"], errors="coerce").dt.strftime("%b %d, %Y")
for col in ["Client","Client Type","Provider","Service_XAxis","Amount","Date"]:
    if col not in display.columns:
        display[col] = ""
display = display.rename(columns={"Service_XAxis":"Service (from Appointments)"})
display = display[["Client","Client Type","Provider","Service (from Appointments)","Amount","Date"]]

csv = filtered.to_csv(index=False).encode("utf-8")
st.download_button("Export CSV", csv, file_name="spa_filtered_line_items.csv", mime="text/csv")
st.dataframe(display, use_container_width=True, height=520)

# ====== Client Lifetime Value ======
st.markdown("#### Client Lifetime Value (Total Spend)")
ltv = pd.DataFrame()
if "Client" in filtered and "Amount" in filtered:
    ltv = filtered.groupby("Client", dropna=False)["Amount"].sum().reset_index().sort_values("Amount", ascending=False)
    ltv.rename(columns={"Client":"Client","Amount":"Total Spend"}, inplace=True)
st.dataframe(ltv, use_container_width=True, height=360)
