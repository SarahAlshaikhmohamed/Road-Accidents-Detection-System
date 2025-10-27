import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pathlib
from components.ui import render_sidebar

# >> path to CSV 
ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_PATH =  ROOT / "datasets" / "analytics" / "Traffic_Accident_Statistics.csv"

# >> page config
st.set_page_config(page_title="RADS | Overview", layout="wide")

# sidebar
render_sidebar() 


# >> small header
st.markdown("""
<style>
header[data-testid="stHeader"]{background:transparent;height:10px;padding:0;box-shadow:none}
header[data-testid="stHeader"] > div{min-height:10px;height:10px}
main .block-container{padding-top:0.2rem}
</style>
""", unsafe_allow_html=True)

# >> sidebar controls 
st.sidebar.header("Settings")
city_filter = st.sidebar.text_input("Filter by City (type exact name)", key="region").strip()

# >> title
st.title("RADS: Road Accident Detection System")

# >> load dataset 
def load_dataset(path: str):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        st.error(f"Failed to load dataset from '{path}'. Error: {e}")
        return None

    for col in ["Damage_Accident", "Injury_Accident", "Death_Accident"]:
        if col not in df.columns:
            df[col] = 0
    week_cols = ["Saturday","Sunday","Monday","Tuesday","Wednesday","Thursday","Friday"]
    for c in week_cols:
        if c not in df.columns:
            df[c] = 0
    if "City" not in df.columns:
        df["City"] = "Unknown"
    if "Year" not in df.columns:
        df["Year"] = 2024
    if "Month" not in df.columns:
        df["Month"] = 1

    df["Total_Accidents"] = df["Damage_Accident"] + df["Injury_Accident"] + df["Death_Accident"]
    df["Weekend"] = df["Friday"] + df["Saturday"]
    df["Weekday"] = df[["Sunday","Monday","Tuesday","Wednesday","Thursday"]].sum(axis=1)

    def sev_row(r):
        if r["Death_Accident"] > 0: return "Fatal"
        if r["Injury_Accident"] > 0: return "Injury"
        return "Minor"
    df["Accident_Severity"] = df.apply(sev_row, axis=1)
    return df

# >> load once
if "df" not in st.session_state:
    st.session_state["df"] = load_dataset(DATA_PATH)
    if st.session_state["df"] is None:
        st.session_state["df"] = pd.DataFrame({
            "Year": [2023,2023,2024,2024],
            "Month": [1,2,1,2],
            "City": ["Riyadh","Jeddah","Riyadh","Dammam"],
            "Damage_Accident": [12,8,14,9],
            "Injury_Accident": [3,2,4,3],
            "Death_Accident": [1,0,1,0],
            "Saturday":[5,4,6,2],"Sunday":[3,1,4,2],"Monday":[2,1,2,1],
            "Tuesday":[2,0,2,1],"Wednesday":[3,1,2,1],"Thursday":[4,1,4,2],"Friday":[5,2,6,3],
            "Over_Speeding":[7,5,8,6],
            "Small_Car":[10,7,11,8], "SUV_Car":[6,5,7,6], "Pickup_Truck":[3,2,3,2],
            "Bus":[1,1,1,1], "Truck":[2,1,2,1], "Water_Truck":[0,0,1,0], "Other_Car":[2,1,2,1],
            "Illegal_Stop":[1,0,1,0], "Illegal_Turn":[1,1,1,1], "Illegal_Overtaking":[1,0,1,0],
            "Running_Red_Light":[2,1,2,1], "Under_Drugs_Influence":[0,0,0,0],
            "Other_Accident_Cause":[1,1,1,1],
            "Day":[20,14,22,15], "Night":[15,10,16,11]
        })
        df = st.session_state["df"]
        df["Total_Accidents"] = df["Damage_Accident"] + df["Injury_Accident"] + df["Death_Accident"]
        df["Weekend"] = df["Friday"] + df["Saturday"]
        df["Weekday"] = df[["Sunday","Monday","Tuesday","Wednesday","Thursday"]].sum(axis=1)

df = st.session_state["df"]
if df is None or len(df) == 0:
    st.stop()

# >> city filter
def apply_city_filter(df_in):
    if df_in is None or not city_filter:
        return df_in
    return df_in[df_in["City"].astype(str).str.strip().str.lower() == city_filter.lower()]

df_adv = df.copy()
if "Injuries" not in df_adv.columns:
    df_adv["Injuries"] = df_adv.get("Injury_Accident", 0)
if "Deaths" not in df_adv.columns:
    df_adv["Deaths"] = df_adv.get("Death_Accident", 0)
if "Uninjured" not in df_adv.columns:
    df_adv["Uninjured"] = df_adv.get("Damage_Accident", 0)

# >> EDA section 

st.header("Overview")

dff = apply_city_filter(df)

# ---- KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Accidents", f"{len(df_adv):,}")
c2.metric("Total Injuries", f"{int(df_adv['Injuries'].sum()):,}")
c3.metric("Total Deaths", f"{int(df_adv['Deaths'].sum()):,}")
c4.metric("Total Uninjured", f"{int(df_adv['Uninjured'].sum()):,}")

st.markdown("---")
# year / month filters
c1, c2 = st.columns(2)
with c1:
    years = sorted(pd.Series(df["Year"]).dropna().unique().tolist())
    year = st.selectbox("Filter: Year", ["All"] + years, key="yr")
with c2:
    months = sorted(pd.Series(df["Month"]).dropna().unique().tolist())
    month = st.selectbox("Filter: Month", ["All"] + months, key="mo")
if year != "All":
    dff = dff[dff["Year"] == year]
if month != "All":
    dff = dff[dff["Month"] == month]

# preview
st.subheader("Dataset Preview")
st.dataframe(dff.head())

# Q2: monthly trend
st.subheader("Monthly trend by outcome")
by_m = df.groupby(["Year","Month"])[["Damage_Accident","Injury_Accident","Death_Accident"]].sum().reset_index()
by_m["Year-Month"] = by_m["Year"].astype(str) + "-" + by_m["Month"].astype(str).str.zfill(2)
fig2 = px.line(by_m, x="Year-Month", y=["Damage_Accident","Injury_Accident","Death_Accident"],
               markers=True, title="Monthly Trend by Outcome")
st.plotly_chart(fig2, use_container_width=True)

# Q3: by weekday
st.subheader("Accidents by day of week")
week_cols = [c for c in ["Saturday","Sunday","Monday","Tuesday","Wednesday","Thursday","Friday"] if c in df.columns]
if week_cols:
    wk = dff[week_cols].sum().rename_axis("Day").reset_index(name="Count")
    fig3 = px.bar(wk, x="Day", y="Count", color="Count", color_continuous_scale="Blues",
                  title="Accidents by Day of Week")
    st.plotly_chart(fig3, use_container_width=True)

# A1: Fatal top cities
st.subheader("Top cities by fatal accidents")
fatal_c = df.groupby("City")["Death_Accident"].sum().sort_values(ascending=False).head(10).reset_index()
st.plotly_chart(px.bar(fatal_c, x="City", y="Death_Accident", color="Death_Accident",
                       color_continuous_scale="Reds"), use_container_width=True)

# A2: city trends
st.subheader("City trends over time")
city_choices = sorted(df["City"].dropna().unique().tolist())[:12]
picks = st.multiselect("Pick cities",
                       city_choices, default=city_choices[:min(3,len(city_choices))])
if picks:
    city_tr = df.groupby(["City","Year","Month"])["Total_Accidents"].sum().reset_index()
    city_tr["YearMonth"] = city_tr["Year"].astype(str) + "-" + city_tr["Month"].astype(str).str.zfill(2)
    figA2 = px.line(city_tr[city_tr["City"].isin(picks)], x="YearMonth", y="Total_Accidents", color="City", markers=True)
    st.plotly_chart(figA2, use_container_width=True)

# A5/A6: vehicle involvement + A6: correlation with deaths
veh_cols_all = ["Small_Car","SUV_Car","Pickup_Truck","Bus","Truck","Water_Truck","Other_Car"]
vcols = [c for c in veh_cols_all if c in df.columns]
if vcols:
    st.subheader("Vehicle involvement")
    veh = df[vcols].sum().sort_values(ascending=False)
    st.plotly_chart(px.bar(veh, x=veh.index, y=veh.values, title="Vehicle Type Distribution"), use_container_width=True)

    st.subheader("Vehicle correlation with deaths")
    corr_list = []
    for v in vcols:
        try:
            corr_val = df[[v,"Death_Accident"]].corr().iloc[0,1]
        except Exception:
            corr_val = 0.0
        corr_list.append((v, corr_val))
    cdf = pd.DataFrame(corr_list, columns=["Vehicle","Correlation_with_Deaths"]).sort_values("Correlation_with_Deaths", ascending=False)
    st.plotly_chart(px.bar(cdf, x="Vehicle", y="Correlation_with_Deaths", title="Correlation with Deaths"),
                    use_container_width=True)

# A7/A8: causes + correlation
cause_cols_all = ["Illegal_Stop","Illegal_Turn","Illegal_Overtaking","Running_Red_Light",
                  "Over_Speeding","Under_Drugs_Influence","Other_Accident_Cause"]
cause_cols = [c for c in cause_cols_all if c in df.columns]
if cause_cols:
    st.subheader("Top 5 causes per year")
    cy = df.groupby("Year")[cause_cols].sum().reset_index().melt(id_vars="Year", var_name="Cause", value_name="Count")
    top5 = cy.sort_values(["Year","Count"], ascending=[True,False]).groupby("Year").head(5)
    st.plotly_chart(px.bar(top5, x="Cause", y="Count", facet_col="Year", color="Cause"),
                    use_container_width=True)

    st.subheader("Causes associated with deaths (correlation)")
    cc = []
    for c in cause_cols:
        try:
            cc_val = df[[c,"Death_Accident"]].corr().iloc[0,1]
        except Exception:
            cc_val = 0.0
        cc.append((c, cc_val))
    ccdf = pd.DataFrame(cc, columns=["Cause","Correlation_with_Deaths"]).sort_values("Correlation_with_Deaths", ascending=False)
    st.plotly_chart(px.bar(ccdf, x="Cause", y="Correlation_with_Deaths"), use_container_width=True)

# >> maps
st.subheader("Geospatial Maps")

CITY_LATLON = {
    "Riyadh": (24.7136, 46.6753),
    "Jeddah": (21.4858, 39.1925),
    "Dammam": (26.4207, 50.0888),
    "Mecca": (21.3891, 39.8579),
    "Medina": (24.5247, 39.5692),
    "Taif": (21.4373, 40.5127),
    "Tabuk": (28.3833, 36.5667),
    "Abha": (18.2164, 42.5053),
    "Khamis Mushait": (18.3053, 42.7360),
    "Hail": (27.5114, 41.7208),
    "Al Kharj": (24.1486, 47.3050),
    "Al Khobar": (26.2794, 50.2083),
    "Buraidah": (26.3260, 43.9750),
    "Najran": (17.5650, 44.2236),
    "Jazan": (16.8892, 42.5700)
}

agg = df.groupby("City", as_index=False)["Total_Accidents"].sum()
agg["lat"] = agg["City"].apply(lambda c: CITY_LATLON.get(str(c), (None,None))[0])
agg["lon"] = agg["City"].apply(lambda c: CITY_LATLON.get(str(c), (None,None))[1])
bubble = agg.dropna(subset=["lat","lon"])
if not bubble.empty:
    figM1 = px.scatter_geo(
        bubble, lat="lat", lon="lon", scope="world",
        size="Total_Accidents", hover_name="City",
        projection="natural earth", title="City Bubble Map (Total Accidents)"
    )
    figM1.update_geos(fitbounds="locations", showcountries=True)
    st.plotly_chart(figM1, use_container_width=True)
else:
    st.info("No known cities in the mapping dictionary to plot. Adjust city names or add coordinates.")

if all(col in df.columns for col in ["Latitude","Longitude"]):
    st.caption("Raw Points Map (using Latitude/Longitude columns)")
    dmap = dff.dropna(subset=["Latitude","Longitude"]).copy()
    if not dmap.empty:
        import plotly.express as px
        dmap["size_var"] = dmap.get("Total_Accidents", pd.Series([5]*len(dmap)))
        figM2 = px.scatter_geo(
            dmap, lat="Latitude", lon="Longitude", scope="world",
            size="size_var", hover_name="City",
            color=dmap.get("Accident_Severity","Minor"),
            projection="natural earth", title="Point Map (Records)"
        )
        figM2.update_geos(center=dict(lat=23.8859, lon=45.0792),
                          lataxis_showgrid=True, lonaxis_showgrid=True)
        st.plotly_chart(figM2, use_container_width=True)


# -------------------------------------------
# Advanced analytics in same data


st.markdown("---")

df_adv = df.copy()
if "Injuries" not in df_adv.columns:
    df_adv["Injuries"] = df_adv.get("Injury_Accident", 0)
if "Deaths" not in df_adv.columns:
    df_adv["Deaths"] = df_adv.get("Death_Accident", 0)
if "Uninjured" not in df_adv.columns:
    df_adv["Uninjured"] = df_adv.get("Damage_Accident", 0)

# Optional: Gregorian year (fallback to Year if not available)
if "Gregorian_Year" not in df_adv.columns:
    try:
        from hijri_converter import convert
        def to_greg(y):
            try:
                y = int(y)
            except:
                return None
            if y >= 1900:
                return y
            return convert.Hijri(y, 1, 1).to_gregorian().year
        df_adv["Gregorian_Year"] = df_adv["Year"].apply(to_greg).fillna(df_adv["Year"])
    except Exception:
        df_adv["Gregorian_Year"] = df_adv["Year"]

# Region mapping
city_to_region = {
    'Riyadh':'Central','Al-Qassim':'Central',
    'Jeddah':'West','Madina':'West','Taif':'West','Makkah':'West','Mecca':'West',
    'Eastern':'East','Dammam':'East','Dhahran':'East','Khobar':'East','Al Khobar':'East','Hofuf':'East',
    'Hail':'North','Tabuk':'North','Qurayyat':'North','Northern Borders':'North','Al-Jawf':'North',
    'Asir':'South','Jazan':'South','Najran':'South','Al-Bahah':'South'
}
df_adv["Region"] = df_adv["City"].map(city_to_region).fillna("Other")

st.markdown("---")

# ---- Top 10 Cities by Total Deaths
st.subheader("Top 10 Cities by Total Deaths")
city_deaths = (
    df_adv.groupby('City', as_index=False)['Deaths']
          .sum()
          .sort_values(by='Deaths', ascending=False)
)
fig_cd = px.bar(
    city_deaths.head(10),
    x='City', y='Deaths', text='Deaths',
    color='Deaths', color_continuous_scale='Reds',
    title='Top 10 Cities by Total Deaths'
)
fig_cd.update_traces(textposition='outside')
fig_cd.update_layout(title_x=0.5)
st.plotly_chart(fig_cd, use_container_width=True)

# ---- Total Deaths by Region
st.subheader("Total Deaths by Region")
region_deaths = (
    df_adv.groupby('Region', as_index=False)['Deaths']
          .sum()
          .sort_values(by='Deaths', ascending=False)
)
fig_rd = px.bar(
    region_deaths, x='Region', y='Deaths', text='Deaths',
    color='Deaths', color_continuous_scale='Reds',
    title='Total Deaths by Region'
)
fig_rd.update_traces(textposition='outside')
fig_rd.update_layout(title_x=0.5)
st.plotly_chart(fig_rd, use_container_width=True)

# ---- Injuries vs Deaths (Overall)
st.subheader("Injuries vs Deaths (Overall Comparison)")
summary = pd.DataFrame({'Metric':['Injuries','Deaths'],
                        'Total':[df_adv['Injuries'].sum(), df_adv['Deaths'].sum()]})
fig_comp = px.bar(
    summary, x='Metric', y='Total', color='Metric', text='Total',
    color_discrete_sequence=['#0080FF','#CC0000'],
    title='Injuries vs Deaths (Overall Comparison)'
)
fig_comp.update_traces(textposition='outside')
fig_comp.update_layout(title_x=0.5)
st.plotly_chart(fig_comp, use_container_width=True)

# ---- Regional Distribution (Grouped)
st.subheader("Regional Distribution of Injuries and Deaths")
region_summary = df_adv.groupby('Region')[['Injuries','Deaths']].sum().reset_index()
fig_region = px.bar(
    region_summary.melt(id_vars='Region', var_name='Metric', value_name='Total'),
    x='Region', y='Total', color='Metric', barmode='group', text='Total',
    color_discrete_sequence=['#0080FF','#CC0000'],
    title='Regional Distribution of Injuries and Deaths'
)
fig_region.update_traces(textposition='outside')
fig_region.update_layout(title_x=0.5)
st.plotly_chart(fig_region, use_container_width=True)

# ---- Gauges (Deaths & Injuries)
st.subheader("Accident Severity Gauges")
fig_gauge = go.Figure()
tot_deaths = float(df_adv['Deaths'].sum())
tot_inj = float(df_adv['Injuries'].sum())
fig_gauge.add_trace(go.Indicator(
    mode="gauge+number", value=max(tot_deaths, 0.0),
    title={"text":"Total Deaths"},
    gauge={'axis': {'range': [None, max(tot_deaths*1.5, 1)]},
           'bar': {'color': "#CC0000"},
           'steps': [
               {'range': [0, max(tot_deaths*0.5, 1e-6)], 'color': "#FFCCCC"},
               {'range': [max(tot_deaths*0.5, 1e-6), max(tot_deaths, 1)], 'color': "#FF6666"}
           ]},
    domain={'x':[0,0.5], 'y':[0,1]}
))
fig_gauge.add_trace(go.Indicator(
    mode="gauge+number", value=max(tot_inj, 0.0),
    title={"text":"Total Injuries"},
    gauge={'axis': {'range': [None, max(tot_inj*1.5, 1)]},
           'bar': {'color': "#0080FF"},
           'steps': [
               {'range': [0, max(tot_inj*0.5, 1e-6)], 'color': "#CCE5FF"},
               {'range': [max(tot_inj*0.5, 1e-6), max(tot_inj, 1)], 'color': "#66B2FF"}
           ]},
    domain={'x':[0.5,1], 'y':[0,1]}
))
fig_gauge.update_layout(title_x=0.5)
st.plotly_chart(fig_gauge, use_container_width=True)

# ---- Treemap: Deaths by Region
st.subheader("Deaths by Region (Treemap)")
fig_tree = px.treemap(
    region_deaths, path=['Region'], values='Deaths',
    color='Deaths', color_continuous_scale='Reds',
    title='Deaths by Region (Treemap)'
)
fig_tree.update_layout(title_x=0.5)
st.plotly_chart(fig_tree, use_container_width=True)

# ---- Trend Over Time (Gregorian Year)
st.subheader("Accident Trend Over Time (Injuries & Deaths)")
trend = df_adv.groupby('Gregorian_Year')[['Injuries','Deaths']].sum().reset_index()
fig_trend = px.line(
    trend, x='Gregorian_Year', y=['Injuries','Deaths'],
    markers=True,
    color_discrete_map={'Injuries':'#0080FF','Deaths':'#CC0000'},
    title='Accident Trend Over Time (Injuries & Deaths)'
)
fig_trend.update_layout(title_x=0.5, xaxis_title="Year (Gregorian)", yaxis_title="Total Count")
st.plotly_chart(fig_trend, use_container_width=True)


# >> day vs night vs weekend/weekday
st.subheader("Day vs Night vs Weekend vs Weekday")
dn_df = pd.DataFrame({
    "Category":["Day","Night","Weekend","Weekday"],
    "Count":[dff.get("Day",pd.Series([0]*len(dff))).sum(),
             dff.get("Night",pd.Series([0]*len(dff))).sum(),
             dff["Weekend"].sum(),
             dff["Weekday"].sum()]
})
fig4 = px.bar(dn_df, x="Category", y="Count", color="Category", title="Time-of-day vs Weekend effect")
st.plotly_chart(fig4, use_container_width=True)


# A3: heatmap Year x Month
st.subheader("Heatmap Year x Month")
heat = df.groupby(["Year","Month"])["Total_Accidents"].sum().reset_index()
if not heat.empty:
    piv = heat.pivot(index="Year", columns="Month", values="Total_Accidents").fillna(0)
    figA3 = px.imshow(piv, aspect="auto", color_continuous_scale="Oranges", title="Total Accidents by Year x Month")
    st.plotly_chart(figA3, use_container_width=True)

# A4: Fatal vs Injury by weekday (estimated)
st.subheader("Fatal vs Injury by weekday (estimated)")
if week_cols:
    wk_base = df[week_cols].sum()
    if wk_base.sum() > 0:
        wk_share = wk_base / wk_base.sum()
        fatal_share = (wk_share * df["Death_Accident"].sum()).reset_index(); fatal_share.columns=["Day","Fatal"]
        inj_share = (wk_share * df["Injury_Accident"].sum()).reset_index(); inj_share.columns=["Day","Injury"]
        stk = fatal_share.merge(inj_share, on="Day")
        figA4 = go.Figure([go.Bar(name="Fatal", x=stk["Day"], y=stk["Fatal"]),
                           go.Bar(name="Injury", x=stk["Day"], y=stk["Injury"])])
        figA4.update_layout(barmode="stack", title="Fatal vs Injury by Weekday (estimated)")
        st.plotly_chart(figA4, use_container_width=True)

# A9: over-speeding vs total accidents
if "Over_Speeding" in df.columns:
    st.subheader("Over-speeding vs total accidents")
    rel = df.groupby(["Year","Month"])[["Over_Speeding","Total_Accidents"]].sum().reset_index()
    if not rel.empty:
        import plotly.graph_objects as go
        x = rel["Over_Speeding"].astype(float).values
        y = rel["Total_Accidents"].astype(float).values
        if len(x) >= 2 and np.std(x) > 0:
            z = np.polyfit(x, y, 1); p = np.poly1d(z)
            xs = np.linspace(x.min(), x.max(), 100); ys = p(xs)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Points"))
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Trendline"))
            fig.update_layout(title="Over-Speeding vs Total Accidents")
            st.plotly_chart(fig, use_container_width=True)
