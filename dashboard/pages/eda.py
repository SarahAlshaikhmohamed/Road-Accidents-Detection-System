import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# >> path to CSV 
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH =  ROOT / "datasets" / "analytics" / "Traffic_Accident_Statistics.csv"

# >> page config
st.set_page_config(page_title="RADS | Overview", layout="wide")

# >> small header/spacing css 
st.markdown("""
<style>
header[data-testid="stHeader"]{background:transparent;height:10px;padding:0;box-shadow:none}
header[data-testid="stHeader"] > div{min-height:10px;height:10px}
main .block-container{padding-top:0.2rem}
</style>
""", unsafe_allow_html=True)

# >> translations 
T = {
    "en": {
        "app_title": "RADS: Road Accident Detection System",
        "sidebar_title": "Settings",
        "choose_lang": "Choose Language",
        "region": "Filter by City (type exact name)",
        "eda_hdr": "Interactive EDA Dashboard",
        "maps_hdr": "Geospatial Maps",
    },
    "ar": {
        "app_title": "نظام توقع وتحليل الحوادث المرورية",
        "sidebar_title": "الإعدادات",
        "choose_lang": "اختر اللغة",
        "region": "تصفية باسم المدينة (اكتبي الاسم مطابق)",
        "eda_hdr": "لوحة EDA تفاعلية",
        "eda_info": "يتم تحميل البيانات من الكود (لا حاجة للرفع).",
        "maps_hdr": "الخرائط الجغرافية",
    }
}

# >> sidebar controls 
st.sidebar.header("Settings")
lang_choice = st.sidebar.selectbox(
    T["en"]["choose_lang"] + " / " + T["ar"]["choose_lang"],
    ["English", "العربية"],
    key="lang"
)
LANG = "ar" if lang_choice == "العربية" else "en"
st.sidebar.header(T[LANG]["sidebar_title"])
city_filter = st.sidebar.text_input(T[LANG]["region"], key="region").strip()

# >> title
st.title(T[LANG]["app_title"])

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

# >> EDA section 

st.header(T[LANG]["eda_hdr"])

dff = apply_city_filter(df)

# year / month filters
c1, c2 = st.columns(2)
with c1:
    years = sorted(pd.Series(df["Year"]).dropna().unique().tolist())
    year = st.selectbox("Filter: Year" if LANG=="en" else "تصفية: السنة", ["All"] + years, key="yr")
with c2:
    months = sorted(pd.Series(df["Month"]).dropna().unique().tolist())
    month = st.selectbox("Filter: Month" if LANG=="en" else "تصفية: الشهر", ["All"] + months, key="mo")
if year != "All":
    dff = dff[dff["Year"] == year]
if month != "All":
    dff = dff[dff["Month"] == month]

# preview
st.subheader("Dataset Preview" if LANG=="en" else "عرض سريع للبيانات")
st.dataframe(dff.head())

# Q1: top cities
st.subheader("Q1. Top cities by total accidents" if LANG=="en" else "س1. أعلى المدن بعدد الحوادث")
top_cities = df.groupby("City")["Total_Accidents"].sum().sort_values(ascending=False).head(15).reset_index()
fig1 = px.bar(top_cities, x="City", y="Total_Accidents", color="Total_Accidents",
              color_continuous_scale="Reds", title="Top Cities by Total Accidents")
st.plotly_chart(fig1, use_container_width=True)

# Q2: monthly trend
st.subheader("Q2. Monthly trend by outcome" if LANG=="en" else "س2. اتجاه شهري حسب المخرجات")
by_m = df.groupby(["Year","Month"])[["Damage_Accident","Injury_Accident","Death_Accident"]].sum().reset_index()
by_m["Year-Month"] = by_m["Year"].astype(str) + "-" + by_m["Month"].astype(str).str.zfill(2)
fig2 = px.line(by_m, x="Year-Month", y=["Damage_Accident","Injury_Accident","Death_Accident"],
               markers=True, title="Monthly Trend by Outcome")
st.plotly_chart(fig2, use_container_width=True)

# Q3: by weekday
st.subheader("Q3. Accidents by day of week" if LANG=="en" else "س3. الحوادث حسب أيام الأسبوع")
week_cols = [c for c in ["Saturday","Sunday","Monday","Tuesday","Wednesday","Thursday","Friday"] if c in df.columns]
if week_cols:
    wk = dff[week_cols].sum().rename_axis("Day").reset_index(name="Count")
    fig3 = px.bar(wk, x="Day", y="Count", color="Count", color_continuous_scale="Blues",
                  title="Accidents by Day of Week")
    st.plotly_chart(fig3, use_container_width=True)

# Q4: day vs night vs weekend/weekday
st.subheader("Q4. Day vs Night vs Weekend vs Weekday" if LANG=="en" else "س4. نهار/ليل/ويكند/أيام عمل")
dn_df = pd.DataFrame({
    "Category":["Day","Night","Weekend","Weekday"],
    "Count":[dff.get("Day",pd.Series([0]*len(dff))).sum(),
             dff.get("Night",pd.Series([0]*len(dff))).sum(),
             dff["Weekend"].sum(),
             dff["Weekday"].sum()]
})
fig4 = px.bar(dn_df, x="Category", y="Count", color="Category", title="Time-of-day vs Weekend effect")
st.plotly_chart(fig4, use_container_width=True)

# A1: Fatal top cities
st.subheader("A1. Top cities by fatal accidents" if LANG=="en" else "م1. أعلى المدن بالوفيات")
fatal_c = df.groupby("City")["Death_Accident"].sum().sort_values(ascending=False).head(10).reset_index()
st.plotly_chart(px.bar(fatal_c, x="City", y="Death_Accident", color="Death_Accident",
                       color_continuous_scale="Reds"), use_container_width=True)

# A2: city trends
st.subheader("A2. City trends over time" if LANG=="en" else "م2. اتجاه المدن عبر الزمن")
city_choices = sorted(df["City"].dropna().unique().tolist())[:12]
picks = st.multiselect("Pick cities" if LANG=="en" else "اختاري مدنًا",
                       city_choices, default=city_choices[:min(3,len(city_choices))])
if picks:
    city_tr = df.groupby(["City","Year","Month"])["Total_Accidents"].sum().reset_index()
    city_tr["YearMonth"] = city_tr["Year"].astype(str) + "-" + city_tr["Month"].astype(str).str.zfill(2)
    figA2 = px.line(city_tr[city_tr["City"].isin(picks)], x="YearMonth", y="Total_Accidents", color="City", markers=True)
    st.plotly_chart(figA2, use_container_width=True)

# A3: heatmap Year x Month
st.subheader("A3. Heatmap Year x Month" if LANG=="en" else "م3. خريطة حرارية سنة × شهر")
heat = df.groupby(["Year","Month"])["Total_Accidents"].sum().reset_index()
if not heat.empty:
    piv = heat.pivot(index="Year", columns="Month", values="Total_Accidents").fillna(0)
    figA3 = px.imshow(piv, aspect="auto", color_continuous_scale="Oranges", title="Total Accidents by Year x Month")
    st.plotly_chart(figA3, use_container_width=True)

# A4: Fatal vs Injury by weekday (estimated)
st.subheader("A4. Fatal vs Injury by weekday (estimated)" if LANG=="en" else "م4. وفيات مقابل إصابات حسب أيام الأسبوع (تقديري)")
if week_cols:
    wk_base = df[week_cols].sum()
    if wk_base.sum() > 0:
        wk_share = wk_base / wk_base.sum()
        fatal_share = (wk_share * df["Death_Accident"].sum()).reset_index(); fatal_share.columns=["Day","Fatal"]
        inj_share = (wk_share * df["Injury_Accident"].sum()).reset_index(); inj_share.columns=["Day","Injury"]
        import plotly.graph_objects as go
        stk = fatal_share.merge(inj_share, on="Day")
        figA4 = go.Figure([go.Bar(name="Fatal", x=stk["Day"], y=stk["Fatal"]),
                           go.Bar(name="Injury", x=stk["Day"], y=stk["Injury"])])
        figA4.update_layout(barmode="stack", title="Fatal vs Injury by Weekday (estimated)")
        st.plotly_chart(figA4, use_container_width=True)

# A5: vehicle involvement + A6: correlation with deaths
veh_cols_all = ["Small_Car","SUV_Car","Pickup_Truck","Bus","Truck","Water_Truck","Other_Car"]
vcols = [c for c in veh_cols_all if c in df.columns]
if vcols:
    st.subheader("A5. Vehicle involvement" if LANG=="en" else "م5. مشاركة المركبات")
    veh = df[vcols].sum().sort_values(ascending=False)
    st.plotly_chart(px.bar(veh, x=veh.index, y=veh.values, title="Vehicle Type Distribution"), use_container_width=True)

    st.subheader("A6. Vehicle correlation with deaths" if LANG=="en" else "م6. ارتباط أنواع المركبات بالوفيات")
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
    st.subheader("A7. Top 5 causes per year" if LANG=="en" else "م7. أعلى 5 أسباب لكل سنة")
    cy = df.groupby("Year")[cause_cols].sum().reset_index().melt(id_vars="Year", var_name="Cause", value_name="Count")
    top5 = cy.sort_values(["Year","Count"], ascending=[True,False]).groupby("Year").head(5)
    st.plotly_chart(px.bar(top5, x="Cause", y="Count", facet_col="Year", color="Cause"),
                    use_container_width=True)

    st.subheader("A8. Causes associated with deaths (correlation)" if LANG=="en" else "م8. الأسباب المرتبطة بالوفيات (ارتباط)")
    cc = []
    for c in cause_cols:
        try:
            cc_val = df[[c,"Death_Accident"]].corr().iloc[0,1]
        except Exception:
            cc_val = 0.0
        cc.append((c, cc_val))
    ccdf = pd.DataFrame(cc, columns=["Cause","Correlation_with_Deaths"]).sort_values("Correlation_with_Deaths", ascending=False)
    st.plotly_chart(px.bar(ccdf, x="Cause", y="Correlation_with_Deaths"), use_container_width=True)

# A9: over-speeding vs total accidents
if "Over_Speeding" in df.columns:
    st.subheader("A9. Over-speeding vs total accidents" if LANG=="en" else "م9. السرعة الزائدة مقابل إجمالي الحوادث")
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

# >> maps
st.subheader(T[LANG]["maps_hdr"])

CITY_LATLON = {
    "Riyadh": (24.7136, 46.6753), "الرياض": (24.7136, 46.6753),
    "Jeddah": (21.4858, 39.1925), "جدة": (21.4858, 39.1925),
    "Dammam": (26.4207, 50.0888), "الدمام": (26.4207, 50.0888),
    "Mecca": (21.3891, 39.8579), "مكة": (21.3891, 39.8579),
    "Medina": (24.5247, 39.5692), "المدينة": (24.5247, 39.5692),
    "Taif": (21.4373, 40.5127), "الطائف": (21.4373, 40.5127),
    "Tabuk": (28.3833, 36.5667), "تبوك": (28.3833, 36.5667),
    "Abha": (18.2164, 42.5053), "أبها": (18.2164, 42.5053),
    "Khamis Mushait": (18.3053, 42.7360), "خميس مشيط": (18.3053, 42.7360),
    "Hail": (27.5114, 41.7208), "حائل": (27.5114, 41.7208),
    "Al Kharj": (24.1486, 47.3050), "الخرج": (24.1486, 47.3050),
    "Al Khobar": (26.2794, 50.2083), "الخبر": (26.2794, 50.2083),
    "Buraidah": (26.3260, 43.9750), "بريدة": (26.3260, 43.9750),
    "Najran": (17.5650, 44.2236), "نجران": (17.5650, 44.2236),
    "Jazan": (16.8892, 42.5700), "جازان": (16.8892, 42.5700)
}

agg = df.groupby("City", as_index=False)["Total_Accidents"].sum()
agg["lat"] = agg["City"].apply(lambda c: CITY_LATLON.get(str(c), (None,None))[0])
agg["lon"] = agg["City"].apply(lambda c: CITY_LATLON.get(str(c), (None,None))[1])
bubble = agg.dropna(subset=["lat","lon"])
if not bubble.empty:
    import plotly.express as px
    figM1 = px.scatter_geo(
        bubble, lat="lat", lon="lon", scope="world",
        size="Total_Accidents", hover_name="City",
        projection="natural earth", title="City Bubble Map (Total Accidents)"
    )
    figM1.update_geos(fitbounds="locations", showcountries=True)
    st.plotly_chart(figM1, use_container_width=True)
else:
    st.info("لا توجد مدن معروفة في القاموس لرسم الخريطة. عدّلي أسماء المدن أو أضيفي إحداثيات.")

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