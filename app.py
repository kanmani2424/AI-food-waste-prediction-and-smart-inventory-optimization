import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import plotly.express as px

st.set_page_config(page_title="AI Food Waste Prediction & Smart Inventory Optimization", page_icon="🍱", layout="wide")
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

def load_file(filename):
    path = MODEL_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Model file not found:\n{path}")
    return joblib.load(path)

try:
    food_waste_model = load_file("food_waste_model.pkl")
    food_waste_encoder = load_file("label_encoder.pkl")
    demand_model = load_file("demand_prediction_model.pkl")
    demand_encoder = load_file("demand_label_encoders.pkl")
    expiry_model = load_file("expiry_risk_model.pkl")
    expiry_encoder = load_file("expiry_label_encoders.pkl")
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(f"❌ Model loading error: {e}")
    st.info("Check that all required .pkl files are inside the models folder.")

st.title("🍱 AI Food Waste Prediction & Smart Inventory Optimization")
st.write("AI-powered system for food waste prediction, demand forecasting, expiry risk detection and smart inventory optimization.")
if models_loaded:
    st.success("✅ All AI Models Loaded Successfully")

st.header("📋 Input Summary")
c1, c2, c3, c4 = st.columns(4)
with c1:
    category = st.selectbox("Category", ["Dairy","Fruits","Vegetables","Bakery","Meat","Beverages"])
with c2:
    region = st.selectbox("Region", ["North","South","East","West"])
with c3:
    initial_stock = st.number_input("Initial Stock", min_value=1, value=100)
with c4:
    daily_demand_input = st.number_input("Daily Demand", min_value=0, value=60)

with st.expander("⚙️ Advanced Input Parameters"):
    c1, c2, c3 = st.columns(3)
    with c1:
        shelf_life_days = st.number_input("Shelf Life Days", min_value=1, value=7)
        days_remaining = st.number_input("Days Remaining at Purchase", min_value=0, value=5)
        storage_temp = st.number_input("Storage Temperature", value=4.0)
        temp_deviation = st.number_input("Temperature Deviation", value=0.0)
        base_price = st.number_input("Base Price", min_value=1.0, value=100.0)
        cost_price = st.number_input("Cost Price", min_value=1.0, value=80.0)
    with c2:
        spoilage_sensitivity = st.number_input("Spoilage Sensitivity", min_value=0, max_value=10, value=1)
        day_of_week = st.number_input("Day of Week", min_value=0, max_value=6, value=1)
        is_weekend = st.selectbox("Weekend", [0,1])
        month = st.number_input("Month", min_value=1, max_value=12, value=1)
        demand_variability = st.number_input("Demand Variability", min_value=0.0, value=10.0)
        temp_abuse_events = st.number_input("Temperature Abuse Events", min_value=0, value=0)
    with c3:
        distribution_hours = st.number_input("Distribution Hours", min_value=0, value=5)
        handling_score = st.number_input("Handling Score", min_value=0, max_value=10, value=8)
        packaging_score = st.number_input("Packaging Score", min_value=0, max_value=10, value=8)
        supplier_score = st.number_input("Supplier Score", min_value=0, max_value=10, value=9)
        is_promoted = st.selectbox("Promoted", [0,1])
        festival = st.selectbox("Festival", ["No","Diwali","Christmas","Pongal","Ramadan"])
        weather = st.selectbox("Weather", ["Sunny","Rainy","Cloudy","Hot"])
        quality_grade = st.selectbox("Quality Grade", ["A","B","C"])

predict_button = st.button("🚀 Predict & Optimize", use_container_width=True)

def encode_frame(frame, encoders):
    for col, encoder in encoders.items():
        if col in frame.columns:
            value = str(frame.loc[0, col])
            frame[col] = encoder.transform(frame[col].astype(str)) if value in encoder.classes_ else 0
    return frame

if predict_button and models_loaded:
    waste_data = pd.DataFrame({
        "category":[category],"region":[region],"shelf_life_days":[shelf_life_days],
        "days_remaining_at_purchase":[days_remaining],"storage_temp":[storage_temp],"temp_deviation":[temp_deviation],
        "base_price":[base_price],"cost_price":[cost_price],"initial_quantity":[initial_stock],
        "spoilage_sensitivity":[spoilage_sensitivity],"day_of_week":[day_of_week],"is_weekend":[is_weekend],
        "month":[month],"daily_demand":[daily_demand_input],"demand_variability":[demand_variability],
        "temp_abuse_events":[temp_abuse_events],"distribution_hours":[distribution_hours],"handling_score":[handling_score],
        "packaging_score":[packaging_score],"supplier_score":[supplier_score],"is_promoted":[is_promoted],
        "festival":[festival],"weather":[weather]
    })
    waste_data = encode_frame(waste_data, food_waste_encoder)
    units_wasted = max(int(round(food_waste_model.predict(waste_data)[0])), 0)
    units_wasted = min(units_wasted, initial_stock)

    demand_data = pd.DataFrame({
        "category":[category],"region":[region],"shelf_life_days":[shelf_life_days],
        "days_remaining_at_purchase":[days_remaining],"storage_temp":[storage_temp],"temp_deviation":[temp_deviation],
        "base_price":[base_price],"cost_price":[cost_price],"initial_quantity":[initial_stock],
        "spoilage_sensitivity":[spoilage_sensitivity],"day_of_week":[day_of_week],"is_weekend":[is_weekend],
        "month":[month],"demand_variability":[demand_variability],"temp_abuse_events":[temp_abuse_events],
        "distribution_hours":[distribution_hours],"handling_score":[handling_score],"packaging_score":[packaging_score],
        "supplier_score":[supplier_score],"is_promoted":[is_promoted],"festival":[festival],"weather":[weather]
    })
    demand_data = encode_frame(demand_data, demand_encoder)
    predicted_demand = max(int(round(demand_model.predict(demand_data)[0])), 0)

    expiry_data = pd.DataFrame({
        "category":[category],"region":[region],"shelf_life_days":[shelf_life_days],
        "days_remaining_at_purchase":[days_remaining],"storage_temp":[storage_temp],"temp_deviation":[temp_deviation],
        "initial_quantity":[initial_stock],"spoilage_sensitivity":[spoilage_sensitivity],"daily_demand":[predicted_demand],
        "demand_variability":[demand_variability],"temp_abuse_events":[temp_abuse_events],"distribution_hours":[distribution_hours],
        "handling_score":[handling_score],"packaging_score":[packaging_score],"quality_grade":[quality_grade],
        "supplier_score":[supplier_score],"is_promoted":[is_promoted],"festival":[festival],"weather":[weather]
    })
    expiry_data = encode_frame(expiry_data, expiry_encoder)
    expiry_risk = str(expiry_model.predict(expiry_data)[0])

    if days_remaining <= 2:
        recommended_quantity = 0
        inventory_message = "🚫 DON'T BUY — Expiry is very near."
        inventory_reason = "Avoid adding more stock because the product may expire before it is sold."
    else:
        stock_gap = predicted_demand - initial_stock
        if stock_gap > 0:
            recommended_quantity = stock_gap
            inventory_message = "🛒 BUY ADDITIONAL STOCK"
            inventory_reason = f"Predicted demand is {predicted_demand} units but current stock is only {initial_stock} units."
        else:
            recommended_quantity = 0
            inventory_message = "📦 MAINTAIN CURRENT INVENTORY"
            inventory_reason = "Current inventory is sufficient for predicted demand."

    if days_remaining <= 1: discount = 30
    elif days_remaining <= 2: discount = 25
    elif days_remaining <= 3: discount = 20
    elif units_wasted >= 40: discount = 15
    elif units_wasted >= 20: discount = 10
    else: discount = 0

    waste_cost = units_wasted * cost_price
    waste_percentage = (units_wasted / initial_stock) * 100 if initial_stock else 0
    waste_risk = "LOW" if waste_percentage < 20 else "MEDIUM" if waste_percentage < 50 else "HIGH"

    st.session_state.update({
        "prediction_done":True,"units_wasted":units_wasted,"predicted_demand":predicted_demand,
        "expiry_risk":expiry_risk,"recommended_quantity":recommended_quantity,
        "inventory_message":inventory_message,"inventory_reason":inventory_reason,"discount":discount,
        "waste_cost":waste_cost,"waste_percentage":waste_percentage,"waste_risk":waste_risk,
        "category":category,"region":region,"initial_stock":initial_stock,"daily_demand":daily_demand_input,
        "festival":festival,"weather":weather,"days_remaining":days_remaining,"storage_temp":storage_temp,
        "base_price":base_price,"cost_price":cost_price
    })

if st.session_state.get("prediction_done", False):
    units_wasted = st.session_state["units_wasted"]
    predicted_demand = st.session_state["predicted_demand"]
    expiry_risk = st.session_state["expiry_risk"]
    recommended_quantity = st.session_state["recommended_quantity"]
    inventory_message = st.session_state["inventory_message"]
    inventory_reason = st.session_state["inventory_reason"]
    discount = st.session_state["discount"]
    waste_cost = st.session_state["waste_cost"]
    waste_percentage = st.session_state["waste_percentage"]
    waste_risk = st.session_state["waste_risk"]

    st.divider()
    st.header("📊 AI Prediction Dashboard")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("🗑️ Units Wasted", units_wasted)
    c2.metric("📈 Predicted Demand", predicted_demand)
    c3.metric("📦 Recommended Stock", recommended_quantity)
    c4.metric("💰 Waste Cost", f"₹{waste_cost:,.0f}")

    st.subheader("🚦 Waste Risk")
    if waste_risk == "LOW": st.success(f"🟢 LOW RISK — Waste Percentage: {waste_percentage:.2f}%")
    elif waste_risk == "MEDIUM": st.warning(f"🟡 MEDIUM RISK — Waste Percentage: {waste_percentage:.2f}%")
    else: st.error(f"🔴 HIGH RISK — Waste Percentage: {waste_percentage:.2f}%")

    st.subheader("⏳ Expiry Risk")
    if expiry_risk.lower() == "high": st.error("🔴 HIGH EXPIRY RISK — Immediate action required.")
    elif expiry_risk.lower() == "medium": st.warning("🟡 MEDIUM EXPIRY RISK — Monitor inventory.")
    else: st.success("🟢 LOW EXPIRY RISK — Inventory is safe.")

    st.subheader("📦 Smart Inventory Recommendation")
    if recommended_quantity > 0:
        st.success(f"🛒 {inventory_message}")
        st.write(f"Recommended additional quantity: **{recommended_quantity} units**")
    else:
        st.info(f"📦 {inventory_message}")
        st.write(inventory_reason)

    st.subheader("💰 Dynamic Discount Recommendation")
    if discount > 0:
        st.warning(f"🏷️ Recommended Discount: **{discount}%**")
        st.write("Discount is recommended to reduce food waste and improve sell-through before expiry.")
    else: st.success("✅ No discount required.")

    st.subheader("🎉 Festival Demand")
    st.write(f"Selected Festival: **{st.session_state['festival']}**")
    st.write(f"Predicted Demand during selected condition: **{predicted_demand} units**")

    # ========================================================
    # FIVE PROFESSIONAL CHARTS
    # ========================================================
    st.divider()
    st.header("📈 Visual Analytics Dashboard")

    chart1 = pd.DataFrame({"Metric":["Current Stock","Daily Demand","Predicted Demand","Recommended Stock"],"Quantity":[initial_stock,daily_demand_input,predicted_demand,recommended_quantity]})
    fig1 = px.bar(chart1,x="Metric",y="Quantity",text="Quantity",title="📈 Demand & Inventory Comparison")
    fig1.update_traces(textposition="outside")
    fig1.update_layout(height=430,xaxis_title="Metric",yaxis_title="Quantity",margin=dict(l=20,r=20,t=60,b=20))
    st.plotly_chart(fig1,use_container_width=True)

    remaining_stock = max(initial_stock-units_wasted,0)
    chart2 = pd.DataFrame({"Status":["Food Wasted","Remaining Stock"],"Quantity":[units_wasted,remaining_stock]})
    fig2 = px.bar(chart2,x="Status",y="Quantity",text="Quantity",title="🗑️ Food Waste vs Remaining Stock")
    fig2.update_traces(textposition="outside")
    fig2.update_layout(height=430,xaxis_title="Stock Status",yaxis_title="Quantity",margin=dict(l=20,r=20,t=60,b=20))
    st.plotly_chart(fig2,use_container_width=True)

    inventory_cost = initial_stock*cost_price
    chart3 = pd.DataFrame({"Cost Type":["Inventory Cost","Waste Cost"],"Amount":[inventory_cost,waste_cost]})
    fig3 = px.bar(chart3,x="Cost Type",y="Amount",text="Amount",title="💰 Inventory Cost vs Waste Cost")
    fig3.update_traces(texttemplate="₹%{text:,.0f}",textposition="outside")
    fig3.update_layout(height=430,xaxis_title="Cost Type",yaxis_title="Amount (₹)",margin=dict(l=20,r=20,t=60,b=20))
    st.plotly_chart(fig3,use_container_width=True)

    def risk_score(value):
        value=str(value).upper()
        return 1 if value=="LOW" else 2 if value=="MEDIUM" else 3
    chart4 = pd.DataFrame({"Risk Type":["Waste Risk","Expiry Risk"],"Risk Score":[risk_score(waste_risk),risk_score(expiry_risk)]})
    fig4 = px.bar(chart4,x="Risk Type",y="Risk Score",text="Risk Score",title="⏳ Waste & Expiry Risk Analysis")
    fig4.update_traces(textposition="outside")
    fig4.update_layout(height=430,xaxis_title="Risk Type",yaxis_title="Risk Level",yaxis=dict(tickmode="array",tickvals=[1,2,3],ticktext=["Low","Medium","High"]),margin=dict(l=20,r=20,t=60,b=20))
    st.plotly_chart(fig4,use_container_width=True)

    chart5 = pd.DataFrame({"Recommendation":["Recommended Discount","No Discount Portion"],"Percentage":[discount,max(100-discount,0)]})
    fig5 = px.pie(chart5,names="Recommendation",values="Percentage",title="🏷️ Dynamic Discount Recommendation")
    fig5.update_layout(height=430,margin=dict(l=20,r=20,t=60,b=20))
    st.plotly_chart(fig5,use_container_width=True)

    st.subheader("📋 Prediction Input Summary")
    summary = pd.DataFrame({"Parameter":["Category","Region","Initial Quantity","Daily Demand","Days Until Expiry","Storage Temperature","Base Price","Cost Price","Festival","Weather"],"Value":[st.session_state["category"],st.session_state["region"],st.session_state["initial_stock"],st.session_state["daily_demand"],st.session_state["days_remaining"],f"{st.session_state['storage_temp']} °C",f"₹{st.session_state['base_price']:.2f}",f"₹{st.session_state['cost_price']:.2f}",st.session_state["festival"],st.session_state["weather"]]})
    st.table(summary)

st.divider()
st.header("🤖 AI Food Waste Assistant")
st.write("Ask questions about demand, waste, inventory, expiry risk, discounts or purchasing.")

def answer_question(question):
    q=question.lower().strip()
    if "demand" in q or "sell" in q or "sold" in q:
        if st.session_state.get("prediction_done",False): return f"📈 Predicted demand is **{st.session_state['predicted_demand']} units**."
        return "Please click **Predict & Optimize** first so I can calculate demand."
    if "waste" in q or "wasted" in q or "spoil" in q:
        if st.session_state.get("prediction_done",False): return f"🗑️ Predicted food waste is **{st.session_state['units_wasted']} units** ({st.session_state['waste_percentage']:.2f}%)."
        return "Please click **Predict & Optimize** first."
    if "buy" in q or "purchase" in q or "inventory" in q or "stock" in q:
        if st.session_state.get("prediction_done",False):
            qty=st.session_state["recommended_quantity"]
            return f"🛒 You should buy approximately **{qty} additional units**." if qty>0 else "📦 You don't need additional stock. Maintain the current inventory."
        return "Please click **Predict & Optimize** first."
    if "expiry" in q or "expire" in q:
        if st.session_state.get("prediction_done",False): return f"⏳ Current expiry risk is **{st.session_state['expiry_risk']}**."
        return "Please click **Predict & Optimize** first."
    if "discount" in q or "offer" in q or "price reduction" in q:
        if st.session_state.get("prediction_done",False):
            d=st.session_state["discount"]
            return f"🏷️ Recommended discount is **{d}%** to reduce waste." if d>0 else "✅ No discount is currently required."
        return "Please click **Predict & Optimize** first."
    if "risk" in q or "danger" in q:
        if st.session_state.get("prediction_done",False): return f"🚦 Waste risk is **{st.session_state['waste_risk']}** and expiry risk is **{st.session_state['expiry_risk']}**."
        return "Please click **Predict & Optimize** first."
    if "hello" in q or "hi" in q or "hey" in q:
        return "👋 Hello! I am your AI Food Waste Assistant. You can ask me about demand, waste, inventory, expiry or discounts."
    if "help" in q or "what can you do" in q:
        return "🤖 I can answer questions like:\n\n- How much will I sell?\n- How much food will be wasted?\n- How much should I buy?\n- What is the expiry risk?\n- What discount should I give?\n- What is the waste risk?"
    return "🤔 I can answer questions about **demand, waste, inventory, expiry risk and discounts**. Try asking: *How much should I buy tomorrow?*"

question=st.chat_input("Example: How much should I buy tomorrow?")
if question:
    with st.chat_message("user"): st.write(question)
    with st.chat_message("assistant"): st.write(answer_question(question))

st.divider()
st.caption("🍱 AI Food Waste Prediction & Smart Inventory Optimization System")
st.caption("Developed using Python • Pandas • Scikit-learn • Random Forest • Streamlit • Plotly")
