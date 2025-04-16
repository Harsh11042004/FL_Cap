import streamlit as st

import streamlit as st
import matplotlib.pyplot as plt

# Import your FL functions
# from capjt_1 import run_hybrid_fl
# from capjt_2_RCS import run_random_client_selection
# from capjt_3_PCS import run_priority_client_selection

st.title("Federated Learning Strategy Comparison")

# Sidebar for user input
st.sidebar.header("Choose FL Strategy")
strategy = st.sidebar.selectbox(
    "Select a method:",
    ("Hybrid HFL+PFL (capjt_1)", "Random Client Selection (capjt_2_RCS)", "Priority Client Selection (capjt_3_PCS)")
)

num_clients = st.sidebar.slider("Number of Clients", 2, 10, 2)
global_rounds = st.sidebar.slider("Global Rounds", 1, 20, 9)

if st.sidebar.button("Run FL Simulation"):
    st.write(f"### Running: {strategy}")
    with st.spinner("Training in progress..."):
        if strategy == "Hybrid HFL+PFL (capjt_1)":
            results = run_hybrid_fl(num_clients, global_rounds)
        elif strategy == "Random Client Selection (capjt_2_RCS)":
            results = run_random_client_selection(num_clients, global_rounds)
        else:
            results = run_priority_client_selection(num_clients, global_rounds)

    # Example: results = (performance_metrics, accuracies, losses)
    performance_metrics, accuracies, losses = results

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(accuracies, marker='o')
    axs[0].set_title("Accuracy per Round")
    axs[1].plot(losses, marker='s', color='r')
    axs[1].set_title("Loss per Round")
    st.pyplot(fig)

    st.write("#### Performance Metrics")
    st.dataframe(performance_metrics)

    st.success("Simulation complete!")
else:
    st.info("Set parameters and click 'Run FL Simulation'.")

st.markdown("""
---
**Legend:**
- **Hybrid HFL+PFL:** Your proposed hybrid federated learning model.
- **Random Client Selection:** Standard FL with clients chosen randomly each round.
- **Priority Client Selection:** FL with clients selected based on a priority metric.
""")

