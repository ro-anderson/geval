import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import aiohttp
import asyncio
import json
from datetime import datetime
import uuid

# Set page configuration
st.set_page_config(
    page_title="G-Eval Playground",
    page_icon="G",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply adaptive theme styling
st.markdown("""
<style>
    /* Remove forced backgrounds to respect user's theme choice */
    .stAlert > div {
        border-radius: 8px;
        border-left: 4px solid #ff6b6b;
    }
    .stSuccess > div {
        border-radius: 8px;
        border-left: 4px solid #4ecdc4;
    }
    .stInfo > div {
        border-radius: 8px;
        border-left: 4px solid #45b7d1;
    }
    .stWarning > div {
        border-radius: 8px;
        border-left: 4px solid #f39c12;
    }
    /* Ensure proper contrast for metrics */
    .metric-container {
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
    }
    /* Fix any potential text visibility issues */
    .stMarkdown, .stText {
        color: inherit;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Helper functions for API calls
async def fetch_api_data(endpoint, method="GET", data=None):
    """Generic API call function"""
    async with aiohttp.ClientSession() as session:
        try:
            if method == "GET":
                async with session.get(f"{API_BASE_URL}/{endpoint}") as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        st.error(f"Error fetching {endpoint}: {response.status}")
                        return None
            elif method == "POST":
                async with session.post(f"{API_BASE_URL}/{endpoint}", json=data) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        st.error(f"Error posting to {endpoint}: {response.status}")
                        return None
            elif method == "PUT":
                async with session.put(f"{API_BASE_URL}/{endpoint}", json=data) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        st.error(f"Error updating {endpoint}: {response.status}")
                        return None
        except Exception as e:
            st.error(f"Connection error: {str(e)}")
            return None

def run_async_function(async_func):
    """Run async functions synchronously"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(async_func)
    finally:
        loop.close()

# Main title
st.title("G-Eval Playground")
st.markdown("**Comprehensive evaluation system for LLM outputs**")

# Check API health
with st.spinner("Checking API connection..."):
    health_check = run_async_function(fetch_api_data(""))
    if health_check:
        st.success("API Connected Successfully!")
    else:
        st.error("Cannot connect to API. Please ensure the server is running on http://localhost:8000")
        st.stop()

# Create tabs
tabs = st.tabs([
    "Dashboard", 
    "Judges Management", 
    "Cases & Metrics", 
    "Documents", 
    "Evaluations", 
    "Models"
])

# Dashboard Tab
with tabs[0]:
    st.header("Evaluation Analytics Dashboard")
    
    # Fetch summary data
    with st.spinner("Loading dashboard data..."):
        judges_data = run_async_function(fetch_api_data("judges"))
        cases_data = run_async_function(fetch_api_data("cases"))
        metrics_data = run_async_function(fetch_api_data("metrics"))
        models_data = run_async_function(fetch_api_data("models"))
        runs_data = run_async_function(fetch_api_data("runs"))
        documents_data = run_async_function(fetch_api_data("documents"))
    
    # Filter Controls Section
    st.subheader("Filters")
    
    if runs_data and len(runs_data) > 0:
        df_all_runs = pd.DataFrame(runs_data)
        df_all_runs['started_at'] = pd.to_datetime(df_all_runs['started_at'])
        
        # Create filter columns
        col_f1, col_f2, col_f3, col_f4, col_f5, col_f6 = st.columns(6)
        
        with col_f1:
            # Judge filter
            judge_options = ["All Judges"] + sorted(list(df_all_runs['judge_name'].unique()))
            selected_judge = st.selectbox("Judge", judge_options, key="dash_judge_filter")
        
        with col_f2:
            # Case filter
            case_options = ["All Cases"] + sorted(list(df_all_runs['case_name'].unique()))
            selected_case = st.selectbox("Case", case_options, key="dash_case_filter")
        
        with col_f3:
            # Metric filter
            metric_options = ["All Metrics"] + sorted(list(df_all_runs['metric_name'].unique()))
            selected_metric = st.selectbox("Metric", metric_options, key="dash_metric_filter")
        
        with col_f4:
            # Model filter
            if 'model_name' in df_all_runs.columns and not df_all_runs['model_name'].isna().all():
                model_options = ["All Models"] + sorted([x for x in df_all_runs['model_name'].unique() if x is not None])
                selected_model = st.selectbox("Model", model_options, key="dash_model_filter")
            else:
                selected_model = "All Models"
                st.selectbox("Model", ["All Models"], key="dash_model_filter", disabled=True, help="Model data not available")
        
        with col_f5:
            # Status filter
            status_options = ["All Status"] + sorted(list(df_all_runs['status'].unique()))
            selected_status = st.selectbox("Status", status_options, key="dash_status_filter")
        
        with col_f6:
            # Evaluation Status filter
            if 'evaluation_status' in df_all_runs.columns:
                eval_status_options = ["All Results"] + sorted([x for x in df_all_runs['evaluation_status'].unique() if x is not None])
                selected_eval_status = st.selectbox("Result", eval_status_options, key="dash_eval_status_filter")
            else:
                selected_eval_status = "All Results"
                st.selectbox("Result", ["All Results"], key="dash_eval_status_filter", disabled=True, help="Evaluation status not available")
        
        # Apply filters
        df_runs = df_all_runs.copy()
        
        if selected_judge != "All Judges":
            df_runs = df_runs[df_runs['judge_name'] == selected_judge]
        if selected_case != "All Cases":
            df_runs = df_runs[df_runs['case_name'] == selected_case]
        if selected_metric != "All Metrics":
            df_runs = df_runs[df_runs['metric_name'] == selected_metric]
        if selected_model != "All Models" and 'model_name' in df_runs.columns:
            df_runs = df_runs[df_runs['model_name'] == selected_model]
        if selected_status != "All Status":
            df_runs = df_runs[df_runs['status'] == selected_status]
        if selected_eval_status != "All Results" and 'evaluation_status' in df_runs.columns:
            df_runs = df_runs[df_runs['evaluation_status'] == selected_eval_status]
        
        # Filter summary
        filter_active = any([
            selected_judge != "All Judges",
            selected_case != "All Cases", 
            selected_metric != "All Metrics",
            selected_model != "All Models" and 'model_name' in df_runs.columns,
            selected_status != "All Status",
            selected_eval_status != "All Results" and 'evaluation_status' in df_runs.columns
        ])
        
        if filter_active:
            st.info(f"Showing {len(df_runs)} runs (filtered from {len(df_all_runs)} total)")
        else:
            st.info(f"Showing all {len(df_runs)} runs")
        
        st.markdown("---")
        
        # Performance Metrics
        if len(df_runs) > 0:
            st.subheader("Performance Overview")
            
            # Key metrics
            col_m1, col_m2, col_m3, col_m4, col_m5, col_m6 = st.columns(6)
            
            completed_runs = df_runs[df_runs['status'] == 'completed']
            
            with col_m1:
                st.metric("Total Runs", len(df_runs))
            
            with col_m2:
                success_rate = len(completed_runs) / len(df_runs) * 100 if len(df_runs) > 0 else 0
                st.metric("Success Rate", f"{success_rate:.1f}%")
            
            with col_m3:
                if len(completed_runs) > 0:
                    avg_score = completed_runs['final_score'].mean()
                    st.metric("Avg Score", f"{avg_score:.3f}")
                else:
                    st.metric("Avg Score", "N/A")
            
            with col_m4:
                if len(completed_runs) > 0:
                    avg_time = completed_runs['execution_time_seconds'].mean()
                    st.metric("Avg Duration", f"{avg_time:.1f}s")
                else:
                    st.metric("Avg Duration", "N/A")
            
            with col_m5:
                if len(completed_runs) > 0:
                    total_tokens = completed_runs['total_usage_tokens'].sum()
                    st.metric("Total Tokens", f"{total_tokens:,}")
                else:
                    st.metric("Total Tokens", "N/A")
            
            with col_m6:
                unique_judges = df_runs['judge_name'].nunique()
                st.metric("Active Judges", unique_judges)
            
            # Charts Section
            st.subheader("Analytics")
            
            # Main performance chart
            hover_data_cols = ['case_name', 'total_usage_tokens']
            if 'model_name' in df_runs.columns:
                hover_data_cols.append('model_name')
            
            fig_main = px.scatter(df_runs,
                                x='started_at',
                                y='final_score',
                                color='judge_name',
                                size='execution_time_seconds',
                                symbol='status',
                                hover_data=hover_data_cols,
                                title="Evaluation Performance Over Time",
                                height=500)
            
            fig_main.update_layout(
                xaxis_title="Time",
                yaxis_title="Final Score",
                margin=dict(l=0, r=0, t=40, b=0)
            )
            
            st.plotly_chart(fig_main, width='stretch')
            
            # Analysis charts in columns
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                # Score distribution by judge
                if df_runs['judge_name'].nunique() > 1:
                    fig_box = px.box(df_runs,
                                   x='judge_name',
                                   y='final_score',
                                   color='judge_name',
                                   title="Score Distribution by Judge")
                    fig_box.update_layout(
                        showlegend=False,
                        xaxis_title="Judge",
                        yaxis_title="Score",
                        margin=dict(l=0, r=0, t=40, b=0)
                    )
                    fig_box.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_box, use_container_width=True)
                else:
                    # Single judge - show score histogram
                    fig_hist = px.histogram(df_runs,
                                          x='final_score',
                                          title="Score Distribution",
                                          nbins=20,
                                          color_discrete_sequence=['#1f77b4'])
                    fig_hist.update_layout(
                        showlegend=False,
                        xaxis_title="Score",
                        yaxis_title="Count",
                        margin=dict(l=0, r=0, t=40, b=0)
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
            
            with col_chart2:
                # Performance vs execution time
                perf_hover_data = ['judge_name', 'status']
                fig_perf = px.scatter(df_runs,
                                    x='execution_time_seconds',
                                    y='final_score',
                                    color='case_name',
                                    size='total_usage_tokens',
                                    hover_data=perf_hover_data,
                                    title="Score vs Execution Time")
                fig_perf.update_layout(
                    xaxis_title="Execution Time (seconds)",
                    yaxis_title="Score",
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                st.plotly_chart(fig_perf, use_container_width=True)
            
            # Additional insights
            col_insight1, col_insight2 = st.columns(2)
            
            with col_insight1:
                # Case performance comparison
                if df_runs['case_name'].nunique() > 1:
                    case_stats = df_runs.groupby('case_name').agg({
                        'final_score': 'mean',
                        'execution_time_seconds': 'mean',
                        'id': 'count'
                    }).round(3)
                    case_stats.columns = ['Avg Score', 'Avg Time (s)', 'Run Count']
                    st.subheader("Performance by Case")
                    st.dataframe(case_stats, width='stretch')
            
            with col_insight2:
                # Judge performance comparison
                if df_runs['judge_name'].nunique() > 1:
                    judge_stats = df_runs.groupby('judge_name').agg({
                        'final_score': ['mean', 'std'],
                        'execution_time_seconds': 'mean',
                        'id': 'count'
                    }).round(3)
                    judge_stats.columns = ['Avg Score', 'Score Std', 'Avg Time (s)', 'Run Count']
                    st.subheader("Performance by Judge")
                    st.dataframe(judge_stats, width='stretch')
            
            # Recent activity table
            st.subheader("Recent Runs")
            recent_runs = df_runs.sort_values('started_at', ascending=False).head(10)
            st.dataframe(
                recent_runs[['started_at', 'judge_name', 'case_name', 'final_score', 'status', 'evaluation_status', 'execution_time_seconds']],
                column_config={
                    "started_at": "Started At",
                    "judge_name": "Judge",
                    "case_name": "Case",
                    "final_score": "Score",
                    "status": "Status",
                    "evaluation_status": "Result",
                    "execution_time_seconds": "Duration (s)"
                },
                width='stretch'
            )
        
        else:
            st.warning("No runs match the selected filters.")
    
    else:
        # No runs available - show system overview
        st.subheader("System Status")
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            judge_count = len(judges_data) if judges_data else 0
            st.metric("Judges", judge_count)
        
        with col2:
            case_count = len(cases_data) if cases_data else 0
            st.metric("Cases", case_count)
        
        with col3:
            metric_count = len(metrics_data) if metrics_data else 0
            st.metric("Metrics", metric_count)
        
        with col4:
            model_count = len(models_data) if models_data else 0
            st.metric("Models", model_count)
        
        with col5:
            run_count = 0
            st.metric("Runs", run_count)
        
        with col6:
            doc_count = len(documents_data) if documents_data else 0
            st.metric("Documents", doc_count)
        
        st.info("No evaluation runs available yet. Create judges and run evaluations to see analytics.")

# Judges Management Tab
with tabs[1]:
    st.header("Judges Management")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Create New Judge")
        
        with st.form("create_judge_form"):
            # Fetch required data for dropdowns
            cases = run_async_function(fetch_api_data("cases")) or []
            metrics = run_async_function(fetch_api_data("metrics")) or []
            models = run_async_function(fetch_api_data("models")) or []
            
            judge_name = st.text_input("Judge Name", placeholder="e.g., Consistency Expert Judge")
            judge_description = st.text_area("Description", placeholder="Describe this judge's specialization...")
            
            if cases:
                case_options = {case['name']: case['id'] for case in cases}
                selected_case = st.selectbox("Case", options=list(case_options.keys()))
                case_id = case_options[selected_case] if selected_case else None
            else:
                st.warning("No cases available. Create a case first.")
                case_id = None
            
            if metrics:
                metric_options = {metric['name']: metric['id'] for metric in metrics}
                selected_metric = st.selectbox("Metric", options=list(metric_options.keys()))
                metric_id = metric_options[selected_metric] if selected_metric else None
            else:
                st.warning("No metrics available. Create a metric first.")
                metric_id = None
            
            if models:
                model_options = {f"{model['name']} ({model['provider']})": model['id'] for model in models}
                selected_model = st.selectbox("Model", options=list(model_options.keys()))
                model_id = model_options[selected_model] if selected_model else None
            else:
                st.warning("No models available. Create a model first.")
                model_id = None
            
            # Parameters section
            st.subheader("Evaluation Parameters")
            col_param1, col_param2 = st.columns(2)
            
            with col_param1:
                temperature = st.slider("Temperature", 0.0, 2.0, 2.0, 0.1)
                max_tokens = st.number_input("Max Tokens", 100, 5000, 2500)
                top_p = st.slider("Top P", 0.0, 1.0, 1.0, 0.1)
                n_responses = st.number_input("N Responses", 1, 50, 10)
            
            with col_param2:
                frequency_penalty = st.slider("Frequency Penalty", 0.0, 2.0, 0.0, 0.1)
                presence_penalty = st.slider("Presence Penalty", 0.0, 2.0, 0.0, 0.1)
                sleep_time = st.number_input("Sleep Time", 0.0, 5.0, 0.0, 0.1)
                rate_limit_sleep = st.number_input("Rate Limit Sleep", 0.0, 10.0, 0.0, 0.1)
            
            submit_judge = st.form_submit_button("Create Judge", type="primary")
            
            if submit_judge and judge_name and case_id and metric_id and model_id:
                parameters = {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p,
                    "frequency_penalty": frequency_penalty,
                    "presence_penalty": presence_penalty,
                    "n_responses": n_responses,
                    "sleep_time": sleep_time,
                    "rate_limit_sleep": rate_limit_sleep
                }
                
                judge_data = {
                    "name": judge_name,
                    "model_id": model_id,
                    "case_id": case_id,
                    "metric_id": metric_id,
                    "parameters": parameters,
                    "description": judge_description
                }
                
                with st.spinner("Creating judge..."):
                    result = run_async_function(fetch_api_data("judges", "POST", judge_data))
                    if result:
                        st.success(f"Judge '{judge_name}' created successfully!")
                        st.rerun()
    
    with col2:
        st.subheader("Existing Judges")
        
        judges = run_async_function(fetch_api_data("judges"))
        if judges:
            for judge in judges:
                with st.expander(f"{judge['name']}"):
                    col_info1, col_info2 = st.columns(2)
                    
                    with col_info1:
                        st.write(f"**Model:** {judge['model_name']}")
                        st.write(f"**Case:** {judge['case_name']}")
                        st.write(f"**Metric:** {judge['metric_name']}")
                    
                    with col_info2:
                        st.write(f"**Created:** {judge['created_at']}")
                        if judge.get('description'):
                            st.write(f"**Description:** {judge['description']}")
                    
                    # Parameters
                    if judge.get('parameters'):
                        st.write("**Parameters:**")
                        params_df = pd.DataFrame(list(judge['parameters'].items()), columns=['Parameter', 'Value'])
                        st.dataframe(params_df, width='stretch')
                    
                    # Update button
                    if st.button(f"Update Judge", key=f"update_{judge['id']}"):
                        st.session_state[f"show_update_{judge['id']}"] = True
                    
                    # Update form
                    if st.session_state.get(f"show_update_{judge['id']}", False):
                        with st.form(f"update_form_{judge['id']}"):
                            st.write("**Update Judge Configuration:**")
                            
                            # Basic information
                            col_basic1, col_basic2 = st.columns(2)
                            with col_basic1:
                                new_name = st.text_input("Name", value=judge['name'])
                            with col_basic2:
                                # Model selector
                                available_models = run_async_function(fetch_api_data("models")) or []
                                if available_models:
                                    model_options = {f"{model['name']} ({model['provider']})": model['id'] for model in available_models}
                                    current_model_display = f"{judge['model_name']} ({judge['model_provider']})"
                                    
                                    # Find current model index
                                    model_names = list(model_options.keys())
                                    current_index = 0
                                    if current_model_display in model_names:
                                        current_index = model_names.index(current_model_display)
                                    
                                    selected_model_display = st.selectbox(
                                        "Model", 
                                        options=model_names,
                                        index=current_index,
                                        key=f"model_{judge['id']}"
                                    )
                                    new_model_id = model_options[selected_model_display]
                                else:
                                    st.warning("No models available")
                                    new_model_id = None
                            
                            new_description = st.text_area("Description", value=judge.get('description', ''))
                            
                            st.write("**Evaluation Parameters:**")
                            # Current parameters
                            current_params = judge.get('parameters', {})
                            
                            col_param1, col_param2 = st.columns(2)
                            with col_param1:
                                new_temperature = st.slider("Temperature", 0.0, 2.0, current_params.get('temperature', 2.0), 0.1, key=f"temp_{judge['id']}")
                                new_max_tokens = st.number_input("Max Tokens", 100, 5000, current_params.get('max_tokens', 2500), key=f"tokens_{judge['id']}")
                                new_top_p = st.slider("Top P", 0.0, 1.0, current_params.get('top_p', 1.0), 0.1, key=f"top_p_{judge['id']}")
                            
                            with col_param2:
                                new_n_responses = st.number_input("N Responses", 1, 50, current_params.get('n_responses', 10), key=f"responses_{judge['id']}")
                                new_frequency_penalty = st.slider("Frequency Penalty", 0.0, 2.0, current_params.get('frequency_penalty', 0.0), 0.1, key=f"freq_{judge['id']}")
                                new_presence_penalty = st.slider("Presence Penalty", 0.0, 2.0, current_params.get('presence_penalty', 0.0), 0.1, key=f"pres_{judge['id']}")
                            
                            col_time1, col_time2 = st.columns(2)
                            with col_time1:
                                new_sleep_time = st.number_input("Sleep Time", 0.0, 5.0, current_params.get('sleep_time', 0.0), 0.1, key=f"sleep_{judge['id']}")
                            with col_time2:
                                new_rate_limit_sleep = st.number_input("Rate Limit Sleep", 0.0, 10.0, current_params.get('rate_limit_sleep', 0.0), 0.1, key=f"rate_{judge['id']}")
                            
                            col_btn1, col_btn2 = st.columns([1, 1])
                            with col_btn1:
                                update_submit = st.form_submit_button("Update Judge", type="primary")
                            with col_btn2:
                                cancel_update = st.form_submit_button("Cancel")
                            
                            if update_submit:
                                new_parameters = current_params.copy()
                                new_parameters.update({
                                    "temperature": new_temperature,
                                    "max_tokens": new_max_tokens,
                                    "top_p": new_top_p,
                                    "frequency_penalty": new_frequency_penalty,
                                    "presence_penalty": new_presence_penalty,
                                    "n_responses": new_n_responses,
                                    "sleep_time": new_sleep_time,
                                    "rate_limit_sleep": new_rate_limit_sleep
                                })
                                
                                update_data = {
                                    "name": new_name,
                                    "parameters": new_parameters,
                                    "description": new_description
                                }
                                
                                # Only include model_id if it changed
                                if new_model_id and new_model_id != judge.get('model_id'):
                                    update_data["model_id"] = new_model_id
                                
                                with st.spinner("Updating judge..."):
                                    result = run_async_function(fetch_api_data(f"judges/{judge['id']}", "PUT", update_data))
                                    if result:
                                        st.success("Judge updated successfully!")
                                        st.session_state[f"show_update_{judge['id']}"] = False
                                        st.rerun()
                            
                            if cancel_update:
                                st.session_state[f"show_update_{judge['id']}"] = False
                                st.rerun()

# Cases & Metrics Tab
with tabs[2]:
    st.header("Cases & Metrics Management")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Metrics")
        
        # Create metric form
        with st.expander("Create New Metric"):
            with st.form("create_metric_form"):
                metric_name = st.text_input("Metric Name", placeholder="e.g., geval, claude_eval")
                
                if st.form_submit_button("Create Metric"):
                    if metric_name:
                        metric_data = {"name": metric_name}
                        with st.spinner("Creating metric..."):
                            result = run_async_function(fetch_api_data("metrics", "POST", metric_data))
                            if result:
                                st.success(f"Metric '{metric_name}' created!")
                                st.rerun()
        
        # List existing metrics
        metrics = run_async_function(fetch_api_data("metrics"))
        if metrics:
            st.write("**Existing Metrics:**")
            for metric in metrics:
                st.write(f"• {metric['name']} (ID: {metric['id'][:8]}...)")
    
    with col2:
        st.subheader("Cases")
        
        # Create case form
        with st.expander("Create New Case"):
            with st.form("create_case_form"):
                case_name = st.text_input("Case Name", placeholder="e.g., consistency, coherence")
                task_intro = st.text_area("Task Introduction", placeholder="Describe the evaluation task...")
                eval_criteria = st.text_area("Evaluation Criteria", placeholder="Specific criteria for evaluation...")
                
                col_score1, col_score2, col_target, col_threshold = st.columns(4)
                with col_score1:
                    min_score = st.number_input("Min Score", 1, 10, 1)
                with col_score2:
                    max_score = st.number_input("Max Score", 1, 10, 5)
                with col_target:
                    requires_reference = st.toggle(
                        "Target Required", 
                        value=False,
                        help="Enable if this case requires comparing actual output with expected output (reference)"
                    )
                with col_threshold:
                    score_threshold = st.slider(
                        "Pass Threshold",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.5,
                        step=0.05,
                        help="Minimum normalized score (0.0-1.0) required to pass evaluation"
                    )
                
                # Add explanation for the target field
                if requires_reference:
                    st.info("**Target Mode**: This case will compare actual output against expected output (reference-based evaluation)")
                else:
                    st.info("**Direct Mode**: This case will evaluate actual output independently (no reference needed)")
                
                if st.form_submit_button("Create Case"):
                    if case_name and task_intro and eval_criteria:
                        case_data = {
                            "name": case_name,
                            "task_introduction": task_intro,
                            "evaluation_criteria": eval_criteria,
                            "min_score": min_score,
                            "max_score": max_score,
                            "requires_reference": requires_reference,
                            "score_threshold": score_threshold
                        }
                        with st.spinner("Creating case..."):
                            result = run_async_function(fetch_api_data("cases", "POST", case_data))
                            if result:
                                st.success(f"Case '{case_name}' created!")
                                st.rerun()
        
        # List existing cases
        cases = run_async_function(fetch_api_data("cases"))
        if cases:
            st.write("**Existing Cases:**")
            for case in cases:
                # Add target indicator to case name
                target_indicator = "Target" if case.get('requires_reference', False) else "Direct"
                with st.expander(f"{case['name']} ({target_indicator})"):
                    st.write(f"**Task:** {case['task_introduction']}")
                    st.write(f"**Criteria:** {case['evaluation_criteria']}")
                    st.write(f"**Score Range:** {case['min_score']} - {case['max_score']}")
                    st.write(f"**Pass Threshold:** {case.get('score_threshold', 0.5):.2f} ({case.get('score_threshold', 0.5)*100:.0f}%)")
                    
                    # Show target mode explanation
                    if case.get('requires_reference', False):
                        st.info("**Target Mode**: Compares actual vs expected output")
                    else:
                        st.info("**Direct Mode**: Evaluates actual output independently")

# Documents Tab
with tabs[3]:
    st.header("Documents Management")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Create New Document")
        
        with st.form("create_document_form"):
            actual_output = st.text_area("Actual Output", placeholder="The text to be evaluated...", height=150)
            expected_output = st.text_area("Expected Output (Optional)", placeholder="Reference text for comparison...", height=150)
            
            if st.form_submit_button("Create Document"):
                if actual_output:
                    doc_data = {
                        "actual_output": actual_output,
                        "expected_output": expected_output if expected_output else None
                    }
                    with st.spinner("Creating document..."):
                        result = run_async_function(fetch_api_data("documents", "POST", doc_data))
                        if result:
                            st.success("Document created successfully!")
                            st.rerun()
    
    with col2:
        st.subheader("Existing Documents")
        
        documents = run_async_function(fetch_api_data("documents"))
        if documents:
            for i, doc in enumerate(documents):
                with st.expander(f"Document {i+1} (ID: {doc['id'][:8]}...)"):
                    st.write("**Actual Output:**")
                    st.write(doc['actual_output'][:200] + "..." if len(doc['actual_output']) > 200 else doc['actual_output'])
                    
                    if doc.get('expected_output'):
                        st.write("**Expected Output:**")
                        st.write(doc['expected_output'][:200] + "..." if len(doc['expected_output']) > 200 else doc['expected_output'])
                    
                    st.write(f"**Created:** {doc['created_at']}")

# Evaluations Tab
with tabs[4]:
    st.header("Evaluations")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Run New Evaluation")
        
        with st.form("run_evaluation_form"):
            # Fetch data for dropdowns
            judges = run_async_function(fetch_api_data("judges")) or []
            cases = run_async_function(fetch_api_data("cases")) or []
            documents = run_async_function(fetch_api_data("documents")) or []
            
            # Step 1: Select Case
            if cases:
                case_options = {case['name']: case['id'] for case in cases}
                selected_case = st.selectbox("Select Case", options=list(case_options.keys()))
                selected_case_id = case_options[selected_case] if selected_case else None
            else:
                st.warning("No cases available. Create a case first.")
                selected_case_id = None
            
            # Step 2: Select Judge (filtered by case)
            if judges and selected_case_id:
                # Filter judges by selected case
                filtered_judges = [judge for judge in judges if judge['case_id'] == selected_case_id]
                
                if filtered_judges:
                    judge_options = {f"{judge['name']} (Model: {judge['model_name']})": judge['id'] for judge in filtered_judges}
                    selected_judge = st.selectbox("Select Judge", options=list(judge_options.keys()))
                    judge_id = judge_options[selected_judge] if selected_judge else None
                else:
                    st.warning(f"No judges available for case '{selected_case}'. Create a judge for this case first.")
                    judge_id = None
            else:
                st.warning("Select a case first to see available judges.")
                judge_id = None
            
            # Step 3: Select Document
            if documents:
                doc_options = {f"Document {i+1} ({doc['actual_output'][:50]}...)": doc['id'] for i, doc in enumerate(documents)}
                selected_doc = st.selectbox("Select Document", options=list(doc_options.keys()))
                document_id = doc_options[selected_doc] if selected_doc else None
            else:
                st.warning("No documents available. Create a document first.")
                document_id = None
            
            if st.form_submit_button("Run Evaluation", type="primary"):
                if judge_id and document_id:
                    eval_data = {
                        "judge_id": judge_id,
                        "document_id": document_id
                    }
                    
                    with st.spinner("Running evaluation... This may take a while."):
                        result = run_async_function(fetch_api_data("eval", "POST", eval_data))
                        if result:
                            st.success("Evaluation completed!")
                            st.json(result)
                            st.rerun()
    
    with col2:
        st.subheader("Evaluation Results")
        
        # Add judge filter for results
        all_judges = run_async_function(fetch_api_data("judges")) or []
        all_runs = run_async_function(fetch_api_data("runs"))
        
        if all_judges and all_runs:
            # Judge selector for filtering results
            judge_filter_options = ["All Judges"] + [f"{judge['name']} ({judge['case_name']})" for judge in all_judges]
            selected_judge_filter = st.selectbox("Filter by Judge", options=judge_filter_options, key="judge_results_filter")
            
            # Filter runs based on selected judge
            if selected_judge_filter == "All Judges":
                runs = all_runs
                chart_title_suffix = "(All Judges)"
            else:
                # Find the selected judge ID
                selected_judge_name = selected_judge_filter.split(" (")[0]
                selected_judge_obj = next((j for j in all_judges if j['name'] == selected_judge_name), None)
                
                if selected_judge_obj:
                    runs = [run for run in all_runs if run['judge_id'] == selected_judge_obj['id']]
                    chart_title_suffix = f"({selected_judge_obj['name']})"
                else:
                    runs = all_runs
                    chart_title_suffix = "(All Judges)"
            
            if runs:
                # Create a summary chart
                df_runs = pd.DataFrame(runs)
                
                if len(df_runs) > 0:
                    # Score distribution charts
                    col_chart1, col_chart2 = st.columns(2)
                    
                    with col_chart1:
                        # Histogram with fixed color
                        fig_hist = px.histogram(df_runs, x='final_score', 
                                              title=f"Score Distribution {chart_title_suffix}",
                                              nbins=15,
                                              color_discrete_sequence=['#1f77b4'])  # Fixed blue color
                        fig_hist.update_layout(
                            showlegend=False,
                            xaxis_title="Final Score",
                            yaxis_title="Count",
                            margin=dict(l=0, r=0, t=40, b=0)
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                    with col_chart2:
                        # Box plot for better statistical overview
                        fig_box = px.box(df_runs, y='final_score', x='status',
                                       title=f"Score by Status {chart_title_suffix}",
                                       color='status',
                                       color_discrete_sequence=['#2ca02c', '#ff7f0e', '#d62728', '#9467bd'])
                        fig_box.update_layout(
                            showlegend=False,
                            xaxis_title="Status",
                            yaxis_title="Final Score",
                            margin=dict(l=0, r=0, t=40, b=0)
                        )
                        st.plotly_chart(fig_box, use_container_width=True)
                    
                    # Score statistics
                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                    with col_stat1:
                        st.metric("Average Score", f"{df_runs['final_score'].mean():.3f}")
                    with col_stat2:
                        st.metric("Median Score", f"{df_runs['final_score'].median():.3f}")
                    with col_stat3:
                        st.metric("Min Score", f"{df_runs['final_score'].min():.3f}")
                    with col_stat4:
                        st.metric("Max Score", f"{df_runs['final_score'].max():.3f}")
                    
                    # Latest runs table
                    st.subheader("Recent Runs")
                    latest_runs = df_runs.sort_values('started_at', ascending=False).head(5)
                    st.dataframe(
                        latest_runs[['started_at', 'judge_name', 'case_name', 'final_score', 'status', 'evaluation_status']],
                        column_config={
                            "started_at": "Started At",
                            "judge_name": "Judge",
                            "case_name": "Case",
                            "final_score": "Score",
                            "status": "Status",
                            "evaluation_status": "Result"
                        },
                        width='stretch'
                    )
                else:
                    st.info(f"No evaluation runs found {chart_title_suffix.lower()}.")
            else:
                st.info("No evaluation runs available. Run an evaluation first.")
        else:
            st.warning("No judges or runs available. Create judges and run evaluations first.")

# Models Tab
with tabs[5]:
    st.header("Models Management")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Create New Model")
        
        with st.form("create_model_form"):
            model_name = st.text_input("Model Name", placeholder="e.g., gpt-4o-2024-08-06")
            model_provider = st.selectbox("Provider", ["openai", "anthropic", "google", "other"])
            
            if st.form_submit_button("Create Model"):
                if model_name and model_provider:
                    model_data = {
                        "name": model_name,
                        "provider": model_provider
                    }
                    with st.spinner("Creating model..."):
                        result = run_async_function(fetch_api_data("models", "POST", model_data))
                        if result:
                            st.success(f"Model '{model_name}' created!")
                            st.rerun()
    
    with col2:
        st.subheader("Existing Models")
        
        models = run_async_function(fetch_api_data("models"))
        if models:
            for model in models:
                with st.expander(f"{model['name']}"):
                    st.write(f"**Provider:** {model['provider']}")
                    st.write(f"**Created:** {model['created_at']}")
                    st.write(f"**ID:** {model['id']}")

# Footer
st.markdown("---")
st.caption("G-Eval Playground v1.0 - Comprehensive LLM Evaluation System")
st.caption("Powered by FastAPI • Built with Streamlit")
