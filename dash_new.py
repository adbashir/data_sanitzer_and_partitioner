import streamlit as st
import boto3
import time as t
import json
import re
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from PIL import Image
from datetime import date

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
athena = boto3.client("athena", region_name="us-east-1")
s3 = boto3.client("s3", region_name="us-east-1")

ATHENA_DB = "newdata"
ATHENA_OUTPUT = "s3://aws-athena-query-results-us-east-1-882540910495/"

HEARTBEATS_COLUMNS = [
    "time", "temperature", "co", "hcho", "etoh", "humidity", "tvoc", "co2",
    "no2", "pm_1", "pm_25", "pm_10", "person_detection", "loitering", "aqi",
    "ehi", "noise_ratio", "crowd_count", "device_id", "date"
]

NUMERIC_COLUMNS = [
    "temperature", "co", "hcho", "etoh", "humidity", "tvoc", "co2",
    "no2", "pm_1", "pm_25", "pm_10", "person_detection", "loitering",
    "aqi", "ehi", "noise_ratio", "crowd_count"
]

def get_sql_and_natural_response(question):
    prompt = (
        "You are an expert AWS Athena SQL assistant. "
        "The database is 'newdata' with tables:\n"
        "- heartbeats: time, temperature, co, hcho, etoh, humidity, tvoc, co2, no2, pm_1, pm_25, pm_10, person_detection, loitering, aqi, ehi, noise_ratio, crowd_count, device_id, date\n\n"
        "Always pick the most relevant table and column(s). "
        "ALWAYS use correct column names and this date format: 'YYYY-MM-DD'. "
        "DO NOT use the database name as a table. "
        "Respond ONLY in this JSON format: {\"sql\": \"...\", \"answer\": \"...\"}\n"
        "EXAMPLES:\n"
        "Q: Return temperature for first day of August 2025\n"
        "A: {\"sql\": \"SELECT temperature FROM heartbeats WHERE date = '2025-08-01';\", \"answer\": \"Temperature on 2025-08-01 from heartbeats table.\"}\n"
        "Q: List alert types for July 2025\n"
        "A: {\"sql\": \"SELECT alert_type, time FROM alerts WHERE date LIKE '2025-07%';\", \"answer\": \"All alert types for July 2025 from alerts table.\"}\n"
        "Q: <insert user question below>\n"
        "A: "
        + question
    )

    body = json.dumps({
        "prompt": prompt,
        "max_tokens": 512,
        "temperature": 0.2
    })

    response = bedrock.invoke_model(
        modelId="cohere.command-light-text-v14",
        contentType="application/json",
        accept="application/json",
        body=body
    )
    response_text = response['body'].read().decode()

    token_info = None
    try:
        resp_json = json.loads(response_text)
        if 'usage' in resp_json:
            token_info = resp_json['usage']
        elif 'token_usage' in resp_json:
            token_info = resp_json['token_usage']
        elif 'total_tokens' in resp_json:
            token_info = {'total_tokens': resp_json['total_tokens']}
    except Exception:
        pass

    try:
        text = resp_json['generations'][0]['text']
        matches = re.findall(r'\{.*?\}', text, re.DOTALL)
        sql = None
        answer = None
        for m in matches:
            try:
                j = json.loads(m)
                if 'sql' in j:
                    sql = j['sql']
                if 'answer' in j:
                    answer = j['answer']
            except Exception:
                continue
        if not sql or not answer:
            raise Exception("Could not extract sql/answer")
    except Exception as e:
        st.error(f"Could not parse model output. Raw output: {response_text}")
        return None, None, None, response_text

    return sql, answer, token_info, response_text

def run_athena_query(query):
    exec_id = athena.start_query_execution(
        QueryString=query,
        QueryExecutionContext={'Database': ATHENA_DB},
        ResultConfiguration={'OutputLocation': ATHENA_OUTPUT}
    )['QueryExecutionId']

    while True:
        status = athena.get_query_execution(QueryExecutionId=exec_id)
        state = status['QueryExecution']['Status']['State']
        if state in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
            break
        t.sleep(2)

    if state != 'SUCCEEDED':
        st.error(f"Query failed with state: {state}")
        return None

    result = athena.get_query_results(QueryExecutionId=exec_id)
    rows = result['ResultSet']['Rows']
    header = [col['VarCharValue'] for col in rows[0]['Data']]
    data = [[col.get('VarCharValue', '') for col in row['Data']] for row in rows[1:]]
    return header, data

def plot_subplots(df, numeric_cols):
    num_plots = len(numeric_cols)
    fig = make_subplots(rows=num_plots, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05,
                        subplot_titles=[col.capitalize() for col in numeric_cols])

    for i, col in enumerate(numeric_cols, start=1):
        fig.add_trace(
            go.Scatter(
                x=df['time'] if 'time' in df.columns else list(range(len(df))),
                y=df[col],
                mode='lines+markers',
                marker=dict(size=4),
                name=col
            ),
            row=i, col=1
        )
        fig.update_yaxes(title_text=col.capitalize(), row=i, col=1)

    fig.update_layout(height=250*num_plots, width=800,
                      title_text="Selected Variables over Time",
                      showlegend=False,
                      hovermode="x unified",
                      margin=dict(t=50))

    st.plotly_chart(fig, use_container_width=True)

# --- Streamlit app start ---

logo = Image.open("image.png")
st.image(logo, use_container_width=False, width=200, output_format='PNG')

st.title("Athena Dashboard")

option = st.radio("Choose query mode:", ["LLM", "NON LLM"])

if option == "LLM":
    user_q = st.text_input("Ask a question about your data:")
    if user_q:
        with st.spinner("Calling Bedrock LLM..."):
            sql, answer, token_info, response_text = get_sql_and_natural_response(user_q)
        if sql and answer:
            st.code(sql, language="sql")
            st.write("LLM Answer:", answer)
            if token_info:
                st.info(f"Token usage info: {token_info}")
            st.write("Querying Athena...")
            result = run_athena_query(sql)
            if result:
                header, data = result
                df = pd.DataFrame(data, columns=header)
                num_rows = len(df)
                st.write(f"**Original prompt:** {user_q}")

                for col in NUMERIC_COLUMNS:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                st.markdown("**Result:**")
                if num_rows > 0:
                    st.markdown("**Row Details:**")
                    for col in df.columns:
                        st.markdown(f"- **{col}**: {df.iloc[0][col]}")

                    if num_rows > 1:
                        if 'time' in df.columns:
                            df['time'] = pd.to_datetime(df['time'], errors='coerce')

                        if 'temperature' in df.columns:
                            plot_subplots(df, ['temperature'])
                        else:
                            st.write("No 'temperature' column available for plotting.")
                    else:
                        st.write("Line chart skipped because only one data row is returned.")

                    st.table(df.head(10))
                else:
                    st.write("No data returned for the query.")

            st.markdown("### Raw Bedrock Response Text (for debugging)")
            st.text(response_text)

elif option == "NON LLM":
    device_ids = [
        "02AAAAAAA003","02AAAAAAA005","02AAAAAAA007","02AAAAAAA009",
        "02AAAAAAA011","02AAAAAAA013","02AAAAAAA015","02AAAAAAA017",
        "02AAAAAAA019","02BBBBBBB003"
    ]

    st.subheader("Select date range, device ID and columns")

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", value=date(2025, 7, 1), min_value=date(2025, 6, 1), max_value=date(2025, 7, 31))
    with col2:
        end_date = st.date_input("End date", value=date(2025, 7, 20), min_value=date(2025, 6, 1), max_value=date(2025, 7, 31))

    device_id = st.selectbox("Select device ID", device_ids)

    default_cols = [col for col in HEARTBEATS_COLUMNS if col not in ['device_id', 'date']]
    selected_cols = st.multiselect(
        "Select columns to query and display (time will always be included)",
        options=default_cols,
        default=["time", "temperature"]
    )

    if "time" not in selected_cols:
        selected_cols.insert(0, "time")

    if start_date > end_date:
        st.error("Start date must be before or equal to End date.")
    else:
        if st.button("Run Query"):
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")

            cols_str = ", ".join(selected_cols)
            sql = f"""
            SELECT {cols_str}, device_id, date FROM heartbeats
            WHERE device_id = '{device_id}'
            AND date BETWEEN '{start_str}' AND '{end_str}'
            ORDER BY date, time
            """

            st.code(sql, language="sql")
            st.write(f"Querying Athena for device {device_id} from {start_str} to {end_str}...")

            result = run_athena_query(sql)
            if result:
                header, data = result
                df = pd.DataFrame(data, columns=header)
                num_rows = len(df)

                st.markdown("**Query Result:**")
                if num_rows > 0:
                    for col in selected_cols:
                        if col in NUMERIC_COLUMNS and col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')

                    if 'time' in df.columns:
                        df['time'] = pd.to_datetime(df['time'], errors='coerce')

                    st.markdown("**Row Details:**")
                    for col in df.columns:
                        st.markdown(f"- **{col}**: {df.iloc[0][col]}")

                    numeric_selected = [c for c in selected_cols if c in NUMERIC_COLUMNS][:5]

                    if num_rows > 1 and numeric_selected:
                        plot_subplots(df, numeric_selected)
                    else:
                        st.write("Line chart skipped because only one data row is returned or no numeric columns selected.")

                    st.table(df[selected_cols].head(10))
                else:
                    st.write("No data found for the selected parameters.")
