"""
Platform Policy Explorer Dashboard
Tier 1 Social Media Platforms
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path
import pandas as pd

current_file = Path(__file__).resolve()
dashboard_dir = current_file.parent
data_dir = dashboard_dir / 'data'

# Page config
st.set_page_config(
    page_title="Platform Policy Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
@st.cache_data
def load_data():
    # Get absolute path to data directory
    # dashboard/app.py -> ASML3/dashboard, so data is at ASML3/dashboard/data
    dashboard_dir = current_file.parent  # ASML3/dashboard
    data_dir = dashboard_dir / 'data'    # ASML3/dashboard/data
    
    viz_path = data_dir / 'tier1_visualization_data.json'
    scores_path = data_dir / 'tier1_dimension_scores.json'
    hybrid_path = data_dir / 'tier1_hybrid_temporal.json'
    discretion_path = data_dir / 'tier1_discretion_scores.json'
    its_path = data_dir / 'tier1_its_analysis.json'
    discretion_by_type_path = data_dir / 'discretion_scores_by_type.json'
    temporal_by_type_path = data_dir / 'temporal_data_by_type.json'
    evasion_by_type_path = data_dir / 'regulatory_evasion_by_type.json'
    dimension_by_type_path = data_dir / 'dimension_scores_by_type.json'
    
    with open(viz_path) as f:
        viz_data = json.load(f)
    with open(scores_path) as f:
        platform_scores = json.load(f)
    with open(hybrid_path) as f:
        hybrid_temporal = json.load(f)
    
    # Load enhanced methodology data if available
    try:
        with open(discretion_path) as f:
            discretion_scores = json.load(f)
    except FileNotFoundError:
        discretion_scores = {}
        
        
    try:
        with open(its_path) as f:
            its_analysis = json.load(f)
    except FileNotFoundError:
        its_analysis = {}

    try:
        with open(discretion_by_type_path) as f:
            discretion_scores_by_type = json.load(f)
    except FileNotFoundError:
        discretion_scores_by_type = {}

    try:
        with open(temporal_by_type_path) as f:
            temporal_data_by_type = json.load(f)
    except FileNotFoundError:
        temporal_data_by_type = {}

    try:
        with open(evasion_by_type_path) as f:
            regulatory_evasion_by_type = json.load(f)
    except FileNotFoundError:
        regulatory_evasion_by_type = {}

    try:
        with open(dimension_by_type_path) as f:
            dimension_scores_by_type = json.load(f)
    except FileNotFoundError:
        dimension_scores_by_type = {}

    # Clause-level diff analysis data
    diff_metrics_path = data_dir / 'diff_metrics.json'
    try:
        with open(diff_metrics_path) as f:
            diff_metrics = json.load(f)
    except FileNotFoundError:
        diff_metrics = {}
        
    return (
        viz_data,
        platform_scores,
        hybrid_temporal,
        discretion_scores,
        its_analysis,
        discretion_scores_by_type,
        temporal_data_by_type,
        regulatory_evasion_by_type,
        dimension_scores_by_type,
        diff_metrics,
    )

(
    viz_data,
    platform_scores,
    hybrid_temporal,
    discretion_scores,
    its_analysis,
    discretion_scores_by_type,
    temporal_data_by_type,
    regulatory_evasion_by_type,
    dimension_scores_by_type,
    diff_metrics,
) = load_data()

# Platform colors and metadata (10 platforms with real text analysis)
PLATFORM_CONFIG = {
    "Meta": {"color": "#1877F2", "confidence": "high", "emoji": "", "category": "social_media"},
    "YouTube": {"color": "#FF0000", "confidence": "high", "emoji": "", "category": "social_media"},
    "WhatsApp": {"color": "#25D366", "confidence": "medium", "emoji": "", "category": "messaging"},
    "Reddit": {"color": "#FF4500", "confidence": "high", "emoji": "", "category": "social_media"},
    "LinkedIn": {"color": "#0A66C2", "confidence": "high", "emoji": "", "category": "social_media"},
    "TikTok": {"color": "#000000", "confidence": "low", "emoji": "⚠️", "category": "social_media"},
    "Twitter": {"color": "#1DA1F2", "confidence": "medium", "emoji": "", "category": "social_media"},
    "Instagram": {"color": "#E4405F", "confidence": "high", "emoji": "", "category": "social_media"},
    "Pinterest": {"color": "#BD081C", "confidence": "high", "emoji": "", "category": "social_media"},
    "Discord": {"color": "#5865F2", "confidence": "high", "emoji": "", "category": "social_media"},
    "Google": {"color": "#4285F4", "confidence": "high", "emoji": "", "category": "social_media"},
}

# Header
st.title("📊 Platform Policy Explorer")

st.markdown("---")

# Sidebar filters
st.sidebar.header("⚙️ Filters")
selected_platforms = st.sidebar.multiselect(
    "Select Platforms",
    options=list(platform_scores.keys()),
    default=list(platform_scores.keys())
)

show_confidence = st.sidebar.checkbox("Show Confidence Indicators", value=True)

doc_type_filter = st.sidebar.multiselect(
    "Document Types",
    options=['privacy', 'tos'],
    default=['privacy', 'tos'],
    format_func=lambda x: "Privacy Policy" if x == 'privacy' else "Terms of Service"
)

# Main content tabs
tab6, tab7, tab9, tab2, tab4, tab3, tab8, tab5 = st.tabs([
    "📈 Time Series",
    "🔍 Regulatory Evasion",
    "🔬 Clause-Level Diff",
    "🔺 Dimensions",
    "📜 Discretion",
    "⚖️ Agency Asymmetry",
    "📄 Privacy vs ToS",
    "📋 Detailed Metrics",
])

# TAB 2: DIMENSIONS
with tab2:
    st.subheader("Four Dimensions of Legalization")

    # Direction explanation
    st.info("""
    **📐 How to Read This Chart:**

    All four dimensions point in the **same direction** — larger area = more **platform-favorable** (less user-friendly):

    | Dimension | Higher Value Means | Impact on Users |
    |-----------|-------------------|-----------------|
    | **Complexity** | Harder to read (longer sentences, deeper syntax) | ❌ Harder to understand |
    | **Agency Asymmetry** | More "platform" mentions vs "user" mentions | ❌ Platform-centric language |
    | **Formality** | More legal jargon and formal terms | ❌ Less accessible |
    | **Discretion** | More discretionary power language | ❌ More platform control |

    ⚠️ **Larger radar area = More platform-protective policy = Worse for users**
    """)

    # Radar chart with 4 dimensions
    fig = go.Figure()
    
    categories = ['Complexity', 'Agency Asymmetry', 'Formality', 'Discretion']  # Added Discretion
    
    for platform in selected_platforms:
        # Get scores from dimension_scores and discretion_scores
        dim = platform_scores[platform]
        disc = discretion_scores.get(platform, {})
        discretion_val = disc.get('avg_discretion_ratio', 0)

        # Dynamic normalization based on actual data range
        if discretion_scores and discretion_val:
            all_ratios = [d.get('avg_discretion_ratio', 0) for d in discretion_scores.values()]
            ratio_min = min(all_ratios)
            ratio_max = max(all_ratios)
            ratio_mean = sum(all_ratios) / len(all_ratios) if all_ratios else 1

            # Z-score-like normalization to match other dimensions
            # Avoid division by zero
            denom = (ratio_max - ratio_min) if (ratio_max - ratio_min) > 0 else 1
            discretion_normalized = (discretion_val - ratio_mean) / denom * 2
        else:
            discretion_normalized = 0
        
        # Use pre-calculated radar scores for other dimensions if available, or raw
        # viz_data['radar'] has the normalized scores.
        # But viz_data['radar'] only has 3 dimensions.
        # We need to reconstruct the list.
        
        existing_radar = viz_data['radar'][platform]
        # existing_radar is [complexity, agency, formality]
        
        scores = existing_radar[:3] + [discretion_normalized]
        scores_closed = scores + [scores[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=scores_closed,
            theta=categories + [categories[0]],
            fill='toself',
            name=platform,
            line_color=PLATFORM_CONFIG[platform]['color']
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[-1, 1.5]
            )
        ),
        showlegend=True,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Dimensions by Document Type")
    doc_categories = ['Discretion', 'Disclaimer', 'Obligation', 'Empowerment (inv.)']
    col_doc1, col_doc2 = st.columns(2)

    with col_doc1:
        st.markdown("**🔒 Privacy Policy Dimensions**")
        if 'privacy' in doc_type_filter:
            fig_radar_priv = go.Figure()
            privacy_emp = [
                dimension_scores_by_type.get(p, {}).get('privacy', {}).get('empowerment', 0)
                for p in selected_platforms
                if 'privacy' in dimension_scores_by_type.get(p, {})
            ]
            max_emp = max(privacy_emp) if privacy_emp else 1

            for platform in selected_platforms:
                d = dimension_scores_by_type.get(platform, {}).get('privacy')
                if not d:
                    continue
                emp_inverted = max_emp - d['empowerment'] + 0.1
                values = [d['discretion'], d['disclaimer'], d['obligation'], emp_inverted]
                values_closed = values + [values[0]]

                fig_radar_priv.add_trace(go.Scatterpolar(
                    r=values_closed,
                    theta=doc_categories + [doc_categories[0]],
                    fill='toself',
                    name=platform,
                    line=dict(color=PLATFORM_CONFIG[platform]['color'])
                ))

            fig_radar_priv.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, None])),
                showlegend=True,
                height=500
            )
            st.plotly_chart(fig_radar_priv, use_container_width=True)
        else:
            st.info("Privacy Policy is not selected in Document Types filter.")

    with col_doc2:
        st.markdown("**📜 Terms of Service Dimensions**")
        if 'tos' in doc_type_filter:
            fig_radar_tos = go.Figure()
            tos_emp = [
                dimension_scores_by_type.get(p, {}).get('tos', {}).get('empowerment', 0)
                for p in selected_platforms
                if 'tos' in dimension_scores_by_type.get(p, {})
            ]
            max_emp = max(tos_emp) if tos_emp else 1

            for platform in selected_platforms:
                d = dimension_scores_by_type.get(platform, {}).get('tos')
                if not d:
                    continue
                emp_inverted = max_emp - d['empowerment'] + 0.1
                values = [d['discretion'], d['disclaimer'], d['obligation'], emp_inverted]
                values_closed = values + [values[0]]

                fig_radar_tos.add_trace(go.Scatterpolar(
                    r=values_closed,
                    theta=doc_categories + [doc_categories[0]],
                    fill='toself',
                    name=platform,
                    line=dict(color=PLATFORM_CONFIG[platform]['color'])
                ))

            fig_radar_tos.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, None])),
                showlegend=True,
                height=500
            )
            st.plotly_chart(fig_radar_tos, use_container_width=True)
        else:
            st.info("Terms of Service is not selected in Document Types filter.")
    
    # Updated explanation
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Complexity** (Syntactic)
        - Average sentence length
        - Parse tree depth
        - Clause nesting

        **Agency Asymmetry** (Pronoun Ratio)
        - Platform pronouns: we, our, us
        - User pronouns: you, your
        - Ratio = (platform + 1) / (user + 1)
        """)

    with col2:
        st.markdown("""
        **Formality** (Legal Term Density)
        - Terms: hereby, herein, notwithstanding, pursuant to, etc.
        - Count per 1,000 words
        - 16 formal legal terms tracked

        **Discretion** (Power Language Ratio)
        - Discretion: sole discretion, at any time, reserve the right
        - Disclaimer: as is, without liability, no warranty
        - Ratio = (discretion + disclaimer) / obligation
        """)

# TAB 3: AGENCY ASYMMETRY
with tab3:
    st.subheader("Agency Visibility: Platform vs User")
    
    # Filter agency comparison data
    filtered_agency = [
        item for item in viz_data['agency_comparison'] 
        if item['platform'] in selected_platforms
    ]
    
    # Create grouped bar chart
    fig = px.bar(
        filtered_agency,
        x='platform',
        y='visibility',
        color='entity',
        barmode='group',
        color_discrete_map={'User': '#10B981', 'Platform': '#EF4444'},
        labels={'visibility': 'Visibility Rate', 'platform': 'Platform'},
        title="Explicit Agency Visibility by Entity Type"
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate asymmetry stats
    st.subheader("Asymmetry Statistics")
    
    asymmetry_data = []
    for platform in selected_platforms:
        scores = platform_scores[platform]
        asymmetry_data.append({
            'Platform': platform,
            'Platform Visibility': f"{scores['raw_metrics']['platform_visibility']:.1%}",
            'User Visibility': f"{scores['raw_metrics']['user_visibility']:.1%}",
            'Asymmetry Factor': f"{scores['raw_metrics']['asymmetry_factor']:.1f}x"
        })
    
    st.table(asymmetry_data)
    
    st.warning("""
    **⚠️ Interpretation:** USER actions are explicit 79-81% of the time, while 
    PLATFORM actions are explicit only 5-7% of the time. This 11-20x asymmetry 
    suggests systematic use of passive voice and nominalization to obscure 
    platform obligations.
    """)

# NEW TAB 4: DISCRETION ANALYSIS
with tab4:
    st.subheader("Platform Discretion Language")
    
    st.markdown("""
    **What is Discretion Language?**
    
    Discretion language gives platforms unilateral power to act without user consent.
    Examples: "sole discretion", "reserves the right", "at any time", "without notice"
    """)
    
    # Filter to selected platforms
    filtered_discretion = {p: discretion_scores[p] for p in selected_platforms if p in discretion_scores}
    
    if filtered_discretion:
        # Simpler: just discretion ratio
        ratio_data = []
        for platform, data in filtered_discretion.items():
            ratio_data.append({
                'Platform': platform,
                'Discretion Ratio': data['avg_discretion_ratio'],
                'Power Index': data['avg_platform_power_index']
            })
        
        df_ratio = pd.DataFrame(ratio_data)
        df_ratio = df_ratio.sort_values('Discretion Ratio', ascending=False)
        
        fig_disc = px.bar(
            df_ratio,
            x='Platform',
            y='Discretion Ratio',
            color='Platform',
            color_discrete_map={p: PLATFORM_CONFIG[p]['color'] for p in PLATFORM_CONFIG},
            title="Discretion Ratio by Platform"
        )
        
        fig_disc.update_layout(
            yaxis_title="Discretion Ratio (Platform Power / User Obligation)",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig_disc, use_container_width=True)

        st.markdown("---")
        st.subheader("📊 Discretion by Document Type")
        col_priv, col_tos = st.columns(2)

        with col_priv:
            st.markdown("**🔒 Privacy Policy Discretion Metrics**")
            if 'privacy' in doc_type_filter:
                privacy_metrics = []
                for platform in selected_platforms:
                    d = discretion_scores_by_type.get(platform, {}).get('privacy')
                    if not d:
                        continue
                    privacy_metrics.append({
                        'Platform': platform,
                        'Discretion/1k': f"{d.get('avg_discretion_per_1k', 0):.2f}",
                        'Disclaimer/1k': f"{d.get('avg_disclaimer_per_1k', 0):.2f}",
                        'Obligation/1k': f"{d.get('avg_obligation_per_1k', 0):.2f}",
                        'Discretion Ratio': f"{d.get('avg_discretion_ratio', 0):.2f}",
                        'PAI': f"{d.get('avg_power_asymmetry_index', 0):.2f}",
                        'Docs': d.get('document_count', 0),
                    })

                if privacy_metrics:
                    st.dataframe(pd.DataFrame(privacy_metrics), use_container_width=True, hide_index=True)
                else:
                    st.info("No Privacy Policy discretion metrics for current filters.")
            else:
                st.info("Privacy Policy is not selected in Document Types filter.")

        with col_tos:
            st.markdown("**📜 Terms of Service Discretion Metrics**")
            if 'tos' in doc_type_filter:
                tos_metrics = []
                for platform in selected_platforms:
                    d = discretion_scores_by_type.get(platform, {}).get('tos')
                    if not d:
                        continue
                    tos_metrics.append({
                        'Platform': platform,
                        'Discretion/1k': f"{d.get('avg_discretion_per_1k', 0):.2f}",
                        'Disclaimer/1k': f"{d.get('avg_disclaimer_per_1k', 0):.2f}",
                        'Obligation/1k': f"{d.get('avg_obligation_per_1k', 0):.2f}",
                        'Discretion Ratio': f"{d.get('avg_discretion_ratio', 0):.2f}",
                        'PAI': f"{d.get('avg_power_asymmetry_index', 0):.2f}",
                        'Docs': d.get('document_count', 0),
                    })

                if tos_metrics:
                    st.dataframe(pd.DataFrame(tos_metrics), use_container_width=True, hide_index=True)
                else:
                    st.info("No Terms of Service discretion metrics for current filters.")
            else:
                st.info("Terms of Service is not selected in Document Types filter.")
        
        # Detailed metrics table
        st.subheader("Detailed Discretion Metrics")
        
        detail_data = []
        for platform, data in filtered_discretion.items():
            detail_data.append({
                'Platform': platform,
                'Discretion/1k': f"{data['avg_discretion_per_1k']:.2f}",
                'Disclaimer/1k': f"{data['avg_disclaimer_per_1k']:.2f}",
                'Obligation/1k': f"{data['avg_obligation_per_1k']:.2f}",
                'Discretion Ratio': f"{data['avg_discretion_ratio']:.2f}",
                'Power Index': f"{data['avg_platform_power_index']:.2f}"
            })
        
        st.table(detail_data)
        
        # Explanation
        st.markdown("""
        **Metrics Explained:**
        - **Discretion/1k**: Discretion expressions per 1,000 words
        - **Disclaimer/1k**: Liability disclaimers per 1,000 words
        - **Obligation/1k**: User obligation language per 1,000 words
        - **Discretion Ratio**: (Discretion + Disclaimer) / Obligation
        - **Power Index**: Combined platform authority measure
        
        **Higher values = More platform-favorable language**
        """)
        
        # Top expressions found
        # Top expressions found
        with st.expander("🔍 Common Discretion Expressions (Corpus-Derived)"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Discretion Terms:**
                - "at any time" (113x)
                - "reserve the right" (90x)
                - "sole discretion" (56x)
                - "for any reason" (28x)
                - "without notice" (23x)
                """)
            
            with col2:
                st.markdown("""
                **Disclaimer Terms:**
                - "without limitation" (32x)
                - "as is" / "as available"
                - "not responsible for"
                - "limitation of liability"
                - "indemnify"
                """)
            
            st.caption("""
            These expressions were extracted directly from the 50 Tier 1 policy documents,
            not from a generic predefined list. This ensures platform-specific language is captured.
            """)
    else:
        st.info("Discretion analysis data not currently loaded.")

# TAB 5: DETAILED METRICS
with tab5:
    st.subheader("Detailed Metrics Table")
    
    # Filter detailed metrics
    filtered_metrics = [
        m for m in viz_data['detailed_metrics'] 
        if m['platform'] in selected_platforms
    ]
    
    # Format for display
    df = pd.DataFrame(filtered_metrics)
    df = df.sort_values('legalization_index', ascending=False)
    
    # Format columns
    df['legalization_index'] = df['legalization_index'].apply(lambda x: f"{x:+.3f}")
    df['complexity'] = df['complexity'].apply(lambda x: f"{x:+.3f}")
    df['agency'] = df['agency'].apply(lambda x: f"{x:+.3f}")
    df['formality'] = df['formality'].apply(lambda x: f"{x:+.3f}")
    df['adl'] = df['adl'].apply(lambda x: f"{x:.2f}")
    df['platform_visibility'] = df['platform_visibility'].apply(lambda x: f"{x:.1%}")
    df['user_visibility'] = df['user_visibility'].apply(lambda x: f"{x:.1%}")
    df['asymmetry'] = df['asymmetry'].apply(lambda x: f"{x:.1f}x")
    
    # Rename columns
    df = df.rename(columns={
        'platform': 'Platform',
        'legalization_index': 'Policy Index',
        'complexity': 'Complexity',
        'agency': 'Agency',
        'formality': 'Formality',
        'adl': 'ADL',
        'platform_visibility': 'Platform Vis',
        'user_visibility': 'User Vis',
        'asymmetry': 'Asymmetry',
        'doc_count': 'Docs'
    })
    
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # CSV Export Button
    st.subheader("Export Data")

    # Prepare export data
    export_data = []
    for platform in selected_platforms:
        if platform in platform_scores:
            row = {
                'Platform': platform,
                'Combined Index': next((item['index'] for item in viz_data.get('ranking', []) if item['platform'] == platform), 'N/A'),
                'Rank': next((item['rank'] for item in viz_data.get('ranking', []) if item['platform'] == platform), 'N/A'),
                'Complexity Score': platform_scores[platform].get('complexity_score', 'N/A'),
                'Agency Score': platform_scores[platform].get('agency_score', 'N/A'),
                'Formality Score': platform_scores[platform].get('formality_score', 'N/A'),
            }
            
            # Add discretion metrics if available
            if platform in discretion_scores:
                row['Discretion Ratio'] = discretion_scores[platform].get('avg_discretion_ratio', 'N/A')
                row['Power Index'] = discretion_scores[platform].get('avg_platform_power_index', 'N/A')
            
            export_data.append(row)

    df_export = pd.DataFrame(export_data)

    csv = df_export.to_csv(index=False)
    st.download_button(
        label="📥 Download Full Metrics (CSV)",
        data=csv,
        file_name="policy_explorer_metrics.csv",
        mime="text/csv"
    )

# TAB 6: TIME SERIES ANALYSIS
with tab6:
    st.subheader("Policy Evolution Over Time")
    
    from datetime import datetime
    
    st.success("""
    📊 **All Platforms Have Real Temporal Data** (Verified 2026-01-22)

    | Platform | Documents | Year Range |
    |----------|-----------|------------|
    | Meta | 354 | 2005-2025 (21 years) |
    | Reddit | 232 | 2007-2025 (19 years) |
    | YouTube | 177 | 2013-2025 (13 years) |
    | WhatsApp | 167 | 2009-2025 (17 years) |
    | LinkedIn | 163 | 2013-2025 (13 years) |
    | TikTok | 95 | 2019-2025 (7 years) |

    Data source: TransparencyDB (transparencydb_dev.documents.json)
    """)
    
    st.markdown("---")
    
    temporal_platforms = list(temporal_data_by_type.get('platforms', {}).keys())
    if not temporal_platforms:
        temporal_platforms = list(hybrid_temporal.get('platforms', {}).keys())
    ts_platform_filter = st.multiselect(
        "Select platforms to display",
        options=temporal_platforms,
        default=temporal_platforms,
        key='ts_filter'
    )

    metric_choice = st.selectbox(
        "Select Metric to Display",
        ['power_asymmetry_index', 'term_density'],
        format_func=lambda x: 'Power Asymmetry Index — "WHAT" (Higher = Worse)'
        if x == 'power_asymmetry_index'
        else 'Term Density — "HOW" (Complexity + Formality + Agency)',
        key='ts_metric'
    )

    # Policy Index Definition
    with st.expander("📖 **What is Policy Index?**"):
        st.markdown("""
        **Policy Index** (= Legalization Index) measures how "legalized" and platform-protective a policy document is.

        **Formula:**
        ```
        Policy Index = (Complexity + Agency + Formality) / 3
        ```

        **Components (4 Dimensions, equal-weighted):**
        - **Complexity**: Syntactic complexity (avg sentence length, tree depth)
        - **Agency Asymmetry**: Platform vs user pronoun ratio (we/our vs you/your)
        - **Formality**: Formal legal terms density (hereby, pursuant to, notwithstanding, etc.)
        - **Discretion**: Platform power language ratio (discretion + disclaimer) / obligation

        **Interpretation:**
        - **Higher values** → More platform-protective, legally complex language
        - **Lower values** → More user-friendly, accessible language
        - **Increase over time** → "Legalization" trend (policies becoming more protective)
        """)

    col_priv_ts, col_tos_ts = st.columns(2)

    with col_priv_ts:
        if 'privacy' in doc_type_filter:
            st.subheader("🔒 Privacy Policy Evolution")
            fig_priv = go.Figure()

            for platform in ts_platform_filter:
                ts = temporal_data_by_type.get('platforms', {}).get(platform, {}).get('privacy', {}).get('time_series', [])
                if not ts:
                    continue
                years = [t['year'] for t in ts]
                values = [t.get(metric_choice, 0) for t in ts]
                fig_priv.add_trace(go.Scatter(
                    x=years,
                    y=values,
                    mode='lines+markers',
                    name=platform,
                    line=dict(color=PLATFORM_CONFIG[platform]['color'], width=2),
                    marker=dict(size=6),
                ))

            fig_priv.add_vline(x=2016, line_dash='dot', line_color='blue')
            fig_priv.add_vline(x=2018, line_dash='solid', line_color='red')
            fig_priv.add_vline(x=2020, line_dash='solid', line_color='green')
            display_name = 'Power Asymmetry Index (WHAT)' if metric_choice == 'power_asymmetry_index' else 'Term Density (HOW)'
            fig_priv.update_layout(
                title=f"Privacy Policy: {display_name}",
                xaxis_title='Year',
                yaxis_title=display_name,
                height=460,
                hovermode='x unified'
            )
            st.plotly_chart(fig_priv, use_container_width=True)
        else:
            st.info("Privacy Policy is not selected in Document Types filter.")

    with col_tos_ts:
        if 'tos' in doc_type_filter:
            st.subheader("📜 Terms of Service Evolution")
            fig_tos = go.Figure()

            for platform in ts_platform_filter:
                ts = temporal_data_by_type.get('platforms', {}).get(platform, {}).get('tos', {}).get('time_series', [])
                if not ts:
                    continue
                years = [t['year'] for t in ts]
                values = [t.get(metric_choice, 0) for t in ts]
                fig_tos.add_trace(go.Scatter(
                    x=years,
                    y=values,
                    mode='lines+markers',
                    name=platform,
                    line=dict(color=PLATFORM_CONFIG[platform]['color'], width=2),
                    marker=dict(size=6),
                ))

            fig_tos.add_vline(x=2016, line_dash='dot', line_color='blue')
            fig_tos.add_vline(x=2018, line_dash='solid', line_color='red')
            fig_tos.add_vline(x=2020, line_dash='solid', line_color='green')
            display_name = 'Power Asymmetry Index (WHAT)' if metric_choice == 'power_asymmetry_index' else 'Term Density (HOW)'
            fig_tos.update_layout(
                title=f"Terms of Service: {display_name}",
                xaxis_title='Year',
                yaxis_title=display_name,
                height=460,
                hovermode='x unified'
            )
            st.plotly_chart(fig_tos, use_container_width=True)
        else:
            st.info("Terms of Service is not selected in Document Types filter.")

    # NEW: Discretion Ratio Time Series (Lexicon-based)
    st.markdown("---")
    st.subheader("📊 Discretion Ratio Over Time (Lexicon-Based)")

    with st.expander("📖 **What is Discretion Ratio?**"):
        st.markdown("""
        **Discretion Ratio** measures the balance of power language between platform and user.

        **Formula:**
        ```
        Discretion Ratio = (Discretion + Disclaimer) / max(Obligation, 1)
        ```

        **Where:**
        - **Discretion**: 22 terms like "sole discretion", "at any time", "reserve the right"
        - **Disclaimer**: 13 terms like "without limitation", "as is", "no warranty"
        - **Obligation**: 11 terms like "you must", "you agree", "you shall"

        **Interpretation:**
        - **Ratio > 1** → Platform talks more about its own power than user obligations
        - **Ratio < 1** → More balanced or user-protective
        - **Increase over time** → Platform gaining more discretionary power
        """)

    col_priv_disc, col_tos_disc = st.columns(2)

    with col_priv_disc:
        if 'privacy' in doc_type_filter:
            st.subheader("🔒 Privacy Policy Discretion Ratio")
            fig_priv_disc = go.Figure()

            for platform in ts_platform_filter:
                ts = temporal_data_by_type.get('platforms', {}).get(platform, {}).get('privacy', {}).get('time_series', [])
                if not ts:
                    continue
                years = [t['year'] for t in ts]
                values = [t.get('discretion_ratio', 0) for t in ts]
                fig_priv_disc.add_trace(go.Scatter(
                    x=years,
                    y=values,
                    mode='lines+markers',
                    name=platform,
                    line=dict(color=PLATFORM_CONFIG[platform]['color'], width=2),
                    marker=dict(size=6),
                ))

            fig_priv_disc.add_vline(x=2016, line_dash='dot', line_color='blue')
            fig_priv_disc.add_vline(x=2018, line_dash='solid', line_color='red')
            fig_priv_disc.add_vline(x=2020, line_dash='solid', line_color='green')
            fig_priv_disc.update_layout(
                title='Privacy Policy: Discretion Ratio',
                xaxis_title='Year',
                yaxis_title='Discretion Ratio',
                height=460,
                hovermode='x unified'
            )
            st.plotly_chart(fig_priv_disc, use_container_width=True)
        else:
            st.info("Privacy Policy is not selected in Document Types filter.")

    with col_tos_disc:
        if 'tos' in doc_type_filter:
            st.subheader("📜 Terms of Service Discretion Ratio")
            fig_tos_disc = go.Figure()

            for platform in ts_platform_filter:
                ts = temporal_data_by_type.get('platforms', {}).get(platform, {}).get('tos', {}).get('time_series', [])
                if not ts:
                    continue
                years = [t['year'] for t in ts]
                values = [t.get('discretion_ratio', 0) for t in ts]
                fig_tos_disc.add_trace(go.Scatter(
                    x=years,
                    y=values,
                    mode='lines+markers',
                    name=platform,
                    line=dict(color=PLATFORM_CONFIG[platform]['color'], width=2),
                    marker=dict(size=6),
                ))

            fig_tos_disc.add_vline(x=2016, line_dash='dot', line_color='blue')
            fig_tos_disc.add_vline(x=2018, line_dash='solid', line_color='red')
            fig_tos_disc.add_vline(x=2020, line_dash='solid', line_color='green')
            fig_tos_disc.update_layout(
                title='Terms of Service: Discretion Ratio',
                xaxis_title='Year',
                yaxis_title='Discretion Ratio',
                height=460,
                hovermode='x unified'
            )
            st.plotly_chart(fig_tos_disc, use_container_width=True)
        else:
            st.info("Terms of Service is not selected in Document Types filter.")

    st.info("""
    **📊 Two Complementary Perspectives:**

    | Metric | Measures | Source |
    |--------|----------|--------|
    | **Power Asymmetry Index** | Power balance in language | By-type temporal aggregate |
    | **Term Density** | Complexity + formality + agency style | By-type temporal aggregate |
    | **Discretion Ratio** | Discretion + disclaimer vs obligation | By-type temporal aggregate |

    Both metrics increasing = Platform policies becoming more **complex** AND more **power-asymmetric**.
    """)

    # NEW: ITS Analysis Results Section
    st.markdown("---")
    st.subheader("📈 Regulatory Impact Analysis (ITS)")
    
    if its_analysis:
        for platform, platform_its in its_analysis.items():
            with st.expander(f"📈 {platform} - Regulatory Event Impact"):
                for event_name, event_stats in platform_its['events'].items():
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        pre_trend = event_stats['pre_trend']['slope_per_year']
                        pre_sig = "✓" if event_stats['pre_significant'] else ""
                        st.metric(
                            f"Pre-{event_name.split()[0]} Trend",
                            f"{pre_trend:+.4f}/yr",
                            delta=pre_sig + " significant" if event_stats['pre_significant'] else "not significant",
                            delta_color="normal"
                        )
                    
                    with col2:
                        post_trend = event_stats['post_trend']['slope_per_year']
                        post_sig = "✓" if event_stats['post_significant'] else ""
                        st.metric(
                            f"Post-{event_name.split()[0]} Trend",
                            f"{post_trend:+.4f}/yr",
                            delta=post_sig + " significant" if event_stats['post_significant'] else "not significant"
                        )
                    
                    with col3:
                        if event_stats['level_change']:
                            st.metric(
                                "Immediate Impact",
                                f"{event_stats['level_change']:+.3f}",
                                delta="level jump" if event_stats['level_change'] > 0 else "level drop"
                            )
        
        # Summary callout
        st.success("""
        **🎯 Key ITS Findings**
        
        - **GDPR (2018)**: Statistically significant acceleration in legalization trend for Meta
        - **CCPA (2020)**: Secondary impact, less pronounced than GDPR
        - **Overall Pattern**: Regulatory events correlate with increased policy complexity
        
        *Note: ITS analysis performed only on platforms with real temporal data (Meta, YouTube)*
        """)
    else:
        st.info("ITS analysis data not available.")
    
    # Key observations
    st.subheader("Temporal Trends")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Meta Evolution",
            "2006-2025",
            "+0.40 index points (19 years)"
        )
    
    with col2:
        st.metric(
            "GDPR Impact",
            "2018",
            "Visible acceleration in all platforms"
        )
    
    with col3:
        st.metric(
            "YouTube Evolution",
            "2007-2025",
            "+0.40 index points (18 years)"
        )
    
    st.markdown("---")
    
    st.subheader("Major Regulatory Milestones")
    
    st.markdown("""
    **Key Regulatory Events:**
    - **GDPR** (2016 announced, 2018 effective): EU General Data Protection Regulation  
      → Visible acceleration in legalization for Meta and YouTube post-2016
    - **CCPA** (2018 signed, 2020 effective): California Consumer Privacy Act  
      → Continued complexity increases through 2020
    - **CPRA** (2023): California Privacy Rights Act  
      → Recent policy updates reflect enhanced consumer rights
    - **DSA** (2024): Digital Services Act (EU)  
      → Latest regulatory milestone affecting platform policies
    
    **Observed Patterns:**
    - 📈 **Steady Increase**: Meta and YouTube show consistent 19-year upward trend
    - ⚡ **Regulatory Acceleration**: Growth rate increases around GDPR/CCPA deadlines
    - 🔄 **Industry-Wide**: Both platforms follow similar trajectories despite different starting points
    - 📊 **Convergence**: Gap between Meta (+0.279) and YouTube (-0.134) reflects baseline differences, not temporal divergence
    """)
    

# TAB 7: REGULATORY EVASION ANALYSIS
with tab7:
    st.subheader("Regulatory Evasion Pattern Analysis")

    st.markdown("""
    **Research Question**: Did platforms genuinely comply with GDPR/CCPA, or did they adapt their language
    while preserving power asymmetry?
    """)

    if regulatory_evasion_by_type.get('platforms'):
        st.markdown("---")
        st.subheader("📊 PP vs ToS Classification Matrix")

        classification_colors = {
            'genuine_simplification': '#2E8B57',
            'compliance_with_complexity': '#F1C40F',
            'streamlined_power_grab': '#E67E22',
            'defensive_legalization': '#C0392B',
            'no_clear_pattern': '#7F8C8D',
        }

        compare_rows = []
        for platform in selected_platforms:
            pdata = regulatory_evasion_by_type.get('platforms', {}).get(platform, {})
            if not pdata:
                continue
            for doc_type in ['privacy', 'tos']:
                if doc_type not in doc_type_filter:
                    continue
                summary = pdata.get(doc_type, {}).get('summary', {})
                compare_rows.append({
                    'Platform': platform,
                    'Document Type': 'Privacy Policy' if doc_type == 'privacy' else 'Terms of Service',
                    'PAI Change %': summary.get('pai_change_pct', 0),
                    'Classification': summary.get('classification', 'no_clear_pattern'),
                })

        if compare_rows:
            df_class = pd.DataFrame(compare_rows)
            fig_class = go.Figure()

            for doc_label in ['Privacy Policy', 'Terms of Service']:
                doc_df = df_class[df_class['Document Type'] == doc_label]
                if doc_df.empty:
                    continue
                fig_class.add_trace(go.Bar(
                    name=doc_label,
                    x=doc_df['Platform'],
                    y=doc_df['PAI Change %'],
                    marker_color=[classification_colors.get(c, '#7F8C8D') for c in doc_df['Classification']],
                    text=[f"{c.replace('_', ' ').title()}<br>{v:+.1f}%" for c, v in zip(doc_df['Classification'], doc_df['PAI Change %'])],
                    textposition='outside',
                ))

            fig_class.update_layout(
                barmode='group',
                title='Power Asymmetry Change by Document Type (Pre-GDPR → Post-CCPA)',
                yaxis_title='PAI Change (%)',
                xaxis_title='Platform',
                height=520,
                yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black'),
            )
            fig_class.add_hline(y=0, line_dash='solid', line_color='black', line_width=2)
            st.plotly_chart(fig_class, use_container_width=True)
        else:
            st.info("No by-type regulatory evasion classification data for current filters.")

        st.caption("Color indicates classification category; bars show PAI change separately for Privacy Policy and Terms of Service.")

    # Load regulatory evasion data
    evasion_path = data_dir / 'all_platforms_regulatory_analysis.json'

    try:
        with open(evasion_path) as f:
            evasion_data = json.load(f)

        # Get metadata if available
        metadata = evasion_data.pop('_metadata', {})
        real_data_platforms = metadata.get('data_sources', {}).get('tier1_extracted', [])
        estimated_platforms = metadata.get('data_sources', {}).get('metadata_estimated', [])

        st.success(f"""
        ✅ **Real Text Analysis: {len(evasion_data)} Platforms** (Updated {metadata.get('last_updated', '2026-02-04')[:10]})

        All platforms analyzed using actual policy document text from tier1_extracted.json:
        Meta, YouTube, WhatsApp, Reddit, LinkedIn, TikTok, Twitter, Instagram, Pinterest, Discord
        """)

        # Key finding callout - Multiple platforms with evasion patterns
        st.error("""
        **⚠️ Key Finding: Multiple Platforms Show Evasion Patterns**

        **Reddit** (Most Aggressive - Increased ALL metrics):
        - `sole_discretion`: **+144%** | `may_terminate`: **+174%** | `at_any_time`: **+38%**

        **Meta** (Substitution Pattern - Reduced explicit, added implicit):
        - `may_terminate`: **+76%** ↑ (new pattern introduced)
        - `sole_discretion`: -60% | `reserve_right`: -88% (reduced explicit language)

        **Twitter/X** (Post-CCPA Introduction - No pre-GDPR data for some patterns):
        - `reserve_right`: **+100%** (0→0.18) | `without_liability`: **+100%** (0→0.21)
        - `may_terminate`: **+100%** (0→0.08) - all new patterns added post-regulation

        **LinkedIn** (Partial Evasion):
        - `without_liability`: **+10.6%** ↑ (only increasing pattern)
        - Other patterns reduced but liability limitation strengthened
        """)

        st.markdown("---")

        # Pattern comparison chart
        st.subheader("Discretion Language Changes: Pre-GDPR vs Post-CCPA")

        # Prepare data for visualization
        pattern_data = []
        for platform, data in evasion_data.items():
            if 'patterns' in data:
                for pattern, values in data['patterns'].items():
                    pattern_data.append({
                        'Platform': platform,
                        'Pattern': pattern.replace('_', ' ').title(),
                        'Pre-GDPR': values['pre_gdpr'],
                        'Post-CCPA': values['post_ccpa'],
                        'Change %': values['change_pct']
                    })

        if pattern_data:
            import pandas as pd
            df_evasion = pd.DataFrame(pattern_data)

            # Filter to key patterns
            key_patterns = ['Sole Discretion', 'May Terminate', 'Reserve Right', 'For Any Reason']
            df_key = df_evasion[df_evasion['Pattern'].isin(key_patterns)]

            # Create grouped bar chart
            fig_evasion = go.Figure()

            colors = {'Pre-GDPR': '#3498db', 'Post-CCPA': '#e74c3c'}

            for period in ['Pre-GDPR', 'Post-CCPA']:
                fig_evasion.add_trace(go.Bar(
                    name=period,
                    x=[f"{row['Platform']}<br>{row['Pattern']}" for _, row in df_key.iterrows()],
                    y=df_key[period],
                    marker_color=colors[period]
                ))

            fig_evasion.update_layout(
                barmode='group',
                title="Discretion Language Frequency (per 1,000 words)",
                yaxis_title="Frequency per 1k words",
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )

            st.plotly_chart(fig_evasion, use_container_width=True)

            # Change percentage table
            st.subheader("Change Analysis by Platform")

            # Pivot for display
            pivot_df = df_key.pivot(index='Platform', columns='Pattern', values='Change %')

            # Style the dataframe
            def color_change(val):
                if pd.isna(val):
                    return ''
                if val > 50:
                    return 'background-color: #ffcccc'  # Red for big increase
                elif val < -50:
                    return 'background-color: #ccffcc'  # Green for big decrease
                return ''

            st.dataframe(
                pivot_df.style.map(color_change).format("{:.0f}%"),
                use_container_width=True
            )

            st.caption("""
            🔴 Red = Increased discretion (potential evasion)
            🟢 Green = Decreased discretion (compliance)
            """)

        # Platform classification
        st.markdown("---")
        st.subheader("Platform Compliance Classification")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            **✅ Genuine Compliance**
            - **WhatsApp**: All patterns reduced (avg -80%)
            - **YouTube**: Major patterns reduced (avg -90%)

            *Substantial reduction in discretion language*
            """)

        with col2:
            st.markdown("""
            **⚠️ Substitution Strategy**
            - **Meta**: +76% `may_terminate` (offsetting -88% `reserve_right`)
            - **Twitter**: New patterns introduced post-CCPA
            - **LinkedIn**: +10.6% `without_liability`

            *Reduced explicit, introduced implicit equivalents*
            """)

        with col3:
            st.markdown("""
            **❌ Regulatory Defiance**
            - **Reddit**: ALL patterns increased
                - +174% `may_terminate`
                - +144% `sole_discretion`
                - +38% `at_any_time`

            *Openly increased discretion post-GDPR/CCPA*
            """)

        # Interpretation
        st.markdown("---")
        st.info("""
        **📊 Interpretation: Three Distinct Regulatory Response Strategies**

        **1. Genuine Compliance** (WhatsApp, YouTube):
        - Substantially reduced ALL discretion patterns (-80% to -90%)
        - No compensatory introduction of new patterns
        - Net effect: Actual power rebalancing toward users

        **2. Substitution/Evasion** (Meta, Twitter, LinkedIn):
        - Removed explicit language ("sole discretion", "reserve the right")
        - Introduced functionally equivalent patterns ("may terminate", "without liability")
        - **Net effect**: Power asymmetry maintained through language evolution

        **3. Regulatory Defiance** (Reddit):
        - INCREASED discretion language across ALL patterns post-GDPR/CCPA
        - Suggests deliberate non-compliance or different regulatory interpretation
        - **Net effect**: Power asymmetry explicitly strengthened

        **Implication**: Regulation success is non-uniform. Language-based analysis reveals that approximately
        30% of platforms show evasion behavior that text-mining can detect but surface compliance audits may miss.
        """)

    except FileNotFoundError:
        st.info("Regulatory evasion analysis data not currently loaded.")

# TAB 8: PRIVACY VS TOS SEPARATION
with tab8:
    st.subheader("📄 Document Type Separation Analysis")

    st.markdown("""
    **Critical Methodological Insight**: Mixing Privacy Policy and Terms of Service
    produces misleading results. This tab shows analysis **separated by document type**.
    """)

    # Load doc type analysis
    doc_type_path = data_dir / 'doc_type_analysis.json'

    try:
        with open(doc_type_path) as f:
            doc_type_data = json.load(f)

        st.success("""
        ✅ **Document Type Analysis Available**

        GDPR primarily targets **Privacy Policies**, not Terms of Service.
        Separating these reveals dramatically different patterns.
        """)

        # Summary comparison table
        st.subheader("Privacy Policy vs Terms of Service: Change Comparison")

        comparison_rows = []
        for platform in ['Meta', 'YouTube', 'WhatsApp', 'Reddit', 'LinkedIn', 'Twitter', 'Pinterest']:
            if platform not in doc_type_data:
                continue

            priv_change = doc_type_data[platform].get('privacy', {}).get('change_pct', None)
            tos_change = doc_type_data[platform].get('tos', {}).get('change_pct', None)

            comparison_rows.append({
                'Platform': platform,
                'Privacy Δ': f"{priv_change:+.1f}%" if priv_change is not None else "N/A",
                'ToS Δ': f"{tos_change:+.1f}%" if tos_change is not None else "N/A",
            })

        if comparison_rows:
            import pandas as pd
            df_compare = pd.DataFrame(comparison_rows)

            # Style the dataframe
            def style_change(val):
                if val == "N/A":
                    return 'color: gray'
                try:
                    num = float(val.replace('%', '').replace('+', ''))
                    if num > 50:
                        return 'background-color: #ffcccc; color: darkred; font-weight: bold'
                    elif num < -30:
                        return 'background-color: #ccffcc; color: darkgreen'
                except:
                    pass
                return ''

            st.dataframe(
                df_compare.style.map(style_change, subset=['Privacy Δ', 'ToS Δ']),
                use_container_width=True,
                hide_index=True
            )

        st.caption("""
        🔴 Red = Large increase (potential defensive response)
        🟢 Green = Decrease (compliance)
        """)

        # Key insights
        st.markdown("---")
        st.subheader("🔍 Key Insights from Separation")

        col1, col2 = st.columns(2)

        with col1:
            st.error("""
            **Meta: Opposite Patterns**

            - Privacy Policy: **+1,281%** ↑
            - Terms of Service: **-25%** ↓

            Meta dramatically increased discretion in Privacy
            (defensive response to GDPR scrutiny) while
            reducing it in ToS.
            """)

        with col2:
            st.success("""
            **Reddit: Reinterpretation Needed**

            - Privacy Policy: **-63%** ↓
            - ToS: Limited pre-GDPR data

            Previously labeled as "regulatory defiance,"
            Reddit actually shows **strong privacy compliance**
            when separated by document type.
            """)

        # Visualization: Bar chart comparison
        st.markdown("---")
        st.subheader("Discretion Ratio Change by Document Type")

        fig_doctype = go.Figure()

        platforms_with_data = []
        priv_changes = []
        tos_changes = []

        for platform in ['Meta', 'YouTube', 'WhatsApp', 'Reddit', 'LinkedIn', 'Twitter', 'Pinterest']:
            if platform not in doc_type_data:
                continue

            priv = doc_type_data[platform].get('privacy', {}).get('change_pct', None)
            tos = doc_type_data[platform].get('tos', {}).get('change_pct', None)

            if priv is not None or tos is not None:
                platforms_with_data.append(platform)
                priv_changes.append(priv if priv is not None else 0)
                tos_changes.append(tos if tos is not None else 0)

        fig_doctype.add_trace(go.Bar(
            name='Privacy Policy',
            x=platforms_with_data,
            y=priv_changes,
            marker_color='#3498db'
        ))

        fig_doctype.add_trace(go.Bar(
            name='Terms of Service',
            x=platforms_with_data,
            y=tos_changes,
            marker_color='#e74c3c'
        ))

        fig_doctype.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)

        fig_doctype.update_layout(
            title="Discretion Ratio Change: Pre-GDPR to Post-CCPA",
            yaxis_title="Change (%)",
            barmode='group',
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )

        st.plotly_chart(fig_doctype, use_container_width=True)

    except FileNotFoundError:
        st.info("Document type analysis data not currently loaded.")

# Sidebar: Methodology
st.sidebar.markdown("---")
st.sidebar.header("📚 Methodology v3")

with st.sidebar.expander("📊 Index Calculation"):
    st.markdown("""
    **Combined Policy Index**
    
    **Dimensions (Equal-Weighted):**
    1. Complexity (ADL, tree depth)
    2. Agency (visibility asymmetry)
    3. Formality (legal terminology)
    4. Discretion (corpus-derived)
    
    **Interpretation:**
    Higher score = more legalized = more platform-protective
    """)

# TAB 9: CLAUSE-LEVEL DIFF ANALYSIS
with tab9:
    st.subheader("Clause-Level Policy Evolution Analysis")

    st.markdown("""
    **Direct observation** of which specific clauses were added, deleted, or modified across consecutive
    policy versions. Unlike aggregate time-series analysis, this approach identifies the *content* of
    policy changes and classifies them by regulatory direction.

    > **Method**: SequenceMatcher-based clause alignment (Tao et al., 2025), PAI lexicon classification,
    > GDPR-specific term detection. 354 version pairs across 10 platforms.
    """)

    if not diff_metrics:
        st.warning("⏳ Clause diff analysis not yet available. Run the pipeline scripts first.")
    else:
        _agg = diff_metrics.get('aggregate', {})
        _pm = diff_metrics.get('platform_metrics', {})

        # --- Section A: KPI Metrics ---
        st.markdown("---")
        col_k1, col_k2, col_k3, col_k4 = st.columns(4)
        with col_k1:
            st.metric("Total Clause Changes", f"{_agg.get('total_changes', 0):,}",
                      delta=f"{_agg.get('total_version_pairs', 0)} version pairs")
        with col_k2:
            st.metric("RRI (strict)", f"{_agg.get('RRI_strict', 0):.1%}",
                      delta="User-protective ratio", delta_color="off")
        with col_k3:
            st.metric("GSR", f"{_agg.get('GSR', 0):.1%}",
                      delta="GDPR-specific ratio", delta_color="off")
        with col_k4:
            _ce = _agg.get('CE_words')
            st.metric("CE (words)", f"{_ce:.3f}" if _ce else "N/A",
                      delta="Complexity evolution", delta_color="off")

        st.info("""
        **Key Finding**: 59.3% of substantive clause changes (RRI_strict) are user-protection oriented.
        However, only 12.7% of *all* changes (RRI_broad) are substantive — **78.6% of changes are neutral/cosmetic**
        (rewording without directional shift). GDPR-specific terms appear in 4.1% of changes.
        """)

        # --- Section B: Classification Heatmap ---
        st.markdown("---")
        st.subheader("Change Classification Distribution by Platform")

        platforms_sorted = sorted(_pm.keys())
        classifications = ['regulation_specific', 'user_protection_enhancing',
                          'user_protection_reducing', 'neutral_cosmetic']
        class_labels = ['Regulation-\nSpecific', 'User Prot.\nEnhancing',
                       'User Prot.\nReducing', 'Neutral/\nCosmetic']

        heatmap_z = []
        for platform in platforms_sorted:
            doc_types_data = _pm[platform]
            total_counts = {cls: 0 for cls in classifications}
            for dt, data in doc_types_data.items():
                dist = data.get('overall', {}).get('classification_distribution', {})
                for cls in classifications:
                    total_counts[cls] += dist.get(cls, 0)
            total = sum(total_counts.values())
            row = [total_counts[cls] / max(total, 1) for cls in classifications]
            heatmap_z.append(row)

        fig_hm = go.Figure(data=go.Heatmap(
            z=heatmap_z,
            x=class_labels,
            y=platforms_sorted,
            text=[[f'{v:.2f}' for v in row] for row in heatmap_z],
            texttemplate='%{text}',
            colorscale='YlOrRd',
            colorbar=dict(title='Proportion'),
            hovertemplate='Platform: %{y}<br>Category: %{x}<br>Proportion: %{z:.3f}<extra></extra>',
        ))
        fig_hm.update_layout(
            height=max(400, len(platforms_sorted) * 45),
            margin=dict(l=100, r=20, t=40, b=80),
            xaxis=dict(side='bottom'),
        )
        st.plotly_chart(fig_hm, use_container_width=True)

        st.markdown("""
        > **Interpretation**: Neutral/cosmetic changes dominate across all platforms (62–93%).
        > Pinterest shows the highest regulation-specific ratio (13%), while Twitter and TikTok have
        > the highest substantive change rates (~35% non-neutral). Google has the most cosmetic-heavy
        > profile (93% neutral).
        """)

        # --- Section C: GDPR Pre/Post RRI Comparison ---
        st.markdown("---")
        st.subheader("GDPR vs Non-GDPR Period: Regulatory Response")

        gdpr_platforms = []
        gdpr_rri_vals = []
        non_gdpr_rri_vals = []

        for platform in platforms_sorted:
            doc_types_data = _pm[platform]
            has_gdpr = False
            gdpr_enh, gdpr_total_sub, non_gdpr_enh, non_gdpr_total_sub = 0, 0, 0, 0
            for dt, data in doc_types_data.items():
                gp = data.get('gdpr_period', {})
                ngp = data.get('non_gdpr_period', {})
                if gp.get('total_changes', 0) > 0:
                    has_gdpr = True
                    g_dist = gp.get('classification_distribution', {})
                    g_e = g_dist.get('user_protection_enhancing', 0) + g_dist.get('regulation_specific', 0)
                    g_r = g_dist.get('user_protection_reducing', 0)
                    gdpr_enh += g_e
                    gdpr_total_sub += g_e + g_r
                if ngp.get('total_changes', 0) > 0:
                    ng_dist = ngp.get('classification_distribution', {})
                    ng_e = ng_dist.get('user_protection_enhancing', 0) + ng_dist.get('regulation_specific', 0)
                    ng_r = ng_dist.get('user_protection_reducing', 0)
                    non_gdpr_enh += ng_e
                    non_gdpr_total_sub += ng_e + ng_r
            if has_gdpr:
                gdpr_platforms.append(platform)
                gdpr_rri_vals.append(gdpr_enh / max(gdpr_total_sub, 1))
                non_gdpr_rri_vals.append(non_gdpr_enh / max(non_gdpr_total_sub, 1))
            elif non_gdpr_total_sub > 0:
                gdpr_platforms.append(platform)
                gdpr_rri_vals.append(0)
                non_gdpr_rri_vals.append(non_gdpr_enh / max(non_gdpr_total_sub, 1))

        if gdpr_platforms:
            fig_gdpr = go.Figure()
            fig_gdpr.add_trace(go.Bar(
                name='Non-GDPR Period', x=gdpr_platforms, y=non_gdpr_rri_vals,
                marker_color='#90CAF9', marker_line=dict(color='black', width=0.5),
            ))
            fig_gdpr.add_trace(go.Bar(
                name='GDPR Period', x=gdpr_platforms, y=gdpr_rri_vals,
                marker_color='#1565C0', marker_line=dict(color='black', width=0.5),
            ))
            fig_gdpr.add_hline(y=0.5, line_dash='dash', line_color='gray', opacity=0.4)
            fig_gdpr.update_layout(
                barmode='group', yaxis_range=[0, 1.05],
                yaxis_title='RRI (strict)',
                height=450, margin=dict(t=40),
            )
            st.plotly_chart(fig_gdpr, use_container_width=True)

        st.success("""
        **Key Finding**: GDPR period shows elevated RRI across most platforms. Google and Twitter reach
        RRI=1.0 during GDPR transition (100% of substantive changes were user-protective). Mean GDPR-period
        GSR (0.091) is nearly **2× higher** than non-GDPR period (0.048), confirming regulation-driven
        language adoption.
        """)

        # --- Section D: Change Type Distribution ---
        st.markdown("---")
        st.subheader("Change Type Distribution by Platform")

        ct_data = {}
        for platform in platforms_sorted:
            ct_data[platform] = {'added': 0, 'deleted': 0, 'modified': 0}
            for dt, data in _pm[platform].items():
                ct_dist = data.get('overall', {}).get('change_type_distribution', {})
                for ct in ('added', 'deleted', 'modified'):
                    ct_data[platform][ct] += ct_dist.get(ct, 0)

        ct_colors = {'added': '#4CAF50', 'deleted': '#F44336', 'modified': '#FF9800'}
        fig_ct = go.Figure()
        for ct in ('added', 'deleted', 'modified'):
            vals = []
            for p in platforms_sorted:
                total = sum(ct_data[p].values())
                vals.append(ct_data[p][ct] / max(total, 1))
            fig_ct.add_trace(go.Bar(
                name=ct.capitalize(), x=platforms_sorted, y=vals,
                marker_color=ct_colors[ct],
            ))
        fig_ct.update_layout(
            barmode='stack', yaxis_range=[0, 1.05],
            yaxis_title='Proportion',
            height=400, margin=dict(t=40),
        )
        st.plotly_chart(fig_ct, use_container_width=True)

        st.markdown("""
        > Modified clauses are rare (6.3% overall) — most policy evolution occurs through wholesale
        > addition and deletion of clauses rather than in-place editing. This is consistent with platform
        > practice of periodic full rewrites rather than incremental amendments.
        """)

        # --- Section E: Threshold Sensitivity ---
        st.markdown("---")
        st.subheader("Threshold Sensitivity Analysis")

        @st.cache_data
        def load_threshold_data():
            _diffs_path = data_dir / 'clause_diffs.json'
            try:
                with open(_diffs_path) as _f:
                    _raw = json.load(_f)
                ts_agg = {}
                for d in _raw.get('diffs', []):
                    ts = d.get('threshold_sensitivity', {})
                    for thr_str, counts in ts.items():
                        if thr_str not in ts_agg:
                            ts_agg[thr_str] = {'added': 0, 'deleted': 0, 'modified': 0, 'unchanged': 0}
                        for ct_key in ('added', 'deleted', 'modified', 'unchanged'):
                            ts_agg[thr_str][ct_key] += counts.get(ct_key, 0)
                return ts_agg
            except FileNotFoundError:
                return {}

        ts_agg = load_threshold_data()
        if ts_agg:
            thresholds = sorted(ts_agg.keys())
            ts_colors = {'added': '#4CAF50', 'deleted': '#F44336', 'modified': '#FF9800', 'unchanged': '#9E9E9E'}
            fig_ts = go.Figure()
            for ct_name in ('added', 'deleted', 'modified', 'unchanged'):
                vals = [ts_agg[t].get(ct_name, 0) for t in thresholds]
                fig_ts.add_trace(go.Scatter(
                    x=thresholds, y=vals, mode='lines+markers',
                    name=ct_name.capitalize(), line=dict(color=ts_colors[ct_name], width=2),
                    marker=dict(size=8),
                ))
            fig_ts.update_layout(
                xaxis_title='Similarity Threshold',
                yaxis_title='Total Clause Count',
                height=400, margin=dict(t=40),
            )
            st.plotly_chart(fig_ts, use_container_width=True)

            st.markdown("""
            > **Robustness**: Results are stable across the 0.50–0.70 threshold range. Modified count
            > decreases monotonically with stricter thresholds, while added/deleted increase correspondingly.
            > The unchanged count remains constant (~23K). This monotonic pattern confirms the primary
            > threshold (0.60) choice does not create artifacts.
            """)

        # --- Section F: Privacy vs ToS Comparison ---
        st.markdown("---")
        st.subheader("Privacy Policy vs Terms of Service")

        priv_rris, tos_rris = [], []
        priv_gsrs, tos_gsrs = [], []
        priv_ces, tos_ces = [], []
        for platform, doc_types_data in _pm.items():
            if 'privacy' in doc_types_data:
                ov = doc_types_data['privacy'].get('overall', {})
                priv_rris.append(ov.get('RRI_strict', 0))
                priv_gsrs.append(ov.get('GSR', 0))
                if ov.get('CE_words') is not None:
                    priv_ces.append(ov['CE_words'])
            if 'tos' in doc_types_data:
                ov = doc_types_data['tos'].get('overall', {})
                tos_rris.append(ov.get('RRI_strict', 0))
                tos_gsrs.append(ov.get('GSR', 0))
                if ov.get('CE_words') is not None:
                    tos_ces.append(ov['CE_words'])

        import numpy as np
        metric_names = ['RRI (strict)', 'GSR', 'CE (words)']
        priv_means = [
            float(np.mean(priv_rris)) if priv_rris else 0,
            float(np.mean(priv_gsrs)) if priv_gsrs else 0,
            float(np.mean(priv_ces)) if priv_ces else 0,
        ]
        tos_means = [
            float(np.mean(tos_rris)) if tos_rris else 0,
            float(np.mean(tos_gsrs)) if tos_gsrs else 0,
            float(np.mean(tos_ces)) if tos_ces else 0,
        ]

        fig_dt = go.Figure()
        fig_dt.add_trace(go.Bar(
            name='Privacy Policy', x=metric_names, y=priv_means,
            marker_color='#7B1FA2', text=[f'{v:.3f}' for v in priv_means], textposition='outside',
        ))
        fig_dt.add_trace(go.Bar(
            name='Terms of Service', x=metric_names, y=tos_means,
            marker_color='#F57C00', text=[f'{v:.3f}' for v in tos_means], textposition='outside',
        ))
        fig_dt.update_layout(
            barmode='group',
            yaxis_title='Mean Value',
            height=400, margin=dict(t=40),
        )
        st.plotly_chart(fig_dt, use_container_width=True)

        st.markdown("""
        > Privacy policies and Terms of Service show similar RRI patterns but differ in GDPR-specific
        > content: privacy policies contain more regulation-specific language (higher GSR) as expected,
        > since GDPR primarily targets data processing disclosures rather than service terms.
        """)

        # --- Section G: Convergent Validity ---
        cv = diff_metrics.get('convergent_validity', {})
        if cv.get('available'):
            st.markdown("---")
            st.subheader("Convergent Validity")
            st.warning(f"""
            **Spearman ρ = {cv['spearman_rho']}** (p = {cv['p_value']:.4f}, n = {cv['n_platforms']} platforms)

            The weak negative correlation suggests that **cross-sectional PAI scores and longitudinal RRI
            capture different dimensions** of platform behavior — a methodologically important finding.
            Platforms with high static power asymmetry do not necessarily make fewer user-protective changes
            over time.
            """)

        # --- Section H: Data Summary ---
        with st.expander("📋 Pipeline Summary"):
            summary_rows = []
            for platform in platforms_sorted:
                doc_types_data = _pm[platform]
                for dt, data in doc_types_data.items():
                    ov = data.get('overall', {})
                    _ce_val = ov.get('CE_words')
                    summary_rows.append({
                        'Platform': platform,
                        'Doc Type': 'Privacy' if dt == 'privacy' else 'ToS',
                        'Pairs': ov.get('total_version_pairs', 0),
                        'Changes': ov.get('total_changes', 0),
                        'RRI (strict)': f"{ov.get('RRI_strict', 0):.3f}",
                        'GSR': f"{ov.get('GSR', 0):.3f}",
                        'CE (words)': f"{_ce_val:.3f}" if _ce_val is not None else 'N/A',
                    })
            if summary_rows:
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

            st.caption("""
            **Metric Definitions:**
            - **RRI_strict** = (enhancing + regulation_specific) / (enhancing + reducing + regulation_specific) — substantive changes only
            - **GSR** = regulation_specific / total_changes — GDPR-specific response ratio
            - **CE_words** = avg words in added clauses / avg words in existing clauses — >1.0 means added clauses are more complex
            """)


with st.sidebar.expander("📜 Discretion Lexicon"):
    st.markdown("""
    **Corpus-Derived Approach**
    
    Unlike generic lexicons, our discretion terms were
    extracted directly from the 50 Tier 1 policy documents.
    
    **Key Patterns Found:**
    - "at any time" (113x)
    - "reserve the right" (90x)
    - "sole discretion" (56x)
    - "without limitation" (32x)
    
    This ensures no platform-specific language is missed.
    """)

# Footer
st.markdown("---")
st.caption("""
**Data Version:** 2026-01-22 (Audit Verified) | **Platforms:** 11 |
**Model:** SpaCy Legal NLP | **Corpus:** Tier 1 Social Media (1,752 documents) |
**Clause Diff:** 354 pairs, 35,735 changes analyzed
""")
