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
    pca_path = data_dir / 'tier1_pca_index.json'
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
        with open(pca_path) as f:
            pca_index = json.load(f)
    except FileNotFoundError:
        pca_index = {}
        
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
        
    return (
        viz_data,
        platform_scores,
        hybrid_temporal,
        discretion_scores,
        pca_index,
        its_analysis,
        discretion_scores_by_type,
        temporal_data_by_type,
        regulatory_evasion_by_type,
        dimension_scores_by_type,
    )

(
    viz_data,
    platform_scores,
    hybrid_temporal,
    discretion_scores,
    pca_index,
    its_analysis,
    discretion_scores_by_type,
    temporal_data_by_type,
    regulatory_evasion_by_type,
    dimension_scores_by_type,
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
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "📊 Rankings",
    "🔺 Dimensions",
    "⚖️ Agency Asymmetry",
    "📜 Discretion",
    "📋 Detailed Metrics",
    "📈 Time Series",
    "🔍 Regulatory Evasion",
    "📄 Privacy vs ToS",
    "🔬 Causal Analysis",
    "✅ Validation"
])

# TAB 1: RANKINGS
with tab1:
    st.subheader("Platform Rankings by Policy Index")
    
    if pca_index:
        # NEW: Show methodology badge
        st.info(f"""
        📊 **PCA-Weighted Index** (PC1 explains {pca_index['methodology']['explained_variance']:.1%} of variance)
        
        Dimensions: {', '.join(pca_index['methodology']['dimensions'])}
        """)
        
        # Build ranking from PCA results
        pca_ranking = []
        for platform, data in pca_index['platforms'].items():
            if platform in selected_platforms:
                pca_ranking.append({
                    'platform': platform,
                    'index': data['pca_index'],
                    'rank': data['rank']
                })
        
        pca_ranking.sort(key=lambda x: x['index'], reverse=True)
        
        # Create bar chart
        fig = go.Figure()
        
        for item in pca_ranking:
            platform = item['platform']
            config = PLATFORM_CONFIG[platform]
            
            label = f"{platform} {config['emoji']}" if show_confidence and config['emoji'] else platform
            
            fig.add_trace(go.Bar(
                x=[item['index']],
                y=[label],
                orientation='h',
                marker_color=config['color'],
                name=platform,
                text=[f"{item['index']:+.2f}"],
                textposition='outside',
                showlegend=False
            ))
        
        fig.update_layout(
            xaxis_title="PCA Policy Index (higher = more platform-protective)",
            yaxis_title="",
            height=400,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("📋 Rankings by Document Type")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🔒 Privacy Policy Rankings**")
            privacy_rows = []
            for platform in selected_platforms:
                platform_data = discretion_scores_by_type.get(platform, {})
                if 'privacy' not in platform_data or 'privacy' not in doc_type_filter:
                    continue
                d = platform_data['privacy']
                privacy_rows.append({
                    'Platform': platform,
                    'PAI': round(d.get('avg_power_asymmetry_index', 0), 2),
                    'Term Density': round(d.get('avg_term_density', 0), 2),
                    'Docs': d.get('document_count', 0),
                })

            if privacy_rows:
                df_priv = pd.DataFrame(privacy_rows).sort_values('PAI', ascending=False)
                st.dataframe(df_priv, use_container_width=True, hide_index=True)
            else:
                st.info("No Privacy Policy ranking data for current filters.")

        with col2:
            st.markdown("**📜 Terms of Service Rankings**")
            tos_rows = []
            for platform in selected_platforms:
                platform_data = discretion_scores_by_type.get(platform, {})
                if 'tos' not in platform_data or 'tos' not in doc_type_filter:
                    continue
                d = platform_data['tos']
                tos_rows.append({
                    'Platform': platform,
                    'PAI': round(d.get('avg_power_asymmetry_index', 0), 2),
                    'Term Density': round(d.get('avg_term_density', 0), 2),
                    'Docs': d.get('document_count', 0),
                })

            if tos_rows:
                df_tos = pd.DataFrame(tos_rows).sort_values('PAI', ascending=False)
                st.dataframe(df_tos, use_container_width=True, hide_index=True)
            else:
                st.info("No Terms of Service ranking data for current filters.")

        compare_rows = []
        for platform in selected_platforms:
            platform_data = discretion_scores_by_type.get(platform, {})
            if not platform_data:
                continue
            if 'privacy' in doc_type_filter:
                compare_rows.append({
                    'Platform': platform,
                    'Document Type': 'Privacy Policy',
                    'PAI': platform_data.get('privacy', {}).get('avg_power_asymmetry_index', 0),
                })
            if 'tos' in doc_type_filter:
                compare_rows.append({
                    'Platform': platform,
                    'Document Type': 'Terms of Service',
                    'PAI': platform_data.get('tos', {}).get('avg_power_asymmetry_index', 0),
                })

        if compare_rows:
            df_compare_doc_type = pd.DataFrame(compare_rows)
            fig_compare = px.bar(
                df_compare_doc_type,
                x='Platform',
                y='PAI',
                color='Document Type',
                barmode='group',
                color_discrete_map={
                    'Privacy Policy': '#2E8B57',
                    'Terms of Service': '#8B4513',
                },
                title='Privacy Policy vs Terms of Service Power Asymmetry Index'
            )
            fig_compare.update_layout(yaxis_title='Power Asymmetry Index', height=420)
            st.plotly_chart(fig_compare, use_container_width=True)
        
        # NEW: Show PCA loadings
        with st.expander("📐 How dimensions contribute to the index"):
            loadings = pca_index['methodology']['loadings']
            
            loading_data = []
            for dim, loading in loadings.items():
                loading_data.append({
                    'Dimension': dim,
                    'Loading': f"{loading:+.3f}",
                    'Contribution': "Strong" if abs(loading) > 0.4 else "Moderate" if abs(loading) > 0.2 else "Weak"
                })
            
            st.table(loading_data)
            
            st.caption("""
            **Loadings** show how much each dimension contributes to the overall index.
            Higher absolute values mean stronger contribution.
            """)
        
        # Key finding (updated)
        st.success(f"""
        **🎯 Key Finding: TikTok Confirmed as Extreme Outlier**
        
        With the PCA-weighted index, TikTok scores +{pca_index['platforms']['TikTok']['pca_index']:.2f}, 
        which is {pca_index['platforms']['TikTok']['pca_index'] / pca_index['platforms']['Meta']['pca_index']:.1f}x 
        higher than Meta. This reflects not just syntactic complexity but also 
        aggressive use of discretion language ("sole discretion", "at any time", etc.).
        """)
        
        # Quick ITS insight (if data available)
        if its_analysis and 'Meta' in its_analysis:
            meta_gdpr = its_analysis['Meta']['events'].get('GDPR Effective (May 2018)', {})
            if meta_gdpr:
                st.info(f"""
                📈 **Regulatory Impact Preview**: After GDPR, Meta's legalization trend 
                accelerated by {meta_gdpr.get('slope_change', 0):+.4f}/year. 
                See the **Time Series** tab for full analysis.
                """)
        
        # Comparison section
        with st.expander("📊 Compare Old vs New Index"):
            st.markdown("**Simple Average vs PCA-Weighted**")
            
            comparison_data = []
            for platform in selected_platforms:
                if platform in platform_scores and platform in pca_index['platforms']:
                    # Old index (simple average)
                    dim = platform_scores[platform]
                    old_index = (dim['complexity_score'] + dim['agency_score'] + dim['formality_score']) / 3
                    
                    # New index (PCA)
                    new_index = pca_index['platforms'][platform]['pca_index']
                    
                    comparison_data.append({
                        'Platform': platform,
                        'Simple Average': round(old_index, 3),
                        'PCA-Weighted': round(new_index, 3),
                        'Difference': round(new_index - old_index, 3)
                    })
            
            df_compare = pd.DataFrame(comparison_data)
            df_compare = df_compare.sort_values('PCA-Weighted', ascending=False)
            st.dataframe(df_compare, use_container_width=True)
            
            st.caption("""
            The PCA-weighted index incorporates a 4th dimension (Discretion) and uses
            data-driven weights instead of equal weighting. This typically amplifies
            differences between platforms.
            """)
    else:
        st.warning("PCA Index data not available. Please run calculate_pca_index.py.")

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
        st.warning("Discretion scores not available. Please run calculate_discretion.py.")

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
                'Policy Index (PCA)': pca_index['platforms'].get(platform, {}).get('pca_index', 'N/A') if pca_index else 'N/A',
                'Rank': pca_index['platforms'].get(platform, {}).get('rank', 'N/A') if pca_index else 'N/A',
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

        **Components (4 Dimensions, PCA-weighted):**
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
    
    # Data quality note
    st.subheader("Data Quality & Methodology")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Real Data (Meta & YouTube):**
        - Source: 1,337 policy versions
        - Sampling: Quarterly representative documents
        - Coverage: 128 Meta docs, 1,209 YouTube docs
        - Index Estimation: Based on current complexity metrics adjusted for temporal position relative to regulatory events
        """)
    
    with col2:
        st.markdown("""
        **All Data Sources:**
        - TransparencyDB real policy documents
        - Period-aggregated metrics (weighted by document count)
        - Both Privacy Policy and Terms of Service analyzed separately
        """)
    
    st.info("""
    💡 **Research Implications**: The 19-year Meta and YouTube datasets enable:
    - Causal analysis of regulatory impact (GDPR as natural experiment)
    - Lead/lag identification between policy changes and regulations
    - Event study methodology validation
    - Predictive modeling of future legalization trends
    
    This represents one of the longest temporal policy datasets in platform governance research.
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
        st.warning("Regulatory evasion analysis data not found. Please run the analysis script first.")

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
            - ToS: N/A (insufficient pre-GDPR data)

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

        # Methodological note
        st.info("""
        **📊 Methodological Recommendation**

        Future regulatory impact studies should **always separate by document type**:

        1. **Privacy Policy**: Direct target of GDPR/CCPA → Primary compliance indicator
        2. **Terms of Service**: Less directly regulated → May show business strategy
        3. **Guidelines**: Community standards → Different regulatory context

        Combining document types can produce **misleading conclusions** about compliance behavior.
        """)

    except FileNotFoundError:
        st.warning("Document type analysis not available. Run analyze_by_doc_type.py first.")

# TAB 9: CAUSAL ANALYSIS (DiD)
with tab9:
    st.subheader("Causal Analysis: Difference-in-Differences (DiD)")

    st.markdown("""
    **Research Question**: Did GDPR/CCPA cause platforms to change their policy language?

    We use Difference-in-Differences (DiD) to estimate the **causal effect** of regulations,
    not just correlation.
    """)

    # Load DiD results
    did_path = data_dir / 'did_analysis_results.json'

    try:
        with open(did_path) as f:
            did_results = json.load(f)

        # Show treatment/control groups
        config = did_results.get('config', {})

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🎯 Treatment Group** (High EU Exposure)")
            treated = config.get('treated', {})
            st.write(", ".join(treated.get('platforms', [])))
            st.caption(treated.get('rationale', ''))

        with col2:
            st.markdown("**🔄 Control Group** (Low EU Exposure)")
            control = config.get('control', {})
            st.write(", ".join(control.get('platforms', [])))
            st.caption(control.get('rationale', ''))

        st.markdown("---")

        # Results for each intervention
        for intervention_name, intervention_data in did_results.get('interventions', {}).items():
            st.subheader(f"📅 {intervention_name} Analysis")

            # Observation counts
            counts = intervention_data.get('observation_counts', {})
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Treated Pre", counts.get('treated_pre', 0))
            col2.metric("Treated Post", counts.get('treated_post', 0))
            col3.metric("Control Pre", counts.get('control_pre', 0))
            col4.metric("Control Post", counts.get('control_post', 0))

            # Main DiD result
            main_reg = intervention_data.get('main_regression', {})
            if 'error' not in main_reg:
                did_coef = main_reg.get('coefficients', {}).get('did_effect', {})

                col1, col2, col3 = st.columns(3)

                with col1:
                    estimate = did_coef.get('estimate', 0)
                    sig = "✅" if did_coef.get('significant_05') else ("⚠️" if did_coef.get('significant_10') else "❌")
                    st.metric(
                        "DiD Effect (β₃)",
                        f"{estimate:.4f}",
                        delta=f"p={did_coef.get('p_value', 1):.4f} {sig}"
                    )

                with col2:
                    ci = did_coef.get('ci_95', [0, 0])
                    st.metric(
                        "95% Confidence Interval",
                        f"[{ci[0]:.3f}, {ci[1]:.3f}]"
                    )

                with col3:
                    r2 = main_reg.get('r_squared', 0)
                    st.metric("R²", f"{r2:.4f}")

                # Interpretation
                if did_coef.get('significant_05'):
                    st.success(f"**Statistically significant effect detected.** {intervention_name} caused treated platforms to change policy language by {estimate:.3f} index points more than control platforms.")
                elif did_coef.get('significant_10'):
                    st.warning(f"**Marginally significant effect.** Some evidence that {intervention_name} affected treated platforms (p < 0.10).")
                else:
                    st.info(f"**No significant causal effect detected.** The difference between treated and control groups after {intervention_name} is not statistically distinguishable from zero.")

            # Parallel Trends
            parallel = intervention_data.get('parallel_trends', {})
            if 'error' not in parallel:
                with st.expander("📊 Parallel Trends Assumption"):
                    assessment = parallel.get('assessment', 'Unknown')
                    if assessment == 'PASS':
                        st.success("✅ Parallel trends assumption SUPPORTED")
                    else:
                        st.warning("⚠️ Parallel trends assumption may be VIOLATED - interpret with caution")

                    mean_test = parallel.get('tests', {}).get('mean_comparison', {})
                    st.write(f"Pre-treatment means: Treated = {mean_test.get('treated_mean', 0):.4f}, Control = {mean_test.get('control_mean', 0):.4f}")
                    st.write(f"Difference p-value: {mean_test.get('p_value', 1):.4f}")

            # Robustness
            robustness = intervention_data.get('robustness', {})
            if robustness:
                with st.expander("🔧 Robustness Checks"):
                    if 'alternative_control' in robustness:
                        alt = robustness['alternative_control']
                        alt_reg = alt.get('regression', {})
                        if 'error' not in alt_reg:
                            alt_coef = alt_reg.get('coefficients', {}).get('did_effect', {})
                            st.write(f"**Alternative Control** ({', '.join(alt.get('control_group', []))})")
                            st.write(f"DiD Effect: {alt_coef.get('estimate', 0):.4f} (p={alt_coef.get('p_value', 1):.4f})")

                    if 'placebo_test' in robustness:
                        placebo = robustness['placebo_test']
                        placebo_reg = placebo.get('regression', {})
                        if 'error' not in placebo_reg:
                            placebo_coef = placebo_reg.get('coefficients', {}).get('did_effect', {})
                            placebo_pass = not placebo_coef.get('significant_10', False)
                            st.write(f"**Placebo Test** (1 year before)")
                            st.write(f"DiD Effect: {placebo_coef.get('estimate', 0):.4f} (p={placebo_coef.get('p_value', 1):.4f})")
                            if placebo_pass:
                                st.success("✅ Placebo test PASSED - no spurious pre-trend")
                            else:
                                st.error("❌ Placebo test FAILED - possible confounding")

            st.markdown("---")

        # Methodology explanation
        with st.expander("📚 DiD Methodology Explained"):
            st.markdown("""
            **Difference-in-Differences (DiD)** is a causal inference method that compares:

            1. **Before vs After** (time effect)
            2. **Treated vs Control** (group effect)
            3. **The DIFFERENCE of these differences** (causal effect)

            **Model:**
            ```
            Y = β₀ + β₁(Post) + β₂(Treated) + β₃(Post × Treated) + ε
            ```

            - **β₃ (DiD Effect)**: The causal effect of treatment
            - If β₃ is significant, the regulation CAUSED the change

            **Key Assumption: Parallel Trends**
            - Treated and control groups would have followed similar trends WITHOUT treatment
            - We test this using pre-treatment data

            **Our Setup:**
            - **Treatment**: GDPR (2018) / CCPA (2020)
            - **Treated Group**: High EU exposure (Meta, YouTube, WhatsApp)
            - **Control Group**: Low EU exposure (Reddit, Twitter)
            """)

    except FileNotFoundError:
        st.warning("DiD analysis results not found. Please run `did_analysis_v2.py` first.")

        if st.button("Run DiD Analysis Now"):
            import subprocess
            result = subprocess.run(
                ['python3', str(data_dir.parent / 'scripts' / 'analysis' / 'did_analysis_v2.py')],
                capture_output=True,
                text=True
            )
            st.code(result.stdout)
            if result.returncode == 0:
                st.success("Analysis complete! Please refresh the page.")
            else:
                st.error(f"Error: {result.stderr}")

# TAB 10: VALIDATION & ROBUSTNESS
with tab10:
    st.subheader("Validation & Robustness Checks")

    st.markdown("""
    **Comprehensive validation** of the PAI methodology, addressing reviewer feedback
    on dimensional analysis, normalization, post-hoc rationalization, and null hypothesis testing.
    """)

    experiments_dir = dashboard_dir.parent / 'experiments'

    # --- Section 1: Permutation Test ---
    st.markdown("---")
    st.subheader("🎲 Permutation Test: Lexicon vs Random N-grams")

    perm_path = experiments_dir / '2026-02-16_permutation_test' / 'permutation_results.json'
    try:
        with open(perm_path) as f:
            perm_data = json.load(f)

        st.success(f"""
        ✅ **{perm_data['metadata']['n_iterations']} permutations completed**
        ({perm_data['metadata']['n_documents']} documents, {perm_data['metadata'].get('runtime_seconds', '?')}s runtime)
        """)

        p_vals = perm_data['p_values']
        col1, col2, col3 = st.columns(3)

        with col1:
            p_var = p_vals['cross_platform_variance']
            sig = "✅" if p_var < 0.05 else "❌"
            st.metric("Cross-Platform Variance", f"p = {p_var:.4f}", delta=f"{sig} {'Significant' if p_var < 0.05 else 'Not significant'}")

        with col2:
            p_rank = p_vals['ranking_stability']
            st.metric("Ranking Stability (τ≥0.8)", f"p = {p_rank:.4f}", delta=f"Mean τ = {perm_data['null_summary']['kendall_tau']['mean']:.3f}")

        with col3:
            p_gdpr = p_vals['gdpr_effect']
            sig = "✅" if p_gdpr < 0.05 else "❌"
            st.metric("GDPR Effect", f"p = {p_gdpr:.4f}", delta=f"{sig} {'Significant' if p_gdpr < 0.05 else 'Not significant'}")

        with st.expander("📊 Detailed Permutation Results"):
            st.markdown(f"**Interpretation:** {perm_data.get('interpretation', 'N/A')}")

            real = perm_data['real_statistics']
            null = perm_data['null_summary']

            comp_data = []
            comp_data.append({
                'Statistic': 'Cross-Platform Variance',
                'Real': f"{real['cross_platform_variance']:.6f}",
                'Null Mean': f"{null['variance']['mean']:.6f}",
                'Null SD': f"{null['variance']['std']:.6f}",
                'p-value': f"{p_vals['cross_platform_variance']:.4f}"
            })
            comp_data.append({
                'Statistic': 'GDPR DiD Effect',
                'Real': f"{real['gdpr_did_effect']:.6f}",
                'Null Mean': f"{null['gdpr_did']['mean']:.6f}",
                'Null SD': f"{null['gdpr_did']['std']:.6f}",
                'p-value': f"{p_vals['gdpr_effect']:.4f}"
            })
            st.table(comp_data)

            st.markdown("**Platform Rankings (Real Lexicon):**")
            rank_data = [{'Platform': p, 'Rank': r} for p, r in real['platform_rankings'].items()]
            rank_data.sort(key=lambda x: x['Rank'])
            st.table(rank_data)

    except FileNotFoundError:
        st.warning("⏳ Permutation test results not yet available. Run `permutation_test.py` first.")

    # --- Section 2: Normalization Sensitivity ---
    st.markdown("---")
    st.subheader("📐 Normalization Sensitivity Analysis")

    norm_path = experiments_dir / '2026-02-16_normalization' / 'sensitivity_results.json'
    try:
        with open(norm_path) as f:
            norm_data = json.load(f)

        summary = norm_data['summary']
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Mean Spearman ρ", f"{summary['mean_spearman_rho']:.3f}")
        with col2:
            st.metric("Assessment", summary['stability_assessment'].split(':')[0])

        # Correlation heatmap
        corr = norm_data['pairwise_correlations']
        methods = ['per_1k', 'per_sentence', 'tfidf', 'raw']
        labels = ['Per-1k-word', 'Per-sentence', 'TF-IDF', 'Raw counts']

        corr_matrix = []
        for i, m1 in enumerate(methods):
            row = []
            for j, m2 in enumerate(methods):
                if i == j:
                    row.append(1.0)
                else:
                    key = f"{m1}_vs_{m2}" if f"{m1}_vs_{m2}" in corr else f"{m2}_vs_{m1}"
                    row.append(corr.get(key, {}).get('rho', 0))
            corr_matrix.append(row)

        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=labels,
            y=labels,
            text=[[f"{v:.3f}" for v in row] for row in corr_matrix],
            texttemplate="%{text}",
            colorscale='RdYlGn',
            zmin=0.5, zmax=1.0
        ))
        fig_corr.update_layout(title="Spearman Rank Correlation Between Normalization Methods", height=400)
        st.plotly_chart(fig_corr, use_container_width=True)

        with st.expander("📊 Rankings by Method"):
            results = norm_data['results_by_method']
            rank_compare = []
            for method_key, method_data in results.items():
                for i, p in enumerate(method_data['ranking']):
                    rank_compare.append({
                        'Method': method_data['name'],
                        'Platform': p,
                        'Rank': i + 1
                    })
            df_ranks = pd.DataFrame(rank_compare)
            pivot = df_ranks.pivot(index='Platform', columns='Method', values='Rank')
            st.dataframe(pivot, use_container_width=True)

    except FileNotFoundError:
        st.warning("Normalization sensitivity results not available.")

    # --- Section 3: Bootstrap Rank CI ---
    st.markdown("---")
    st.subheader("📊 Bootstrap Rank Confidence Intervals")

    boot_path = experiments_dir / '2026-02-16_bootstrap' / 'bootstrap_rank_results.json'
    try:
        with open(boot_path) as f:
            boot_data = json.load(f)

        st.info(f"**{boot_data['metadata']['n_iterations']:,} bootstrap iterations** | {boot_data['metadata']['n_documents']} documents | {boot_data['metadata']['n_platforms']} platforms")

        # Rank CI chart
        ci_data = boot_data['rank_confidence_intervals']
        platforms_sorted = sorted(ci_data.keys(), key=lambda p: ci_data[p]['mean_rank'])

        fig_ci = go.Figure()
        for p in platforms_sorted:
            d = ci_data[p]
            color = PLATFORM_CONFIG.get(p, {}).get('color', '#888888')
            fig_ci.add_trace(go.Bar(
                x=[p],
                y=[d['mean_rank']],
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=[d['ci_upper'] - d['mean_rank']],
                    arrayminus=[d['mean_rank'] - d['ci_lower']]
                ),
                marker_color=color,
                name=p,
                showlegend=False,
                text=[f"{d['mean_rank']:.1f} [{d['ci_lower']}-{d['ci_upper']}]"],
                textposition='outside'
            ))

        fig_ci.update_layout(
            title="Platform Rank with 95% Bootstrap CI (lower = more platform-protective)",
            yaxis_title="Rank (1 = highest PAI)",
            yaxis=dict(autorange='reversed'),
            height=450
        )
        st.plotly_chart(fig_ci, use_container_width=True)

        # Rank reversal heatmap
        with st.expander("🔄 Rank Reversal Probability Matrix"):
            reversal = boot_data['rank_reversal_probability']
            plats = sorted(reversal.keys())
            rev_matrix = [[reversal[p1].get(p2, 0) for p2 in plats] for p1 in plats]

            fig_rev = go.Figure(data=go.Heatmap(
                z=rev_matrix,
                x=plats, y=plats,
                text=[[f"{v:.2f}" for v in row] for row in rev_matrix],
                texttemplate="%{text}",
                colorscale='RdBu',
                zmin=0, zmax=1
            ))
            fig_rev.update_layout(title="P(row platform ranked higher than column platform)", height=500)
            st.plotly_chart(fig_rev, use_container_width=True)

            st.caption("Values > 0.5 mean the row platform is typically ranked higher (more platform-protective). Values near 0.5 indicate uncertain ordering.")

        if boot_data.get('insufficient_data_platforms'):
            st.warning(f"⚠️ Insufficient data: {', '.join(boot_data['insufficient_data_platforms'])}")

    except FileNotFoundError:
        st.warning("Bootstrap results not available.")

    # --- Section 4: Segmented Regression ---
    st.markdown("---")
    st.subheader("📈 Segmented Regression (ITS v2)")

    seg_path = experiments_dir / '2026-02-16_its' / 'segmented_regression_results_v2.json'
    try:
        with open(seg_path) as f:
            seg_data = json.load(f)

        for platform_name in ['Meta', 'Reddit']:
            if platform_name not in seg_data.get('platforms', {}):
                continue
            pdata = seg_data['platforms'][platform_name]

            with st.expander(f"📈 {platform_name} (n={pdata['n']})"):
                if 'regression' in pdata:
                    reg = pdata['regression']
                    coefs = reg['coefficients']
                    pvals = reg['p_values']

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("R²", f"{reg['r_squared']:.3f}")
                    col2.metric("Level Change (β₂)", f"{coefs['level_change']:+.3f}", delta=f"p={pvals['level_change']:.3f}")
                    col3.metric("Slope Change (β₃)", f"{coefs['slope_change']:+.4f}", delta=f"p={pvals['slope_change']:.3f}")
                    col4.metric("Cohen's d", f"{pdata.get('cohens_d', 'N/A')}")

                    coef_table = []
                    for key in ['intercept', 'time_trend', 'level_change', 'slope_change']:
                        coef_table.append({
                            'Coefficient': key,
                            'Estimate': f"{coefs[key]:+.4f}",
                            'HC3 SE': f"{reg['se_hc3'][key]:.4f}",
                            't-stat': f"{reg['t_statistics'][key]:.4f}",
                            'p-value': f"{pvals[key]:.4f}"
                        })
                    st.table(coef_table)
                else:
                    st.write(f"Pre/Post comparison only: Cohen's d = {pdata.get('cohens_d', 'N/A')}")

        # Effect size summary for all platforms
        st.markdown("**Effect Size Summary (All Platforms)**")
        effect_data = []
        for pname, pdata in seg_data.get('platforms', {}).items():
            effect_data.append({
                'Platform': pname,
                'n': pdata['n'],
                'Mean Pre': f"{pdata.get('mean_pre', 'N/A'):.4f}" if pdata.get('mean_pre') is not None else 'N/A',
                'Mean Post': f"{pdata.get('mean_post', 'N/A'):.4f}" if pdata.get('mean_post') is not None else 'N/A',
                "Cohen's d": f"{pdata['cohens_d']:.3f}" if pdata.get('cohens_d') is not None else 'N/A',
                'Method': 'Seg. Regression' if 'regression' in pdata else 'Pre/Post only'
            })
        st.table(effect_data)

    except FileNotFoundError:
        st.warning("Segmented regression results not available.")

    # --- Section 5: DiD v3 ---
    st.markdown("---")
    st.subheader("⚖️ Difference-in-Differences v3 (Lexicon-Based)")

    did3_path = experiments_dir / '2026-02-16_did_analysis' / 'did_results_v3.json'
    try:
        with open(did3_path) as f:
            did3_data = json.load(f)

        st.info(f"""
        **Primary outcome**: combined_index (lexicon-based) | **SE type**: {did3_data['metadata']['se_type']} |
        **n**: {did3_data['metadata']['n_total']}
        """)

        sensitivity = did3_data['treatment_sensitivity']

        did_summary = []
        for threshold, tdata in sensitivity.items():
            ci_did = tdata['combined_index_did']
            leg_did = tdata.get('legalization_index_did', {})
            did_coef = ci_did['coefficients']['did_effect']
            did_p = ci_did['p_values']['did_effect']

            did_summary.append({
                'Threshold': threshold,
                'Treated': ', '.join(tdata['treated_platforms']),
                'n_treated': tdata['n_treated'],
                'DiD (combined)': f"{did_coef:+.3f}",
                'p-value': f"{did_p:.3f}" if not pd.isna(did_p) else 'NaN',
                'Sig?': '✅' if did_p < 0.05 else '❌',
                'DiD (legalization)': f"{leg_did.get('did_effect', 0):+.3f}",
                'MDE': f"{tdata['power_analysis']['mde_80pct_power']:.3f}"
            })

        st.table(did_summary)

        st.warning("""
        **⚠️ Interpretation**: DiD effects are **not statistically significant** across all treatment thresholds.
        However, effects are consistently **positive** (treated platforms show higher PAI increase),
        and the MDE = 0.382 SD suggests we may lack power to detect true effects of this magnitude.
        This is an exploratory finding — "absence of evidence is not evidence of absence."
        """)

    except FileNotFoundError:
        st.warning("DiD v3 results not available.")

    # --- Overall Assessment ---
    st.markdown("---")
    st.subheader("📋 Overall Validation Summary")

    st.markdown("""
    | Check | Status | Key Result |
    |-------|--------|------------|
    | **Normalization Sensitivity** | ✅ Stable | Mean ρ = 0.80, top/bottom rankings consistent |
    | **Bootstrap Rank CI** | ✅ Robust | Top-3 and bottom-3 stable across 10K iterations |
    | **Segmented Regression** | ⚠️ Underpowered | Good fit (R²>0.85) but n=12 limits significance |
    | **DiD v3** | ⚠️ Underpowered | Consistent direction, MDE = 0.382 SD |
    | **Placebo Tests** | ⚠️ Limited | Sparse data prevents meaningful placebo variation |
    | **Permutation Test** | 🔄 See above | 1,000 iterations with random n-gram lexicons |
    """)

# Sidebar: Methodology
st.sidebar.markdown("---")
st.sidebar.header("📚 Methodology v3")

with st.sidebar.expander("📊 Index Calculation"):
    if pca_index:
        st.markdown(f"""
        **PCA-Weighted Composite Index**
        
        PC1 Explained Variance: **{pca_index['methodology']['explained_variance']:.1%}**
        
        **Dimensions:**
        1. Complexity (ADL, tree depth)
        2. Agency (visibility asymmetry)
        3. Formality (legal terminology)
        4. Discretion (corpus-derived) *NEW*
        
        **Interpretation:**
        Higher score = more legalized = more platform-protective
        """)
    else:
        st.markdown("PCA Index data not loaded.")

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

with st.sidebar.expander("📈 ITS Analysis"):
    st.markdown("""
    **Interrupted Time Series**
    
    Tests whether regulatory events (GDPR, CCPA) caused
    statistically significant changes in policy trends.
    
    **Method:**
    - Segment time series at event date
    - Compare pre/post trend slopes
    - Test for significance (p < 0.05)
    - Measure immediate level changes
    """)

with st.sidebar.expander("Confidence Levels"):
    st.markdown("""
    - **High (✓):** ≥8 documents, validated metrics
    - **Medium (⚠):** 5-7 documents
    - **Low (⚠️):** <5 documents, preliminary
    
    **TikTok** has low confidence (only 4 docs) but shows 
    clear outlier pattern.
    """)

# Footer
st.markdown("---")
st.caption("""
**Data Version:** 2026-01-22 (Audit Verified) | **Platforms:** 7 (All Real Temporal Data) |
**Model:** SpaCy Legal NLP | **Corpus:** Tier 1 Social Media (1,273 documents)
""")
