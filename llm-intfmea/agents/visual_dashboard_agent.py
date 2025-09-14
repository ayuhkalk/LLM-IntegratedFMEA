"""
Visual Dashboard - Handles all visualization components
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


class VisualDashboard:
    """Agent responsible for creating visualizations and dashboards"""
    
    def __init__(self):
        pass
    
    def apply_feedback(self, df: pd.DataFrame, memory: dict) -> pd.DataFrame:
        """Apply user feedback to classification results"""
        df = df.copy()
        overrides = memory.get("overrides", {})
        for term, label in overrides.items():
            mask = df["canonical"].str.lower().eq(term.lower()) | df["term"].str.lower().eq(term.lower())
            df.loc[mask, "label"] = label
            df.loc[mask, "final_confidence"] = df.loc[mask, "final_confidence"].clip(lower=0.85)
        
        subs = set(memory.get("subsystem_lexicon", []))
        comps = set(memory.get("component_lexicon", []))
        df.loc[df["canonical"].str.lower().isin({s.lower() for s in subs}), "label"] = "Subsystem"
        df.loc[df["canonical"].str.lower().isin({c.lower() for c in comps}), "label"] = "Component"
        return df

    def create_pipeline_visualizations(self, results):
        """Create visualizations from pipeline results"""
        if not results or results['validated'].empty:
            return None
        
        df = results['validated'].copy()
        
        # Apply feedback
        df = self.apply_feedback(df, st.session_state.feedback_memory)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Label distribution
            label_counts = df['label'].value_counts()
            fig_labels = px.pie(
                values=label_counts.values, 
                names=label_counts.index,
                title="Distribution of Extracted Terms"
            )
            st.plotly_chart(fig_labels, use_container_width=True)
            
            # Confidence distribution
            fig_conf = px.histogram(
                df, 
                x='final_confidence', 
                nbins=20, 
                title="Confidence Score Distribution"
            )
            st.plotly_chart(fig_conf, use_container_width=True)
        
        with col2:
            # Top terms by confidence
            top_terms = df.nlargest(15, 'final_confidence')
            fig_top = px.bar(
                top_terms, 
                x='final_confidence', 
                y='canonical', 
                color='label',
                orientation='h',
                title="Top Terms by Confidence"
            )
            st.plotly_chart(fig_top, use_container_width=True)
            
            # Score vs Confidence scatter
            fig_scatter = px.scatter(
                df, 
                x='score', 
                y='final_confidence', 
                color='label',
                hover_data=['canonical'],
                title="Score vs Confidence"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        return df

    def create_enhanced_fmea_visualizations(self, fmea_df):
        """Create enhanced FMEA-specific visualizations"""
        if fmea_df.empty:
            return
        
        # Enhanced metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            high_risk = len(fmea_df[fmea_df["RPN"] >= 200])
            st.metric("High Risk Items", high_risk, help="RPN ≥ 200")
        with col2:
            avg_severity = fmea_df["Severity (S)"].mean()
            st.metric("Avg Severity", f"{avg_severity:.1f}")
        with col3:
            critical_components = len(fmea_df[fmea_df["Severity (S)"] >= 8])
            st.metric("Critical Components", critical_components, help="Severity ≥ 8")
        with col4:
            historical_count = len(fmea_df[fmea_df["Data Source"].str.contains("Historical", na=False)])
            st.metric("Historical Data Items", historical_count)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 RPN Distribution by Data Source")
            fig_rpn_source = px.histogram(
                fmea_df, 
                x='RPN', 
                color='Data Source',
                nbins=20, 
                title="Risk Priority Number Distribution by Data Source"
            )
            st.plotly_chart(fig_rpn_source, use_container_width=True)
            
            st.subheader("🎯 Top Risk Components")
            top_risks = fmea_df.nlargest(10, 'RPN')
            fig_risks = px.bar(
                top_risks, 
                x='RPN', 
                y='Component', 
                color='Severity Level',
                orientation='h',
                title="Components by Risk Priority Number"
            )
            st.plotly_chart(fig_risks, use_container_width=True)
        
        with col2:
            st.subheader("🔍 3D Risk Factor Analysis")
            fig_scatter = px.scatter_3d(
                fmea_df, 
                x='Severity (S)', 
                y='Occurrence (O)', 
                z='Detection (D)',
                size='RPN', 
                color='Severity Level', 
                hover_data=['Component'],
                title="3D Risk Analysis (S×O×D)"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            st.subheader("📋 Severity Level Distribution")
            severity_counts = fmea_df['Severity Level'].value_counts()
            fig_severity = px.pie(
                values=severity_counts.values,
                names=severity_counts.index,
                title="Distribution by Severity Level"
            )
            st.plotly_chart(fig_severity, use_container_width=True)

    def render_pipeline_tab_visualizations(self, results):
        """Render Tab 2 visualizations and results table"""
        if results:
            # Enhanced metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Candidates", len(results['candidates']))
            with col2:
                st.metric("Validated Terms", len(results['validated']))
            with col3:
                components_count = len(results['validated'][results['validated']['label'] == 'Component'])
                st.metric("Components", components_count)
            with col4:
                subsystems_count = len(results['validated'][results['validated']['label'] == 'Subsystem'])
                st.metric("Subsystems", subsystems_count)
            
            # Visualizations
            st.subheader("📊 Extraction Results Visualization")
            self.create_pipeline_visualizations(results)
            
            # Results table
            st.subheader("✅ Validated Terms")
            validated_df = results['validated'].copy()
            validated_df = self.apply_feedback(validated_df, st.session_state.feedback_memory)
            
            # Filters
            col1, col2 = st.columns(2)
            with col1:
                label_filter = st.multiselect("Filter by Label", 
                                            options=validated_df['label'].unique(),
                                            default=validated_df['label'].unique())
            with col2:
                min_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.5, 0.05)
            
            filtered_df = validated_df[
                (validated_df['label'].isin(label_filter)) &
                (validated_df['final_confidence'] >= min_confidence)
            ]
            
            st.dataframe(filtered_df[['canonical', 'label', 'final_confidence', 'score']], use_container_width=True)
            
            # Pipeline logs
            with st.expander("📋 Pipeline Logs"):
                for log in results.get('logs', []):
                    st.write(log)

    def render_export_tab_visualizations(self, results):
        """Render Tab 5 export visualizations and metrics"""
        if results:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Files Processed", results['extraction_metadata']['total_files'])
            with col2:
                st.metric("Text Chunks", results['extraction_metadata']['total_chunks'])
            with col3:
                st.metric("Candidates Found", results['extraction_metadata']['total_candidates'])
            with col4:
                validated_count = len(results['validated'])
                st.metric("Validated Terms", validated_count)
            
            # Enhanced metrics if FMEA is available
            if st.session_state.fmea_table is not None:
                st.subheader("🎯 FMEA Metrics")
                fmea_df = st.session_state.fmea_table
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("FMEA Entries", len(fmea_df))
                with col2:
                    historical_count = len(fmea_df[fmea_df["Data Source"].str.contains("Historical", na=False)])
                    st.metric("Historical Data Items", historical_count)
                with col3:
                    high_severity = len(fmea_df[fmea_df["Severity (S)"] >= 8])
                    st.metric("High Severity Items", high_severity)
                with col4:
                    critical_rpn = len(fmea_df[fmea_df["RPN"] >= 200])
                    st.metric("Critical RPN Items", critical_rpn)

