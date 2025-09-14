"""
FMEA Generator Agent - Handles FMEA analysis and generation
"""

import streamlit as st
import pandas as pd


class FMEAGeneratorAgent:
    """Agent responsible for FMEA generation and analysis"""
    
    def __init__(self, document_agent, visual_dashboard):
        self.document_agent = document_agent
        self.visual_dashboard = visual_dashboard
    
    def apply_feedback(self, df: pd.DataFrame, memory: dict) -> pd.DataFrame:
        """Apply user feedback to classification results"""
        return self.visual_dashboard.apply_feedback(df, memory)
    
    def display_enhanced_fmea_details(self, fmea_df):
        """Display detailed FMEA information with enhanced features"""
        if fmea_df.empty:
            return
        
        st.subheader("🔍 Enhanced FMEA Analysis Details")
        
        # Risk Level Filtering
        risk_filter = st.selectbox(
            "Filter by Risk Level:",
            ["All Risk Levels", "High Risk (RPN ≥ 200)", "Medium Risk (RPN 50-199)", "Low Risk (RPN < 50)"],
            index=0
        )
        
        # Start with full dataset (no source filter applied)
        filtered_fmea = fmea_df.copy()
        
        # Apply risk filter
        if risk_filter == "High Risk (RPN ≥ 200)":
            filtered_fmea = filtered_fmea[filtered_fmea["RPN"] >= 200]
        elif risk_filter == "Medium Risk (RPN 50-199)":
            filtered_fmea = filtered_fmea[(filtered_fmea["RPN"] >= 50) & (filtered_fmea["RPN"] < 200)]
        elif risk_filter == "Low Risk (RPN < 50)":
            filtered_fmea = filtered_fmea[filtered_fmea["RPN"] < 50]
        
        # Sort by RPN descending
        filtered_fmea = filtered_fmea.sort_values("RPN", ascending=False)
        
        st.write(f"**Showing {len(filtered_fmea)} entries (filtered from {len(fmea_df)} total)**")
        
        # Display FMEA table with enhanced formatting
        st.dataframe(
            filtered_fmea.style.format({
                "Severity (S)": "{:.0f}",
                "Occurrence (O)": "{:.0f}",
                "Detection (D)": "{:.0f}",
                "RPN": "{:.0f}"
            }).background_gradient(subset=["RPN"], cmap="Reds"),
            use_container_width=True,
            height=400
        )
        
        # Detailed view for high-risk items
        high_risk_items = filtered_fmea[filtered_fmea["RPN"] >= 200]
        if not high_risk_items.empty:
            st.subheader("⚠️ High Risk Items - Detailed Analysis")
            
            for idx, row in high_risk_items.iterrows():
                with st.expander(f"🚨 {row['Component']} - {row['Potential Failure Mode']} (RPN: {row['RPN']})"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**🎯 Risk Assessment:**")
                        st.write(f"• **Severity:** {row['Severity (S)']} ({row['Severity Level']}) - {row['Severity Reasoning']}")
                        st.write(f"• **Occurrence:** {row['Occurrence (O)']} - {row['Occurrence Reasoning']}")
                        st.write(f"• **Detection:** {row['Detection (D)']} - {row['Detection Reasoning']}")
                        
                        st.write("**🔍 Failure Analysis:**")
                        st.write(f"• **Causes:** {row['Potential Causes']}")
                        st.write(f"• **Effects:** {row['Potential Effects']}")
                        st.write(f"• **Data Source:** {row['Data Source']}")
                    
                    with col2:
                        st.write("**🛠️ Detailed Recommended Actions:**")
                        actions = row['Detailed Recommended Actions'].split(" | ")
                        for action in actions:
                            if action.strip():
                                st.write(f"• {action.strip()}")

    def render_fmea_tab(self):
        """Render the complete FMEA analysis tab"""
        st.header("🎯 FMEA Analysis & Intelligent Risk Assessment")
        
        if st.session_state.pipeline_results:
            validated_df = st.session_state.pipeline_results['validated'].copy()
            validated_df = self.apply_feedback(validated_df, st.session_state.feedback_memory)
            components_df = validated_df[validated_df['label'] == 'Component']
            
            if not components_df.empty:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("🔧 Generate FMEA Table")
                    
                    if st.button("🎯 Generate FMEA", use_container_width=True):
                        with st.spinner("Generating FMEA with historical data analysis..."):
                            try:
                                fmea_table = st.session_state.agent.generate_intelligent_fmea_table(components_df)
                                st.session_state.fmea_table = fmea_table
                                st.success(f"✅ Generated FMEA table with {len(fmea_table)} entries!")
                            except Exception as e:
                                st.error(f"Failed to generate FMEA: {e}")
                
                with col2:
                    st.subheader("📊 Component Overview")
                    st.metric("Components Found", len(components_df))
                    
                    if st.session_state.fmea_table is not None:
                        fmea_df = st.session_state.fmea_table
                        avg_rpn = fmea_df['RPN'].mean()
                        max_rpn = fmea_df['RPN'].max()
                        high_risk_count = len(fmea_df[fmea_df['RPN'] > 100])
                        
                        st.metric("Average RPN", f"{avg_rpn:.1f}")
                        st.metric("Maximum RPN", max_rpn)
                        st.metric("High Risk Items (RPN>100)", high_risk_count)
                
                # Display FMEA table if generated
                if st.session_state.fmea_table is not None:
                    fmea_df = st.session_state.fmea_table
                    
                    # Enhanced FMEA Visualizations
                    st.subheader("📈 Risk Analysis Visualizations")
                    self.visual_dashboard.create_enhanced_fmea_visualizations(fmea_df)
                    
                    # Detailed FMEA Display
                    self.display_enhanced_fmea_details(fmea_df)
                    
                    # Export FMEA table
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        csv_data = fmea_df.to_csv(index=False)
                        st.download_button(
                            "💾 Download FMEA CSV",
                            csv_data,
                            "enhanced_fmea_analysis.csv",
                            "text/csv",
                            use_container_width=True
                        )
                    
                    with col2:
                        high_risk_items = fmea_df[fmea_df['RPN'] >= 200]
                        if not high_risk_items.empty:
                            high_risk_csv = high_risk_items.to_csv(index=False)
                            st.download_button(
                                "🚨 Download High Risk Items CSV",
                                high_risk_csv,
                                "high_risk_fmea_items.csv",
                                "text/csv",
                                use_container_width=True
                            )
                        else:
                            st.button("🚨 No High Risk Items", disabled=True, use_container_width=True)
                    
                    with col3:
                        json_data = fmea_df.to_json(orient="records", indent=2)
                        st.download_button(
                            "📋 Download FMEA JSON",
                            json_data,
                            "enhanced_fmea_analysis.json",
                            "application/json",
                            use_container_width=True
                        )
                
            else:
                st.warning("No components found. Please run the extraction pipeline first.")
        else:
            st.warning("Please run the extraction pipeline first in the 'Multi-Agent Pipeline' tab.")

