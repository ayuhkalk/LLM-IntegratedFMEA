"""
Agent Coordinator - Main file that orchestrates all agents and handles the Streamlit app
Combines document upload, pipeline execution, and coordination between specialized agents
"""

import streamlit as st
import pandas as pd
import json
import tempfile
import os
from pathlib import Path
from datetime import datetime
import shutil

# Import our agents and document agent
from document_agent import EnhancedIntegratedDocumentAgent
from chat_agent import ChatAgent
from visual_dashboard_agent import VisualDashboard
from fmea_generator_agent import FMEAGeneratorAgent

# Configure Streamlit
st.set_page_config(
    page_title="🔧 LLM-Integrated FMEA System",
    page_icon="🔧",
    layout="wide"
)

class AgentCoordinator:
    """Main coordinator class that manages all agents and the application flow"""
    
    def __init__(self):
        self.initialize_session_state()
        self.visual_dashboard = VisualDashboard()
        self.chat_agent = None
        self.fmea_agent = None
    
    def initialize_session_state(self):
        """Initialize all session state variables"""
        if 'agent' not in st.session_state:
            st.session_state.agent = None
        if 'documents_loaded' not in st.session_state:
            st.session_state.documents_loaded = False
        if 'indexes_built' not in st.session_state:
            st.session_state.indexes_built = False
        if 'pipeline_results' not in st.session_state:
            st.session_state.pipeline_results = None
        if 'fmea_table' not in st.session_state:
            st.session_state.fmea_table = None
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'feedback_memory' not in st.session_state:
            st.session_state.feedback_memory = {"overrides": {}, "subsystem_lexicon": [], "component_lexicon": []}

    def initialize_agent(self):
        """Initialize the EnhancedIntegratedDocumentAgent and specialized agents"""
        with st.spinner("Initializing Integrated FMEA Agent..."):
            agent = EnhancedIntegratedDocumentAgent(
                storage_dir="./streamlit_enhanced_storage",
                use_local_llm=st.session_state.get('use_local_llm', True),
                local_model=st.session_state.get('local_model', 'llama3.2:1b'),
                embedding_model=st.session_state.get('embedding_model', 'all-MiniLM-L6-v2')
            )
            st.session_state.agent = agent
            
            # Initialize specialized agents
            self.chat_agent = ChatAgent(agent)
            self.fmea_agent = FMEAGeneratorAgent(agent, self.visual_dashboard)
            
            return agent

    def process_uploaded_files(self, uploaded_files):
        """Process uploaded files and load into agent"""
        if not uploaded_files:
            return False
        
        temp_dir = Path(tempfile.mkdtemp())
        file_paths = []
        
        try:
            # Save uploaded files to temp directory
            for uploaded_file in uploaded_files:
                file_path = temp_dir / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                file_paths.append(str(file_path))
            
            # Load documents
            with st.spinner(f"Loading {len(file_paths)} documents..."):
                st.session_state.agent.load_documents(file_paths)
                st.session_state.documents_loaded = True
            
            return True
            
        except Exception as e:
            st.error(f"Error processing files: {e}")
            return False
        finally:
            # Clean up temp files
            try:
                shutil.rmtree(temp_dir)
            except:
                pass

    def run_extraction_pipeline(self):
        """Run the integrated extraction pipeline"""
        if not st.session_state.documents_loaded:
            st.error("Please upload and process documents first")
            return None
        
        with st.spinner("Running enhanced multi-agent extraction pipeline... This may take a few minutes."):
            try:
                # Get pipeline settings from sidebar
                top_k = st.session_state.get('top_k', 300)
                sim_threshold = st.session_state.get('sim_threshold', 0.80)
                use_llm = st.session_state.get('use_pipeline_llm', False)
                ollama_url = st.session_state.get('ollama_url', 'http://localhost:11434/api/generate')
                ollama_model = st.session_state.get('ollama_model', 'llama3.2:1b')
                
                results = st.session_state.agent.run_extraction_pipeline(
                    top_k=top_k,
                    sim_threshold=sim_threshold,
                    use_llm=use_llm,
                    ollama_url=ollama_url,
                    ollama_model=ollama_model
                )
                st.session_state.pipeline_results = results
                return results
            except Exception as e:
                st.error(f"Pipeline execution failed: {e}")
                return None

    def render_sidebar(self):
        """Render the sidebar configuration"""
        st.sidebar.header("⚙️Configuration")

        # LLM Settings
        st.sidebar.subheader("🤖 LLM Settings")
        use_local_llm = st.sidebar.checkbox("Use Local LLM (Ollama)", value=True)
        st.session_state.use_local_llm = use_local_llm

        if use_local_llm:
            local_models = ["llama3.2:1b", "llama3.1:8b", "gemma:2b", "mistral:7b"]
            selected_model = st.sidebar.selectbox("Local Model", local_models)
            st.session_state.local_model = selected_model
        else:
            st.sidebar.info("Using OpenAI GPT-4. Make sure OPENAI_API_KEY is set.")

        # Embedding Settings
        embedding_models = [
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "../local_models/bge-small-en-v1.5"
        ]
        selected_embedding = st.sidebar.selectbox("Embedding Model", embedding_models)
        st.session_state.embedding_model = selected_embedding

        # Pipeline Settings
        st.sidebar.subheader("⚙️ Pipeline Settings")
        use_pipeline_llm = st.sidebar.checkbox("Use LLM for Classification", value=False)
        st.session_state.use_pipeline_llm = use_pipeline_llm

        if use_pipeline_llm:
            ollama_url = st.sidebar.text_input("Ollama URL", value="http://localhost:11434/api/generate")
            ollama_model = st.sidebar.text_input("Classification Model", value="llama3.2:1b")
            st.session_state.ollama_url = ollama_url
            st.session_state.ollama_model = ollama_model

        top_k = st.sidebar.slider("Candidates (top-k)", 50, 1000, 300, 50)
        st.session_state.top_k = top_k

        sim_threshold = st.sidebar.slider("Merge similarity", 0.6, 0.95, 0.8, 0.01)
        st.session_state.sim_threshold = sim_threshold

        # Initialize agent button
        if st.sidebar.button("🚀 Initialize Agent"):
            self.initialize_agent()
            st.sidebar.success("Agent initialized!")

        # Enhanced sidebar status and tips
        st.sidebar.markdown("---")
        st.sidebar.header("🔧 System Status")
        
        if st.session_state.agent:
            st.sidebar.success("✅ Agent Ready")
            
            if st.session_state.documents_loaded:
                summary = st.session_state.agent.get_document_summary()
                st.sidebar.info(f"📄 {summary.get('total_documents', 0)} documents loaded")
            
            if st.session_state.pipeline_results:
                validated_count = len(st.session_state.pipeline_results['validated'])
                st.sidebar.info(f"⚙️ {validated_count} terms extracted")
            
            if st.session_state.fmea_table is not None:
                fmea_count = len(st.session_state.fmea_table)
                st.sidebar.info(f"🎯 {fmea_count} FMEA entries generated")
            
            if st.session_state.indexes_built:
                st.sidebar.info("🔍 Search indexes built")
            
            if st.session_state.chat_history:
                st.sidebar.info(f"💬 {len(st.session_state.chat_history)} chat interactions")
        else:
            st.sidebar.warning("⚠️ Agent not initialized")
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("**💡 Tips:**")
        st.sidebar.markdown("• Upload historical maintenance data for intelligent FMEA analysis")
        st.sidebar.markdown("• Include Excel/CSV files with component lists for better extraction")
        st.sidebar.markdown("• Historical data enables severity extraction and detailed recommendations")
        st.sidebar.markdown("• Use feedback system to improve classification accuracy")
        st.sidebar.markdown("• Build search indexes for advanced document chat features")
        st.sidebar.markdown("• Export enhanced FMEA results with risk prioritization")
        st.sidebar.markdown("• Review high-risk items (RPN ≥ 200) for immediate action")

        # Debug information (only show in development)
        if st.sidebar.checkbox("Show Debug Info"):
            st.sidebar.subheader("Debug Information")
            st.sidebar.write("Session State Keys:", list(st.session_state.keys()))
            
            if st.session_state.pipeline_results:
                st.sidebar.write("Pipeline Results Keys:", list(st.session_state.pipeline_results.keys()))
            
            if st.session_state.fmea_table is not None:
                st.sidebar.write("FMEA Table Shape:", st.session_state.fmea_table.shape)
                st.sidebar.write("FMEA Columns:", list(st.session_state.fmea_table.columns))
    
    def sync_agents(self):
        if st.session_state.agent:
            self.chat_agent = ChatAgent(st.session_state.agent)
            self.fmea_agent = FMEAGeneratorAgent(st.session_state.agent, self.visual_dashboard)

    def render_upload_tab(self):
        """Render Tab 1: Upload & Process"""
        st.header("📋 Document Upload and Processing")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload Documents (including historical maintenance data)",
            type=['pdf', 'docx', 'xlsx', 'csv', 'txt'],
            accept_multiple_files=True,
            help="Supported formats: PDF, DOCX, XLSX, CSV, TXT. Include historical maintenance data for enhanced FMEA analysis."
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 Process Documents", disabled=not uploaded_files):
                if not st.session_state.agent:
                    st.session_state.agent = self.initialize_agent()
                
                success = self.process_uploaded_files(uploaded_files)
                if success:
                    st.success(f"✅ Processed {len(uploaded_files)} documents successfully!")
                    
                    # Show document summary
                    summary = st.session_state.agent.get_document_summary()
                    with st.expander("📊 Document Summary"):
                        st.json(summary)
                else:
                    st.error("Failed to process documents")
        
        with col2:
            # Status indicators
            st.subheader("🔧 Processing Status")
            
            if st.session_state.agent:
                st.success("🤖 Agent: Ready")
            else:
                st.warning("🤖 Agent: Not initialized")
            
            if st.session_state.documents_loaded:
                st.success("📄 Documents: Loaded")
            else:
                st.warning("📄 Documents: Not loaded")
            
            if st.session_state.pipeline_results:
                st.success("⚙️ Pipeline: Complete")
            else:
                st.warning("⚙️ Pipeline: Not run")
            
            if st.session_state.fmea_table is not None:
                st.success("🎯 FMEA: Generated")
            else:
                st.warning("🎯 Enhanced FMEA: Not generated")

    def render_pipeline_tab(self):
        """Render Tab 2: Multi-Agent Pipeline"""
        st.header("🤖 Multi-Agent Extraction Pipeline")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("""
            **Pipeline Stages:**
            1. **Ingestion** - Load and chunk documents with full text preservation
            2. **Candidate Extraction** - TF-IDF + NLP + Pattern matching
            3. **Classification** - Heuristic or LLM-based classification
            4. **Normalization** - Similarity-based term grouping
            5. **Validation** - Confidence filtering and ranking
            6. **📋 Historical Analysis** - Extract patterns from maintenance data
            7. **🎯 Intelligent FMEA** - Generate FMEA with historical integration
            """)
        
        with col2:
            if st.button("🚀 Run Pipeline", disabled=not st.session_state.documents_loaded):
                results = self.run_extraction_pipeline()
                if results:
                    st.success("✅ Pipeline completed!")
        
        # Display pipeline results using visual dashboard
        if st.session_state.pipeline_results:
            self.visual_dashboard.render_pipeline_tab_visualizations(st.session_state.pipeline_results)

    def render_export_tab(self):
        """Render Tab 5: Results & Export"""
        st.header("📤 Export Results")
        
        if st.session_state.pipeline_results:
            results = st.session_state.pipeline_results
            
            # Use visual dashboard for metrics
            self.visual_dashboard.render_export_tab_visualizations(results)
            
            # Feedback and corrections
            st.subheader("🔧 Feedback & Corrections")
            with st.expander("Correct Classification Results"):
                st.markdown("**Adjust term classifications and save feedback for future runs:**")
                
                validated_df = results['validated'].copy()
                validated_df = self.visual_dashboard.apply_feedback(validated_df, st.session_state.feedback_memory)
                
                # Show top terms for feedback
                feedback_df = validated_df.head(20)  # Show top 20 for feedback
                
                for idx, row in feedback_df.iterrows():
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.write(f"**{row['canonical']}** (confidence: {row['final_confidence']:.3f})")
                    with col2:
                        current_label = row['label']
                        new_label = st.selectbox(
                            f"Label for '{row['canonical']}'",
                            options=['Subsystem', 'Component', 'Irrelevant'],
                            index=['Subsystem', 'Component', 'Irrelevant'].index(current_label),
                            key=f"feedback_{idx}"
                        )
                        if new_label != current_label:
                            st.session_state.feedback_memory['overrides'][row['canonical']] = new_label
                    with col3:
                        if st.button(f"Reset", key=f"reset_{idx}"):
                            if row['canonical'] in st.session_state.feedback_memory['overrides']:
                                del st.session_state.feedback_memory['overrides'][row['canonical']]
                
                if st.button("💾 Save Feedback"):
                    st.success("Feedback saved! It will be applied in future pipeline runs.")
            
            # Export options
            st.subheader("📁 Export Options")
            
            # Apply current feedback to results
            final_results = validated_df.copy()
            final_results = self.visual_dashboard.apply_feedback(final_results, st.session_state.feedback_memory)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Export validated terms
                csv_data = final_results.to_csv(index=False)
                st.download_button(
                    "📄 Download Validated Terms CSV",
                    csv_data,
                    "validated_terms.csv",
                    "text/csv"
                )
            
            with col2:
                # Export full pipeline results
                try:
                    # Prepare JSON export
                    export_data = {}
                    for key, value in results.items():
                        if isinstance(value, pd.DataFrame):
                            if key == 'validated':
                                export_data[key] = self.visual_dashboard.apply_feedback(value, st.session_state.feedback_memory).to_dict('records')
                            else:
                                export_data[key] = value.to_dict('records')
                        else:
                            export_data[key] = value
                    
                    json_data = json.dumps(export_data, indent=2, default=str)
                    st.download_button(
                        "🗂️ Download Full Results JSON",
                        json_data,
                        "enhanced_pipeline_results.json",
                        "application/json"
                    )
                except Exception as e:
                    st.error(f"Error preparing JSON export: {e}")
            
            with col3:
                # Export Enhanced FMEA table
                if st.session_state.fmea_table is not None:
                    fmea_csv = st.session_state.fmea_table.to_csv(index=False)
                    st.download_button(
                        "🎯 Download FMEA CSV",
                        fmea_csv,
                        "enhanced_fmea_analysis.csv",
                        "text/csv"
                    )
                else:
                    st.button("🎯 No FMEA Generated", disabled=True)
            
            # Performance metrics
            with st.expander("Pipeline Performance Metrics"):
                metadata = results.get('extraction_metadata', {})
                st.json(metadata)
            
        else:
            st.info("No pipeline results available. Please run the extraction pipeline first.")

    def run_app(self):
        """Main application runner"""
        # App header
        st.title("🔧 LLM-Integrated FMEA System")
        st.markdown("Multi-agent pipeline for component extraction + RAG-based document chat + FMEA Analysis")

        # Render sidebar
        self.render_sidebar()

        # Main Content Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📤 Upload & Process", 
            "🤖 Multi-Agent Pipeline", 
            "🎯 FMEA Analysis",
            "💬 Chat with Documents", 
            "📈 Results & Export"
        ])
        self.sync_agents()

        # Tab 1: Upload & Process
        with tab1:
            self.render_upload_tab()

        # Tab 2: Multi-Agent Pipeline
        with tab2:
            self.render_pipeline_tab()

        # Tab 3: Enhanced FMEA Analysis
        with tab3:
            if self.fmea_agent:
                self.fmea_agent.render_fmea_tab()
            else:
                st.warning("Please initialize the agent first.")

        # Tab 4: Chat with Documents
        with tab4:
            if self.chat_agent:
                self.chat_agent.render_chat_interface()
            else:
                st.warning("Please initialize the agent first.")

        # Tab 5: Results & Export
        with tab5:
            self.render_export_tab()

        # Footer
        st.markdown("---")
        st.markdown(
            """
            **LLM-Integrated FMEA System**
            
            🚀 **Core Features:** TF-IDF extraction, NLP processing, pattern matching, LLM classification, hybrid search, RAG chat  
            🎯 **Enhanced FMEA:** Historical data analysis, intelligent severity extraction, detailed recommendations  
            📚 **Supports:** PDF, DOCX, XLSX, CSV documents with historical maintenance data integration  
            🤖 **Models:** Local (Ollama)
            ⚙️ **Pipeline:** Ingestion → Extraction → Classification → Normalization → Validation → Historical Analysis → Intelligent FMEA
            """
        )


def main():
    """Main entry point"""
    coordinator = AgentCoordinator()
    coordinator.run_app()


if __name__ == "__main__":
    main()

