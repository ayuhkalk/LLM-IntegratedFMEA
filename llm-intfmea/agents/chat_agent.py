"""
Chat Agent - Handles document chat functionality using RAG
"""

import streamlit as st
from datetime import datetime


class ChatAgent:
    """Agent responsible for document chat functionality"""
    
    def __init__(self, document_agent):
        self.document_agent = document_agent
        
    def render_chat_interface(self):
        """Render the chat interface tab"""
        st.header("💬 Chat with Your Documents")
        
        if not st.session_state.documents_loaded:
            st.warning("Please upload and process documents first.")
        else:
            # Build indexes if not already built
            if not st.session_state.indexes_built:
                if st.button("🔍 Build Search Indexes"):
                    with st.spinner("Building search indexes..."):
                        try:
                            st.session_state.agent.build_indexes()
                            st.session_state.indexes_built = True
                            st.success("✅ Indexes built successfully!")
                        except Exception as e:
                            st.error(f"Failed to build indexes: {e}")
            
            if st.session_state.indexes_built:
                # Predefined example questions
                st.subheader("❓ Quick Questions")
                example_questions = [
                    "What components were extracted from the documents?",
                    "What are the main subsystems identified?",
                    "What maintenance procedures are mentioned?",
                    "Which components might need regular inspection?",
                    "What safety systems are described?",
                    "What are the common failure modes mentioned?",
                    "What historical maintenance issues were found?",
                    "Which components have the highest failure risk?"
                ]
                
                cols = st.columns(4)
                for i, question in enumerate(example_questions):
                    with cols[i % 4]:
                        if st.button(question, key=f"example_{i}"):
                            st.session_state.current_question = question
                
                # Chat interface
                st.subheader("💭 Ask Your Question")
                user_question = st.text_input(
                    "Question:",
                    value=st.session_state.get('current_question', ''),
                    placeholder="Ask anything about your documents..."
                )
                
                if st.button("🔍 Ask") and user_question:
                    with st.spinner("Searching documents and generating response..."):
                        try:
                            response = st.session_state.agent.chat_with_documents(user_question)
                            
                            # Add to chat history
                            st.session_state.chat_history.append({
                                'question': user_question,
                                'answer': response['answer'],
                                'sources': response['sources'],
                                'timestamp': response['timestamp']
                            })
                            
                            # Display response
                            st.success("Response:")
                            st.write(response['answer'])
                            
                            # Show sources
                            with st.expander(f"📚 Sources ({len(response['sources'])} documents)"):
                                for i, source in enumerate(response['sources']):
                                    st.write(f"**Source {i+1}:** {source['source_document']} (Score: {source['score']:.3f})")
                                    st.write(f"*Content:* {source['content']}")
                                    st.write("---")
                        
                        except Exception as e:
                            st.error(f"Error processing question: {e}")
                
                # Chat history
                if st.session_state.chat_history:
                    st.subheader("💬 Chat History")
                    for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
                        with st.expander(f"Q: {chat['question'][:50]}..."):
                            st.write(f"**Question:** {chat['question']}")
                            st.write(f"**Answer:** {chat['answer']}")
                            st.write(f"**Time:** {chat['timestamp']}")
            else:
                st.info("Click 'Build Search Indexes' to enable document chat.")

