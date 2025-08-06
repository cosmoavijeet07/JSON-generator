import nltk
import ssl

# Fix SSL certificate issues with NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data at startup
def download_nltk_data():
    """Download required NLTK data packages"""
    packages = ['punkt', 'punkt_tab', 'stopwords', 'averaged_perceptron_tagger']
    for package in packages:
        try:
            nltk.download(package, quiet=True)
        except:
            pass

# Call this before initializing services
download_nltk_data()

import sys
import os
from pathlib import Path

# Setup NLTK data directory
import nltk
nltk_data_dir = Path.home() / "nltk_data"
nltk_data_dir.mkdir(exist_ok=True)
nltk.data.path.append(str(nltk_data_dir))

# Download required NLTK data at startup
def initialize_nltk():
    """Initialize NLTK with required data packages"""
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    packages = ['punkt', 'punkt_tab', 'stopwords', 'averaged_perceptron_tagger']
    for package in packages:
        try:
            nltk.data.find(f'tokenizers/{package}')
        except LookupError:
            try:
                nltk.download(package, quiet=True, download_dir=str(nltk_data_dir))
            except:
                pass

# Initialize NLTK before anything else
initialize_nltk()


import streamlit as st
import json
import os
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from core import (
    LLMInterface, SchemaValidator, JSONExtractor, PromptEngine,
    LoggerService, SessionManager, TokenEstimator, EmbeddingService,
    PipelineManager
)

from processors import (
    TextProcessor, SchemaProcessor, ChunkManager, MergeManager
)

from pipelines import SimplePipeline, ExtensivePipeline

from utils import MetricsCollector, FileHandler, DisplayManager

# Initialize services
@st.cache_resource
def init_services():
    """Initialize all services"""
    return {
        "llm": LLMInterface(),
        "validator": SchemaValidator(),
        "extractor": JSONExtractor(),
        "prompt_engine": PromptEngine(),
        "logger": LoggerService(),
        "session_manager": SessionManager(),
        "token_estimator": TokenEstimator(),
        "embedding_service": EmbeddingService(),
        "text_processor": TextProcessor(),
        "schema_processor": SchemaProcessor(),
        "chunk_manager": ChunkManager(),
        "merge_manager": MergeManager(),
        "metrics": MetricsCollector(),
        "file_handler": FileHandler(),
        "display": DisplayManager()
    }

def main():
    st.set_page_config(
        page_title="Advanced JSON Extractor",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 Advanced JSON Extraction System")
    st.markdown("Convert unstructured text to structured JSON with AI")
    
    # Initialize services
    services = init_services()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Model selection
        model_options = {
            "GPT 4.1": "gpt-4.1-2025-04-14",
            "GPT O4 Mini": "o4-mini-2025-04-16",
            "Claude Sonnet 4": "claude-sonnet-4-20250514"
        }
        
        selected_model = st.selectbox(
            "Select Model",
            options=list(model_options.keys()),
            index=1
        )
        
        model = model_options[selected_model]
        
        # Pipeline selection will be automatic based on analysis
        
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📄 Input")
        
        # Schema input
        st.subheader("JSON Schema")
        schema_input_method = st.radio(
            "Schema Input Method",
            ["Upload File", "Paste JSON"],
            horizontal=True
        )
        
        schema = None
        if schema_input_method == "Upload File":
            schema_file = st.file_uploader("Upload JSON Schema", type=["json"])
            if schema_file:
                schema = json.loads(schema_file.read())
        else:
            schema_text = st.text_area(
                "Paste JSON Schema",
                height=200,
                placeholder='{"type": "object", "properties": {...}}'
            )
            if schema_text:
                try:
                    schema = json.loads(schema_text)
                except:
                    st.error("Invalid JSON schema")
        
        # Text input
        st.subheader("Text Document")
        text_input_method = st.radio(
            "Text Input Method",
            ["Upload File", "Paste Text"],
            horizontal=True
        )
        
        text = None
        if text_input_method == "Upload File":
            text_file = st.file_uploader("Upload Text File", type=["txt", "md"])
            if text_file:
                text = text_file.read().decode()
        else:
            text = st.text_area(
                "Paste Text",
                height=200,
                placeholder="Enter your unstructured text here..."
            )
    
    with col2:
        st.header("📊 Analysis & Output")
        
        if schema and text:
            # Analyze inputs
            if st.button("🔍 Analyze", type="primary"):
                with st.spinner("Analyzing..."):
                    # Token estimation
                    token_metrics = services["token_estimator"].analyze_content(
                        text, schema, model
                    )
                    
                    # Schema analysis
                    schema_metrics = services["schema_processor"].analyze_schema(schema)
                    
                    # Display metrics
                    st.subheader("📈 Analysis Results")
                    
                    col2_1, col2_2 = st.columns(2)
                    
                    with col2_1:
                        st.metric("Total Tokens", token_metrics["total_tokens"])
                        st.metric("Schema Complexity", f"{schema_metrics['complexity_score']:.1f}")
                        st.metric("Schema Depth", schema_metrics["max_depth"])
                    
                    with col2_2:
                        st.metric("Text Tokens", token_metrics["text_tokens"])
                        st.metric("Schema Fields", schema_metrics["total_fields"])
                        st.metric("Estimated Chunks", token_metrics["estimated_chunks"])
                    
                    # Pipeline recommendation
                    recommended = token_metrics["recommended_pipeline"]
                    st.info(f"💡 Recommended Pipeline: **{recommended.upper()}**")
                    
                    # Store in session state
                    st.session_state["analysis"] = {
                        "token_metrics": token_metrics,
                        "schema_metrics": schema_metrics,
                        "recommended_pipeline": recommended
                    }
    
    # Extraction section
    if schema and text and "analysis" in st.session_state:
        st.header("🚀 Extraction")
        
        col3, col4 = st.columns([1, 2])
        
        with col3:
            # Pipeline selection
            pipeline_type = st.radio(
                "Select Pipeline",
                ["simple", "extensive"],
                index=0 if st.session_state["analysis"]["recommended_pipeline"] == "simple" else 1,
                help="Simple: Fast, single-pass extraction\nExtensive: Multi-pass with chunking and merging"
            )
            
            # Advanced options for extensive pipeline
            if pipeline_type == "extensive":
                num_passes = st.slider("Number of Passes", 2, 5, 3)
            else:
                num_passes = 1
            
            # Extract button
            if st.button("🎯 Extract JSON", type="primary"):
                # Create session
                session_id = services["session_manager"].create_session({
                    "model": model,
                    "pipeline": pipeline_type,
                    "timestamp": str(os.path.getctime(Path.cwd()))
                })
                
                services["logger"].start_session(session_id)
                
                # Prepare input
                input_data = {
                    "schema": schema,
                    "text": text,
                    "model": model,
                    "session_id": session_id,
                    "num_passes": num_passes
                }
                
                # Run pipeline
                with st.spinner(f"Running {pipeline_type} pipeline..."):
                    if pipeline_type == "simple":
                        pipeline = SimplePipeline(
                            logger=services["logger"],
                            session_manager=services["session_manager"],
                            llm_interface=services["llm"],
                            json_extractor=services["extractor"],
                            schema_validator=services["validator"],
                            prompt_engine=services["prompt_engine"],
                            token_estimator=services["token_estimator"]
                        )
                    else:
                        pipeline = ExtensivePipeline(
                            logger=services["logger"],
                            session_manager=services["session_manager"],
                            llm_interface=services["llm"],
                            json_extractor=services["extractor"],
                            schema_validator=services["validator"],
                            prompt_engine=services["prompt_engine"],
                            token_estimator=services["token_estimator"],
                            text_processor=services["text_processor"],
                            schema_processor=services["schema_processor"],
                            chunk_manager=services["chunk_manager"],
                            merge_manager=services["merge_manager"],
                            embedding_service=services["embedding_service"]
                        )
                    
                    result = pipeline.run(input_data)
                
                services["logger"].end_session()
                
                # Store result
                st.session_state["result"] = result
                st.session_state["session_id"] = session_id
        
        with col4:
            # Display results
            if "result" in st.session_state:
                result = st.session_state["result"]
                
                if result["success"]:
                    st.success("✅ Extraction Successful!")
                    
                    # Display JSON
                    st.subheader("📋 Extracted JSON")
                    st.json(result["output"])
                    
                    # Display metrics
                    if "metrics" in result:
                        st.subheader("📊 Metrics")
                        st.text(services["display"].format_metrics(result["metrics"]))
                    
                    # Download buttons
                    col4_1, col4_2 = st.columns(2)
                    
                    with col4_1:
                        st.download_button(
                            "📥 Download JSON",
                            data=json.dumps(result["output"], indent=2),
                            file_name=f"{st.session_state['session_id']}_output.json",
                            mime="application/json"
                        )
                    
                    with col4_2:
                        # Get logs
                        log_dir = Path("logs") / st.session_state["session_id"]
                        if log_dir.exists():
                            log_files = list(log_dir.glob("*.log"))
                            if log_files:
                                log_content = "\n\n".join(
                                    f.read_text() for f in log_files
                                )
                                st.download_button(
                                    "📄 Download Logs",
                                    data=log_content,
                                    file_name=f"{st.session_state['session_id']}_logs.txt",
                                    mime="text/plain"
                                )
                else:
                    st.error(f"❌ Extraction Failed: {result.get('error', 'Unknown error')}")
                    
                    if "metrics" in result:
                        st.subheader("📊 Metrics")
                        st.text(services["display"].format_metrics(result["metrics"]))
                        
if __name__ == "__main__":
    main()