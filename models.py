import openai
import streamlit as st
import tiktoken
from typing import Dict, Any, List, Optional
import time
import json
from config import Config

class ModelInterface:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.config = Config()
    
    def count_tokens(self, text: str, model: str = "gpt-4") -> int:
        """Count tokens in text"""
        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except:
            # Fallback estimation
            return len(text.split()) * 1.3
    
    def estimate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Estimate API cost"""
        model_config = self.config.MODELS.get(model, {})
        input_cost = (input_tokens / 1000) * model_config.get('cost_per_1k_input', 0)
        output_cost = (output_tokens / 1000) * model_config.get('cost_per_1k_output', 0)
        return input_cost + output_cost
    
    def call_model(self, 
                   prompt: str, 
                   model: str, 
                   max_tokens: int = 4000,
                   temperature: float = 0.1,
                   json_mode: bool = True) -> Dict[str, Any]:
        """Call OpenAI model with retry logic"""
        
        messages = [
            {
                "role": "system",
                "content": "You are an expert at extracting structured data from text and converting it to valid JSON format. Always respond with valid JSON only."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ]
        
        input_tokens = self.count_tokens(str(messages), model)
        
        for attempt in range(self.config.MAX_RETRIES):
            try:
                response_format = {"type": "json_object"} if json_mode else None
                
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_format=response_format
                )
                
                output_tokens = response.usage.completion_tokens
                cost = self.estimate_cost(input_tokens, output_tokens, model)
                
                return {
                    'success': True,
                    'content': response.choices[0].message.content,
                    'usage': {
                        'input_tokens': input_tokens,
                        'output_tokens': output_tokens,
                        'total_tokens': input_tokens + output_tokens,
                        'estimated_cost': cost
                    },
                    'attempt': attempt + 1
                }
                
            except Exception as e:
                if attempt == self.config.MAX_RETRIES - 1:
                    return {
                        'success': False,
                        'error': str(e),
                        'attempt': attempt + 1
                    }
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return {'success': False, 'error': 'Max retries exceeded'}
    
    def validate_json_output(self, output: str) -> tuple[bool, Optional[dict], str]:
        """Validate and parse JSON output"""
        try:
            # Clean the output
            cleaned_output = output.strip()
            if cleaned_output.startswith('```'):
                cleaned_output = cleaned_output[7:]
            if cleaned_output.endswith('```'):
                cleaned_output = cleaned_output[:-3]
            cleaned_output = cleaned_output.strip()
            
            # Parse JSON
            json_data = json.loads(cleaned_output)
            return True, json_data, "Valid JSON"
        except json.JSONDecodeError as e:
            return False, None, f"JSON Parse Error: {str(e)}"
        except Exception as e:
            return False, None, f"Validation Error: {str(e)}"
    
    def get_model_recommendation(self, text_length: int, schema_complexity: str) -> str:
        """Recommend best model based on input characteristics"""
        if schema_complexity == "high" or text_length > 50000:
            return "gpt-4.1"
        elif text_length < 10000 and schema_complexity == "low":
            return "gpt-4o-mini"
        else:
            return "o3"

class ModelSelector:
    @staticmethod
    def render_model_selection():
        """Render model selection UI with detailed comparison"""
        st.subheader("🤖 Model Selection")
        
        config = Config()
        
        # Model comparison table
        st.write("### Model Comparison")
        
        comparison_data = []
        for model_key, model_info in config.MODELS.items():
            comparison_data.append({
                "Model": model_info['name'],
                "Input Cost (per 1K tokens)": f"${model_info['cost_per_1k_input']:.5f}",
                "Output Cost (per 1K tokens)": f"${model_info['cost_per_1k_output']:.5f}",
                "Max Tokens": f"{model_info['max_tokens']:,}",
                "Best For": model_info['best_for']
            })
        
        st.table(comparison_data)
        
        # Model selection
        col1, col2 = st.columns([1, 2])
        
        with col1:
            selected_model = st.selectbox(
                "Choose Model:",
                options=list(config.MODELS.keys()),
                format_func=lambda x: config.MODELS[x]['name'],
                help="Select the AI model for JSON extraction"
            )
        
        with col2:
            model_info = config.MODELS[selected_model]
            st.info(f"**{model_info['name']}**\n\n{model_info['description']}")
        
        # Detailed model info
        with st.expander(f"📋 {model_info['name']} Details"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Advantages:**")
                for pro in model_info['pros']:
                    st.write(f"✅ {pro}")
            
            with col2:
                st.write("**Considerations:**")
                for con in model_info['cons']:
                    st.write(f"⚠️ {con}")
            
            st.write(f"**Recommended for:** {model_info['best_for']}")
        
        return selected_model
