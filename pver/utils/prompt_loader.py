"""
Prompt Loader Utility for loading prompt templates from YAML files.
"""

import os
import yaml
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class PromptTemplate:
    """Simple prompt template holder."""
    template: str
    schema: str = ""
    fusion_schema: str = ""


def load_prompts(cfg) -> Dict[str, PromptTemplate]:
    """
    Load prompt templates based on config.
    
    Args:
        cfg: OmegaConf config with prompt section
    
    Returns:
        Dict of prompt name -> PromptTemplate
    """
    prompts = {}
    
    # Get prompt config
    prompt_cfg = cfg.get("prompt", {})
    
    # Base directory for prompts
    # Try relative to project root, then absolute
    prompt_base = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                               "configs", "prompts")
    
    # Load each prompt file
    prompt_files = {
        "evidence": prompt_cfg.get("evidence_file", "evidence_v1.yaml"),
        "extract": prompt_cfg.get("extract_file", "extract_v1.yaml"),
        "verify": prompt_cfg.get("verify_file", "verify_v1.yaml"),
        "nav": prompt_cfg.get("nav_file", "nav_attr_v1.yaml"),
        "category": prompt_cfg.get("category_file", "category_v1.yaml"),
        "fusion": prompt_cfg.get("fusion_file", None)
    }
    
    for name, filename in prompt_files.items():
        if not filename:
            continue
            
        filepath = os.path.join(prompt_base, filename)
        
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = yaml.safe_load(f)
                
                prompts[name] = PromptTemplate(
                    template=content.get("template", ""),
                    schema=content.get("schema", ""),
                    fusion_schema=content.get("fusion_schema", "")
                )
            except Exception as e:
                print(f"[PromptLoader] Failed to load {filename}: {e}")
        else:
            print(f"[PromptLoader] Prompt file not found: {filepath}")
    
    return prompts
