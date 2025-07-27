"""
Utility functions for file operations.
"""
import json
from pathlib import Path
from typing import Dict, List, Any

def ensure_directory(directory_path: Path) -> Path:
    """
    Ensure that a directory exists.
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        Path to the directory
    """
    directory_path.mkdir(parents=True, exist_ok=True)
    return directory_path

def load_json_file(file_path: Path) -> Dict[str, Any]:
    """
    Load a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary containing the JSON data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_file(data: Dict[str, Any], file_path: Path) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to the JSON file
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def list_json_files(directory_path: Path) -> List[Path]:
    """
    List all JSON files in a directory.
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        List of paths to JSON files
    """
    return list(directory_path.glob("*.json"))

def load_all_json_files(directory_path: Path) -> List[Dict[str, Any]]:
    """
    Load all JSON files from a directory.
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        List of dictionaries containing the JSON data
    """
    json_files = list_json_files(directory_path)
    return [load_json_file(file_path) for file_path in json_files] 