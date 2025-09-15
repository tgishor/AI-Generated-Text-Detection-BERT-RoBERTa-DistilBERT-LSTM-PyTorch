import json
import sys

def fix_notebook_widgets(input_file, output_file):
    """Fix notebook by removing problematic widgets metadata"""
    
    with open(input_file, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Remove widgets from metadata if present
    if 'metadata' in notebook:
        if 'widgets' in notebook['metadata']:
            print("Removing widgets from notebook metadata...")
            del notebook['metadata']['widgets']
    
    # Clean up cell metadata - remove widget references
    for cell in notebook.get('cells', []):
        if 'metadata' in cell:
            # Remove colab widget references
            if 'colab' in cell['metadata']:
                if 'referenced_widgets' in cell['metadata']['colab']:
                    print("Removing referenced_widgets from cell...")
                    del cell['metadata']['colab']['referenced_widgets']
            
            # Remove any widget-related metadata
            if 'widgets' in cell['metadata']:
                del cell['metadata']['widgets']
    
    # Write the cleaned notebook
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f"✅ Fixed notebook saved as: {output_file}")
    print("The notebook should now render properly on GitHub!")

if __name__ == "__main__":
    input_file = "AI-Generated-Text-Detection-BERT-RoBERTa-DistilBERT-LSTM-PyTorch.ipynb"
    output_file = "AI-Generated-Text-Detection-BERT-RoBERTa-DistilBERT-LSTM-PyTorch.ipynb"
    
    fix_notebook_widgets(input_file, output_file)
