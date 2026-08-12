import nbformat as nbf
import os

os.makedirs('notebooks', exist_ok=True)

nb1 = nbf.v4.new_notebook()
text1 = """# Results Comparison\n\nThis notebook pulls the final evaluation metrics from all models and creates comparison tables and plots."""
code1 = """import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport os\n\nresults_path = '../outputs/comparison_results.csv'\nif os.path.exists(results_path):\n    df = pd.read_csv(results_path)\n    display(df)\n    \n    # Plotting\n    df_melted = df.melt(id_vars=['model_name'], value_vars=['dsc', 'iou', 'f1', 'hd95'], var_name='Metric', value_name='Score')\n    plt.figure(figsize=(12, 6))\n    sns.barplot(data=df_melted, x='model_name', y='Score', hue='Metric')\n    plt.title('Model Performance Comparison')\n    plt.xticks(rotation=45)\n    plt.tight_layout()\n    plt.show()\nelse:\n    print('Run scripts/evaluate_all.py first to generate the comparison results.')"""
nb1['cells'] = [nbf.v4.new_markdown_cell(text1), nbf.v4.new_code_cell(code1)]
nbf.write(nb1, 'notebooks/04_results_comparison.ipynb')

nb2 = nbf.v4.new_notebook()
text2 = """# Inference Visualization\n\nThis notebook allows interactive visualization of predictions from all models."""
code2 = """import matplotlib.pyplot as plt\nimport os\nfrom PIL import Image\n\nmodels = ['unet', 'unet_plusplus', 'attention_unet', 'nnunet', 'segformer', 'unetformer', 'swin_unet']\nsample_idx = 0\n\nplt.figure(figsize=(20, 10))\nfor i, model in enumerate(models):\n    pred_path = f'../outputs/{model}/predictions/pred_{sample_idx}.png'\n    if os.path.exists(pred_path):\n        img = Image.open(pred_path)\n        plt.subplot(2, 4, i+1)\n        plt.imshow(img)\n        plt.title(model)\n        plt.axis('off')\nplt.tight_layout()\nplt.show()"""
nb2['cells'] = [nbf.v4.new_markdown_cell(text2), nbf.v4.new_code_cell(code2)]
nbf.write(nb2, 'notebooks/05_inference_visualization.ipynb')

print("Notebooks created successfully!")
