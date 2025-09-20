# iris_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import numpy as np

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_iris_data():
    """Load and prepare the Iris dataset"""
    try:
        # Load the iris dataset
        iris = load_iris()
        
        # Create DataFrame
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['species'] = iris.target
        df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        
        print("✅ Dataset loaded successfully!")
        return df
    
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

def explore_dataset(df):
    """Explore the dataset structure"""
    print("\n" + "="*50)
    print("DATASET EXPLORATION")
    print("="*50)
    
    # Display first few rows
    print("\n1. First 5 rows of the dataset:")
    print(df.head())
    
    # Dataset info
    print("\n2. Dataset information:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Data types and missing values
    print("\n3. Data types and missing values:")
    print(df.info())
    
    # Check for missing values
    print("\n4. Missing values summary:")
    missing_values = df.isnull().sum()
    print(missing_values)
    
    # Since Iris dataset is clean, we'll just confirm no missing values
    if missing_values.sum() == 0:
        print("✅ No missing values found!")
    else:
        # If there were missing values, we'd handle them here
        df = df.dropna()  # or use df.fillna() for specific strategies
        print("⚠️ Missing values handled")

def basic_analysis(df):
    """Perform basic data analysis"""
    print("\n" + "="*50)
    print("BASIC DATA ANALYSIS")
    print("="*50)
    
    # Basic statistics
    print("\n1. Basic statistics for numerical columns:")
    print(df.describe())
    
    # Group by species and compute means
    print("\n2. Mean values grouped by species:")
    species_means = df.groupby('species').mean()
    print(species_means)
    
    # Additional insights
    print("\n3. Interesting findings:")
    print(f"- Setosa has the smallest petal dimensions")
    print(f"- Virginica has the largest measurements overall")
    print(f"- Versicolor is intermediate in most measurements")
    
    return species_means

def create_visualizations(df):
    """Create the required visualizations"""
    print("\n" + "="*50)
    print("DATA VISUALIZATION")
    print("="*50)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Iris Dataset Analysis - Visualizations', fontsize=16, fontweight='bold')
    
    # 1. Line chart (simulating trends - we'll use index as pseudo-time)
    axes[0, 0].plot(df.index[:50], df['sepal length (cm)'][:50], label='Sepal Length', marker='o')
    axes[0, 0].plot(df.index[:50], df['petal length (cm)'][:50], label='Petal Length', marker='s')
    axes[0, 0].set_title('1. Line Chart: Measurements Trend (First 50 Samples)')
    axes[0, 0].set_xlabel('Sample Index')
    axes[0, 0].set_ylabel('Length (cm)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Bar chart - average sepal length by species
    avg_by_species = df.groupby('species')['sepal length (cm)'].mean()
    axes[0, 1].bar(avg_by_species.index, avg_by_species.values, 
                  color=['skyblue', 'lightgreen', 'lightcoral'])
    axes[0, 1].set_title('2. Bar Chart: Average Sepal Length by Species')
    axes[0, 1].set_xlabel('Species')
    axes[0, 1].set_ylabel('Average Sepal Length (cm)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(avg_by_species.values):
        axes[0, 1].text(i, v + 0.05, f'{v:.2f}', ha='center', va='bottom')
    
    # 3. Histogram - distribution of sepal length
    axes[1, 0].hist(df['sepal length (cm)'], bins=15, alpha=0.7, 
                   color='lightblue', edgecolor='black')
    axes[1, 0].set_title('3. Histogram: Distribution of Sepal Length')
    axes[1, 0].set_xlabel('Sepal Length (cm)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Scatter plot - sepal length vs petal length
    colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
    for species in df['species'].unique():
        species_data = df[df['species'] == species]
        axes[1, 1].scatter(species_data['sepal length (cm)'], 
                          species_data['petal length (cm)'],
                          label=species, alpha=0.7)
    
    axes[1, 1].set_title('4. Scatter Plot: Sepal Length vs Petal Length')
    axes[1, 1].set_xlabel('Sepal Length (cm)')
    axes[1, 1].set_ylabel('Petal Length (cm)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('iris_visualizations.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Visualizations created and saved as 'iris_visualizations.png'")

def main():
    """Main function to run the analysis"""
    print("Starting Iris Dataset Analysis...")
    
    # Task 1: Load and explore dataset
    df = load_iris_data()
    if df is None:
        return
    
    explore_dataset(df)
    
    # Task 2: Basic data analysis
    species_means = basic_analysis(df)
    
    # Task 3: Data visualization
    create_visualizations(df)
    
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE!")
    print("="*50)
    print("Summary of findings:")
    print("- Three distinct species clusters visible in scatter plot")
    print("- Setosa has significantly smaller petals than other species")
    print("- Virginica has the largest sepals on average")
    print("- Measurements show clear separation between species")

if __name__ == "__main__":
    main()
