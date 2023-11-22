from hmac import new
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


################################################################################################################################################
# This first section of code enters in the data from the test sequences for visualization. There is an entry for each sequence in the primate
# and the rodent tests, and it includes the accesnion number, their log-viterbi score, which type of sequence, and what Hidden Markov Model they 
# were tested on. I then used pandas to enter this data into a dataframe.
################################################################################################################################################

data = {
    'Sequence': ['XM_003799497.3', 'XM_030795717.1', 
                'XM_030942663.1', 'XM_025367945.1', 'XM_028839202.1', 'XM_011735912.2',
                 'XM_003799497.3', 'XM_030795717.1', 'XM_030942663.1', 
                 'XM_025367945.1', 'XM_028839202.1', 'XM_011735912.2',
                 'XM_031385595.1', 'XM_027945926.1', 'XM_035457778.1', 
                 'XM_027430875.2', 'XM_034499372.1', 'XM_010623359.2',
                 'XM_031385595.1', 'XM_027945926.1', 'XM_035457778.1', 
                 'XM_027430875.2', 'XM_034499372.1', 'XM_010623359.2'],
    'Log-Viterbi Score': [-282.045, -99.265, -104.445, -100.027, -98.55, -96.59, 
                        -419.95, -438.59, -423.48, -432.10, -434.66, -433.311,
                          -419.955, -438.59, -423.48, -432.10, -434.66, -433.31, 
                          -219.03, -480.08, -292.97, -262.15, -202.13, -663.44],
    'Sequence Type': ['Primate', 'Primate', 'Primate', 'Primate', 'Primate', 'Primate', 
                        'Rodent', 'Rodent', 'Rodent', 'Rodent', 'Rodent', 'Rodent',
                      'Primate', 'Primate', 'Primate', 'Primate', 'Primate', 'Primate', 
                      'Rodent', 'Rodent', 'Rodent', 'Rodent', 'Rodent', 'Rodent'],
    'HMM': ['Primate', 'Primate', 'Primate', 'Primate', 'Primate', 'Primate', 'Primate', 
            'Primate', 'Primate', 'Primate', 'Primate', 'Primate',
            'Rodent', 'Rodent', 'Rodent', 'Rodent', 'Rodent', 'Rodent', 
            'Rodent', 'Rodent', 'Rodent', 'Rodent', 'Rodent', 'Rodent']
}

# Create DataFrame
df = pd.DataFrame(data)

################################################################################################################################################
# The first plot is a categorical swarm plot showing the log viterbi scores and which Hidden Markov Model they were tested on. The are also 
# seperated by color as shown in the legend.
################################################################################################################################################


# Create a catplot to show the sequence types and log viterbi scores of the tests
plt.figure(figsize=(12, 8))
sns.catplot(x='Sequence Type', y='Log-Viterbi Score', hue='HMM', kind='swarm', data=df)
plt.title('Cat Plot of Log-Viterbi Scores')
plt.xlabel('Sequence Type')
plt.ylabel('Log-Viterbi Score')
plt.show()


#rotate 
plt.xticks(rotation=45, ha='right') 
plt.show()



################################################################################################################################################
# This function calcualtes the interquartile range
################################################################################################################################################
def calc_interquartile_range(x):
    return x.quantile(0.75) - x.quantile(0.25)


################################################################################################################################################
# This plot uses a boxplot to represent the data. Using different plots to view the data can give us new insights
################################################################################################################################################
plt.figure(figsize=(10, 6))
plt.figure(figsize=(10, 6))
sns.boxplot(x='Log-Viterbi Score', 
            y='Sequence', 
            hue='HMM', 
            data=df)
plt.title('Box Plot of Log-Viterbi Scores')
#labeling the plot
plt.xlabel('Sequence Type')
plt.ylabel('Log-Viterbi Score')
plt.legend(title='HMM')
plt.show()

################################################################################################################################################
# The next plot is a violin plot. Again data represented in differnt ways 
################################################################################################################################################
plt.figure(figsize=(10, 6))
sns.violinplot(x='Sequence Type', y='Log-Viterbi Score', hue='HMM', data=df, split=True)
plt.title('Violin Plot of Log-Viterbi Scores')
plt.xlabel('Sequence Type')
plt.ylabel('Log-Viterbi Score')
plt.legend(title='HMM')
plt.show()

################################################################################################################################################
# The next plot is a violin plot. Again data represented in differnt ways 
################################################################################################################################################

new_values = df.groupby(['Sequence Type', 'HMM']).agg({
    'Log-Viterbi Score': [
        'mean',       # Mean of Log-Viterbi Scores
        'median',     # Median of Log-Viterbi Scores
        'std',        # Standard Deviation of Log-Viterbi Scores
        lambda x: x.quantile(0.25),   # 25th Percentile of Log-Viterbi Scores
        lambda x: x.quantile(0.75)    # 75th Percentile of Log-Viterbi Scores
    ]
}).reset_index()

# Rename the columns 
new_values.columns = ['Sequence Type', 'HMM', 'Mean', 'Median', 'Std Dev', '25th Percentile', '75th Percentile']

# Print the summary table
print("Summary Table:")
print(new_values)

################################################################################################################################################
# Box plot showing the standard deviation of the Log-Viterbi Scores and a Bar plot representing the standard deviation of the scores
################################################################################################################################################

plt.figure(figsize=(10, 7))
sns.boxplot(x='Sequence Type', y='Log-Viterbi Score', hue='HMM', data=df)
plt.title('Box Plot of Log-Viterbi Scores - Standard Deviation')
plt.show()

# Create a bar plot to compare standard deviations
plt.figure(figsize=(12, 8))
sns.barplot(x='HMM', y='Std Dev', hue='Sequence Type', data=new_values)
plt.title('Bar Plot of Standard Deviations')
plt.show()


################################################################################################################################################
# Box plot showing the mean of the Log-Viterbi scores
################################################################################################################################################

plt.figure(figsize=(12, 8))
sns.barplot(x='Sequence Type', y='Mean', hue='HMM', data=new_values, ci=None)
plt.title('Bar Plot of Mean Log-Viterbi Scores')
#label the plots
plt.xlabel('Sequence Type')
plt.ylabel('Mean Log-Viterbi Score')
plt.legend(title='HMM')
plt.show()

################################################################################################################################################
# This chunk of code displays the new_values table in one plot. First we use melt to convert the table to long format. After that we use
# x, y, hue, col, and kind to show the different types of data such as mean, standard deviation, 25th percentile, and 75th percentile.
################################################################################################################################################

melted_summary = new_values.melt(id_vars=['Sequence Type', 'HMM'],
                                     var_name='Statistic',
                                     value_name='Value')

# Create a bar plot for mean, median, standard deviation, 25th, and 75th percentiles
plt.figure(figsize=(14, 8))
sns.catplot(x='HMM', y='Value', hue='Statistic', col='Sequence Type', kind='bar', data=melted_summary)
plt.suptitle('Bar Plot of Log-Viterbi Scores Statistics')
plt.show()


################################################################################################################################################
# THis plot is a double plot and shows the distribution of Log-Viterbi Scores using the Stack feature. 
################################################################################################################################################

plt.figure(figsize=(10, 8))
#use stack
sns.histplot(df, x='Log-Viterbi Score', hue='HMM', multiple='stack', kde=True)
#labels the graph axis and plot title
plt.title('Distribution of Log-Viterbi Scores')
plt.xlabel('Log-Viterbi Score')
plt.ylabel('Frequency')
plt.legend(title='HMM')
plt.show()

################################################################################################################################################
# This code takes the data frame and converts it into a format readable for heatmaps by using the pivot table function. For this heat map, we are 
# using the log-Viterbi mean scores. 
################################################################################################################################################

heat_map = df[['Log-Viterbi Score', 
                'Sequence Type', 'HMM']].pivot_table(index='Sequence Type', columns='HMM', values='Log-Viterbi Score', aggfunc='mean')
sns.heatmap(heat_map, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap of Log-Viterbi Mean Scroress')
plt.show()