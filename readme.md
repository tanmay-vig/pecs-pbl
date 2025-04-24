# Bayesian Optimization with Branin Function

This project implements Bayesian Optimization using the Branin function as a benchmark. The optimization process leverages Gaussian Process Regression and an Expected Improvement acquisition function to iteratively find the minimum of the Branin function.

## Project Structure

-   **`assignment3.py`**: The main script that implements the Bayesian Optimization process.
-   **`branin_sample_points_50.csv`**: A CSV file containing initial sample points for the Branin function.
-   **`requirements.txt`**: A list of Python dependencies required to run the project.

## Features

-   **Branin Function**: A well-known benchmark function for optimization problems.
-   **Bayesian Optimization**: Iterative optimization using Gaussian Process Regression.
-   **Visualization**: Plots the sampled points and their corresponding Branin values.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/tanmay-vig/pecs-pbl
    ```
2. Install the required dependencies:

```code
pip install -r requirements.txt
```

Usage
Ensure the branin_sample_points_50.csv file is in the same directory as assignment3.py.

Run the script:

```
python assignment3.py
```

The script will output the best value found during the optimization process and display a scatter plot of the sampled points.

## File Descriptions
assignment3.py:
<ol>
<li>Implements the Branin function.</li>
<li>Defines the Expected Improvement acquisition function.</li>
<li>Proposes new sampling points using the L-BFGS-B optimization method.
</li>
<li>Loads initial sample points from the CSV file and performs Bayesian Optimization.</li>
</ol>

branin_sample_points_50.csv:
<ol>
<li>Contains 50 initial sample points (x1, x2) and their corresponding Branin function values.</li>
</ol>

requirements.txt: <br>
Lists the Python libraries required for the project:
```code
scikit-learn
scipy
numpy
matplotlib
pandas
```

## Output
<ul>
<li>The script prints the best value found during each iteration of the optimization process.</li>
<li>
A scatter plot is displayed, showing the sampled points and their Branin values.</li>
</ul>

## License

This project is licensed under the MIT License. See the LICENSE file for details.

# Acknowledgments

<ul>
<li>The Branin function is a standard benchmark function for optimization problems.</li>
<li>
The implementation uses concepts from Gaussian Process Regression and Bayesian Optimization.</li>
</ul>

## Code by:
<ol>
<li>Tanmay Vig 21803003</li>
<li>Ansh Mishra 21803011</li>
<li>Sanat Walia 21803012</li>
<li>Vivek Shaurya 21803013</li>
</ol>
