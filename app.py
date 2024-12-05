from flask import Flask, request, render_template
import re
import numpy as np
from simplex.simplex_method import simplex_method

app = Flask(__name__)

def parse_problem(problem_text):
    try:
        print("Parsing the problem...")
        objective_line = re.search(r'P\s*=\s*(.+)', problem_text)
        if not objective_line:
            raise ValueError("Objective function not found")
        
        # Extracting variables and their coefficients
        variables = re.findall(r'([\d\.]*)([a-z])', objective_line.group(1))
        print("Objective function variables:", variables)

        c = []
        var_map = {}
        for i, (coef, var) in enumerate(variables):
            c.append(float(coef) if coef else 1.0)
            var_map[var] = i
        print("Parsed objective coefficients:", c)

        # Extracting constraints
        constraints_lines = re.findall(r'(.+?)≤\s*(\d+)', problem_text)
        print("Parsed constraints:", constraints_lines)

        A = []
        b = []
        for line, rhs in constraints_lines:
            constraint_vars = re.findall(r'([\d\.]*)([a-z])', line)
            row = [0.0] * len(c)
            for coef, var in constraint_vars:
                row[var_map[var]] = float(coef) if coef else 1.0
            A.append(row)
            b.append(float(rhs))
        
        print("Constraint coefficients (A):", A)
        print("RHS values (b):", b)

        return np.array(c), np.array(A), np.array(b)
    
    except Exception as e:
        print(f"Error parsing problem: {e}")
        raise ValueError("Error parsing problem. Please check your input format.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/solve', methods=['POST'])
def solve():
    print("Solve route hit!")  # Debugging line to check if route is being hit
    
    try:
        problem_text = request.form['problem']
        print(f"Received problem text:\n{problem_text}")  # Log input
    except KeyError as e:
        return f"Error: Missing key in form data. {e}", 400

    try:
        c, A, b = parse_problem(problem_text)
    except Exception as e:
        return f"Error parsing the problem: {e}", 400

    try:
        solution, max_value = simplex_method(c, A, b)
    except Exception as e:
        return f"Error solving the problem: {e}", 500

    return render_template('result.html', problem=problem_text, solution=solution, max_value=max_value)

if __name__ == '__main__':
    app.run(debug=True)
