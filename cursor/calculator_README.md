# 🧮 Advanced Calculator with Streamlit

A feature-rich calculator application built with Python and Streamlit, offering both basic arithmetic and scientific functions.

## Features

### Basic Operations
- Addition, subtraction, multiplication, division
- Decimal point support
- Clear and delete functions
- Parentheses for grouping expressions

### Scientific Functions
- **Trigonometric functions**: sin, cos, tan (in degrees)
- **Square root**: √
- **Logarithms**: ln (natural), log (base 10)
- **Power function**: ^
- **Constants**: π (pi)

### Advanced Features
- Real-time expression display
- Calculation history (last 10 calculations)
- Error handling for invalid expressions
- Modern, responsive interface
- Session state management

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## How to Run

1. Start the calculator app:
```bash
streamlit run calculator_app.py
```

2. Open your browser and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

## How to Use

### Basic Calculations
1. Click the number buttons to input numbers
2. Use the operator buttons (+, -, ×, ÷) for arithmetic
3. Press = to calculate the result
4. Use C to clear or ⌫ to delete the last character

### Scientific Functions
- **sin, cos, tan**: Click the function button, then input the angle in degrees
- **√**: Click √, then input the number
- **ln/log**: Click the function, then input the number
- **^**: Input base number, click ^, then input exponent
- **π**: Click π to insert the value of pi

### Examples
- `2 + 3 × 4` = 14
- `sin(30)` = 0.5
- `√(16)` = 4
- `2^3` = 8
- `ln(2.718)` ≈ 1

## Files

- `calculator.py`: Contains the Calculator class with all mathematical operations
- `calculator_app.py`: Streamlit frontend application
- `requirements.txt`: Python dependencies
- `calculator_README.md`: This file

## Technical Details

- **Expression Evaluation**: Uses Python's `eval()` function with custom preprocessing
- **Function Handling**: Regular expressions to convert mathematical notation to Python syntax
- **History Management**: Session state to persist calculation history
- **Error Handling**: Graceful handling of invalid expressions

## Safety Features

- Input validation for mathematical expressions
- Error messages for invalid calculations
- Session state management for data persistence
- Responsive design for different screen sizes

Enjoy calculating! 🧮✨ 