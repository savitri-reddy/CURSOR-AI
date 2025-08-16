# 🧮 Calculator Apps with Python

This repository contains two different calculator applications built with Python:

## 1. Advanced Calculator with Streamlit (Web App)

A feature-rich calculator with scientific functions and a modern web interface.

### Features:
- **Basic Operations**: Addition, subtraction, multiplication, division
- **Scientific Functions**: sin, cos, tan, √, ln, log, power (^), π
- **Advanced Features**: Calculation history, real-time display, error handling
- **Modern Interface**: Responsive web design with Streamlit

### Files:
- `calculator.py`: Core calculator logic
- `calculator_app.py`: Streamlit web interface
- `requirements.txt`: Dependencies

### How to Run:
```bash
pip install -r requirements.txt
streamlit run calculator_app.py
```

## 2. Simple Calculator with Tkinter (Desktop App)

A lightweight desktop calculator with a clean GUI interface.

### Features:
- **Basic Operations**: All standard arithmetic operations
- **Clean Interface**: Native desktop application
- **Easy to Use**: Simple button-based interface
- **No Dependencies**: Uses only Python's built-in tkinter

### Files:
- `simple_calculator.py`: Complete desktop calculator app

### How to Run:
```bash
python simple_calculator.py
```

## Installation

### For Streamlit Calculator:
```bash
pip install -r requirements.txt
```

### For Tkinter Calculator:
No additional installation required - uses Python's built-in tkinter library.

## Usage Examples

### Streamlit Calculator:
- `2 + 3 × 4` = 14
- `sin(30)` = 0.5
- `√(16)` = 4
- `2^3` = 8
- `ln(2.718)` ≈ 1

### Tkinter Calculator:
- Basic arithmetic operations
- Clear (C) and delete (⌫) functions
- Error handling for invalid expressions

## Features Comparison

| Feature | Streamlit Calculator | Tkinter Calculator |
|---------|-------------------|-------------------|
| Interface | Web-based | Desktop GUI |
| Scientific Functions | ✅ | ❌ |
| Calculation History | ✅ | ❌ |
| Dependencies | Streamlit, NumPy | None |
| Cross-platform | ✅ | ✅ |
| Real-time Updates | ✅ | ✅ |

## Technical Details

### Streamlit Calculator:
- **Backend**: Custom Calculator class with mathematical function handling
- **Frontend**: Streamlit web framework
- **Expression Evaluation**: Python eval() with custom preprocessing
- **History Management**: Session state persistence

### Tkinter Calculator:
- **Backend**: Simple expression evaluation
- **Frontend**: Native tkinter GUI
- **Expression Evaluation**: Python eval() with symbol replacement
- **Interface**: Grid-based button layout

## Safety Features

- Input validation for mathematical expressions
- Error handling for invalid calculations
- Graceful error messages
- Session state management (Streamlit version)

Choose the calculator that best fits your needs! 🧮✨ 