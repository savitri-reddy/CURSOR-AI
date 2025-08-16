import math
import re

class Calculator:
    def __init__(self):
        self.history = []
        self.current_expression = ""
        self.result = 0
        
    def add_to_expression(self, value):
        """Add a value to the current expression"""
        self.current_expression += str(value)
        
    def clear(self):
        """Clear the current expression"""
        self.current_expression = ""
        self.result = 0
        
    def delete_last(self):
        """Delete the last character from the expression"""
        self.current_expression = self.current_expression[:-1]
        
    def calculate(self):
        """Calculate the result of the current expression"""
        try:
            # Replace mathematical symbols with Python operators
            expression = self.current_expression.replace('×', '*').replace('÷', '/')
            
            # Handle special functions
            expression = self._handle_functions(expression)
            
            # Evaluate the expression
            result = eval(expression)
            
            # Add to history
            self.history.append({
                'expression': self.current_expression,
                'result': result
            })
            
            self.result = result
            self.current_expression = str(result)
            return result
            
        except Exception as e:
            return "Error"
    
    def _handle_functions(self, expression):
        """Handle mathematical functions like sqrt, sin, cos, etc."""
        # Square root
        expression = re.sub(r'sqrt\(([^)]+)\)', r'math.sqrt(\1)', expression)
        
        # Trigonometric functions
        expression = re.sub(r'sin\(([^)]+)\)', r'math.sin(math.radians(\1))', expression)
        expression = re.sub(r'cos\(([^)]+)\)', r'math.cos(math.radians(\1))', expression)
        expression = re.sub(r'tan\(([^)]+)\)', r'math.tan(math.radians(\1))', expression)
        
        # Power
        expression = re.sub(r'(\d+)\^(\d+)', r'\1**\2', expression)
        
        # Natural logarithm
        expression = re.sub(r'ln\(([^)]+)\)', r'math.log(\1)', expression)
        
        # Common logarithm
        expression = re.sub(r'log\(([^)]+)\)', r'math.log10(\1)', expression)
        
        return expression
    
    def get_history(self):
        """Get calculation history"""
        return self.history
    
    def clear_history(self):
        """Clear calculation history"""
        self.history = [] 